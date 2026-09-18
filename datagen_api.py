"""
datagen_with_api.py

Identical to datagen.py for Stage 0 (activation extraction, still needs the
local GPU + target model, since that's where the hidden states come from).

Stage 1 (teacher explanation generation) is replaced: instead of loading a
small local instruct model that ignores formatting instructions, this calls
a real API. Output schema and file format are IDENTICAL to datagen.py's --
same columns, same parquet layout -- so anything downstream doesn't need to
know which script (or which provider) produced the data.

Supports three providers, all through the same OpenAI-compatible
chat-completions shape -- switch with config.yaml's `api_teacher.provider`:

  - "groq"       (default, recommended): no credit card ever required.
                 Free tier is generous and per-model: ~1,000 requests/day
                 for large models (gpt-oss-120b, llama-3.3-70b-versatile),
                 up to ~14,400/day for smaller ones (gpt-oss-20b). Fastest
                 of the three (LPU hardware). Get a key at console.groq.com.
  - "google"     Google AI Studio / Gemini. No card required. Flash models
                 get ~15 requests/min, ~1,500/day for free. Get a key at
                 aistudio.google.com/apikey.
  - "openrouter" Aggregates many providers' `:free` models behind one key,
                 but the free tier is a single shared pool: 20 req/min,
                 50/day (or 1,000/day after a one-time $10 credit purchase
                 that never expires). Get a key at openrouter.ai/keys.

IMPORTANT: free-tier request/token limits shift over time and rotate
per-model without notice on all three platforms. The numbers above and the
defaults in config.yaml are a reasonable starting point as of when this was
written -- if you hit unexpected 429s, check the provider's current limits
(console.groq.com/docs/rate-limits, ai.google.dev/gemini-api/docs/rate-limits,
openrouter.ai/docs/api-reference/limits) and adjust config.yaml.

Setup:
    1. Pick a provider, get a free key (no card needed for any of the three).
    2. On Kaggle: Add-ons -> Secrets -> add the matching env var name below,
       then attach it to the notebook. Locally: `export GROQ_API_KEY=...`
       (or GOOGLE_API_KEY / OPENROUTER_API_KEY, matching your chosen provider).
    3. Set `api_teacher.provider` in config.yaml.

Usage:
    python datagen_with_api.py
    # Safe to interrupt and rerun -- already-generated rows are skipped.
"""

from pathlib import Path
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from common import load_config
from datagen import (
    extract_dataset,
    make_warmstart_schema,
    parse_explanation_and_emotion,
    load_teacher_prompt_template,
)


# Per-provider connection details. `needs_free_filter` is True only for
# OpenRouter, which mixes free and paid models behind one endpoint -- Groq
# and Google's endpoints are inherently the free tier (no billing enabled),
# so any model returned by their /models listing is fair game.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "signup_url": "https://console.groq.com/keys",
        "default_candidates": [
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
        ],
        "default_requests_per_minute": 28,
        "default_daily_request_budget": 1000,
        "needs_free_filter": False,
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "signup_url": "https://aistudio.google.com/apikey",
        "default_candidates": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ],
        "default_requests_per_minute": 14,
        "default_daily_request_budget": 1500,
        "needs_free_filter": False,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "signup_url": "https://openrouter.ai/keys",
        "default_candidates": [
            "openai/gpt-oss-120b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "z-ai/glm-4.6:free",
            "qwen/qwen3-235b-a22b:free",
        ],
        "default_requests_per_minute": 18,
        "default_daily_request_budget": 50,
        "needs_free_filter": True,
    },
}


def get_api_key(provider: str) -> str:
    env_var = PROVIDERS[provider]["api_key_env"]
    key = os.environ.get(env_var)
    if key:
        return key

    # Kaggle Secrets fallback, so the notebook doesn't need to manually
    # os.environ[...] = ... before importing this module.
    # You can replace this with your own secret management if you want to run locally without env vars.
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret(env_var)
        if key:
            os.environ[env_var] = key
            return key
    except Exception:
        pass

    raise RuntimeError(
        f"{env_var} not found. On Kaggle: Add-ons -> Secrets -> add a "
        f"secret named {env_var} and attach it to this notebook. "
        f"Locally: export {env_var}=... "
        f"(get a free key at {PROVIDERS[provider]['signup_url']})"
    )


def fetch_model_ids(provider: str, base_url: str, api_key: str) -> set:
    """
    Hit the provider's /models listing to see what's actually available
    right now. For OpenRouter this also filters down to models priced at
    $0 -- for Groq/Google, everything listed is free-tier by default.
    """
    resp = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()

    ids = set()
    for model in resp.json().get("data", []):
        model_id = model.get("id", "")
        # Google's native listing sometimes prefixes with "models/".
        model_id = model_id.removeprefix("models/")

        if PROVIDERS[provider]["needs_free_filter"]:
            pricing = model.get("pricing", {})
            is_free = model_id.endswith(":free") or (
                str(pricing.get("prompt", "1")) in ("0", "0.0")
                and str(pricing.get("completion", "1")) in ("0", "0.0")
            )
            if not is_free:
                continue

        ids.add(model_id)

    return ids


def resolve_model(provider: str, base_url: str, api_key: str, candidates: list) -> str:
    """
    Pick the first candidate that's currently available (and, for
    OpenRouter, free). Raises with a clear message rather than a confusing
    404 mid-run if none of the candidates check out.
    """
    try:
        available = fetch_model_ids(provider, base_url, api_key)
    except requests.RequestException as e:
        print(f"WARNING: could not verify model availability ({e}); "
              f"proceeding with the first candidate unverified.")
        return candidates[0]

    for model_id in candidates:
        if model_id in available:
            print(f"Using teacher model: {model_id} (via {provider})")
            return model_id

    raise RuntimeError(
        f"None of the configured candidate models are currently available "
        f"on {provider}: {candidates}. Free-tier model rosters shift over "
        f"time -- check the provider's model list and update config.yaml's "
        f"api_teacher.{provider}.model / fallback_models."
    )


class RateLimiter:
    """Simple client-side throttle so we don't even attempt to exceed the
    provider's requests-per-minute cap."""

    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def call_chat_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
):
    """
    One OpenAI-compatible chat-completion call with retry/backoff. Works
    unchanged against Groq, Google AI Studio, and OpenRouter -- all three
    speak the same request/response shape. Returns the raw text on success,
    or None if every retry was exhausted (caller decides how to handle a
    failed row -- we don't want one bad row to kill a multi-hour run).
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    backoff = 2.0

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
        except requests.RequestException as e:
            print(f"  request error ({e}), retrying in {backoff:.0f}s...")
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                print(f"  malformed response body: {data}")
                return None

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else backoff
            print(f"  429 rate-limited, waiting {wait_s:.0f}s...")
            time.sleep(wait_s)
            backoff *= 2
            continue

        if resp.status_code in (500, 502, 503, 504):
            print(f"  server error {resp.status_code}, retrying in {backoff:.0f}s...")
            time.sleep(backoff)
            backoff *= 2
            continue

        # 4xx other than 429 (bad request, model gone, etc.) -- not worth
        # retrying, surface it and move on.
        print(f"  non-retryable error {resp.status_code}: {resp.text[:300]}")
        return None

    print("  giving up after max retries")
    return None


def load_existing_output(output_path: Path):
    """
    Resume support: if a previous run already wrote some rows, load them so
    we can skip re-generating (and re-spending quota on) the same rows.
    Keyed on (doc_id, n_raw_tokens) since that uniquely identifies a sampled
    position within a document.
    """
    if not output_path.exists():
        return None, set()

    existing = pq.read_table(output_path).to_pandas()
    done_keys = set(zip(existing["doc_id"], existing["n_raw_tokens"]))
    print(f"Resuming: found {len(existing)} already-generated rows in {output_path}")
    return existing, done_keys


def generate_explanations_via_api(config, base_path):
    """
    Stage 1 (API version): same job as datagen.py's generate_explanations,
    but the teacher call goes to a hosted API (Groq / Google / OpenRouter)
    instead of a local model.
    """
    datagen_cfg = config["datagen"]
    api_teacher_cfg = config.get("api_teacher", {})

    provider = api_teacher_cfg.get("provider", "groq")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown api_teacher.provider '{provider}'. "
            f"Choose one of: {list(PROVIDERS)}"
        )

    provider_defaults = PROVIDERS[provider]
    provider_cfg = api_teacher_cfg.get(provider, {})

    base_url = provider_defaults["base_url"]
    api_key = get_api_key(provider)

    candidates = [
        provider_cfg.get("model", provider_defaults["default_candidates"][0])
    ] + provider_cfg.get(
        "fallback_models", provider_defaults["default_candidates"][1:]
    )
    model = resolve_model(provider, base_url, api_key, candidates)

    requests_per_minute = provider_cfg.get(
        "requests_per_minute", provider_defaults["default_requests_per_minute"]
    )
    daily_request_budget = provider_cfg.get(
        "daily_request_budget", provider_defaults["default_daily_request_budget"]
    )
    max_retries = api_teacher_cfg.get("max_retries", 5)
    timeout_seconds = api_teacher_cfg.get("timeout_seconds", 60)
    checkpoint_every = api_teacher_cfg.get("checkpoint_every", 50)
    max_rows = datagen_cfg.get("api_max_rows")  # None = process everything

    teacher_max_output_tokens = datagen_cfg.get("teacher_max_output_tokens", 200)
    template = load_teacher_prompt_template(config)

    base_table = pq.read_table(base_path)
    d_model = base_table.schema.field("activation_vector").type.list_size
    warmstart_schema = make_warmstart_schema(d_model)

    output_path = Path(datagen_cfg["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df, done_keys = load_existing_output(output_path)

    all_rows = base_table.to_pylist()
    pending_rows = [
        r for r in all_rows
        if (r["doc_id"], r["n_raw_tokens"]) not in done_keys
    ]

    if max_rows is not None:
        remaining_budget = max(0, max_rows - len(done_keys))
        pending_rows = pending_rows[:remaining_budget]

    print(f"Provider                        : {provider}")
    print(f"Total activation rows available : {len(all_rows)}")
    print(f"Already done (resumed)          : {len(done_keys)}")
    print(f"Pending this run                : {len(pending_rows)}")
    print(f"Daily request budget            : {daily_request_budget} "
          f"(check the provider's current limits for this model -- these shift over time)")

    if not pending_rows:
        print("Nothing to do -- output already complete.")
        return output_path

    rate_limiter = RateLimiter(requests_per_minute)
    new_records = []
    calls_made = 0
    failed = 0

    progress = tqdm(total=len(pending_rows), desc=f"Stage 1 ({provider}): generating explanations")

    try:
        for row in pending_rows:
            if calls_made >= daily_request_budget:
                print(
                    f"\nHit the daily request budget ({daily_request_budget}). "
                    "Stopping here -- rerun this script tomorrow (or after "
                    "your quota resets) to continue; already-done rows will "
                    "be skipped automatically."
                )
                break

            prompt = template.format(context=row["detokenized_text_truncated"])

            rate_limiter.wait()
            raw_output = call_chat_api(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=teacher_max_output_tokens,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
            calls_made += 1

            if raw_output is None:
                failed += 1
                explanation, emotion = "", ""
            else:
                explanation, emotion = parse_explanation_and_emotion(raw_output)

            record = dict(row)
            record["explanation"] = explanation
            record["emotion"] = emotion
            new_records.append(record)

            progress.update(1)
            progress.set_postfix(failed=failed, calls=calls_made)

            if len(new_records) % checkpoint_every == 0:
                _write_checkpoint(existing_df, new_records, warmstart_schema, output_path)

    finally:
        progress.close()

    final_path = _write_checkpoint(existing_df, new_records, warmstart_schema, output_path)

    print()
    print("Stage 1 (API) complete for this run.")
    print(f"Provider                : {provider}")
    print(f"Model                   : {model}")
    print(f"API calls made this run : {calls_made}")
    print(f"Failed rows (empty explanation, worth re-running) : {failed}")
    print(f"Output                  : {final_path}")

    return final_path


def _write_checkpoint(existing_df, new_records, schema, output_path):
    """Overwrite output_path with existing + newly-generated rows. Cheap
    enough at this scale (thousands of rows) to just rewrite the whole file
    each checkpoint rather than managing append-only shards."""
    import pandas as pd

    new_df = pd.DataFrame(new_records, columns=schema.names)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df

    table = pa.Table.from_pandas(combined, schema=schema, preserve_index=False)
    pq.write_table(table, output_path)
    return output_path


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Stage 0 (activation extraction). "
            "CPU fallback is disabled by design."
        )

    config = load_config("config.yaml")

    base_path = extract_dataset(config)               # Stage 0: local GPU, unchanged
    generate_explanations_via_api(config, base_path)   # Stage 1: hosted API