from pathlib import Path
import hashlib
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from common import load_config, load_model_info, get_dtype


MIN_POSITION = 50


def resolve_decoder_layers(model):
    """
    Find the decoder block list for common Hugging Face
    decoder-only architectures.

    We fail loudly for unsupported architectures rather than
    silently extracting from the wrong layer.
    """
    candidates = [
        "model.layers",
        "language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "decoder.layers",
    ]

    for path in candidates:
        obj = model

        try:
            for part in path.split("."):
                obj = getattr(obj, part)

            if isinstance(obj, torch.nn.ModuleList):
                return obj

        except AttributeError:
            continue

    raise ValueError(
        "Could not locate the decoder transformer layers. "
        "This model architecture is not currently supported."
    )


class ActivationExtractor:
    """
    Extract raw layer-K activations from a batch of documents.

    This follows the authors' Stage-0 idea:
        document -> target model -> layer K -> hidden states

    A forward hook captures ONLY the requested decoder layer,
    avoiding output_hidden_states=True and therefore avoiding
    storing every layer's activations.
    """

    def __init__(
        self,
        model_name,
        layer_index,
        dtype,
        max_length,
        batch_size=4,
        trust_remote_code=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # The original extractor requires right padding/truncation
        # so token positions remain aligned with the original text.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).eval()

        self.model.to("cuda")

        self.layers = resolve_decoder_layers(self.model)

        if not 0 <= layer_index < len(self.layers):
            raise ValueError(
                f"Extraction layer {layer_index} is invalid for "
                f"model with {len(self.layers)} decoder layers."
            )

        self.layer_index = layer_index
        self.max_length = max_length
        self.batch_size = batch_size

        self.d_model = self.model.config.hidden_size

        self._captured = None

        self.hook_handle = self.layers[layer_index].register_forward_hook(
            self._hook
        )

    def _hook(self, _module, _inputs, output):
        """
        Capture the output of decoder block K.

        This corresponds to the residual stream after block K,
        which the authors identify with hidden_states[K+1] in
        Hugging Face's hidden-state indexing.
        """
        hidden = output[0] if isinstance(output, tuple) else output

        self._captured = hidden.detach().float().cpu()

    @torch.no_grad()
    def extract(self, texts):
        """
        Extract layer-K hidden states for a list of documents.

        Returns:
            list of dictionaries containing:
                hidden_states: [sequence_length, d_model]
                token_ids: unpadded token IDs
        """
        results = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")

            self._captured = None

            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            if self._captured is None:
                raise RuntimeError(
                    f"Forward hook for layer {self.layer_index} "
                    "did not fire."
                )

            lengths = attention_mask.sum(dim=1).cpu().tolist()

            for i, seq_len in enumerate(lengths):
                results.append(
                    {
                        "hidden_states": self._captured[i, :seq_len].clone(),
                        "token_ids": (
                            input_ids[i, :seq_len]
                            .cpu()
                            .tolist()
                        ),
                    }
                )

        return results

    def close(self):
        self.hook_handle.remove()


def sample_positions(
    token_ids,
    n_positions,
    special_token_ids,
    doc_id,
    seed,
):
    """
    Reproduce the authors' deterministic per-document position sampling.

    Only positions >= 50 are eligible, avoiding extremely early tokens.

    The RNG is seeded from:
        seed + document ID

    so the same document receives the same positions across runs.
    """
    rng_seed = hashlib.sha256(
        f"{seed}|{doc_id}".encode("utf-8")
    ).digest()

    rng = random.Random(rng_seed)

    candidates = [
        i
        for i, token_id in enumerate(token_ids)
        if i >= MIN_POSITION
        and token_id not in special_token_ids
    ]

    if not candidates:
        return []

    k = min(n_positions, len(candidates))

    return rng.sample(candidates, k=k)


def make_base_schema(d_model):
    """
    Stage-0 (raw activations only) Parquet schema.
    """
    return pa.schema(
        [
            ("n_raw_tokens", pa.int64()),
            ("detokenized_text_truncated", pa.string()),
            (
                "activation_vector",
                pa.list_(pa.float32(), d_model),
            ),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
        ]
    )


def make_warmstart_schema(d_model):
    """
    Stage-1 (final warmstart) Parquet schema: Stage-0 columns
    plus the teacher-generated natural-language explanation that
    the AV model is warm-started to imitate, and a separate
    emotion/affect field describing the tone of the text itself
    at the probed position.
    """
    return pa.schema(
        [
            ("n_raw_tokens", pa.int64()),
            ("detokenized_text_truncated", pa.string()),
            (
                "activation_vector",
                pa.list_(pa.float32(), d_model),
            ),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
            ("explanation", pa.string()),
            ("emotion", pa.string()),
        ]
    )

_ANALYSIS_RE = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", flags=re.IGNORECASE | re.DOTALL)
_EMOTION_LINE_RE = re.compile(
    r"^.*\b(emotion|emotional|affect|tone|mood|sentiment)\b.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

def parse_explanation_and_emotion(raw_text: str):
    match = _ANALYSIS_RE.search(raw_text)
    content = match.group(1).strip() if match else raw_text.strip()

    emotion_match = _EMOTION_LINE_RE.search(content)
    emotion = emotion_match.group(0).strip() if emotion_match else ""

    return content, emotion


def load_corpus(datagen_cfg):
    """
    Load the requested corpus in streaming mode.

    Streaming is our Kaggle memory/download adaptation.
    """
    dataset_name = datagen_cfg["dataset"]
    dataset_config = datagen_cfg.get("dataset_config")
    split = datagen_cfg.get("split", "train")

    return load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
    )


def extract_dataset(config):
    """
    Stage 0:

        corpus
          ↓
        tokenize
          ↓
        target model
          ↓
        layer K
          ↓
        sample 5 positions/document
          ↓
        save RAW activations

    No normalization is performed here. No explanation text is
    generated here either -- that is Stage 1 (generate_explanations).
    """
    model_cfg = config["model"]
    extraction_cfg = config["extraction"]
    datagen_cfg = config["datagen"]

    model_name = model_cfg["base_model"]
    dtype = get_dtype(model_cfg["dtype"])

    # Obtain model-independent architecture information.
    model_info = load_model_info(config)

    layer_index = model_info["extraction_layer"]
    d_model = model_info["hidden_size"]

    num_documents = datagen_cfg["num_documents"]
    vectors_per_document = extraction_cfg["vectors_per_document"]
    max_context_tokens = extraction_cfg["max_context_tokens"]
    seed = datagen_cfg["random_seed"]

    # Current config may not yet contain these fields.
    text_column = datagen_cfg.get("text_column", "text")
    extraction_batch_size = datagen_cfg.get(
        "extraction_batch_size",
        4,
    )

    base_path = Path(datagen_cfg["output_file"])

    # Stage 0 uses an intermediate file.
    output_path = base_path.with_name(
        f"{base_path.stem}_base{base_path.suffix}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema = make_base_schema(d_model)

    extractor = ActivationExtractor(
        model_name=model_name,
        layer_index=layer_index,
        dtype=dtype,
        max_length=max_context_tokens,
        batch_size=extraction_batch_size,
        trust_remote_code=model_cfg.get(
            "trust_remote_code",
            False,
        ),
    )

    special_token_ids = set(
        extractor.tokenizer.all_special_ids
    )

    corpus = load_corpus(datagen_cfg)
    iter_corpus = iter(corpus)
    corpus_exhausted = False

    writer = pq.ParquetWriter(
        str(output_path),
        schema,
    )

    total_rows = 0
    processed_documents = 0
    skipped_documents = 0
    short_sample_documents = 0

    progress = tqdm(total=num_documents, desc="Stage 0: extracting activations")

    try:
        while processed_documents < num_documents and not corpus_exhausted:
            text_batch = []

            while (
                len(text_batch) < extraction_batch_size
                and processed_documents + len(text_batch)
                < num_documents
            ):
                try:
                    example = next(iter_corpus)
                except StopIteration:
                    # Streamed corpus ran out before we hit num_documents.
                    # Finish gracefully with whatever we collected instead
                    # of crashing -- the metadata below records the true
                    # document count either way.
                    corpus_exhausted = True
                    break

                text = example[text_column]

                if not isinstance(text, str) or not text.strip():
                    continue

                text_batch.append(text)

            if not text_batch:
                break

            results = extractor.extract(text_batch)

            rows = {
                name: []
                for name in schema.names
            }

            for local_idx, result in enumerate(results):
                doc_idx = processed_documents + local_idx

                doc_id = (
                    f"{datagen_cfg['dataset']}:"
                    f"{datagen_cfg.get('split', 'train')}:"
                    f"{doc_idx}"
                )

                token_ids = result["token_ids"]

                positions = sample_positions(
                    token_ids=token_ids,
                    n_positions=vectors_per_document,
                    special_token_ids=special_token_ids,
                    doc_id=doc_id,
                    seed=seed,
                )

                if not positions:
                    skipped_documents += 1
                    continue

                if len(positions) < vectors_per_document:
                    short_sample_documents += 1

                tokenizer = extractor.tokenizer

                for position in positions:
                    vector = result["hidden_states"][position]

                    n_raw_tokens = position + 1

                    truncated_ids = token_ids[:n_raw_tokens]

                    truncated_text = tokenizer.decode(
                        truncated_ids,
                        skip_special_tokens=True,
                    )

                    rows["n_raw_tokens"].append(
                        n_raw_tokens
                    )

                    rows[
                        "detokenized_text_truncated"
                    ].append(
                        truncated_text
                    )

                    # IMPORTANT:
                    # Keep raw activations.
                    # Do not L2-normalize here.
                    rows["activation_vector"].append(
                        vector.tolist()
                    )

                    rows["activation_layer"].append(
                        layer_index
                    )

                    rows["doc_id"].append(
                        doc_id
                    )

                    total_rows += 1

            if rows["doc_id"]:
                writer.write_table(
                    pa.Table.from_pydict(
                        rows,
                        schema=schema,
                    )
                )

            progress.update(len(text_batch))
            processed_documents += len(text_batch)

    finally:
        progress.close()
        writer.close()
        extractor.close()

    if corpus_exhausted and processed_documents < num_documents:
        print(
            f"WARNING: corpus stream exhausted after {processed_documents} "
            f"documents (requested {num_documents}). Continuing with fewer "
            f"rows than configured."
        )

    metadata = {
        "stage": "base",
        "base_model": model_name,
        "model_type": model_info["model_type"],
        "hidden_size": d_model,
        "num_hidden_layers": model_info[
            "num_hidden_layers"
        ],
        "extraction_layer": layer_index,
        "ar_num_hidden_layers": model_info[
            "ar_num_hidden_layers"
        ],
        "normalization": "none",
        "dataset": datagen_cfg["dataset"],
        "split": datagen_cfg.get("split", "train"),
        "num_documents": processed_documents,
        "vectors_per_document": vectors_per_document,
        "max_context_tokens": max_context_tokens,
        "random_seed": seed,
        "row_count": total_rows,
    }

    import yaml

    with open(
        f"{output_path}.nla_meta.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
        )

    print()
    print("Stage 0 complete.")
    print(f"Documents processed : {processed_documents}")
    print(f"Activation rows     : {total_rows}")
    print(f"Skipped documents   : {skipped_documents}")
    print(
        f"Short-sampled docs  : "
        f"{short_sample_documents}"
    )
    print(f"Output              : {output_path}")
    print(
        f"Metadata            : "
        f"{output_path}.nla_meta.yaml"
    )

    return output_path


def load_teacher_prompt_template(config):
    prompts_cfg = config.get("prompts", {})
    template_path = Path(
        prompts_cfg.get(
            "teacher_template_file",
            "prompts/teacher_prompt.txt",
        )
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Teacher prompt template not found: {template_path.resolve()}"
        )

    return template_path.read_text(encoding="utf-8")


@torch.no_grad()
def generate_explanations(config, base_path):
    """
    Stage 1:

        Stage-0 parquet (context + raw activation)
          ↓
        teacher model, prompted per-row with teacher_prompt.txt
          ↓
        natural-language explanation of "what the model is
        representing/thinking" at that context cutoff
          ↓
        final warmstart parquet (context + activation + explanation)

    This is the piece that turns Stage-0's raw activation dump into
    an actual (input, activation, explanation) warmstart triple that
    the AV model is later SFT'd to imitate.
    """
    datagen_cfg = config["datagen"]

    teacher_model_name = datagen_cfg["teacher_model"]
    teacher_max_output_tokens = datagen_cfg.get(
        "teacher_max_output_tokens", 200
    )
    teacher_batch_size = datagen_cfg.get("teacher_batch_size", 8)
    trust_remote_code = config["model"].get("trust_remote_code", False)

    template = load_teacher_prompt_template(config)

    base_table = pq.read_table(base_path)
    d_model = base_table.schema.field("activation_vector").type.list_size

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding so `generate` can be batched safely for a decoder-only model.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=get_dtype(config["model"]["dtype"]),
        trust_remote_code=trust_remote_code,
    ).eval()

    model.to("cuda")

    output_path = Path(datagen_cfg["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warmstart_schema = make_warmstart_schema(d_model)
    writer = pq.ParquetWriter(str(output_path), warmstart_schema)

    num_rows = base_table.num_rows
    progress = tqdm(total=num_rows, desc="Stage 1: generating explanations")
    total_malformed = 0

    try:
        for start in range(0, num_rows, teacher_batch_size):
            end = min(start + teacher_batch_size, num_rows)
            chunk = base_table.slice(start, end - start).to_pylist()

            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": template.format(
                                context=row["detokenized_text_truncated"]
                            ),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in chunk
            ]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda")

            generated = model.generate(
                **encoded,
                max_new_tokens=teacher_max_output_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            raw_outputs = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )

            rows = {name: [] for name in warmstart_schema.names}
            malformed_count = 0

            for row, raw_output in zip(chunk, raw_outputs):
                explanation, emotion = parse_explanation_and_emotion(raw_output)

                if not emotion:
                    malformed_count += 1

                for name in warmstart_schema.names:
                    if name == "explanation":
                        rows[name].append(explanation)
                    elif name == "emotion":
                        rows[name].append(emotion)
                    else:
                        rows[name].append(row[name])

            writer.write_table(
                pa.Table.from_pydict(rows, schema=warmstart_schema)
            )

            progress.update(end - start)
            total_malformed += malformed_count
            if malformed_count:
                progress.set_postfix(malformed_emotion_tag=total_malformed)

    finally:
        progress.close()
        writer.close()

    print()
    print("Stage 1 complete.")
    print(f"Explanations written : {num_rows}")
    print(f"Teacher model         : {teacher_model_name}")
    print(f"Rows missing an emotion tag (format not followed): {total_malformed} ({100*total_malformed/num_rows:.1f}%)")
    print(f"Output                : {output_path}")

    return output_path


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. "
            "CPU fallback is disabled by design."
        )

    config = load_config("config.yaml")

    base_path = extract_dataset(config)
    generate_explanations(config, base_path)