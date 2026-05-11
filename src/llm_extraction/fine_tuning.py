"""
fine_tuning.py — QLoRA fine-tuning utilities for Mistral 7B (EarningsLens).

Uses spaCy-extracted targets as weak-supervision labels to instruction-tune
Mistral 7B in 4-bit precision via PEFT LoRA + BitsAndBytes. The resulting
checkpoint is designed for deployment with vLLM.

Heavy dependencies (transformers, peft, bitsandbytes, datasets) are imported
lazily inside each function so the module can be imported in inference-only
environments without them installed.

Typical usage:
    from llm_extraction.fine_tuning import prepare_training_data, train, export_for_vllm

    dataset  = prepare_training_data("data/processed/spacy_targets.parquet")
    qlora    = setup_qlora_config()
    train("mistralai/Mistral-7B-Instruct-v0.2", dataset, "checkpoints/ft-mistral-earnings")
    export_for_vllm("checkpoints/ft-mistral-earnings/best", "models/mistral-earnings-merged")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """\
<s>[INST] <<SYS>>
You are a financial analyst extracting performance targets from earnings call \
transcripts. Return ONLY a valid JSON array of target objects. Each object has: \
metric_name, raw_text, numerical_value, trend_direction, unit, temporal_framing, \
is_financial, confidence.
<</SYS>>

Transcript segment:
{transcript_text}
[/INST]
{target_json}
</s>"""

DEFAULT_TRAINING_ARGS: Dict[str, Any] = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 25,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": False,
    "bf16": True,         # better for Ampere+ GPUs; falls back gracefully
    "dataloader_num_workers": 4,
    "report_to": "none",  # disable W&B by default; set to "wandb" to enable
}

LORA_CONFIG: Dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

BNBS_CONFIG: Dict[str, Any] = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",  # resolved to torch.bfloat16 at runtime
}


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    spacy_targets_path: str,
    transcript_text_col: str = "text",
    targets_col: str = "targets",
    test_size: float = 0.1,
    seed: int = 42,
) -> "datasets.DatasetDict":  # type: ignore[name-defined]
    """
    Convert spaCy-extracted targets into instruction-tuning format.

    Reads a Parquet file where each row is one transcript component with:
      - *transcript_text_col* : raw text of the segment
      - *targets_col*         : JSON-serialised list of target dicts

    Returns a HuggingFace ``DatasetDict`` with ``"train"`` and ``"test"`` splits.

    Parameters
    ----------
    spacy_targets_path  : str   — path to Parquet file with spaCy outputs
    transcript_text_col : str   — column name for transcript text
    targets_col         : str   — column name for JSON target list
    test_size           : float — fraction of data held out for evaluation
    seed                : int   — random seed for reproducible splits

    Returns
    -------
    datasets.DatasetDict — ``{"train": Dataset, "test": Dataset}``
    """
    try:
        import pandas as pd
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "pandas and datasets are required. Install with: "
            "pip install pandas datasets"
        ) from exc

    logger.info("prepare_training_data | loading %s", spacy_targets_path)
    df = pd.read_parquet(spacy_targets_path)

    required = {transcript_text_col, targets_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from {spacy_targets_path}: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Drop rows with null text or empty targets
    df = df.dropna(subset=[transcript_text_col, targets_col])
    df = df[df[targets_col].apply(
        lambda v: bool(v) and (isinstance(v, (list, str)) and len(v) > 2)
    )]

    logger.info("prepare_training_data | %d usable rows after filtering", len(df))

    def _format_row(row: pd.Series) -> Dict[str, str]:
        text = row[transcript_text_col]
        targets = row[targets_col]

        # Normalise targets column — may already be a list or a JSON string
        if isinstance(targets, str):
            try:
                targets = json.loads(targets)
            except json.JSONDecodeError:
                targets = []
        if not isinstance(targets, list):
            targets = []

        target_json = json.dumps(targets, ensure_ascii=False)
        formatted = INSTRUCTION_TEMPLATE.format(
            transcript_text=text.strip(),
            target_json=target_json,
        )
        return {"text": formatted}

    records = df.apply(_format_row, axis=1).tolist()
    hf_dataset = Dataset.from_list(records)

    split = hf_dataset.train_test_split(test_size=test_size, seed=seed)
    logger.info(
        "prepare_training_data | train=%d  test=%d",
        len(split["train"]), len(split["test"]),
    )
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ---------------------------------------------------------------------------
# 2. LoRA / QLoRA configuration
# ---------------------------------------------------------------------------

def setup_qlora_config() -> "peft.LoraConfig":  # type: ignore[name-defined]
    """
    Build and return a PEFT ``LoraConfig`` for 4-bit QLoRA fine-tuning.

    Targets all four attention projection matrices (q, k, v, o) with:
      r=16, lora_alpha=32, lora_dropout=0.05

    Returns
    -------
    peft.LoraConfig — ready to pass to ``get_peft_model()``

    Raises
    ------
    ImportError — if the ``peft`` package is not installed
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "peft is required. Install with: pip install peft"
        ) from exc

    config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(
        "setup_qlora_config | r=%d  alpha=%d  dropout=%.2f  modules=%s",
        config.r, config.lora_alpha, config.lora_dropout, config.target_modules,
    )
    return config


def _build_bnb_config() -> "transformers.BitsAndBytesConfig":  # type: ignore[name-defined]
    """
    Build a 4-bit BitsAndBytes quantisation config.

    Returns
    -------
    transformers.BitsAndBytesConfig
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "transformers and bitsandbytes are required. "
            "Install with: pip install transformers bitsandbytes"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=BNBS_CONFIG["load_in_4bit"],
        bnb_4bit_use_double_quant=BNBS_CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=BNBS_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    dataset: "datasets.DatasetDict",  # type: ignore[name-defined]
    output_dir: str,
    training_args_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Fine-tune *model_name* with QLoRA on *dataset* and save to *output_dir*.

    Training configuration:
      - 3 epochs, per-device batch size 4, gradient accumulation steps 4
      - Learning rate 2e-4 with 100 warmup steps
      - 4-bit NF4 quantisation via BitsAndBytes
      - Best checkpoint saved based on eval loss

    Parameters
    ----------
    model_name             : str              — HuggingFace model ID or local path
                                               (e.g. "mistralai/Mistral-7B-Instruct-v0.2")
    dataset                : DatasetDict      — ``{"train": ..., "test": ...}``
    output_dir             : str              — directory to save checkpoints
    training_args_overrides: Dict, optional   — override any DEFAULT_TRAINING_ARGS keys

    Returns
    -------
    None — checkpoints written to *output_dir*

    Raises
    ------
    ImportError — if transformers, peft, or bitsandbytes are not installed
    """
    try:
        from peft import get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers and peft are required. "
            "Install with: pip install transformers peft bitsandbytes"
        ) from exc

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Merge training args ────────────────────────────────────────────────
    args_dict = {**DEFAULT_TRAINING_ARGS}
    if training_args_overrides:
        args_dict.update(training_args_overrides)

    logger.info("train | model=%s  output=%s", model_name, output_dir)
    logger.info("train | effective training args: %s", args_dict)

    # ── Load tokeniser ─────────────────────────────────────────────────────
    logger.info("train | loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Tokenise dataset ───────────────────────────────────────────────────
    def _tokenize(examples: Dict) -> Dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    logger.info("train | tokenising dataset …")
    tokenised = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    # ── Load model in 4-bit ────────────────────────────────────────────────
    logger.info("train | loading model in 4-bit quantisation …")
    bnb_config = _build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── Apply LoRA adapters ────────────────────────────────────────────────
    logger.info("train | applying QLoRA adapters …")
    lora_config = setup_qlora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── TrainingArguments ──────────────────────────────────────────────────
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        **{k: v for k, v in args_dict.items() if k not in ("bf16", "fp16")},
        bf16=args_dict.get("bf16", True),
        fp16=args_dict.get("fp16", False),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("train | starting training …")
    trainer.train()

    # Save the best checkpoint's adapter weights
    best_ckpt = os.path.join(output_dir, "best")
    model.save_pretrained(best_ckpt)
    tokenizer.save_pretrained(best_ckpt)
    logger.info("train | best checkpoint saved to %s", best_ckpt)


# ---------------------------------------------------------------------------
# 4. vLLM export
# ---------------------------------------------------------------------------

def export_for_vllm(
    checkpoint_path: str,
    output_path: str,
) -> None:
    """
    Merge LoRA adapter weights into the base model and save for vLLM serving.

    The merged full-precision model is saved to *output_path* in a format
    directly loadable by vLLM (``--model output_path``).

    Parameters
    ----------
    checkpoint_path : str — path to the saved PEFT adapter checkpoint
                            (typically ``output_dir/best``)
    output_path     : str — destination directory for the merged model

    Returns
    -------
    None

    Raises
    ------
    ImportError   — if peft or transformers are not available
    FileNotFoundError — if checkpoint_path does not exist
    """
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "peft and transformers are required. "
            "Install with: pip install peft transformers"
        ) from exc

    checkpoint_path = str(Path(checkpoint_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("export_for_vllm | loading PEFT model from %s …", checkpoint_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info("export_for_vllm | merging LoRA weights …")
    merged_model = model.merge_and_unload()

    logger.info("export_for_vllm | saving merged model to %s …", output_path)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.save_pretrained(output_path)

    logger.info(
        "export_for_vllm | done. Serve with: "
        "vllm serve %s --served-model-name mistral-earnings",
        output_path,
    )
