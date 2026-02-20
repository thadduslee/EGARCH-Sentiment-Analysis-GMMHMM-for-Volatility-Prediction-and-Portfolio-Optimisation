
"""
8-bit LoRA finetuning script for sentiment classification with a CausalLM (Llama 3 8B style).
- Loads base model in 8-bit (bitsandbytes)
- Trains ONLY LoRA adapters (PEFT)
- Uses a CSV with columns: combined_text, sentiment  (sentiment in {negative, neutral, positive})
- Trains with Hugging Face Trainer

Run:
  python train_sentiment_lora_int8.py \
    --csv_path /path/to/train.csv \
    --out_dir outputs/lora_int8_sentiment \
    --base_model meta-llama/Meta-Llama-3-8B \
    --max_len 512 \
    --epochs 1

Notes:
- If you have a separate eval CSV, pass --eval_csv_path
- If you want to continue from an existing adapter, pass --init_adapter (optional)
"""
import json
from datetime import datetime
from transformers import TrainerCallback
import argparse, math, os, random, torch
import numpy as np 
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
from reddit_dataset import SentimentCSVDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

LABELS = ["negative", "neutral", "positive"]


def build_prompt(text: str) -> str:
    # Keep your exact inference prompt format
    return (
        "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\n"
        f"Input: {text}\n"
        "Answer:"
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@dataclass
class PadCollator:
    """
    Pads input_ids/attention_mask/labels to the longest sequence in batch.
    Labels padded with -100.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        # Pad labels manually to same length
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab

        batch["labels"] = padded_labels
        return batch

class EvalLoggerCallback(TrainerCallback):
    """
    Appends evaluation metrics to a log file (JSONL) after each eval.
    One JSON object per line.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "global_step": int(state.global_step),
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

            
def load_model_and_tokenizer(
    base_model: str,
    load_in_8bit: bool = True,
    init_adapter: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # compute dtype
    )

    # Prepare for k-bit training (casts norms, enables grad checkpoint compatibility, etc.)
    model = prepare_model_for_kbit_training(model)

    if init_adapter:
        # Continue training from an existing LoRA adapter
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        # Create a fresh LoRA adapter
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


@torch.no_grad()
def score_labels_batch(model, tokenizer, prompts, labels=LABELS, max_length=512, device=None):
    """
    Returns:
      scores: [B, C] mean log-prob per label-token (length-normalized)
      probs : [B, C] softmax over labels
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompts as a batch
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids_prompt = tok["input_ids"].to(device)         # [B, P]
    attn_prompt = tok["attention_mask"].to(device)         # [B, P]
    B, P = input_ids_prompt.shape
    C = len(labels)

    # Pre-tokenize labels (add leading space to match your training target)
    label_ids_list = [
        tokenizer(" " + l, add_special_tokens=False).input_ids for l in labels
    ]
    maxL = max(len(x) for x in label_ids_list)

    # Pad label ids to maxL so we can batch them
    lab_ids = torch.full((C, maxL), fill_value=tokenizer.pad_token_id, dtype=torch.long, device=device)
    lab_attn = torch.zeros((C, maxL), dtype=attn_prompt.dtype, device=device)
    for ci, ids in enumerate(label_ids_list):
        lab_ids[ci, :len(ids)] = torch.tensor(ids, device=device)
        lab_attn[ci, :len(ids)] = 1

    # Expand so we evaluate all (prompt, label) pairs in one forward:
    # prompts: [B, P] -> [B*C, P]
    # labels : [C, L] -> [B*C, L]
    input_ids_prompt_exp = input_ids_prompt.unsqueeze(1).expand(B, C, P).reshape(B * C, P)
    attn_prompt_exp = attn_prompt.unsqueeze(1).expand(B, C, P).reshape(B * C, P)

    lab_ids_exp = lab_ids.unsqueeze(0).expand(B, C, maxL).reshape(B * C, maxL)
    lab_attn_exp = lab_attn.unsqueeze(0).expand(B, C, maxL).reshape(B * C, maxL)

    input_ids = torch.cat([input_ids_prompt_exp, lab_ids_exp], dim=1)   # [B*C, P+L]
    attn = torch.cat([attn_prompt_exp, lab_attn_exp], dim=1)            # [B*C, P+L]

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [B*C, P+L, V]

    # Compute token log-prob for label tokens only
    # predictor position for token t is logits at t-1
    # label tokens are at positions [P, P+L-1], predictors at [P-1, P+L-2]
    # We'll compute for all maxL and mask with lab_attn_exp.
    pred_positions = torch.arange(P - 1, P + maxL - 1, device=device)  # [maxL]
    label_positions = torch.arange(P, P + maxL, device=device)         # [maxL]

    logprobs = F.log_softmax(logits[:, pred_positions, :], dim=-1)     # [B*C, maxL, V]
    target = input_ids[:, label_positions]                              # [B*C, maxL]
    token_logp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1) # [B*C, maxL]

    # Mask out padded label tokens and length-normalize (mean over real label tokens)
    mask = lab_attn_exp.float()                                         # [B*C, maxL]
    token_logp = token_logp * mask
    lengths = mask.sum(dim=1).clamp_min(1.0)
    seq_score = token_logp.sum(dim=1) / lengths                         # [B*C]

    scores = seq_score.view(B, C)                                       # [B, C]
    probs = F.softmax(scores, dim=1)
    return scores, probs


def evaluate_accuracy_softmax(model, tokenizer, df, batch_size=16, max_length=512, device=None):
    """
    Computes accuracy by scoring 3 candidate labels with log-probs and softmax.
    df must have columns: combined_text, sentiment
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # normalize GT
    gt = df["sentiment"].astype(str).str.strip().str.lower().tolist()

    correct = 0
    total = 0

    for start in tqdm(range(0, len(df), batch_size)):
        end = min(start + batch_size, len(df))
        texts = df.iloc[start:end]["combined_text"].astype(str).tolist()
        prompts = [build_prompt(t) for t in texts]

        _, probs = score_labels_batch(
            model, tokenizer, prompts,
            labels=LABELS,
            max_length=max_length,
            device=device
        )

        pred_idx = probs.argmax(dim=1).tolist()
        preds = [LABELS[i] for i in pred_idx]

        for p, y in zip(preds, gt[start:end]):
            if p == y:
                correct += 1
            total += 1

    acc = correct / total if total else 0.0
    print(f"\nValidation Accuracy (softmax over labels): {acc:.4f}")
    return acc


def eval_from_df(model, tokenizer, df, max_len=512):
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device

    for i in tqdm(range(len(df))):
        text = str(df.iloc[i]["combined_text"])
        gt = str(df.iloc[i]["sentiment"]).strip().lower()

        prompt = build_prompt(text)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

        # normalize prediction to one of LABELS
        pred = None
        for lab in LABELS:
            if gen_text.startswith(lab) or lab in gen_text.split():
                pred = lab
                break

        if pred == gt:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    print(f"\nValidation Accuracy: {acc:.4f}")
    return acc
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--out_dir", type=str, default="outputs/lora_int8_sentiment")
    parser.add_argument("--max_len", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--batch_size", type=int, default=8)  
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--init_adapter", type=str, default=None, help="Path or HF id to an existing LoRA adapter")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_log_path", type=str, default=None,
                        help="Where to write eval metrics (JSONL). Default: <out_dir>/eval_metrics.jsonl")
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.eval_log_path is None:
        args.eval_log_path = os.path.join(args.out_dir, "eval_metrics.jsonl")
    # Load data
    full_df = pd.read_csv(args.csv_path)
    
    # Normalize labels
    full_df["sentiment"] = (
        full_df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    
    # Keep only valid labels
    full_df = full_df[full_df["sentiment"].isin(LABELS)].reset_index(drop=True)
    
    print("Label distribution (full dataset):")
    print(full_df["sentiment"].value_counts())
    
    # Stratified split (important for class balance)
    train_df, eval_df = train_test_split(
        full_df,
        test_size=0.1,                 # 10% validation
        random_state=args.seed,
        stratify=full_df["sentiment"], # keeps class balance
    )
    
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    print("\nTrain size:", len(train_df))
    print("Eval size:", len(eval_df))
    print("\nTrain label distribution:")
    print(train_df["sentiment"].value_counts())
    print("\nEval label distribution:")
    print(eval_df["sentiment"].value_counts())


    
    print("Label counts:\n", train_df["sentiment"].value_counts(dropna=False))
    

    # Load model/tokenizer (8-bit base + LoRA trainable)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        load_in_8bit=True,
        init_adapter=args.init_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Optional but recommended on 24GB
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # important when using grad checkpointing

    train_ds = SentimentCSVDataset(train_df, tokenizer, max_len=args.max_len)
    eval_ds = SentimentCSVDataset(eval_df, tokenizer, max_len=args.max_len) if eval_df is not None else None

    collator = PadCollator(tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        report_to="none",
        optim="paged_adamw_8bit",  # key for bitsandbytes-friendly optimizer
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True, 
        eval_accumulation_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        #compute_metrics=compute_metrics,
        callbacks=[EvalLoggerCallback(args.eval_log_path)],
    )

    trainer.train()


    print("\nRunning final evaluation...")
    eval_from_df(model, tokenizer, eval_df, max_len=1024)

    # Save adapter + tokenizer
    trainer.save_model(args.out_dir)  # for PEFT this saves adapter weights/config
    tokenizer.save_pretrained(args.out_dir)

    print(f"\nSaved LoRA adapter to: {args.out_dir}")

    evaluate_accuracy_softmax(model, tokenizer, eval_df, batch_size=16, max_length=1024)
    
if __name__ == "__main__":
    main()
