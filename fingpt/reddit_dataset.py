import torch, pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional


LABELS = ["negative", "neutral", "positive"]


def build_prompt(text: str) -> str:
    # Keep your exact inference prompt format
    return (
        "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\n"
        f"Input: {text}\n"
        "Answer:"
    )


class SentimentCSVDataset(Dataset):
    """
    Produces tokenized examples where loss is computed ONLY on the answer tokens
    (i.e., the label word + eos), not on the prompt.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int = 2048,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Basic checks
        if "combined_text" not in self.df.columns or "sentiment" not in self.df.columns:
            raise ValueError("CSV must contain columns: combined_text, sentiment")

        # Normalize sentiment strings
        self.df["sentiment"] = self.df["sentiment"].astype(str).str.strip().str.lower()
        bad = ~self.df["sentiment"].isin(LABELS)
        if bad.any():
            bad_vals = sorted(self.df.loc[bad, "sentiment"].unique().tolist())
            raise ValueError(f"Found sentiment values not in {LABELS}: {bad_vals}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.df.iloc[idx]["combined_text"])
        label = str(self.df.iloc[idx]["sentiment"]).strip().lower()

        prompt = build_prompt(text)

        # We want model to generate: "<label><eos>"
        # Add a leading space before label to match common tokenization behavior.
        answer = " " + label + self.tokenizer.eos_token

        # Tokenize prompt and answer separately so we can mask prompt tokens in labels
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]

        # Leave room for answer tokens within max_len
        ans_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]

        # If prompt is too long, truncate it to fit answer
        if len(prompt_ids) + len(ans_ids) > self.max_len:
            prompt_ids = prompt_ids[: self.max_len - len(ans_ids)]

        input_ids = prompt_ids + ans_ids
        attention_mask = [1] * len(input_ids)

        # labels: mask prompt part with -100 so loss only computed on answer tokens
        labels = [-100] * len(prompt_ids) + ans_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


