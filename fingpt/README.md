# Instructions for FinGPT usage

Full example code for loading model, non-batched, batched inference is in `test_fingpt.ipynb`. 

## Quick Start 
`utils.py` provides the following functions: 
- `load_fingpt`: Loads model from LoRA adapter path. Returns tokenizer and model.
- `sentiment_probs`: Inference on a piece of text. Returns scores and probability distribution across labels.
- `sentiment_probs_batched`: Batched inference. Returns `[B, 3]` shape of probability distributions.

### Example Usage for `sentiment_probs`
```python 
from utils import load_fingpt, sentiment_probs, sentiment_probs_batch, LABELS

# specify path to LoRa adapter
peft_model = "./outputs/lora_int8_sentiment_llama"
# load model using loader function
tokenizer, model = load_fingpt(peft_model, 'cuda')

combined_text = '''MSTR Puts

Opened a couple of weeks ago.. was hoping for bitcoin to crash in Q3.. things happened much faster. I think the floor will be around 35-40K for Bitcoin.

not this one.. don't see where the 25% will come from with the price of bitcoin'''

# inference 
_, probs = sentiment_probs(tokenizer, model, combined_text)

# print prob. distribution
for i, p in enumerate(probs.tolist()):
    for lab, pr in zip(LABELS, p):
        print(f"  {lab:8s}: {pr:.4f}")
```

### Example Usage for `sentiment_probs_batched`
```python 
import pandas as pd
from tqdm.auto import tqdm 
from utils import load_fingpt, sentiment_probs, sentiment_probs_batch, LABELS

# Load CSV data
device='cuda'
df = pd.read_csv('../labelled_posts.csv')
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df = df[df["sentiment"].isin(LABELS)].reset_index(drop=True)

batch_size = 2  
preds = []
prob_rows = []
texts = df["combined_text"].astype(str).tolist()


for start in tqdm(range(0, len(texts), batch_size), desc="Infer (batched)"):
    # batch by creating a list
    batch = texts[start:start + batch_size]
    _, probs = sentiment_probs_batch(
        tokenizer, model,
        input_sentences=batch,
        labels=LABELS,
        max_length=1024,
        device=device,
    )
    probs_cpu = probs.detach().cpu()  # [B, 3]
    pred_idx = probs_cpu.argmax(dim=1).tolist()

    preds.extend([LABELS[i] for i in pred_idx])
    prob_rows.extend(probs_cpu.tolist())

df_out = df.copy()
df_out["pred"] = preds
df_out["p_negative"] = [p[0] for p in prob_rows]
df_out["p_neutral"]  = [p[1] for p in prob_rows]
df_out["p_positive"] = [p[2] for p in prob_rows]
df_out["correct"] = (df_out["pred"] == df_out["sentiment"]).astype(int)

acc = df_out["correct"].mean() if len(df_out) else 0.0
print(f"\nAccuracy: {acc:.4f}")

```


### Training 
Training was done by running `train.py` on `h100-47` with the following args:
```bash 
python fingpt/train.py --max_len 1024 --out_dir fingpt/outputs/lora_int8_sentiment_fingpt --lr 1e-4 --batch_size 5 --init_adapter FinGPT/fingpt-mt_llama3-8b_lora --csv_path labelled_posts.csv 
```