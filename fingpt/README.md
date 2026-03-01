## Instructions for FinGPT usage

Full example code for loading model, non-batched, batched inference is in `test_fingpt.ipynb`. 

### Quick Start 
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


### Training 
Training was done by running `train.py` on `h100-47` with the following args:
```bash 
python fingpt/train.py --max_len 1024 --out_dir fingpt/outputs/lora_int8_sentiment_fingpt --lr 1e-4 --batch_size 4 --init_adapter FinGPT/fingpt-mt_llama3-8b_lora --csv_path labelled_posts.csv 
```