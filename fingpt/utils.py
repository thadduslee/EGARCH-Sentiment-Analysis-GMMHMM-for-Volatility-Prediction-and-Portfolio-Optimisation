from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import torch.nn.functional as F 

# DO NOT CHANGE
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
PEFT_MODEL = "./outputs/lora_int8_sentiment"

LABELS = ["negative", "neutral", "positive"]


def load_fingpt(lora_dir, device='auto'):
    """
    Loads the finetuned FinGPT model. 
    Returns: HF tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    
    # turn off kv cache. If we have alot of GPU mem, we can turn this on
    model.config.use_cache = False

    return tokenizer, model

@torch.no_grad() 
def sentiment_probs(
    tokenizer, model,
    input_sentence, labels=LABELS, max_length=1024, device='cuda'):

    """
    Performs sentiment analysis using the Fingpt model. 
    Returns: probability distribution over labels

    """
    if tokenizer is None or model is None: 
        raise ValueError('Specify tokenizer or model for sentiment analysis.')

    prompt_template = [
        f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
        Input: {input_sentence}
        Answer:"""
    ]

    # tokenize prompt
    prompt_tok = tokenizer(
        prompt_template, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    input_ids_prompt = prompt_tok["input_ids"].to(device)
    attn_prompt = prompt_tok["attention_mask"].to(device)

    B = input_ids_prompt.size(0)
    C = len(labels)

    # pre-tokenize the labels for comparison
    label_ids_list = [
        tokenizer(l, add_special_tokens=False).input_ids for l in labels
    ]

    # Compute scores: shape [B, C]
    scores = torch.empty((B, C), device=device, dtype=torch.float32)

    for ci, lab_ids in enumerate(label_ids_list):
        lab = torch.tensor(lab_ids, device=device).unsqueeze(0).repeat(B, 1)  # [B, L]
        L = lab.size(1)

        # Build combined input: [prompt, label]
        input_ids = torch.cat([input_ids_prompt, lab], dim=1)  # [B, P+L]
        attn = torch.cat([attn_prompt, torch.ones((B, L), device=device, dtype=attn_prompt.dtype)], dim=1)

        # Forward
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [B, P+L, V]

        # We want log p(label_token_i | prompt + previous label tokens)
        # The probability for token at position t is predicted from logits at position t-1.
        P = input_ids_prompt.size(1)
        # label tokens are at positions [P, P+L-1]
        # their predictors are logits at positions [P-1, P+L-2]
        pred_positions = torch.arange(P - 1, P + L - 1, device=device)  # length L
        label_positions = torch.arange(P, P + L, device=device)         # length L

        # Gather logprobs for the label tokens
        logprobs = F.log_softmax(logits[:, pred_positions, :], dim=-1)  # [B, L, V]
        target = input_ids[:, label_positions]                          # [B, L]
        token_logp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, L]

        # Mask out any padding in the prompt doesn't matter here; label tokens are always "real"
        # Length-normalize to reduce short-label bias
        scores[:, ci] = token_logp.mean(dim=1)

    probs = F.softmax(scores, dim=1)  # [B, C]
    return scores, probs

@torch.no_grad()
def sentiment_probs_batch(
    tokenizer,
    model,
    input_sentences,
    labels=LABELS,
    max_length=1024,
    device=None,
):
    """
    input_sentences: List[str] length B
    Returns:
      scores: [B, C] (mean log-prob per label token)
      probs : [B, C]
    """
    if device is None:
        device = next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)

    # Build prompts (exact format)
    prompts = [
        f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
Input: {s}
Answer:"""
        for s in input_sentences
    ]

    # Tokenize prompts as a batch
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids_prompt = tok["input_ids"].to(device)        # [B, P]
    attn_prompt = tok["attention_mask"].to(device)        # [B, P]
    B, P = input_ids_prompt.shape
    C = len(labels)

    # Pre-tokenize labels (IMPORTANT: add leading space to match training/inference convention)
    label_ids_list = [tokenizer(" " + l, add_special_tokens=False).input_ids for l in labels]
    maxL = max(len(x) for x in label_ids_list)

    # Pad label token ids to maxL so we can expand+cat
    lab_ids = torch.full((C, maxL), fill_value=tokenizer.pad_token_id, dtype=torch.long, device=device)
    lab_attn = torch.zeros((C, maxL), dtype=attn_prompt.dtype, device=device)
    for ci, ids in enumerate(label_ids_list):
        lab_ids[ci, :len(ids)] = torch.tensor(ids, device=device)
        lab_attn[ci, :len(ids)] = 1

    # Expand prompts across labels and labels across batch:
    # prompts: [B, P] -> [B*C, P]
    # labels : [C, L] -> [B*C, L]
    input_ids_prompt_exp = input_ids_prompt.unsqueeze(1).expand(B, C, P).reshape(B * C, P)
    attn_prompt_exp = attn_prompt.unsqueeze(1).expand(B, C, P).reshape(B * C, P)

    lab_ids_exp = lab_ids.unsqueeze(0).expand(B, C, maxL).reshape(B * C, maxL)
    lab_attn_exp = lab_attn.unsqueeze(0).expand(B, C, maxL).reshape(B * C, maxL)

    input_ids = torch.cat([input_ids_prompt_exp, lab_ids_exp], dim=1)  # [B*C, P+L]
    attn = torch.cat([attn_prompt_exp, lab_attn_exp], dim=1)           # [B*C, P+L]

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [B*C, P+L, V]

    # Gather log-probs for label tokens only
    pred_positions = torch.arange(P - 1, P + maxL - 1, device=device)  # [maxL]
    label_positions = torch.arange(P, P + maxL, device=device)         # [maxL]

    logprobs = F.log_softmax(logits[:, pred_positions, :], dim=-1)     # [B*C, maxL, V]
    target = input_ids[:, label_positions]                              # [B*C, maxL]
    token_logp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1) # [B*C, maxL]

    # Mask padded label tokens + length-normalize
    mask = lab_attn_exp.float()                                        # [B*C, maxL]
    token_logp = token_logp * mask
    lengths = mask.sum(dim=1).clamp_min(1.0)
    seq_score = token_logp.sum(dim=1) / lengths                        # [B*C]

    scores = seq_score.view(B, C)
    probs = F.softmax(scores, dim=1)
    return scores, probs
