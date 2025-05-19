import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset
from tqdm import tqdm

import os, gc
import pandas as pd
import time
import hashlib

from yaml_parser import load_config

# primitive bart model
def load_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)

def batch_summarize(data, summarizer, batch_size=8, max_input_length=1024):
    summaries = []
    for i in tqdm(range(0, len(data), batch_size), desc="Summarizing"):
        batch = data[i:i+batch_size]

        # Prepare input texts with prompt + joined sentences
        inputs = []
        for item in batch:
            prompt = item["prompt"].strip()
            dream = "dream description: " + item["dream"].strip()
            text = "dream symbols: " + " ".join(item["interpretations"]).strip()
            full_input = f"{prompt}\n{dream}\n{text}"
            inputs.append(full_input[:max_input_length])  # truncate safely

        outputs = summarizer(inputs, max_length=1024, min_length=20, do_sample=False)
        summaries.extend([out['summary_text'] for out in outputs])

    return summaries



# Format prompt for causal model
# slightly better flan-T5 model
# Step 2: Define input formatting
# Universal prompt formatter
def format_prompt(prompt, dream, symbols, model_family="decoder"):
    if model_family == "decoder":  # For Mistral, Llama, GPT-style
        return f"""### Instruction:
{prompt.strip()}

### Dream:
{dream.strip()}

### Symbols:
{symbols.strip()}

### Interpretation:"""
    elif model_family == "encoder":  # For T5, FLAN-T5
        return f"Interpret this dream: {dream.strip()}\nSymbols: {symbols.strip()}"
    else:
        raise ValueError(f"Unknown model_family: {model_family}")


def attach_meanings(dream_df, keywords_df):

    config = load_config()
    sym_col = "symbol"
    kw_col = config['data']['keywords_column']
    interp_col = config['data']['interpretation_column']

    #for i, ex in dream_df.iterrows():
        #print(ex)

    def get_meanings(dream):
        keys = dream[kw_col].split(";")[:5]
        
        #print(keys)
        syms = keywords_df[keywords_df[kw_col].isin(keys)]

        descr = syms.apply(lambda r: f' - {r[sym_col]}: {r[kw_col]}{r[interp_col]}', axis = 1)
        meanings = "\n".join(descr)
        return meanings 

    dream_df["meanings"] = dream_df.apply(get_meanings, axis = 1)

    return dream_df


        

def format_input(dataset, prompt, formatter, tokenizer):
    # yes, not very elegant
    print("formatting input")
    config = load_config()
    dream = config['data']['dream_text_column']
    #interp_col = config['data']['interpretation_column']

    dataset["input"] = dataset.apply(lambda r: formatter.format(prompt, r[dream], r["meanings"]), axis = 1)
    dataset["len"] = dataset["input"].str.len()
    dataset["input_tokens"] = dataset.input.apply(lambda prmt: tokenizer.tokenize(prmt, truncation=False, max_length=1024))
    dataset["input_tokens_len"] = dataset.input_tokens.apply(len)
    dataset

    dataset.drop(columns=["input_tokens"] ,inplace=True)

    def get_hash(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    dataset["hash"] = dataset["input"].apply(get_hash) 

    dataset.sort_values("input_tokens_len", ascending=False, inplace=True)
    return dataset


# Load flan-T5 model
def load_causal_model(model_name = "google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


# Load Mistral 7B in 4-bit mode
def load_mistral_4bit_model(model_name="mistralai/Mistral-7B-Instruct"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def find_max_batch_size(model, tokenizer, sample_prompt, task="text-generation", max_possible=256, max_new_tokens=512, ):
    low, high = 1, max_possible
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            print(f"Trying batch_size = {mid}...", end=" ")
            test_pipeline = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                batch_size=mid,
                truncation=False,
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            prompts = [sample_prompt] * mid
            _ = test_pipeline(prompts)
            best = mid
            low = mid + 1
            print("✅ success")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("❌ OOM")
                torch.cuda.empty_cache()
                gc.collect()
                high = mid - 1
            else:
                raise e

    print(f"\n✅ Optimal batch size: {best}")
    return best

# PromptFormatter object
class PromptFormatter:
    def __init__(self, model_family="decoder"):
        self.model_family = model_family

    def format(self, prompt, dream, symbols):
        if self.model_family == "decoder":
            return f"""### Instruction:
{prompt.strip()}

### Dream:
{dream.strip()}

### Symbols:
{symbols.strip()}

### Interpretation:"""
        elif self.model_family == "encoder":
            return f"Interpret this dream: {dream.strip()}\nSymbols: {symbols.strip()}"
        else:
            raise ValueError(f"Unknown model_family: {self.model_family}")

    def unformat(self, output):
        if self.model_family == "decoder":
            return output.split("### Interpretation:")[-1].strip()
        elif self.model_family == "encoder":
            return output.strip()
        else:
            raise ValueError(f"Unknown model_family: {self.model_family}")



# Efficient batched processing using dataset.map

# Explanation:
# HuggingFace Datasets `map()` supports batching, but does NOT automatically clean up GPU memory.
# Since transformers pipeline and large models may leave residual allocations, we force memory cleanup
# after each batch using `torch.cuda.empty_cache()` and `gc.collect()`.
# This prevents out-of-memory (OOM) errors in large datasets or constrained VRAM environments.

# Improved: generic, resumable, disk-saving batch generation

def batch_generate_interpretations(df, model_pipeline, 
                                   input_column="input_text", output_column="interpretation", 
                                   batch_size=8, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # Read processed hashes from existing CSVs
    processed_hashes = set()
    for f in os.listdir(save_dir):
        if f.endswith(".csv"):
            try:
                existing_df = pd.read_csv(os.path.join(save_dir, f), usecols=["hash"])
                processed_hashes.update(existing_df["hash"].tolist())
            except Exception:
                continue

    # Filter out already processed samples
    initial_len = len(df)
    df = df[~df["hash"].isin(processed_hashes)].reset_index(drop=True)
    print(f"✅ Already processed: {initial_len - len(df)} / {initial_len} entries")

    # Now work in batches
    for start in tqdm(range(0, len(df), batch_size), desc="Generating batches"):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end].copy()
        if batch.empty:
            continue

        inputs = batch[input_column].tolist()
        try:
            outputs = model_pipeline(inputs)
            if type(outputs[0]) is list:
                outputs = sum(outputs,[])
            outputs = [out["generated_text"] for out in outputs]
        except Exception as e:
            print(f"❌ Error in batch {start}-{end}: {e}")
            continue

        batch[output_column] = outputs
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_file = os.path.join(save_dir, f"{timestamp}_batch_{start}_{end}.csv")
        batch.to_csv(batch_file, index=False)

        torch.cuda.empty_cache()
        gc.collect()

    print("🏁 Batch generation complete.")
    return None
