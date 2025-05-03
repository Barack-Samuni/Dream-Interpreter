import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset
from tqdm import tqdm

import os, gc
import pandas as pd


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



def find_max_batch_size(model, tokenizer, sample_prompt, task="text-generation", max_possible=256, max_length=1024):
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
                max_length=max_length,
                do_sample=False
            )
            prompts = [sample_prompt] * mid
            _ = test_pipeline(prompts, max_length=max_length)
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

def batch_generate_interpretations(dataset: Dataset, model_pipeline, formatter, batch_size=8, max_length=250, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    def model_inference(batch):
        prompts = [formatter.format(p, d, s) for p, d, s in zip(batch["prompt"], batch["dream"], batch["symbols"])]
        outputs = model_pipeline(prompts, max_length=max_length)
        batch["interpretation"] = [formatter.unformat(out[0]["generated_text"]) for out in outputs]

        # Free GPU memory between batches
        torch.cuda.empty_cache()
        gc.collect()

        return batch

    # Map over dataset with batching and caching enabled
    result_dataset = dataset.map(
        function=model_inference,
        batched=True,
        batch_size=batch_size,
        # cache_file_name= os.path.join(save_dir, "intermediate_cache.arrow"),
        # load_from_cache_file=True,
        remove_columns=[]
    )

    # Save final results to disk
    result_dataset.to_csv(os.path.join(save_dir, "results.csv"))

    return result_dataset.to_pandas()

