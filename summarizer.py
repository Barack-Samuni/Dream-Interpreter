import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

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


# Example usage
if __name__ == "__main__":
    # Mock dataset
    dataset = [
        {"prompt": "Explain what this dream means.", 
         "sentences": [
             "I saw a large snake in my garden.",
             "It slithered toward me but didn’t attack.",
             "I felt a mix of fear and curiosity.",
             "My dog was barking loudly in the background.",
             "Then I woke up suddenly."
         ]}
    ] * 100  # simulate 100 entries for testing

    summarizer = load_summarizer()
    results = batch_summarize(dataset, summarizer, batch_size=8)

    print("Example summary:", results[0])

# Format prompt for causal model
# slightly better flan-T5 model
# Step 2: Define input formatting
def format_input(prompt, dream, symbols):
    return (f"Instruction: {prompt.strip()}\n\n"            
        f"Dream: {dream.strip()}\n\n"
        f"Symbols:\n{symbols.strip()}\n\n"
        "Interpretation:"
    )

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

    return model, tokenizer


# Step 3: Batch interpret function
def batch_generate_interpretations(df, model_pipeline, batch_size=4, **kwargs):
    interpretations = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating Interpretations"):
        batch_df = df.iloc[i:i+batch_size]

        inputs = [
            format_input(row["prompt"], row["dream"], row["symbols"])
            for _, row in batch_df.iterrows()
        ]
        max_input_tokens=model_pipeline.tokenizer.model_max_length
        for prompt in inputs:
            token_count = len(model_pipeline.tokenizer.encode(prompt))
            if token_count > max_input_tokens:
                print(f"⚠️ Prompt truncated: {token_count} tokens (limit = {max_input_tokens})")

        outputs = model_pipeline(inputs, **kwargs)
        interpretations.extend(outputs)

    df["interpretation"] = interpretations
    return df

