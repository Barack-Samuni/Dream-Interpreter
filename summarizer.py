import torch
from transformers import pipeline
from tqdm import tqdm

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
