from transformers import AutoTokenizer
from datasets import load_dataset
from joblib import Parallel, delayed
import pandas as pd
import time

start_time = time.time()

# Model for tokenization
model = "bigscience/mt0-xxl-mt" # Replace with desired model/tokenier
name_for_csv = 'mt0' # For lines 66 and 67

# Load FLORES dataset and convert to Pandas DataFrame
flores_plus_dev = load_dataset("openlanguagedata/flores_plus", split='dev').to_pandas()
flores_plus_dev = flores_plus_dev.rename(columns={'iso_15924': 'language'})

# Clean and prepare text data
def clean_texts(ids, texts):
    cleaned_ids = []
    cleaned_texts = []
    for id_, text in zip(ids, texts):
        if isinstance(text, str) and text.strip():
            cleaned_ids.append(id_)
            cleaned_texts.append(text)
    return cleaned_ids, cleaned_texts

# Function to compute fertility for a single text
def fertility(text, tokenizer):
    tokenized = tokenizer.tokenize(text)
    num_words = len(text.split())
    fertility_score = len(tokenized) / num_words if num_words > 0 else 0
    return fertility_score, tokenized

# Function to compute fertility in parallel batches
def parallel_fertility_batches(texts, tokenizer, batch_size=32, n_jobs=-1):
    def process_batch(batch):
        return [fertility(text, tokenizer) for text in batch]

    # Split texts into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(batch) for batch in batches)
    # Flatten the list of results
    results_flat = [item for sublist in results for item in sublist]
    return zip(*results_flat)  # Separate into fertility scores and tokens

# Main function
def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Clean and prepare text and ID data
    ids = flores_plus_dev['id'].tolist()
    raw_texts = flores_plus_dev['text'].tolist()
    cleaned_ids, cleaned_texts = clean_texts(ids, raw_texts)

    # Compute fertility scores and tokenized texts in parallel
    fertility_scores, tokenized_texts = parallel_fertility_batches(cleaned_texts, tokenizer, batch_size=32, n_jobs=-1)

    # Save results to CSV
    output_df = pd.DataFrame({
        'id': cleaned_ids,
        'text': cleaned_texts,
        'fertility_score': fertility_scores,
        'tokens': tokenized_texts  # Keep tokens as a list
    })
    output_df.to_csv(f"flores_fertilized_and_tokenized_with_{name_for_csv}.csv", index=False)
    print(f"Results saved to 'flores_fertilized_and_tokenized_with_{name_for_csv}.csv'")

if __name__ == "__main__":
    main()

end_time = time.time()
print(f"Runtime: {end_time-start_time} seconds")
