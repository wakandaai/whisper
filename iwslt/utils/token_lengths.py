import pandas as pd
import whisper

# === CONFIG ===
CSV_PATH = "corpora/train/metadata.csv"
OUTPUT_CSV = "corpora/metadata_with_token_lengths.csv"
TOKEN_LIMIT = 448 - 20   # Reserve some margin for special tokens
MODEL = "small"

df = pd.read_csv(CSV_PATH)

# Load Whisper model and tokenizer
model = whisper.load_model(MODEL)
tokenizer = whisper.tokenizer.get_tokenizer(
    multilingual=True, 
    task="transcribe"
)

def count_tokens(text):
    if pd.isna(text):
        return 0
    # whisper tokenizer returns list of IDs
    ids = tokenizer.encode(text)
    return len(ids)

# Compute token lengths
df["token_length"] = df["text"].astype(str).apply(count_tokens)

# Flag rows exceeding limit
df["too_long"] = df["token_length"] > TOKEN_LIMIT

# Save
df.to_csv(OUTPUT_CSV, index=False)

# Summary
n_too_long = df["too_long"].sum()
print(f"Total rows: {len(df)}")
print(f"Rows exceeding {TOKEN_LIMIT} tokens: {n_too_long}")
print(f"Saved output to {OUTPUT_CSV}")
