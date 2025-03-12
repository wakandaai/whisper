import argparse
from whisper.tokenizer import get_tokenizer
from whisper.audio import load_audio
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Text Preprocessing")
    parser.add_argument("--text", type=str, help="Text to preprocess")
    parser.add_argument("--tokenizer", type=str, default="char", help="Tokenizer")
    parser.add_argument("--case_standardization", type=str, default="lower", help="Case standardization")
    parser.add_argument("--wav_scp", type=str, help="Path to wav.scp")
    return parser.parse_args()

def main(args):
    # text = args.text
    # if not text:
    #     raise ValueError("Please provide a text to preprocess")
    # tokenizer = get_tokenizer(multilingual=True, language="sw", task="transcribe")
    # text_tokens = tokenizer.encode(text)
    # language_token = tokenizer.language_token
    # token_sequence = [tokenizer.sot] + [language_token] + text_tokens + [tokenizer.eot]
    # print(f"{tokenizer.tokenize(text)}")
    # print(f"{tokenizer.decode(token_sequence)}")
    # # print all special tokens sorted by their token values
    # special_tokens = tokenizer.special_tokens
    # for token in sorted(special_tokens, key=lambda x: special_tokens[x], reverse=True):
    #     print(f"{token}: {special_tokens[token]}")

    # test all audio
    if args.wav_scp:
        # read file
        with open(args.wav_scp, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            audio_name, audio_path = line.split("\t")
            print(f"Processing {audio_name}")
            _ = load_audio(audio_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)