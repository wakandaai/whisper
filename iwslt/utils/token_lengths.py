import os
import argparse
import pandas as pd
import whisper

def parse_args():
    parser = argparse.ArgumentParser(description='Filter corpus entries exceeding Whisper token limit')
    parser.add_argument('--corpus_dir', type=str, required=True, help='Path to processed corpus directory (e.g., corpora/cv/sw_to_sw_en)')
    parser.add_argument('--token_limit', type=int, default=428, help='Max tokens allowed (default: 428 = 448 - 20 margin)')
    parser.add_argument('--model', type=str, default='small', help='Whisper model for tokenizer (default: small)')
    parser.add_argument('--splits', type=str, default='train,dev', help='Comma-separated splits to process')
    args = parser.parse_args()
    return args

def main(args):
    splits = [s.strip() for s in args.splits.split(',')]
    
    # Load tokenizer
    print(f"Loading Whisper tokenizer (model: {args.model})...")
    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True,
        task="transcribe"
    )
    
    def count_tokens(text):
        if pd.isna(text) or text == "":
            return 0
        return len(tokenizer.encode(str(text)))
    
    total_removed = 0
    total_kept = 0
    
    for split in splits:
        split_dir = os.path.join(args.corpus_dir, split)
        metadata_file = os.path.join(split_dir, 'metadata.csv')
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        
        if not os.path.isfile(metadata_file):
            print(f"Warning: {metadata_file} not found, skipping {split}")
            continue
        
        # Read metadata
        df = pd.read_csv(metadata_file)
        original_count = len(df)
        
        # Count tokens
        print(f"\n{split}: Computing token lengths for {original_count} entries...")
        df['token_length'] = df['text'].apply(count_tokens)
        
        # Filter
        df_filtered = df[df['token_length'] <= args.token_limit].copy()
        df_filtered = df_filtered.drop(columns=['token_length'])
        
        removed = original_count - len(df_filtered)
        total_removed += removed
        total_kept += len(df_filtered)
        
        # Get unique audio IDs that remain
        remaining_audio_ids = set(df_filtered['audio_id'].unique())
        
        # Determine output paths
        out_metadata = metadata_file
        out_wav_scp = wav_scp_file
        # Write filtered metadata
        df_filtered.to_csv(out_metadata, index=False)
        
        # Filter and write wav.scp
        if os.path.isfile(wav_scp_file):
            with open(wav_scp_file, 'r') as f:
                wav_lines = f.readlines()
            
            filtered_wav_lines = []
            for line in wav_lines:
                audio_id = line.split('\t')[0].strip()
                if audio_id in remaining_audio_ids:
                    filtered_wav_lines.append(line)
            
            with open(out_wav_scp, 'w') as f:
                f.writelines(filtered_wav_lines)
        
        print(f"{split}: {original_count} -> {len(df_filtered)} entries ({removed} removed, {len(remaining_audio_ids)} unique audio files)")
        
        # Show some stats on what was removed
        df_removed = df[df['token_length'] > args.token_limit]
        if len(df_removed) > 0:
            max_tokens = df_removed['token_length'].max()
            avg_tokens = df_removed['token_length'].mean()
            print(f"  Removed entries: max={max_tokens} tokens, avg={avg_tokens:.1f} tokens")
    
    print(f"\n{'='*50}")
    print(f"Total: {total_kept} kept, {total_removed} removed (limit: {args.token_limit} tokens)")

if __name__ == '__main__':
    args = parse_args()
    main(args)