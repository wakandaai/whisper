import os
import sys
import argparse
import csv

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(21474836)

# Mapping from directory names to two-letter language codes
LANG_CODE_MAP = {
    'eng_Latn': 'en',
    'fra_Latn': 'fr',
    'kin_Latn': 'rw',
    'swa_Latn': 'sw'
}
INVERTED_LANG_CODE_MAP = {v: k for k, v in LANG_CODE_MAP.items()}

def parse_args():
    parser = argparse.ArgumentParser(description='CommonVoice Multitask Data Preparation for ASR and AST')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to CommonVoice dataset')
    parser.add_argument('--audio_dir', type=str, default='clips_16kHz', help='Audio directory name (default: clips_16kHz)')
    parser.add_argument('--output_dir', type=str, default='corpora/cv', help='Output directory')
    parser.add_argument('--source_lang', type=str, required=True, help='Source language code (e.g., sw)')
    parser.add_argument('--tasks', type=str, required=True, help='Comma-separated tasks (e.g., transcribe,translate)')
    parser.add_argument('--target_langs', type=str, default='', help='Comma-separated target language codes for translation (e.g., en,fr)')
    args = parser.parse_args()
    return args

def read_tsv(filepath):
    """Read a CommonVoice TSV file and return a dict mapping path (without extension) to sentence"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Strip extension to make audio_id extension-agnostic
            audio_id = os.path.splitext(row['path'])[0]
            sentence = row['sentence']
            data[audio_id] = sentence
    return data

def get_lang_code(lang_dir):
    """Convert language directory name to two-letter code"""
    if lang_dir in LANG_CODE_MAP:
        return LANG_CODE_MAP[lang_dir]
    # Fallback: try to extract from the directory name
    print(f"Warning: Language directory '{lang_dir}' not found in LANG_CODE_MAP, using fallback (first two letters).")
    return lang_dir[:2].lower()

def main(args):
    tasks = [task.strip() for task in args.tasks.split(',')]
    
    # Validate tasks
    for task in tasks:
        if task not in ['transcribe', 'translate']:
            raise ValueError(f"Invalid task: {task}. Must be 'transcribe' or 'translate'")
    
    # Parse target languages for translation
    target_lang_dirs = []
    if 'translate' in tasks:
        if not args.target_langs:
            raise ValueError("--target_langs is required when 'translate' task is specified")
        target_langs = [lang.strip() for lang in args.target_langs.split(',')]
        target_lang_dirs = [INVERTED_LANG_CODE_MAP.get(lang, lang) for lang in target_langs]
        
        # Validate translation directories exist
        for lang_dir in target_lang_dirs:
            lang_path = os.path.join(args.base_path, lang_dir)
            if not os.path.isdir(lang_path):
                raise ValueError(f"Translation directory not found: {lang_path}")
    
    # Find audio files
    audio_path = os.path.join(args.base_path, args.audio_dir)
    if not os.path.isdir(audio_path):
        raise ValueError(f"Audio directory not found: {audio_path}")
    
    audio_files = set(f for f in os.listdir(audio_path) if f.endswith('.wav'))
    print(f"Found {len(audio_files)} audio files in {audio_path}")
    
    # Create mapping from base filename (no extension) to full wav path
    wav_scp_dict = {}
    for wav_file in audio_files:
        base_name = os.path.splitext(wav_file)[0]
        wav_scp_dict[base_name] = os.path.abspath(os.path.join(audio_path, wav_file))
    
    # Process each split
    splits = ['train', 'dev', 'test']
    split_data = {}
    
    for split in splits:
        # Read source transcriptions
        source_tsv = os.path.join(args.base_path, f'{split}.tsv')
        if not os.path.isfile(source_tsv):
            print(f"Warning: {source_tsv} not found, skipping split {split}")
            continue
        
        source_data = read_tsv(source_tsv)
        print(f"Read {len(source_data)} entries from {source_tsv}")
        
        # Read translations for each target language
        translation_data = {}
        for lang_dir in target_lang_dirs:
            trans_tsv = os.path.join(args.base_path, lang_dir, f'{split}.tsv')
            if not os.path.isfile(trans_tsv):
                print(f"Warning: {trans_tsv} not found, skipping {lang_dir} for split {split}")
                continue
            translation_data[lang_dir] = read_tsv(trans_tsv)
            print(f"Read {len(translation_data[lang_dir])} entries from {trans_tsv}")
        
        split_data[split] = []
        
        # Create entries for each audio file
        for audio_id, transcription in source_data.items():
            # Add transcription task
            if 'transcribe' in tasks:
                split_data[split].append({
                    'audio_id': audio_id,
                    'task': 'transcribe',
                    'source_language': args.source_lang,
                    'target_language': args.source_lang,
                    'text': transcription.strip()
                })
            
            # Add translation tasks
            if 'translate' in tasks:
                for lang_dir in target_lang_dirs:
                    if lang_dir not in translation_data:
                        continue
                    if audio_id not in translation_data[lang_dir]:
                        continue
                    
                    target_lang = get_lang_code(lang_dir)
                    translation = translation_data[lang_dir][audio_id]
                    
                    split_data[split].append({
                        'audio_id': audio_id,
                        'task': 'translate',
                        'source_language': args.source_lang,
                        'target_language': target_lang,
                        'text': translation.strip()
                    })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write output files for each split
    for split in splits:
        if split not in split_data:
            continue
        
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        metadata_file = os.path.join(split_dir, 'metadata.csv')
        
        written_audio_ids = set()
        valid_entries = 0
        skipped_entries = 0
        
        with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['audio_id', 'task', 'source_language', 'target_language', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in split_data[split]:
                audio_id = entry['audio_id']
                text = entry['text']
                
                # Skip invalid entries
                if text in [".", "", "NOT PLAYING", "NOT FOUND"] or not text:
                    skipped_entries += 1
                    continue
                
                if audio_id not in wav_scp_dict:
                    skipped_entries += 1
                    continue
                
                # Write to CSV
                writer.writerow(entry)
                written_audio_ids.add(audio_id)
                valid_entries += 1
        
        # Write wav.scp with only the audio files that were written
        with open(wav_scp_file, 'w') as f:
            for audio_id in sorted(written_audio_ids):
                f.write(f"{audio_id}\t{wav_scp_dict[audio_id]}\n")
        
        print(f"{split}: {valid_entries} valid entries, {skipped_entries} skipped, {len(written_audio_ids)} unique audio files")
    
    print("\nMultitask data preparation complete")
    print(f"Tasks: {', '.join(tasks)}")
    if target_lang_dirs:
        print(f"Target languages: {', '.join(target_lang_dirs)} -> {', '.join(get_lang_code(l) for l in target_lang_dirs)}")

if __name__ == '__main__':
    args = parse_args()
    main(args)