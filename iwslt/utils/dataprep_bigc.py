import os
import sys
import argparse
import json
import re
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='BigC Multitask Data Preparation')
    parser.add_argument('--base_path', type=str, default='corpora/bigc/bem', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    parser.add_argument('--tasks', type=str, required=True, help='Comma-separated tasks (e.g., transcribe,translate)')
    args = parser.parse_args()
    return args

def main(args):
    tasks = [task.strip() for task in args.tasks.split(',')]
    
    # Validate tasks
    for task in tasks:
        if task not in ['transcribe', 'translate']:
            raise ValueError(f"Invalid task: {task}. Must be 'transcribe' or 'translate'")
    
    # Find audio files
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    
    wav_scp_dict = {}
    for audio_file in audio_files:
        wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
    # Read the jsonl files for each split
    splits = ['train', 'dev', 'test']
    split_data = {}
    
    for split in splits:
        json_file = os.path.join(f"{args.base_path}/splits", f'{split if split != "dev" else "valid"}.jsonl')
        split_data[split] = []
        
        with open(json_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub(r'\\n', '', line)
                data = json.loads(line)
                audio_file = data['audio_id']
                
                # Create entries for each task
                for task in tasks:
                    if task == 'transcribe':
                        text = data['bem_transcription']
                        source_lang = 'sw'
                        target_lang = 'sw'
                    else:  # translate
                        text = data['en_translation']
                        source_lang = 'sw'
                        target_lang = 'en'
                    
                    split_data[split].append({
                        'audio_id': audio_file,
                        'task': task,
                        'source_language': source_lang,
                        'target_language': target_lang,
                        'text': text.strip()
                    })
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Write output files for each split
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Write wav.scp
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        
        # Write metadata CSV
        metadata_file = os.path.join(split_dir, 'metadata.csv')
        
        written_audio_ids = set()
        
        with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['audio_id', 'task', 'source_language', 'target_language', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in split_data[split]:
                audio_file = entry['audio_id']
                text = entry['text']
                
                # Skip invalid entries
                if text in [".", "", "NOT PLAYING", "NOT FOUND"]:
                    print(f"Skipping audio file {audio_file} with invalid text")
                    continue
                
                if audio_file not in wav_scp_dict:
                    print(f"Audio file {audio_file} not found in the audio directory")
                    continue
                
                # Write to CSV
                writer.writerow(entry)
                written_audio_ids.add(audio_file)
        
        # Write wav.scp with only the audio files that were written
        with open(wav_scp_file, 'w') as f:
            for audio_id in sorted(written_audio_ids):
                f.write(f"{audio_id}\t{wav_scp_dict[audio_id]}\n")
    
    print("Multitask data preparation complete")
    print(f"Tasks: {', '.join(tasks)}")
    for split in splits:
        print(f"{split}: {len(split_data[split])} samples")

if __name__ == '__main__':
    args = parse_args()
    main(args)