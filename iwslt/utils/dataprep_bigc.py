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
Updated Dataset
python# Update to whisper/trainer.py - replace WhisperDataset class

class WhisperDataset(Dataset):
    def __init__(
        self,
        wav_scp: str,
        metadata_csv: str,
        model: Whisper,
    ):
        """
        Dataset for Whisper multitask fine-tuning.
        
        Parameters
        ----------
        wav_scp : str
            Path to wav.scp file with audio paths
        metadata_csv : str
            Path to CSV file with columns: audio_id, task, source_language, target_language, text
        model : Whisper
            The Whisper model
        """
        self.model = model
        self.dims = model.dims
        
        # Read wav.scp
        with open(wav_scp, 'r') as f:
            audio_lines = [line.strip().split("\t") for line in f]
        
        # Create audio_id to filepath mapping
        self.audio_map = {id: path for id, path in audio_lines}
        
        # Read metadata CSV
        self.samples = []
        with open(metadata_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                audio_id = row['audio_id']
                if audio_id in self.audio_map:
                    self.samples.append({
                        'audio_id': audio_id,
                        'audio_path': self.audio_map[audio_id],
                        'task': row['task'],
                        'source_language': row['source_language'],
                        'target_language': row['target_language'],
                        'text': row['text']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        audio_id = sample['audio_id']
        audio_path = sample['audio_path']
        task = sample['task']
        source_language = sample['source_language']
        target_language = sample['target_language']
        text = sample['text']
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.dims.n_mels).to(torch.float)
        
        # Create tokenizer for this sample
        tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=target_language,
            task=task
        )
        
        # Get special tokens
        sot_token = [tokenizer.sot]
        source_lang_token = [tokenizer.to_language_token(source_language)]
        target_lang_token = [tokenizer.to_language_token(target_language)]
        task_token = [tokenizer.transcribe if task == "transcribe" else tokenizer.translate]
        eot_token = [tokenizer.eot]
        
        # Encode the text
        text_tokens = tokenizer.encode(" " + text.strip())
        
        # Create input and target tokens with language ID
        # Input: SOT + target_lang + task + text tokens (for teacher forcing)
        input_tokens = torch.tensor(sot_token + target_lang_token + task_token + text_tokens)
        
        # Target: source_lang + task + text tokens + EOT (shifted right from input)
        target_tokens = torch.tensor(source_lang_token + task_token + text_tokens + eot_token)
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_id": audio_id,
            "task": task,
            "source_language": source_language,
            "target_language": target_language,
            "reference_text": text.strip()
        }
    
    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"] for item in batch]
        input_tokens = [item["input_tokens"] for item in batch]
        target_tokens = [item["target_tokens"] for item in batch]
        audio_ids = [item["audio_id"] for item in batch]
        tasks = [item["task"] for item in batch]
        source_languages = [item["source_language"] for item in batch]
        target_languages = [item["target_language"] for item in batch]
        reference_texts = [item["reference_text"] for item in batch]
        
        mels = torch.stack(mels)
        
        # Get EOT token from first sample (all should be the same)
        tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=target_languages[0],
            task=tasks[0]
        )
        
        # Pad token sequences
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=tokenizer.eot)
        target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=tokenizer.eot)
        
        return {
            "mel": mels,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_ids": audio_ids,
            "tasks": tasks,
            "source_languages": source_languages,
            "target_languages": target_languages,
            "reference_texts": reference_texts
        }