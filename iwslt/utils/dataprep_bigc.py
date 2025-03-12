import os
import sys
import argparse
import json
import re

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='BembaSpeech Data Preparation')
    parser.add_argument('--base_path', type=str, default='corpora/bigc/bem', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    parser.add_argument('--task', type=str, required=True, help='Task')
    args = parser.parse_args()
    return args

# main
def main(args):
    task = args.task.lower()
    if task not in ['transcribe', 'translate']:
        raise ValueError("Task must be either 'transcribe' or 'translate'")

    # find the audio files and make a wav.scp where the key is the filename and the value is the absolute path to the audio file
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    wav_scp_dict = {}
    for audio_file in audio_files:
        wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
    # read the jsonl files
    train_json = os.path.join(f"{args.base_path}/splits", 'train.jsonl')
    train_text_dict = {}
    with open(train_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # remove any \n literal in the json line
            line = re.sub(r'\\n', '', line)
            data = json.loads(line)
            audio_file = data['audio_id']
            if task == 'transcribe':
                text = data['bem_transcription']
            else:
                text = data['en_translation']

            train_text_dict[audio_file] = text.strip()
    
    dev_json = os.path.join(f"{args.base_path}/splits", 'valid.jsonl')
    dev_text_dict = {}
    with open(dev_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # remove any \n literal in the json line
            line = re.sub(r'\\n', '', line)
            data = json.loads(line)
            audio_file = data['audio_id']
            if task == 'transcribe':
                text = data['bem_transcription']
            else:
                text = data['en_translation']

            dev_text_dict[audio_file] = text.strip()

    test_json = os.path.join(f"{args.base_path}/splits", 'test.jsonl')
    test_text_dict = {}
    with open(test_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # remove any \n literal in the json line
            line = re.sub(r'\\n', '', line)
            data = json.loads(line)
            audio_file = data['audio_id']
            if task == 'transcribe':
                text = data['bem_transcription']
            else:
                text = data['en_translation']

            test_text_dict[audio_file] = text.strip()

    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # make the partitions
    splits = ['train', 'dev', 'test']
    for split in splits:
      # make the subdirectory for each split
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # make the text file and wav.scp file
        text_file = os.path.join(split_dir, 'text.tsv')
        if split == 'train':
            text_dict = train_text_dict
        elif split == 'dev':
            text_dict = dev_text_dict
        else:
            text_dict = test_text_dict
        
        # make the wav.scp file
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        with open(wav_scp_file, 'w') as f, open(text_file, 'w') as t:
            for audio_file, text in text_dict.items():
                if text == "." or text == "":
                    print(f"Skipping audio file {audio_file} with empty text")
                    continue
                if audio_file not in wav_scp_dict:
                    print(f"Audio file {audio_file} not found in the audio directory")
                    continue
                if text == "NOT PLAYING" or text == "NOT FOUND":
                    print(f"Skipping audio file {audio_file} which doesn't play")
                    continue
                
                f.write(f"{audio_file}\t{wav_scp_dict[audio_file]}\n")
                t.write(f"{audio_file}\t{text}\n")

    # make a single large text file for the entire training set, will be useful for training the tokenizer
    train_text_file = os.path.join(output_dir, 'train_text.tsv')
    with open(train_text_file, 'w') as f:
        for audio_file, sentence in train_text_dict.items():
            f.write(f"{sentence}\n")
    
    print("Data preparation complete")

if __name__ == '__main__':
    args = parse_args()
    main(args)