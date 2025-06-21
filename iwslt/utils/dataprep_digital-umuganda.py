import os
import sys
import argparse
import json
import re
import unicodedata
# parser
def parse_args():
    parser = argparse.ArgumentParser(description='Digital Umudanda Data Preparation')
    parser.add_argument('--base_path', type=str, default='540h', help='Base path')
    parser.add_argument('--output_dir', type=str, default='corpora', help='Output directory')
    args = parser.parse_args()
    return args

# main
def main(args):

    # find the audio files and make a wav.scp where the key is the filename and the value is the absolute path to the audio file
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path)]

    # wav_scp_dict = {}
    # for audio_file in audio_files:
    #     wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
    train_json = os.path.join(args.base_path, 'train.json')
    dev_json = os.path.join(args.base_path, 'dev_test.json')
    test_json = os.path.join(args.base_path, 'test.json')
    
    # read the json file
    train_dict = {}
    with open(train_json, 'r') as f:
        dataset = json.load(f)
    for id in dataset.keys():
        audio_path = dataset[id]['audio_path']
        transcription = dataset[id]['transcription'].strip().lower()
        transcription = transcription.replace('\n', '')
        # Remove unwanted characters (keep letters, numbers, spaces)
        transcription = ''.join(char for char in transcription if unicodedata.category(char).startswith(('L', 'N', 'Z')))
        # Normalize whitespace to single spaces
        transcription = re.sub(r'\s+', ' ', transcription).strip()
        train_dict[id] = {
            'audio_path': audio_path,
            'transcription': transcription
        }
    dev_dict = {}
    with open(dev_json, 'r') as f:
        dataset = json.load(f)
    for id in dataset.keys():
        audio_path = dataset[id]['audio_path']
        transcription = dataset[id]['transcription'].strip().lower()
        transcription = transcription.replace('\n', '')
        # Remove unwanted characters (keep letters, numbers, spaces)
        transcription = ''.join(char for char in transcription if unicodedata.category(char).startswith(('L', 'N', 'Z')))
        # Normalize whitespace to single spaces
        transcription = re.sub(r'\s+', ' ', transcription).strip()
        dev_dict[id] = {
            'audio_path': audio_path,
            'transcription': transcription
        }
    test_dict = {}
    with open(test_json, 'r') as f:
        dataset = json.load(f)
    for id in dataset.keys():
        audio_path = dataset[id]['audio_path']
        test_dict[id] = {
            'audio_path': audio_path
        }

    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # make the partitions
    splits = ['train', 'dev', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # make the text file and wav.scp file for appending
        text_file = os.path.join(split_dir, 'text.tsv')
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        if split == 'train':
            data_dict = train_dict
        elif split == 'dev':
            data_dict = dev_dict
        elif split == 'test':
            data_dict = test_dict

        if split != 'test':
            with open(wav_scp_file, 'w') as f, open(text_file, 'w') as t:
                for id, data in data_dict.items():
                    # Check if transcription exists first
                    if 'transcription' not in data:
                        continue
                        
                    audio_file = data['audio_path']
                    audio_path = os.path.abspath(os.path.join(args.base_path, audio_file))
                    
                    # Check if audio file exists
                    if not os.path.exists(audio_path):
                        continue
                    
                    # Both exist - write to both files
                    transcription = data['transcription']
                    f.write(f"{id}\t{audio_path}\n")
                    t.write(f"{id}\t{transcription}\n")
        else:
            with open(wav_scp_file, 'w') as f:
                for id, data in data_dict.items():
                    audio_file = data['audio_path']
                    audio_path = os.path.abspath(os.path.join(args.base_path, audio_file))
                    
                    # Check if audio file exists
                    if not os.path.exists(audio_path):
                        continue
                    
                    # Write to wav.scp file
                    f.write(f"{id}\t{audio_path}\n")
                
    
    print("Data preparation complete")

if __name__ == '__main__':
    args = parse_args()
    main(args)