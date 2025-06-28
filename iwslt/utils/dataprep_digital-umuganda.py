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
    parser.add_argument('--allow_mismatch', action='store_true', help='Allow mismatch between audio and transcription')
    args = parser.parse_args()
    return args

# main
def main(args):

    # find the audio files and make a wav.scp where the key is the filename and the value is the absolute path to the audio file
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path)]
    
    train_json = os.path.join(args.base_path, 'train.json')
    dev_json = os.path.join(args.base_path, 'dev_test.json')
    test_json = os.path.join(args.base_path, 'test.json')
    
    # train
    train_dict = {}
    with open(train_json, 'r') as f:
        dataset = json.load(f)
    for id in dataset.keys():
        audio_path = dataset[id]['audio_path']
        transcription = dataset[id]['transcription']
        if transcription is not None:
            transcription = transcription.strip().lower()
            transcription = transcription.replace('\n', '')
            # remove punctuation
            transcription = re.sub(r'[^\w\s\']', '', transcription)
            # Normalize whitespace to single spaces
            transcription = re.sub(r'\s+', ' ', transcription).strip()
            train_dict[id] = {
                'audio_path': audio_path,
                'transcription': transcription
            }
        else:
            # If transcription is None, we still want to keep the audio path
            train_dict[id] = {
                'audio_path': audio_path
            }
    # dev
    dev_dict = {}
    with open(dev_json, 'r') as f:
        dataset = json.load(f)
    for id in dataset.keys():
        audio_path = dataset[id]['audio_path']
        transcription = dataset[id]['transcription']
        if transcription is not None:
            transcription = transcription.strip().lower()
            transcription = transcription.replace('\n', '')
            # remove punctuation
            transcription = re.sub(r'[^\w\s\']', '', transcription)
            # Normalize whitespace to single spaces
            transcription = re.sub(r'\s+', ' ', transcription).strip()
            dev_dict[id] = {
                'audio_path': audio_path,
                'transcription': transcription
            }
        else:
            # If transcription is None, we still want to keep the audio path
            dev_dict[id] = {
                'audio_path': audio_path
            }
    # test
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
                        
                    audio_file = data['audio_path']
                    audio_path = os.path.abspath(os.path.join(args.base_path, audio_file))
                    
                    try:
                        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                            print(f"Skipping invalid/empty file: {audio_path}")
                            continue
                    except OSError:
                        print(f"Skipping inaccessible file: {audio_path}")
                        continue

                    # Check if transcription exists
                    if 'transcription' not in data:
                        if not args.allow_mismatch:
                            continue
                        else:
                            # If allow_mismatch is True, write only the audio path
                            f.write(f"{id}\t{audio_path}\n")
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