import os
import sys
import argparse
import pandas as pd

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='BembaSpeech Data Preparation')
    parser.add_argument('--base_path', type=str, default='mymy', help='Base path')
    parser.add_argument('--output_dir', type=str, default='ffstc', help='Output directory')
    args = parser.parse_args()
    return args

# main
def main(args):
    for split in ['train', 'valid']:
        audio_path = os.path.join(args.base_path, split)
        audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
        wav_scp_dict = {}
        for audio_file in audio_files:
            wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
        # read the csv file
        csv = os.path.join(args.base_path, f'{split}.csv')
        text_dict = {}
        df = pd.read_csv(csv)
        # csv structure is ID,utterance,filename,duration
        for index, row in df.iterrows():
            audio_file = row['filename']
            text = row['utterance']
            text_dict[audio_file] = text.strip()

        # create the output directory if it does not exist
        if split == 'valid':
            output_dir = os.path.join(args.output_dir, 'dev')
        else:
            output_dir = os.path.join(args.output_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        # write the text file and wav.scp
        wav_scp_file = os.path.join(output_dir, 'wav.scp')
        text_file = os.path.join(output_dir, 'text.tsv')
        with open(text_file, 'w') as f, open(wav_scp_file, 'w') as w:
            for audio_file, text in text_dict.items():
                f.write(f'{audio_file}\t{text}\n')
                w.write(f'{audio_file}\t{wav_scp_dict[audio_file]}\n')
        
    print("Data preparation complete")

if __name__ == '__main__':
    args = parse_args()
    main(args)