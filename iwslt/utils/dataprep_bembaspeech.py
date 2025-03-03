import os
import sys
import argparse

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='BembaSpeech Data Preparation')
    parser.add_argument('--base_path', type=str, default='corpora/BembaSpeech/bem', help='Base path')
    parser.add_argument('--output_dir', type=str, default='bembaspeech', help='Output directory')
    args = parser.parse_args()
    return args

# main
def main(args):
    # find the audio files and make a wav.scp where the key is the filename and the value is the absolute path to the audio file
    audio_path = os.path.join(args.base_path, 'audio')
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    wav_scp_dict = {}
    for audio_file in audio_files:
        wav_scp_dict[audio_file] = os.path.abspath(os.path.join(audio_path, audio_file))
    
    # read the tsvs
    train_tsv = os.path.join(args.base_path, 'train.tsv')

    # first line is audio	sentence
    # the rest are the audio file name and the corresponding sentence
    train_text_dict = {}
    with open(train_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            audio_file, sentence = line.strip().split('\t')
            train_text_dict[audio_file] = sentence
    
    dev_tsv = os.path.join(args.base_path, 'dev.tsv')
    dev_text_dict = {}
    with open(dev_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            audio_file, sentence = line.strip().split('\t')
            dev_text_dict[audio_file] = sentence

    test_tsv = os.path.join(args.base_path, 'test.tsv')
    test_text_dict = {}
    with open(test_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            audio_file, sentence = line.strip().split('\t')
            test_text_dict[audio_file] = sentence

    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # make the partitions
    splits = ['train', 'dev', 'test']
    for split in splits:
      # make the subdirectory for each split
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # make the text file
        text_file = os.path.join(split_dir, 'text.tsv')
        if split == 'train':
            text_dict = train_text_dict
        elif split == 'dev':
            text_dict = dev_text_dict
        else:
            text_dict = test_text_dict
        with open(text_file, 'w') as f:
            for audio_file, sentence in text_dict.items():
                f.write(f"{audio_file}\t{sentence}\n")
        # make the wav.scp file
        wav_scp_file = os.path.join(split_dir, 'wav.scp')
        with open(wav_scp_file, 'w') as f:
            for audio_file, _ in text_dict.items():
                f.write(f"{audio_file}\t{wav_scp_dict[audio_file]}\n")

    # make a single large text file for the entire training set, will be useful for training the tokenizer
    train_text_file = os.path.join(output_dir, 'train_text.tsv')
    with open(train_text_file, 'w') as f:
        for audio_file, sentence in train_text_dict.items():
            f.write(f"{sentence}\n")
    
    print("Data preparation complete")

if __name__ == '__main__':
    args = parse_args()
    main(args)