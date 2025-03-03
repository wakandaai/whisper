import sys
import argparse
import os
import whisper
from tqdm import tqdm
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--data', type=str, default='corpora/test', help='Path to the data')
    parser.add_argument('--task', type=str, default='transcribe', help='Task')
    parser.add_argument('--model', type=str, default='turbo', help='Model')
    parser.add_argument('--lang', type=str, default='sw', help='Language')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size for inference')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for inference')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    return args

# Dataset
class AudioDataset(Dataset):
    def __init__(self, audio_files, audio_paths, model_dims):
        self.audio_files = audio_files
        self.audio_paths = audio_paths
        self.model_dims = model_dims
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_path = self.audio_paths[idx]
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model_dims.n_mels)
        
        return audio_file, mel

# main
def main(args):
    
    model = whisper.load_model(args.model)

    data_dir = args.data
    # ensure that text.tsv and wav.scp exist
    assert os.path.exists(os.path.join(data_dir, 'text.tsv')), "text.tsv not found"
    assert os.path.exists(os.path.join(data_dir, 'wav.scp')), "wav.scp not found"

    # check task
    task = args.task.lower()
    assert task in ['transcribe', 'translate'], "Task must be either 'transcribe' or 'translate'"

    # load the data
    # read the text.tsv
    text_dict = {}
    with open(os.path.join(data_dir, 'text.tsv'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            audio_file, sentence = line.strip().split('\t')
            text_dict[audio_file] = sentence
    # read the wav.scp
    wav_scp_dict = {}
    with open(os.path.join(data_dir, 'wav.scp'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            audio_file, audio_path = line.strip().split('\t')
            wav_scp_dict[audio_file] = audio_path
    
    # Prepare lists for the dataset
    utterances = list(wav_scp_dict.keys())
    audio_paths = [wav_scp_dict[utt] for utt in utterances]

    # make the output directory
    # for example, if data_dir is corpora/test, the output directory will be results/test
    output_dir = os.path.join('results', os.path.basename(data_dir))
    os.makedirs(output_dir, exist_ok=True)

    # make the output files: text, and stats
    # text file will have the hypothesis, and detected language
    # stats file will have the WER, CER, BLEU, etc.
    output_text = os.path.join(output_dir, 'text.tsv')
    output_stats = os.path.join(output_dir, 'stats.tsv')

    # Create dataset and dataloader
    dataset = AudioDataset(utterances, audio_paths, model.dims)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    wers = []
    with open(output_text, 'w') as f_text, open(output_stats, 'w') as f_stats:
        f_stats.write("audio\tWER\tCER\tBLEU\n")
        # Process in batches
        for batch_uttIDs, batch_mels in tqdm(dataloader, desc=f"Evaluating {os.path.basename(data_dir)} in language {args.lang} with {args.model}"):
            # Move to device
            batch_mels = batch_mels.to(model.device)
            # Detect language
            _, probs = model.detect_language(batch_mels)
            detected_languages = [max(p, key=p.get) for p in probs]
            # Decode
            options = whisper.DecodingOptions(task=task, language=args.lang, beam_size=args.beam_size, temperature=args.temperature)
            results = whisper.decode(model, batch_mels, options)
            # Compute metrics
            for uttID, result, detected_lang in zip(batch_uttIDs, results, detected_languages):
                gen_text = result.text.lower().strip().strip('.,!?')
                f_text.write(f"{uttID}\t{detected_lang}\t{gen_text}\n")
                wer = torchaudio.functional.edit_distance(text_dict[uttID].split(), gen_text.split()) / len(text_dict[uttID].split())
                wers.append(wer)
                f_stats.write(f"{uttID}\t{wer}\t-\t-\n")
        # Write average WER
        with open(os.path.join(output_dir, 'average_wer.txt'), 'w') as f:
            f.write(str(sum(wers) / len(wers)))
        print(f"Average WER: {sum(wers) / len(wers)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)