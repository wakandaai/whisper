import sys
import argparse
import os
import whisper
from tqdm import tqdm
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

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
    parser.add_argument('--sample_len', type=int, default=256, help='Maximum number of tokens to decode')
    parser.add_argument('--length_penalty', type=float, default=None, help='Length penalty for beam search')
    parser.add_argument('--no_labels', action='store_true', help='Run inference only without computing metrics (no ground truth labels required)')
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
    
    # Check required files based on whether we have labels or not
    if not args.no_labels:
        # When we have labels, require text.tsv for evaluation
        assert os.path.exists(os.path.join(data_dir, 'text.tsv')), "text.tsv not found"
    
    # wav.scp is always required
    assert os.path.exists(os.path.join(data_dir, 'wav.scp')), "wav.scp not found"

    # check task
    task = args.task.lower()
    assert task in ['transcribe', 'translate'], "Task must be either 'transcribe' or 'translate'"

    # load the data
    text_dict = {}
    if not args.no_labels:
        # read the text.tsv only if we have labels
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
            # check if the audio file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {audio_path} does not exist. Skipping {audio_file}.")
                continue
            wav_scp_dict[audio_file] = audio_path
    
    # Prepare lists for the dataset
    utterances = list(wav_scp_dict.keys())
    audio_paths = [wav_scp_dict[utt] for utt in utterances]

    # make the output directory
    # for example, if data_dir is corpora/test, the output directory will be results/test
    output_dir = os.path.join('results', os.path.basename(data_dir))
    os.makedirs(output_dir, exist_ok=True)

    # make the output files: text, and stats (only if we have labels)
    # text file will have the hypothesis, and detected language
    # stats file will have the WER, CER, BLEU, etc.
    output_text = os.path.join(output_dir, 'text.tsv')
    if not args.no_labels:
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

    # Initialize metric lists only if we have labels
    if not args.no_labels:
        wers = []
        cers = []
        bleus = []
        chrFs = []

    # Simple counters for running stats
    total_processed = 0

    # Open output files
    if args.no_labels:
        # Only open text output file
        with open(output_text, 'w') as f_text:
            # Process in batches
            for batch_uttIDs, batch_mels in tqdm(dataloader, desc=f"Generating predictions for {os.path.basename(data_dir)} in language {args.lang} with {args.model}"):
                # Move to device
                batch_mels = batch_mels.to(model.device)
                # Detect language
                _, probs = model.detect_language(batch_mels)
                detected_languages = [max(p, key=p.get) for p in probs]
                # Decode
                options = whisper.DecodingOptions(task=task, language=args.lang, beam_size=args.beam_size, temperature=args.temperature, length_penalty=args.length_penalty, sample_len=args.sample_len)
                results = whisper.decode(model, batch_mels, options)
                # Write predictions
                for uttID, result, detected_lang in zip(batch_uttIDs, results, detected_languages):
                    gen_text = result.text.strip()
                    f_text.write(f"{uttID}\t{detected_lang}\t{gen_text}\n")
                
                total_processed += len(batch_uttIDs)
                print(f"Processed {total_processed} utterances")
    else:
        # Open both text and stats files for evaluation
        with open(output_text, 'w') as f_text, open(output_stats, 'w') as f_stats:
            f_stats.write("audio\tWER\tCER\tScore\tchrF\tBLEU\n")
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
                    reference = text_dict[uttID].lower().strip().strip('.,!?')
                    f_text.write(f"{uttID}\t{detected_lang}\t{gen_text}\n")
                    if task == 'transcribe':
                        reference_words = reference.split()
                        pred_words = gen_text.split()
                        
                        # Calculate WER
                        wer = torchaudio.functional.edit_distance(reference_words, pred_words) / len(reference_words)
                        wers.append(wer)
                        
                        # Calculate CER (character-level edit distance)
                        cer = torchaudio.functional.edit_distance(list(reference), list(gen_text)) / len(reference)
                        cers.append(cer)
                        
                        # Calculate combined score
                        combined_error = 0.4 * wer + 0.6 * cer
                        score = (1 - combined_error) * 100
                        
                        f_stats.write(f"{uttID}\t{wer}\t{cer}\t{score}\t_\t_\n")
                    elif task == 'translate':
                        # Create references and hypotheses
                        hypothesis = result.text.strip()
                        reference = text_dict[uttID].strip()
                        
                        # Calculate BLEU using sacrebleu - with version reporting
                        bleu = sacrebleu.corpus_bleu(
                            [hypothesis], 
                            [[reference]], 
                            lowercase=True,
                            tokenize='13a'  # Moses tokenizer (standard)
                        )
                        
                        # Calculate chrF using sacrebleu
                        chrf = sacrebleu.corpus_chrf(
                            [hypothesis], 
                            [[reference]]
                        )
                        
                        # Get the scores
                        bleu_score = bleu.score
                        chrf_score = chrf.score

                        bleus.append(bleu_score)
                        chrFs.append(chrf_score)
                        f_stats.write(f"{uttID}\t-\t-\t{chrf_score}\t{bleu_score}\t-\n")

                total_processed += len(batch_uttIDs)
                
                # Print running stats every batch
                if task == 'transcribe' and len(wers) > 0:
                    running_wer = sum(wers) / len(wers)
                    running_cer = sum(cers) / len(cers)
                    running_score = (1 - (0.4 * running_wer + 0.6 * running_cer)) * 100
                    print(f"Processed {total_processed} | Running WER: {running_wer:.4f} | Running CER: {running_cer:.4f} | Running Score: {running_score:.2f}")
                elif task == 'translate' and len(bleus) > 0:
                    running_bleu = sum(bleus) / len(bleus)
                    running_chrf = sum(chrFs) / len(chrFs)
                    print(f"Processed {total_processed} | Running BLEU: {running_bleu:.2f} | Running chrF: {running_chrf:.2f}")

        # Write average metrics only if we computed them
        if len(wers) > 0:
            avg_wer = sum(wers) / len(wers)
            avg_cer = sum(cers) / len(cers)
            avg_combined_error = 0.4 * avg_wer + 0.6 * avg_cer
            avg_score = (1 - avg_combined_error) * 100
            
            with open(os.path.join(output_dir, 'average_wer.txt'), 'w') as f:
                f.write(str(avg_wer))
            with open(os.path.join(output_dir, 'average_cer.txt'), 'w') as f:
                f.write(str(avg_cer))
            with open(os.path.join(output_dir, 'average_score.txt'), 'w') as f:
                f.write(str(avg_score))
                
            print(f"Average WER: {avg_wer}")
            print(f"Average CER: {avg_cer}")
            print(f"Average Score: {avg_score}")
        elif len(bleus) > 0:  # write average BLEU and chrF
            with open(os.path.join(output_dir, 'average_bleu.txt'), 'w') as f:
                f.write(str(sum(bleus) / len(bleus)))
            with open(os.path.join(output_dir, 'average_chrF.txt'), 'w') as f:
                f.write(str(sum(chrFs) / len(chrFs)))
            print(f"Average BLEU: {sum(bleus) / len(bleus)}")
            print(f"Average chrF: {sum(chrFs) / len(chrFs)}")

    # Print completion message
    if args.no_labels:
        print(f"Inference completed. Predictions saved to {output_text}")
    else:
        print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)