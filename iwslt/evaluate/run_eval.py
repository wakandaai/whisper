import sys
import argparse
import os
import whisper
from tqdm import tqdm
import torchaudio

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--data', type=str, default='iwslt/corpora/test', help='Path to the data')
    parser.add_argument('--task', type=str, default='asr', help='Task')
    parser.add_argument('--model', type=str, default='turbo', help='Model')
    args = parser.parse_args()
    return args

# main
def main(args):
    
    model = whisper.load_model(args.model)

    data_dir = args.data
    # ensure that text.tsv and wav.scp exist
    assert os.path.exists(os.path.join(data_dir, 'text.tsv')), "text.tsv not found"
    assert os.path.exists(os.path.join(data_dir, 'wav.scp')), "wav.scp not found"

    # check task
    if args.task.lower() == 's2tt':
        raise NotImplementedError("Speech-to-Text Translation not yet implemented")

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
    
    
    # make the output directory
    # for example, if data_dir is iwslt/corpora/test, the output directory will be iwslt/results/test
    output_dir = os.path.join('iwslt/results', os.path.basename(data_dir))
    os.makedirs(output_dir, exist_ok=True)

    # make the output files: text, and stats
    # text file will have the hypothesis, and detected language
    # stats file will have the WER, CER, BLEU, etc.
    output_text = os.path.join(output_dir, 'text.tsv')
    output_stats = os.path.join(output_dir, 'stats.tsv')

    wers = []
    with open(output_text, 'w') as f_text, open(output_stats, 'w') as f_stats:
        f_stats.write("audio\tWER\tCER\tBLEU\n")
        for audio_file, audio_path in tqdm(wav_scp_dict.items(), desc=f"Evaluating {os.path.basename(data_dir)} with {args.model}"):
            # load the audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            # make the log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            # detect the spoken language
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            # decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)
            # write the hypothesis and detected language to the text file
            f_text.write(f"{audio_file}\t{detected_language}\t{result.text}\n")
            # compute the WER
            wer = torchaudio.functional.edit_distance(text_dict[audio_file].split(), result.text.split()) / len(text_dict[audio_file].split())
            wers.append(wer)

            # write stats
            f_stats.write(f"{audio_file}\t{wer}\t-\t-\n")

        # write the average WER
        with open("iwslt/results/average_wer.txt", 'w') as f:
            f.write(str(sum(wers) / len(wers)))
        
        print(f"Average WER: {sum(wers) / len(wers)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)