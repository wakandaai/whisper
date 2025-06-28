#!/usr/bin/env python3

import argparse
import json
import torchaudio
import sacrebleu
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Simple offline evaluation')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions text file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to JSON metadata file')
    parser.add_argument('--task', type=str, default='transcribe', choices=['transcribe', 'translate'], help='Task type')
    parser.add_argument('--output', type=str, default='results/stats_trackB.tsv', help='Output stats file')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    return parser.parse_args()

def load_predictions(pred_file):
    """Load predictions from text file"""
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                audio_id = parts[0]
                text = parts[-1]  # Last column is the text
                predictions[audio_id] = text
    return predictions

def load_metadata(metadata_file):
    """Load ground truth from JSON metadata"""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def compute_transcription_metrics(item):
    """Compute WER/CER for a single audio-prediction pair"""
    audio_id, pred_text, ref_text = item
    
    # Clean texts
    pred_clean = pred_text.lower().strip().strip('.,!?')
    ref_clean = ref_text.lower().strip().strip('.,!?')
    
    # WER calculation
    ref_words = ref_clean.split()
    pred_words = pred_clean.split()
    wer = 0.0
    if len(ref_words) > 0:
        wer = torchaudio.functional.edit_distance(ref_words, pred_words) / len(ref_words)
    
    # CER calculation
    cer = 0.0
    if len(ref_clean) > 0:
        cer = torchaudio.functional.edit_distance(list(ref_clean), list(pred_clean)) / len(ref_clean)
    
    # Combined score
    combined_error = 0.4 * wer + 0.6 * cer
    score = (1 - combined_error) * 100
    
    return {
        'audio': audio_id,
        'WER': wer,
        'CER': cer,
        'Score': score,
        'chrF': '_',
        'BLEU': '_'
    }

def compute_translation_metrics(item):
    """Compute BLEU/chrF for a single audio-prediction pair"""
    audio_id, pred_text, ref_text = item
    
    if not (ref_text.strip() and pred_text.strip()):
        return None
    
    # Calculate BLEU for individual pair
    bleu = sacrebleu.corpus_bleu([pred_text.strip()], [[ref_text.strip()]], lowercase=True, tokenize='13a')
    
    # Calculate chrF for individual pair  
    chrf = sacrebleu.corpus_chrf([pred_text.strip()], [[ref_text.strip()]])
    
    bleu_score = bleu.score
    chrf_score = chrf.score
    
    return {
        'audio': audio_id,
        'WER': '-',
        'CER': '-', 
        'Score': chrf_score,
        'chrF': chrf_score,
        'BLEU': bleu_score
    }

def evaluate_transcription(predictions, metadata, output_file, num_workers):
    """Evaluate transcription task (WER/CER) with multiprocessing"""
    # Prepare data for multiprocessing
    items = []
    for audio_id, pred_text in predictions.items():
        if audio_id in metadata:
            ref_text = metadata[audio_id].get('transcription', '')
            items.append((audio_id, pred_text, ref_text))
    
    print(f"Processing {len(items)} items with {num_workers} workers...")
    start_time = time.time()
    
    # Process in parallel with progress bar
    with Pool(num_workers) as pool:
        results = []
        with tqdm(total=len(items), desc="Computing transcription metrics", unit="items") as pbar:
            for result in pool.imap(compute_transcription_metrics, items):
                results.append(result)
                pbar.update(1)
                
                # Update ETA every 10 items
                if len(results) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed
                    remaining = (len(items) - len(results)) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        'Rate': f'{rate:.1f} items/s',
                        'ETA': f'{remaining:.0f}s'
                    })
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    total_time = time.time() - start_time
    print(f"Completed in {total_time:.1f} seconds ({len(results)/total_time:.1f} items/s)")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Write stats file
    with open(output_file, 'w') as f:
        f.write("audio\tWER\tCER\tScore\tchrF\tBLEU\n")
        for result in results:
            f.write(f"{result['audio']}\t{result['WER']:.6f}\t{result['CER']:.6f}\t{result['Score']:.2f}\t{result['chrF']}\t{result['BLEU']}\n")
    
    if results:
        wers = [r['WER'] for r in results if r['WER'] > 0]
        cers = [r['CER'] for r in results if r['CER'] > 0]
        
        if wers and cers:
            avg_wer = sum(wers) / len(wers)
            avg_cer = sum(cers) / len(cers)
            combined_score = (1 - (0.4 * avg_wer + 0.6 * avg_cer)) * 100
            
            print(f"Transcription Results:")
            print(f"Average WER: {avg_wer:.4f}")
            print(f"Average CER: {avg_cer:.4f}")
            print(f"Combined Score: {combined_score:.2f}")
            print(f"Stats saved to: {output_file}")
        else:
            print("No valid metrics computed")
    else:
        print("No valid transcription pairs found for evaluation")

def evaluate_translation(predictions, metadata, output_file, num_workers):
    """Evaluate translation task (BLEU/chrF) with multiprocessing"""
    # Prepare data for multiprocessing
    items = []
    for audio_id, pred_text in predictions.items():
        if audio_id in metadata:
            ref_text = metadata[audio_id].get('text', '')
            items.append((audio_id, pred_text, ref_text))
    
    print(f"Processing {len(items)} items with {num_workers} workers...")
    start_time = time.time()
    
    # Process in parallel with progress bar
    with Pool(num_workers) as pool:
        results = []
        with tqdm(total=len(items), desc="Computing translation metrics", unit="items") as pbar:
            for result in pool.imap(compute_translation_metrics, items):
                if result is not None:
                    results.append(result)
                pbar.update(1)
                
                # Update ETA every 10 items
                if pbar.n % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = pbar.n / elapsed
                    remaining = (len(items) - pbar.n) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        'Rate': f'{rate:.1f} items/s',
                        'ETA': f'{remaining:.0f}s'
                    })
    
    total_time = time.time() - start_time
    print(f"Completed in {total_time:.1f} seconds ({len(results)/total_time:.1f} items/s)")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Write stats file
    with open(output_file, 'w') as f:
        f.write("audio\tWER\tCER\tScore\tchrF\tBLEU\n")
        for result in results:
            f.write(f"{result['audio']}\t{result['WER']}\t{result['CER']}\t{result['Score']:.2f}\t{result['chrF']:.2f}\t{result['BLEU']:.2f}\n")
    
    if results:
        bleu_scores = [r['BLEU'] for r in results]
        chrf_scores = [r['chrF'] for r in results]
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_chrf = sum(chrf_scores) / len(chrf_scores)
        
        print(f"Translation Results:")
        print(f"BLEU Score: {avg_bleu:.2f}")
        print(f"chrF Score: {avg_chrf:.2f}")
        print(f"Stats saved to: {output_file}")
    else:
        print("No valid translation pairs found for evaluation")

def main():
    args = parse_args()
    
    # Set number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Load data
    predictions = load_predictions(args.predictions)
    metadata = load_metadata(args.metadata)
    
    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(metadata)} metadata entries")
    
    # Evaluate based on task
    if args.task == 'transcribe':
        evaluate_transcription(predictions, metadata, args.output, num_workers)
    elif args.task == 'translate':
        evaluate_translation(predictions, metadata, args.output, num_workers)

if __name__ == '__main__':
    main()