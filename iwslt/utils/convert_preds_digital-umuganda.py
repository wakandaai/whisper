import os
import sys
import argparse
import csv

# parser
def parse_args():
    parser = argparse.ArgumentParser(description='Digital Umudanda Data Preparation')
    parser.add_argument('--input_file', type=str, default='results/text.tsv', help='Base path')
    parser.add_argument('--output_file', type=str, default='results/submission.csv', help='Output directory')
    args = parser.parse_args()
    return args

# main
def main(args):

    # check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        sys.exit(1)
    # read the input file
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    # process the lines
    results = {}
    for line in lines:
        line = line.strip()
        if not line:
            print("Skipping empty line")
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            parts.append("no speech")
        id,_, hypothesis = parts
        hypothesis = hypothesis.strip()
        # remove trailing .
        if hypothesis.endswith('.'):
            hypothesis = hypothesis[:-1]

        results[id] = hypothesis

    # write the results to a csv file with two columns: id and transcription
    with open(args.output_file, 'w', newline='') as csvfile:
        fieldnames = ['id', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for id, transcription in results.items():
            writer.writerow({'id': id, 'transcription': transcription})

if __name__ == '__main__':
    args = parse_args()
    main(args)