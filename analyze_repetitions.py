import re
from collections import Counter
from typing import List, Dict, Tuple

class ASRFileAnalyzer:
    def __init__(self, min_repetitions=3, max_keep=2):
        """
        Initialize the ASR file analyzer.
        
        Args:
            min_repetitions: Minimum number of repetitions to flag as problematic
            max_keep: Maximum number of repetitions to keep when cleaning
        """
        self.min_repetitions = min_repetitions
        self.max_keep = max_keep
    
    def detect_repetitions_in_text(self, text: str) -> List[Dict]:
        """Detect all repetitive sequences of 1-3 words in text."""
        repetitions = []
        
        # Check for 1, 2, and 3-word sequences
        for seq_length in [3, 2, 1]:
            pattern_repetitions = self._detect_sequence_repetitions(text, seq_length)
            repetitions.extend(pattern_repetitions)
        
        # Remove overlapping detections (prefer longer sequences)
        repetitions = self._remove_overlaps(repetitions)
        return repetitions
    
    def _detect_sequence_repetitions(self, text: str, seq_length: int) -> List[Dict]:
        """Detect repetitions for sequences of specific length."""
        if seq_length == 1:
            pattern = r'\b(\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        elif seq_length == 2:
            pattern = r'\b(\w+\s+\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        else:  # seq_length == 3
            pattern = r'\b(\w+\s+\w+\s+\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        
        repetitions = []
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            sequence = match.group(1)
            full_match = match.group(0)
            
            sequence_parts = sequence.split()
            full_parts = full_match.split()
            count = len(full_parts) // len(sequence_parts)
            
            repetitions.append({
                'sequence': sequence,
                'sequence_length': seq_length,
                'count': count,
                'start': match.start(),
                'end': match.end(),
                'full_text': full_match
            })
        
        return repetitions
    
    def _remove_overlaps(self, repetitions: List[Dict]) -> List[Dict]:
        """Remove overlapping repetitions, preferring longer sequences."""
        if not repetitions:
            return repetitions
        
        repetitions.sort(key=lambda x: (x['start'], -x['sequence_length']))
        
        non_overlapping = []
        for rep in repetitions:
            overlaps = False
            for accepted in non_overlapping:
                if (rep['start'] < accepted['end'] and rep['end'] > accepted['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(rep)
        
        return non_overlapping
    
    def clean_text_repetitions(self, text: str) -> str:
        """Clean repetitions in text."""
        for seq_length in [3, 2, 1]:
            text = self._clean_sequence_length(text, seq_length)
        
        return re.sub(r'\s+', ' ', text).strip()
    
    def _clean_sequence_length(self, text: str, seq_length: int) -> str:
        """Clean repetitions for a specific sequence length."""
        if seq_length == 1:
            pattern = r'\b(\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        elif seq_length == 2:
            pattern = r'\b(\w+\s+\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        else:
            pattern = r'\b(\w+\s+\w+\s+\w+)(\s+\1){' + str(self.min_repetitions - 1) + ',}'
        
        def replace_repetition(match):
            sequence = match.group(1)
            return ' '.join([sequence] * self.max_keep)
        
        return re.sub(pattern, replace_repetition, text, flags=re.IGNORECASE)
    
    def analyze_line(self, line: str, line_number: int) -> Dict:
        """Analyze a single line from the file."""
        parts = line.strip().split('\t')
        
        if len(parts) < 3:
            return {
                'line_number': line_number,
                'id': 'INVALID',
                'lang': 'INVALID',
                'text': line.strip(),
                'has_repetitions': False,
                'error': 'Invalid format - expected ID, lang, text separated by tabs'
            }
        
        line_id = parts[0]
        lang = parts[1]
        text = '\t'.join(parts[2:])
        
        # Detect repetitions
        repetitions = self.detect_repetitions_in_text(text)
        has_repetitions = len(repetitions) > 0
        
        # Basic text statistics
        char_count = len(text)
        word_count = len(text.split())
        words = text.split()
        unique_words = len(set(words))
        
        # Calculate repetition statistics
        total_repeated_words = sum(rep['count'] * rep['sequence_length'] for rep in repetitions)
        repetition_percentage = (total_repeated_words / word_count * 100) if word_count > 0 else 0
        
        result = {
            'line_number': line_number,
            'id': line_id,
            'lang': lang,
            'text': text,
            'has_repetitions': has_repetitions,
            'repetitions': repetitions,
            'repetition_count': len(repetitions),
            'char_count': char_count,
            'word_count': word_count,
            'unique_words': unique_words,
            'total_repeated_words': total_repeated_words,
            'repetition_percentage': repetition_percentage
        }
        
        # Add cleaned text if repetitions found
        if has_repetitions:
            result['cleaned_text'] = self.clean_text_repetitions(text)
            result['cleaned_word_count'] = len(result['cleaned_text'].split())
            result['words_removed'] = word_count - result['cleaned_word_count']
        
        return result
    
    def analyze_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze entire file and return flagged lines and summary.
        
        Returns:
            Tuple of (flagged_lines, all_lines)
        """
        all_lines = []
        flagged_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():
                        analysis = self.analyze_line(line, line_num)
                        all_lines.append(analysis)
                        
                        if analysis['has_repetitions']:
                            flagged_lines.append(analysis)
        
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found.")
            return [], []
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return [], []
        
        return flagged_lines, all_lines

def print_flagged_line_analysis(analysis: Dict, show_cleaned: bool = True):
    """Print detailed analysis for a flagged line."""
    
    print(f"\n{'='*100}")
    print(f"🚨 FLAGGED LINE {analysis['line_number']} | ID: {analysis['id']} | Lang: {analysis['lang']}")
    print(f"{'='*100}")
    
    if 'error' in analysis:
        print(f"❌ ERROR: {analysis['error']}")
        return
    
    # Show original text
    print(f"📝 ORIGINAL TEXT:")
    print(f"   {analysis['text']}")
    
    # Show repetitions found
    print(f"\n🔍 REPETITIONS DETECTED:")
    for i, rep in enumerate(analysis['repetitions'], 1):
        print(f"   {i}. '{rep['sequence']}' → repeated {rep['count']} times")
        print(f"      Sequence length: {rep['sequence_length']} word(s)")
        print(f"      Full repetition: '{rep['full_text'][:100]}{'...' if len(rep['full_text']) > 100 else ''}'")
    
    # Show statistics
    print(f"\n📊 STATISTICS:")
    print(f"   • Total characters: {analysis['char_count']:,}")
    print(f"   • Total words: {analysis['word_count']:,}")
    print(f"   • Unique words: {analysis['unique_words']:,}")
    print(f"   • Repeated words: {analysis['total_repeated_words']:,}")
    print(f"   • Repetition percentage: {analysis['repetition_percentage']:.1f}%")
    
    # Show cleaned version if available
    if show_cleaned and 'cleaned_text' in analysis:
        print(f"\n✨ CLEANED TEXT:")
        print(f"   {analysis['cleaned_text']}")
        print(f"\n📉 CLEANING RESULTS:")
        print(f"   • Words removed: {analysis['words_removed']:,}")
        print(f"   • Original → Cleaned: {analysis['word_count']:,} → {analysis['cleaned_word_count']:,} words")
        compression = (analysis['words_removed'] / analysis['word_count'] * 100) if analysis['word_count'] > 0 else 0
        print(f"   • Compression ratio: {compression:.1f}%")

def print_summary_statistics(flagged_lines: List[Dict], all_lines: List[Dict]):
    """Print summary statistics for the analysis."""
    
    print(f"\n{'='*100}")
    print(f"📈 SUMMARY STATISTICS")
    print(f"{'='*100}")
    
    total_lines = len(all_lines)
    flagged_count = len(flagged_lines)
    clean_lines = total_lines - flagged_count
    
    print(f"📄 FILE OVERVIEW:")
    print(f"   • Total lines processed: {total_lines:,}")
    print(f"   • Lines with repetitions: {flagged_count:,} ({flagged_count/total_lines*100:.1f}%)")
    print(f"   • Clean lines: {clean_lines:,} ({clean_lines/total_lines*100:.1f}%)")
    
    if flagged_lines:
        # Statistics for flagged lines
        total_repetitions = sum(len(line['repetitions']) for line in flagged_lines)
        avg_repetitions_per_line = total_repetitions / len(flagged_lines)
        
        repetition_percentages = [line['repetition_percentage'] for line in flagged_lines]
        avg_repetition_percentage = sum(repetition_percentages) / len(repetition_percentages)
        max_repetition_percentage = max(repetition_percentages)
        
        print(f"\n🚨 REPETITION ANALYSIS:")
        print(f"   • Total repetitive sequences found: {total_repetitions:,}")
        print(f"   • Average repetitions per flagged line: {avg_repetitions_per_line:.1f}")
        print(f"   • Average repetition percentage: {avg_repetition_percentage:.1f}%")
        print(f"   • Highest repetition percentage: {max_repetition_percentage:.1f}%")
        
        # Show worst offenders
        worst_lines = sorted(flagged_lines, key=lambda x: x['repetition_percentage'], reverse=True)[:3]
        print(f"\n🥇 TOP OFFENDERS:")
        for i, line in enumerate(worst_lines, 1):
            print(f"   {i}. Line {line['line_number']} (ID: {line['id']}): {line['repetition_percentage']:.1f}% repetition")
        
        # Language breakdown
        lang_counts = Counter(line['lang'] for line in flagged_lines)
        print(f"\n🌍 LANGUAGES WITH REPETITIONS:")
        for lang, count in lang_counts.most_common():
            percentage = count / len(flagged_lines) * 100
            print(f"   • {lang}: {count} lines ({percentage:.1f}%)")

def write_cleaned_output_file(input_file_path: str, all_lines: List[Dict], analyzer: ASRFileAnalyzer, output_file: str = None):
    """
    Write a cleaned version of the input file with repetitions removed.
    
    Args:
        input_file_path: Original input file path
        all_lines: List of all analyzed lines
        analyzer: ASRFileAnalyzer instance
        output_file: Output file path (if None, will create one based on input file)
    """
    
    # Generate output file name if not provided
    if output_file is None:
        import os
        base_name = os.path.splitext(input_file_path)[0]
        extension = os.path.splitext(input_file_path)[1]
        output_file = f"{base_name}_cleaned{extension}"
    
    try:
        lines_cleaned = 0
        total_words_removed = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line_data in all_lines:
                if 'error' in line_data:
                    # Write invalid lines as-is
                    out_file.write(f"{line_data['text']}\n")
                else:
                    # Clean the text if it has repetitions
                    if line_data['has_repetitions']:
                        cleaned_text = analyzer.clean_text_repetitions(line_data['text'])
                        lines_cleaned += 1
                        if 'words_removed' in line_data:
                            total_words_removed += line_data['words_removed']
                    else:
                        cleaned_text = line_data['text']
                    
                    # Write in original format: ID\tlang\ttext
                    out_file.write(f"{line_data['id']}\t{line_data['lang']}\t{cleaned_text}\n")
        
        print(f"\n💾 CLEANED FILE CREATED:")
        print(f"   📁 Output file: {output_file}")
        print(f"   🔧 Lines cleaned: {lines_cleaned:,}")
        print(f"   📉 Total words removed: {total_words_removed:,}")
        print(f"   ✅ File ready for use!")
        
    except Exception as e:
        print(f"❌ Error writing cleaned file: {e}")

def analyze_asr_file(file_path: str, 
                    min_repetitions: int = 3, 
                    max_keep: int = 2,
                    show_cleaned: bool = True,
                    show_summary: bool = True,
                    write_cleaned_file: bool = True,
                    output_file: str = None):
    """
    Main function to analyze ASR file for repetitions.
    
    Args:
        file_path: Path to the text file
        min_repetitions: Minimum repetitions to flag as problematic
        max_keep: Maximum repetitions to keep when cleaning
        show_cleaned: Whether to show cleaned versions
        show_summary: Whether to show summary statistics
        write_cleaned_file: Whether to write a cleaned version to a new file
        output_file: Output file path (if None, will create one based on input file)
    """
    
    print(f"🔍 Analyzing ASR file for repetitions: {file_path}")
    print(f"📋 Settings: min_repetitions={min_repetitions}, max_keep={max_keep}")
    
    # Initialize analyzer
    analyzer = ASRFileAnalyzer(min_repetitions, max_keep)
    
    # Analyze file
    flagged_lines, all_lines = analyzer.analyze_file(file_path)
    
    if not all_lines:
        print("❌ No data to analyze.")
        return
    
    if not flagged_lines:
        print("✅ No repetitions found! All lines are clean.")
        if write_cleaned_file:
            print("ℹ️  Since no repetitions were found, no cleaned file will be created.")
        return
    
    print(f"\n🎯 Found {len(flagged_lines)} lines with repetitions out of {len(all_lines)} total lines")
    
    # Show detailed analysis for each flagged line
    for analysis in flagged_lines:
        print_flagged_line_analysis(analysis, show_cleaned)
    
    # Show summary statistics
    if show_summary:
        print_summary_statistics(flagged_lines, all_lines)
    
    # Write cleaned file
    if write_cleaned_file:
        write_cleaned_output_file(file_path, all_lines, analyzer, output_file)

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "results/text.tsv"
    
    # Analyze with default settings
    analyze_asr_file(file_path)
    
    # Example: More strict detection (flag 2+ repetitions)
    # analyze_asr_file(file_path, min_repetitions=2)
    
    # Example: Keep only 1 occurrence when cleaning
    # analyze_asr_file(file_path, max_keep=1)
    
    # Example: Just show flagged lines without cleaned versions
    # analyze_asr_file(file_path, show_cleaned=False)