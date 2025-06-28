#!/usr/bin/env python3
"""
Audio File Validator - Check for corrupted/empty audio files and log them
"""

import os
import whisper
import json
from datetime import datetime
from tqdm import tqdm
import argparse

def validate_audio_files(audio_dir, output_file="corrupted_files.json"):
    """
    Validate all audio files in a directory and save corrupted ones to a file
    
    Args:
        audio_dir: Directory containing audio files
        output_file: File to save the list of corrupted files
    """
    
    # Get all files in the audio directory
    all_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} files to validate in {audio_dir}")
    
    corrupted_files = {
        "timestamp": datetime.now().isoformat(),
        "audio_directory": audio_dir,
        "total_files_checked": len(all_files),
        "empty_files": [],
        "ffmpeg_errors": [],
        "other_errors": [],
        "valid_files_count": 0
    }
    
    for file_path in tqdm(all_files, desc="Validating audio files"):
        try:
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                corrupted_files["empty_files"].append({
                    "path": file_path,
                    "relative_path": os.path.relpath(file_path, audio_dir),
                    "size": 0,
                    "error": "File is empty (0 bytes)"
                })
                continue
            
            # Try to load with whisper (which uses ffmpeg internally)
            try:
                audio = whisper.load_audio(file_path)
                corrupted_files["valid_files_count"] += 1
                
            except Exception as e:
                error_str = str(e)
                
                # Categorize the error
                if "ffmpeg" in error_str.lower() or "Failed to load audio" in error_str:
                    corrupted_files["ffmpeg_errors"].append({
                        "path": file_path,
                        "relative_path": os.path.relpath(file_path, audio_dir),
                        "size": os.path.getsize(file_path),
                        "error": error_str
                    })
                else:
                    corrupted_files["other_errors"].append({
                        "path": file_path,
                        "relative_path": os.path.relpath(file_path, audio_dir),
                        "size": os.path.getsize(file_path),
                        "error": error_str
                    })
                    
        except Exception as e:
            # Handle file system errors (permissions, etc.)
            corrupted_files["other_errors"].append({
                "path": file_path,
                "relative_path": os.path.relpath(file_path, audio_dir),
                "size": "unknown",
                "error": f"File system error: {str(e)}"
            })
    
    # Calculate totals
    total_corrupted = (len(corrupted_files["empty_files"]) + 
                      len(corrupted_files["ffmpeg_errors"]) + 
                      len(corrupted_files["other_errors"]))
    
    corrupted_files["total_corrupted"] = total_corrupted
    corrupted_files["corruption_rate"] = total_corrupted / len(all_files) if all_files else 0
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(corrupted_files, f, indent=2)
    
    # Also save a simple text list of corrupted file paths
    txt_output = output_file.replace('.json', '_paths.txt')
    with open(txt_output, 'w') as f:
        f.write(f"# Corrupted audio files found on {datetime.now().isoformat()}\n")
        f.write(f"# Total files checked: {len(all_files)}\n")
        f.write(f"# Total corrupted: {total_corrupted}\n\n")
        
        if corrupted_files["empty_files"]:
            f.write("# EMPTY FILES:\n")
            for item in corrupted_files["empty_files"]:
                f.write(f"{item['path']}\n")
            f.write("\n")
        
        if corrupted_files["ffmpeg_errors"]:
            f.write("# FFMPEG ERRORS:\n")
            for item in corrupted_files["ffmpeg_errors"]:
                f.write(f"{item['path']}\n")
            f.write("\n")
        
        if corrupted_files["other_errors"]:
            f.write("# OTHER ERRORS:\n")
            for item in corrupted_files["other_errors"]:
                f.write(f"{item['path']}\n")
    
    # Print summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total files checked: {len(all_files)}")
    print(f"Valid files: {corrupted_files['valid_files_count']}")
    print(f"Empty files: {len(corrupted_files['empty_files'])}")
    print(f"FFmpeg errors: {len(corrupted_files['ffmpeg_errors'])}")
    print(f"Other errors: {len(corrupted_files['other_errors'])}")
    print(f"Total corrupted: {total_corrupted}")
    print(f"Corruption rate: {corrupted_files['corruption_rate']:.2%}")
    print(f"\nDetailed results saved to: {output_file}")
    print(f"File paths saved to: {txt_output}")
    
    return corrupted_files

def remove_corrupted_files(corrupted_files_json, dry_run=True):
    """
    Remove corrupted files based on the validation results
    
    Args:
        corrupted_files_json: Path to the JSON file with corrupted file info
        dry_run: If True, only print what would be deleted without actually deleting
    """
    
    with open(corrupted_files_json, 'r') as f:
        data = json.load(f)
    
    all_corrupted = (data["empty_files"] + 
                    data["ffmpeg_errors"] + 
                    data["other_errors"])
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Found {len(all_corrupted)} corrupted files to remove")
    
    removed_count = 0
    for item in all_corrupted:
        file_path = item["path"]
        if os.path.exists(file_path):
            print(f"{'[DRY RUN] Would remove: ' if dry_run else 'Removing: '}{file_path}")
            if not dry_run:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            else:
                removed_count += 1
        else:
            print(f"File already missing: {file_path}")
    
    print(f"{'[DRY RUN] Would remove' if dry_run else 'Removed'} {removed_count} files")
    
    if dry_run:
        print("\nTo actually remove files, run with --remove --no-dry-run")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate audio files and handle corrupted ones")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("-o", "--output", default="corrupted_files.json", 
                       help="Output file for corrupted file list (default: corrupted_files.json)")
    parser.add_argument("--remove", action="store_true", 
                       help="Remove corrupted files after validation")
    parser.add_argument("--no-dry-run", action="store_true",
                       help="Actually remove files (use with --remove)")
    
    args = parser.parse_args()
    
    # Validate files
    print("Starting audio file validation...")
    corrupted_data = validate_audio_files(args.audio_dir, args.output)
    
    # Optionally remove corrupted files
    if args.remove:
        dry_run = not args.no_dry_run
        remove_corrupted_files(args.output, dry_run=dry_run)