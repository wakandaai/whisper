#!/usr/bin/env python3

def count_language_tags(file_path):
    """
    Count the frequency of language tags in the second column of a text file.
    
    Args:
        file_path: Path to the file containing the data
        
    Returns:
        Dictionary with language tags as keys and their frequencies as values
    """
    # Initialize an empty dictionary to store the counts
    language_counts = {}
    
    try:
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip empty lines
                if line.strip() == '':
                    continue
                
                # Split the line by tabs
                columns = line.strip().split('\t')
                
                # Make sure we have at least 2 columns
                if len(columns) >= 2:
                    # Get the language tag (second column)
                    language = columns[1]
                    
                    # Increment the count for this language
                    if language in language_counts:
                        language_counts[language] += 1
                    else:
                        language_counts[language] = 1
        # sort the dictionary by value
        language_counts = dict(sorted(language_counts.items(), key=lambda item: item[1], reverse=True
        ))
        
        return language_counts
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {}

def main():
    # Replace with your actual file path
    file_path = "results/text.tsv"
    
    # Get the language counts
    language_counts = count_language_tags(file_path)
    
    # Print the results
    if language_counts:
        print("Language frequencies:")
        for language, count in language_counts.items():
            print(f"{language}: {count}")
    else:
        print("No language data found.")

if __name__ == "__main__":
    main()