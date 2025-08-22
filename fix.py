import pandas as pd
import os
import re
from pathlib import Path

def extract_original_filename(label_studio_path):
    """
    Extract original filename from Label Studio path by removing hash prefix
    Example: '/data/upload/1/d78d3c22-fridge_e_3.jpeg' -> 'fridge_e_3.jpeg'
    """
    # Get just the filename from the full path
    filename = os.path.basename(label_studio_path)
    
    # Remove hash prefix (pattern: hash- at the beginning)
    # Look for pattern like 'd78d3c22-' and remove it
    match = re.match(r'^[a-f0-9]+-(.+)$', filename)
    if match:
        original_name = match.group(1)
        # Replace underscores with spaces (common transformation)
        original_name = original_name.replace('_', ' ')
        return original_name
    else:
        # If no hash prefix found, just replace underscores with spaces
        return filename.replace('_', ' ')

def find_best_match(original_name, directory_files):
    """
    Find the best matching file in the directory
    """
    # Direct match first
    if original_name in directory_files:
        return original_name
    
    # Try case-insensitive match
    for file in directory_files:
        if original_name.lower() == file.lower():
            return file
    
    # Try partial matching (for files with special characters that might have been changed)
    original_base = os.path.splitext(original_name)[0].lower()
    for file in directory_files:
        file_base = os.path.splitext(file)[0].lower()
        if original_base in file_base or file_base in original_base:
            return file
    
    return None

# Read the CSV
df = pd.read_csv('data/labels.csv')  # Replace with your CSV filename

# Get list of files in your images directory
image_directory = 'data/images'  # Replace with your directory path
directory_files = os.listdir(image_directory)

# Create mapping
mapping_results = []

for idx, row in df.iterrows():
    label_studio_filename = row['image']
    original_filename = extract_original_filename(label_studio_filename)
    matched_file = find_best_match(original_filename, directory_files)
    
    mapping_results.append({
        'annotation_id': row['annotation_id'],
        'label_studio_path': label_studio_filename,
        'extracted_original': original_filename,
        'matched_file': matched_file,
        'choice': row['choice'],
        'match_found': matched_file is not None
    })

# Create results DataFrame
results_df = pd.DataFrame(mapping_results)

# Show summary
print(f"Total files: {len(results_df)}")
print(f"Matched files: {results_df['match_found'].sum()}")
print(f"Unmatched files: {(~results_df['match_found']).sum()}")

# Show unmatched files for manual review
unmatched = results_df[~results_df['match_found']]
if len(unmatched) > 0:
    print("\nUnmatched files:")
    for idx, row in unmatched.iterrows():
        print(f"  {row['extracted_original']}")

# Save the mapping to CSV for review
results_df.to_csv('filename_mapping.csv', index=False)
print("\nMapping saved to 'filename_mapping.csv'")

# Create a corrected CSV with proper filenames
corrected_df = df.copy()
corrected_df['original_filename'] = results_df['matched_file']
corrected_df['mapping_successful'] = results_df['match_found']

# Only keep rows where mapping was successful
successful_mappings = corrected_df[corrected_df['mapping_successful']].copy()
successful_mappings = successful_mappings.drop(['mapping_successful'], axis=1)

successful_mappings.to_csv('corrected_annotations.csv', index=False)
print(f"Corrected CSV saved to 'corrected_annotations.csv' with {len(successful_mappings)} successfully mapped entries")
