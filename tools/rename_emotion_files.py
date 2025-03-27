import os
import csv

# Set your folder here
folder = "./recordings"
mapping_file = "emotions_map.csv"

# Load the mapping
mapping = {}
with open(mapping_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        mapping[row['EMOTIONS'].lower()] = row['TRADUCTION'].lower()

# Apply the renaming
for filename in os.listdir(folder):
    name, ext = os.path.splitext(filename)
    name_lower = name.lower()
    if name_lower in mapping:
        new_name = mapping[name_lower] + ext
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        print(f"Renaming: {filename} -> {new_name}")
        os.rename(old_path, new_path)
