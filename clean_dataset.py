#!/usr/bin/env python3
"""
Run ONLY locally
Create clean english.txt and japanese.txt
"""

import unicodedata
from pathlib import Path
import re

# === PATH CONFIGURATION ===
INPUT_FILE = Path("data/raw/en-ja.dataset.txt")   # original file
OUT_EN = Path("data/processed/english.txt")
OUT_JP = Path("data/processed/japanese.txt")

MAX_LINES = 1_500_000  # max lines to process (None = all, ~8GB, requires 16-20GB RAM)

if MAX_LINES is not None:
    print(f"!!! LIMIT SET: The program will stop at {MAX_LINES} lines !!!")
else:
    print("!!! WARNING: No limit set. Processing the entire file !!!")

# Create processed folder if it doesn't exist
OUT_EN.parent.mkdir(parents=True, exist_ok=True)

# normalization function: NFKC + strip + remove internal newlines + collapse multiple spaces
def normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s.strip())
    # removes internal newlines
    s = s.replace("\n", " ").replace("\r", " ")
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

count = 0
bad = 0

print(f"Reading: {INPUT_FILE}") 
print(f"Writing: {OUT_EN} and {OUT_JP}")

# limit on max valid pairs (after cleaning). None = no limit, process everything.
if MAX_LINES:
    print(f"Limit set to: {MAX_LINES:,} valid pairs")
print("Processing... (may take several minutes with 8GB)")

# Process line by line to avoid RAM issues, write valid pairs to separate files
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUT_EN, "w", encoding="utf-8") as f_en, \
     open(OUT_JP, "w", encoding="utf-8") as f_jp:

    # Read line by line, split by tab, extract columns 4 and 5, normalize, sanity check, write to separate files
    for i, line in enumerate(fin):
        # Stop if we reached the max valid pairs
        if MAX_LINES and count >= MAX_LINES:
            print(f"\n[INFO] limit reached: {MAX_LINES:,} valid pairs. Interrupting...")
            break

        # Progress report every 500k lines
        if i % 500_000 == 0 and i > 0:
            print(f"  Processed {i:,} lines, valid: {count:,}")
        
        # Split line by tab and check if we have at least 5 columns
        parts = line.split("\t")
        if len(parts) < 5:
            bad += 1
            continue
        
        # Extract columns 4 and 5 (0-indexed: 3 and 4)
        en = normalize(parts[3])
        jp = normalize(parts[4])

        # Sanity checks
        if len(en) < 3 or len(jp) < 2:
            bad += 1
            continue
        if en == "" or jp == "":
            bad += 1
            continue

        f_en.write(en + "\n")
        f_jp.write(jp + "\n")
        count += 1

# Final report
print(f"\n{'='*50}")                                         
print(f"Valid pairs:   {count:,}")                       # should be close to MAX_LINES if set, otherwise the total valid pairs in the file
print(f"Discarded:     {bad:,}")                         # could be due to malformed lines, missing columns, or failed sanity checks
print(f"Total lines:   {count + bad:,}")                 # should be close to the original file line count, minus any malformed lines
print(f"{'='*50}")
print(f"Files created:")
print(f"  {OUT_EN} ({OUT_EN.stat().st_size / 1e9:.2f} GB)")
print(f"  {OUT_JP} ({OUT_JP.stat().st_size / 1e9:.2f} GB)")