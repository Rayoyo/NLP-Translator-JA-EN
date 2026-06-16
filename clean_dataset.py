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

OUT_TEST_EN = Path("data/processed/englishTEST.txt")    # for testing phase
OUT_TEST_JP = Path("data/processed/japaneseTEST.txt")

MAX_LINES = 1_500_000  # max lines to process (None = all, ~8GB, requires 16-20GB RAM) -> better to set limit
TEST_LINES = 5_000    # number of lines for test files (None = no separate test set)

if MAX_LINES is not None:
    print(f"!!! LIMIT SET: The program will stop at {MAX_LINES} lines !!!")
else:
    print("!!! WARNING: No limit set. Processing the entire file!!!")

if TEST_LINES:
    print(f"!!! TEST SET: Will extract {TEST_LINES} additional lines for testing !!!")

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
test_count = 0
bad = 0

print(f"Reading: {INPUT_FILE}") 
print(f"Writing: {OUT_EN} and {OUT_JP}")

# limit on max valid pairs (after cleaning). None = no limit, process everything.
if MAX_LINES:
    print(f"Limit set to: {MAX_LINES:,} valid pairs")

if TEST_LINES:
    print(f"Writing Test: {OUT_TEST_EN} and {OUT_TEST_JP}")

print("Processing... (may take several minutes with 8GB)")

# Process line by line to avoid RAM issues, write valid pairs to separate files
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUT_EN, "w", encoding="utf-8") as f_en, \
     open(OUT_JP, "w", encoding="utf-8") as f_jp, \
     open(OUT_TEST_EN, "w", encoding="utf-8") as f_test_en, \
     open(OUT_TEST_JP, "w", encoding="utf-8") as f_test_jp:

    # Read line by line, split by tab, extract columns 4 and 5, normalize, sanity check, write to separate files
    for i, line in enumerate(fin):
        
        # Progress report every 500k lines
        if i % 500_000 == 0 and i > 0:
            print(f"  Processed {i:,} lines, valid: {count:,} | Test pairs: {test_count:,}")
        
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

        # main dataset
        if MAX_LINES is None or count < MAX_LINES:
            f_en.write(en + "\n")
            f_jp.write(jp + "\n")
            count += 1
        
        # test dataset
        elif TEST_LINES is not None and test_count < TEST_LINES:
            f_test_en.write(en + "\n")
            f_test_jp.write(jp + "\n")
            test_count += 1
        
        # if both limits are reached, we can stop immediately without processing the rest of the file
        else:
            print(f"\n[INFO] Target limits reached (Main: {count:,}, Test: {test_count:,}). Interrupting...")
            break


# Final report
print(f"\n{'='*50}")      

print(f"Main dataset pairs: {count:,}") 
if TEST_LINES:
    print(f"Test dataset pairs: {test_count:,}")                           # should be close to TEST_LINES if set, otherwise 0
print(f"Discarded pairs:    {bad:,}")                                      # could be due to malformed lines, missing columns, or failed sanity checks
print(f"Total lines read:   {count + test_count + bad:,}")                 # should be close to the original file line count, minus any malformed lines
print(f"{'='*50}")
print(f"Files created:")
print(f"   {OUT_EN} ({OUT_EN.stat().st_size / 1e9:.4f} GB)")               # should be close to count * average line size in GB
print(f"   {OUT_JP} ({OUT_JP.stat().st_size / 1e9:.4f} GB)")               # should be close to count * average line size in GB
if TEST_LINES and test_count > 0:
    print(f"   {OUT_TEST_EN} ({OUT_TEST_EN.stat().st_size / 1e6:.2f} MB)") # should be close to test_count * average line size in MB
    print(f"   {OUT_TEST_JP} ({OUT_TEST_JP.stat().st_size / 1e6:.2f} MB)") # should be close to test_count * average line size in MB
