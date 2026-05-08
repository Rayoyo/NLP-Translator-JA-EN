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

# Crea cartella processed se non esiste
OUT_EN.parent.mkdir(parents=True, exist_ok=True)

def normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s.strip())
    # Rimuovi newline interni
    s = s.replace("\n", " ").replace("\r", " ")
    # Rimuovi spazi multipli
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

count = 0
bad = 0

print(f"Reading: {INPUT_FILE}")
print(f"Writing: {OUT_EN} and {OUT_JP}")
print("Processing... (può richiedere diversi minuti con 8GB)")

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUT_EN, "w", encoding="utf-8") as f_en, \
     open(OUT_JP, "w", encoding="utf-8") as f_jp:

    for i, line in enumerate(fin):
        if i % 500_000 == 0 and i > 0:
            print(f"  Processed {i:,} lines, valid: {count:,}")
        
        parts = line.split("\t")
        if len(parts) < 5:
            bad += 1
            continue
        
        # Estrai colonne 4 e 5 (0-indexed: 3 e 4)
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

print(f"\n{'='*50}")
print(f"Valid pairs:   {count:,}")
print(f"Discarded:     {bad:,}")
print(f"Total lines:   {count + bad:,}")
print(f"{'='*50}")
print(f"Files created:")
print(f"  {OUT_EN} ({OUT_EN.stat().st_size / 1e9:.2f} GB)")
print(f"  {OUT_JP} ({OUT_JP.stat().st_size / 1e9:.2f} GB)")