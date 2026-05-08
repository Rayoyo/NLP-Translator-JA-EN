# NLP-Translator-JA-EN
My own very first translator, by using python and working on a jupiter notebook

Here I will add some info about how to see the project and start it on your computer

---
## Project layout

```
├── data/
│   ├── raw/                    # Original file (NON su GitHub, solo .gitignore)
│   │   └── en-ja.dataset.txt
│   └── processed/              # Clean file (NOT on GitHub)
│       ├── english.txt
│       └── japanese.txt
├── models/                     # Saved weights (NOT on GitHub)
│   └── .gitkeep
├── notebooks/
│   └── main_training.ipynb     # Notebook Colab principal
├── src/
│   ├── __init__.py
│   ├── transformer.py          # Transformer architecture
│   ├── dataset.py              # Dataset memory-efficient
│   ├── tokenizer.py            # Tokenizer menaging
│   ├── train.py                # Training loop
│   ├── evaluate.py             # BLEU score & comparison
│   └── gui.py                  # Gradio interface
├── app.py                      # Entry point for local GUI 
├── requirements.txt
├── .gitignore
└── README.md
```
---
## Some general info
```bash
# 1. Original dataset info
The original dataset used was taken from JParaCrawl at the following url: https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/#citation
Once you download it, you can add to the data folder, in order to let the code correclty run. Plus you have to rename it as "en-ja.dataset.txt"

# 2. Generayed files from the dataset
The first run of the code will generate 2 separate file: one for the japanese sentences and one for the english ones.
Check their names:
```
### To run on Google Colab
File setup on Colab
```bash
MyDrive/
└── project/
    ├── data/
    │   ├── processed/
    │   │   ├── english.txt        ← clean file
    │   │   └── japanese.txt       ← clean file
    │   └── raw/
    │       └── en-ja.dataset.txt  ← (optional, original)
    └── models/                    ← weights will be saved here
```
