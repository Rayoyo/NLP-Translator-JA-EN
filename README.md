# NLP-Translator-JA-EN
My own very first translator, by using python and working on a jupiter notebook

Here I will add some info about how to see the project and start it on your computer

---
## Project layout

```
├── data/
│   ├── raw/                    # Original file 
│   │   ├── CITATION
│   │   ├── LICENSE
│   │   └── en-ja.dataset.txt   # Original dataset (NOT on GitHub, only .gitignore)
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
├── clean_dataset.py            # To clean the dataset and select a part of it (1 500 000 sentences)
├── local_tokenizer_train.py    # To avoind RAM limit problem of Colab, cound be avoided if you use the ones provided by JParaCrawl
├── requirements.txt
├── .gitignore
└── README.md
```
---
# Some general info

## 1. Original dataset info
The original dataset used was taken from [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/#citation), one of the most popular, largest publicly available English-Japanese parallel corpora built specifically for training and evaluating machine translation models.\
_Whats in the site?_
- _**The Dataset Downloads:**_ Links to download the aligned text pairs in both translation directions (English-to-Japanese and Japanese-to-English).
- _**Pre-trained Models:**_ Machine translation models that NTT already trained using this data, which you can use as a baseline.
  - **"Big" Set (The Raw Archive):** This contains the maximum number of unique sentence pairs found (over 21 million pairs in version 3.0). It includes sentences with lower alignment confidence scores. It has the most vocabulary but also contains the most "noise" (imperfect translations).
  - **"Base" Set (The Balanced Choice):** This filters out the lower-scoring sentences, keeping a medium-to-high confidence threshold. It is the standard balance used for training most baseline translation models.
  - **"Small" Set (The High-Confidence Core):** This includes only the absolute highest-weighted sentence pairs. The alignment score threshold is set very strictly
   > If you donwload it, extract and place on the `model` folder of your drive, you can avoid to use the tokenizer, because it already has it.
   > But it is important to check the relative `file name` before running the full code.
    ```
    x enja_spm_models/
    x enja_spm_models/spm.en.nopretok.vocab     --> en_tokenizer.vocab
    x enja_spm_models/spm.en.nopretok.model     --> en_tokenizer.model
    x enja_spm_models/spm.ja.nopretok.model     --> jp_tokenizer.model
    x enja_spm_models/spm.ja.nopretok.vocab     --> jp_tokenizer.vocab
    ```
- _**The "Citation" Section:**_ The specific academic bibtex block to reference their paper ("JParaCrawl: A Large Scale Web-Based English-Japanese Parallel Corpus") if you use their data for research.
> Terms of Use: A license stating the data is free for non-commercial, research-focused information analysis.
Once you download it, you can add to the data folder, in order to let the code correclty run. \
> Plus you have to rename it as "`en-ja.dataset.txt`"

## 2. General workflow of the project
It is better not to direclty use Google Colab due to the RAM limit (8GB), so the first and the second phases were runned on local pc with 32GB RAM.
1. Firstly, run the code in the file `clean_dataset.py` as follows:
   > Note: I preferred to use a virtual machine to avoid install mandatory package on my computer.
   ```
   cd Desktop\NLP-Translator-JA-EN # to enter the project folder, depends on where you place it
   python -m venv venv             # to create the virtual environment
       venv\Scripts\activate       # start the virtual environment -- WINDOWS
       source venv/bin/activate    # start the virtual environment -- LINUX
   pip install tqdm                # install dependencies useful to run the clear data code 
   python clean_dataset.py         # execute the script
   ```
2. Now it is necessary to create a Tokenizer - AI models (like Transformers) don't read words like we do, they read numbers. \
   The Tokenizer is used to:
   - Break down text: Take a sentence and divide it into "pieces" called tokens. Often, these aren't whole words, but fragments - This helps the model understand word roots and handle unseen words
   - Build a vocabulary: Assign each piece a unique number (ID)
   - Handle Japanese: Unlike English, Japanese doesn't use spaces between words. Without a dedicated tokenizer (like SentencePiece), the AI ​​would only see an infinite string of meaningless symbols \
    Can be done in two ways:
   1. Create `model` and `vocab` by running the file `local_tokenizer_train.py` in the following way:
      - as before, enter the folder if you are not in it already, then create and start the virtual env before run the code
        ```
        pip install sentencepiece         # install dependencies to create the tokenizer
        python local_tokenizer_train.py   # execute the script
        ```
      - on the folder model will be created the necessary files
        
   2. Download `model` and `vocab` from the site [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/#citation)
      - Download the size you prefer (depends on how precise you want the tokenizer to be: small - more precise, base - normal precise, big - very noisy)
      - Then, extract the result and put the four files on the `models` folder

    Now, you have those files on your `model` folder:
    ```
    en_tokenizer.vocab
    en_tokenizer.model
    jp_tokenizer.model
    jp_tokenizer.vocab
    ```
    - copy them on the Google drive's folder file, it should look like this:
        ```bash
        MyDrive/
        └── Project-NLP_Translator/
            ├── data/
            │   ├── processed/
            │   │   ├── english.txt        ← clean file
            │   │   └── japanese.txt       ← clean file
            │   └── raw/
            │       └── en-ja.dataset.txt  ← (optional, original)
            └── models/                    ← weights will be saved here as checkpoints
                ├── en_tokenizer.model
                ├── en_tokenizer.vocab
                ├── jp_tokenizer.model
                └── jp_tokenizer.vocab
        ```
