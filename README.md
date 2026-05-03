# NLP-Translator-JA-EN
My own very first translator, by using python and working on a jupiter notebook

Here I will add some info about how to see the project and start it on your computer

---
## Project layout

```
├── data/               # I tuoi file .txt (originali e processati)
├── checkpoints/        # Per salvare i pesi del modello durante il training
├── src/                # Script Python (.py)
│   ├── tokenizer.py    # Codice per il training del tokenizer
│   ├── model.py        # Definizione dell'architettura Transformer
│   ├── dataset.py      # Pytorch Dataset e DataLoader
│   └── train.py        # Loop di addestramento
├── notebooks/          # Jupyter Notebooks per test rapidi e visualizzazioni
├── app.py              # La GUI (Streamlit o Gradio)
└── requirements.txt    # Librerie necessarie
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
