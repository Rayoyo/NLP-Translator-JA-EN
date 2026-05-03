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
