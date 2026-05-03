# Data Directory

Place your Japanese–English parallel corpus files here.

## Expected Format

The training pipeline expects two plain-text files — one sentence per line — where
line `i` of the Japanese file corresponds to line `i` of the English file:

```
data/
├── train.ja      # Japanese training sentences
├── train.en      # English training sentences
├── val.ja        # Japanese validation sentences
├── val.en        # English validation sentences
├── corpus.txt    # (optional) combined corpus used to train the SentencePiece tokenizer
```

### Example (`train.ja`)
```
猫が窓の外を見ています。
今日は良い天気ですね。
```

### Example (`train.en`)
```
The cat is looking out the window.
The weather is nice today.
```

## Recommended Public Datasets

| Dataset | Description |
|---------|-------------|
| [JParaCrawl](http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/) | Large-scale web-crawled JA–EN corpus |
| [JESC](https://nlp.stanford.edu/projects/jesc/) | Japanese–English Subtitle Corpus |
| [Tatoeba](https://tatoeba.org/) | Community-contributed sentence pairs |

After downloading, preprocess the files (strip HTML, normalise whitespace, etc.)
and save them in the format described above.
