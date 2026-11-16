# Emotion Classification from Text: TF-IDF vs RNN, LSTM, GRU and BiGRU

A deep learning project that classifies short English texts into one of six emotions
(**sadness, joy, love, anger, fear, surprise**). It compares a classical TF-IDF baseline
with four recurrent neural networks, plus one ablation on padding masks, and includes a
Streamlit app to try the models live.

## Dataset

- **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)
- **Content:** English tweets, labelled automatically from emotion hashtags (some label noise)
- **Split:** 16,000 train / 2,000 validation / 2,000 test
- **Imbalance:** the classes are not equally frequent, so the project reports **macro-F1**
  (each emotion counts equally) next to accuracy and trains with **class weights**

## Models

| Model | What it is |
|---|---|
| TF-IDF + LogReg | Baseline. Weighted words and word pairs fed to logistic regression. No real word order. |
| SimpleRNN | Reads the text left to right with a single memory. Forgets early words. |
| LSTM | Recurrent network with gates and a separate long-term memory. |
| GRU | Lighter gated network with fewer parameters than LSTM. |
| BiGRU | A GRU that reads forwards and backwards. |
| LSTM (no mask) | Ablation: the same LSTM with `mask_zero=False`, to test the effect of padding. |

Every model outputs **6 probabilities** (one per emotion). The highest one (`argmax`) is the predicted emotion.

### Shared neural architecture

`Embedding (10,000 x 128)` -> `recurrent layer (64 units)` -> `Dropout (0.4)` -> `Dense (6, softmax)`

- Loss: sparse categorical cross-entropy; optimizer: Adam; batch size 64; up to 20 epochs
- Early stopping on validation loss (patience 3), best weights restored
- Class weights; fixed seed 42; `mask_zero=True` (padding is ignored)

### Preprocessing

Lowercase and strip punctuation -> vocabulary of the top 10,000 words (0 = padding, 1 = unknown)
-> pad or cut every text to 50 word ids. The same `clean()` and `encode()` are used for
training and for new sentences.

## Results

| Model | Val acc | Test acc | Test macro-F1 | Parameters | Train time (s) |
|---|---|---|---|---|---|
| **GRU** | 0.921 | 0.9205 | **0.880** | 1,317,638 | 15.0 |
| LSTM | 0.906 | 0.9075 | 0.867 | 1,329,798 | 18.7 |
| BiGRU | 0.9105 | 0.902 | 0.861 | 1,355,270 | 16.4 |
| LSTM (no mask) | 0.8845 | 0.8775 | 0.845 | 1,329,798 | 39.1 |
| TF-IDF + LogReg | 0.889 | 0.882 | 0.837 | 197,970 | 19.5 |
| SimpleRNN | 0.792 | 0.772 | 0.708 | 1,292,742 | 16.8 |

**Findings**
- GRU is the best model on every metric (about 4.3 macro-F1 points above the TF-IDF baseline) and was also chosen by validation accuracy.
- Gated cells (LSTM, GRU) clearly beat the plain SimpleRNN.
- BiGRU did not beat GRU on these short texts.
- Ignoring the padding (mask) helps: removing it cost about 2.2 macro-F1 points and doubled training time.
- Each model was trained once (one seed), so small gaps (for example GRU vs LSTM) may change across runs.

## Project structure

```
.
├── emotion_classifier.ipynb          # training, evaluation, error analysis, predictions
├── app.py                            # Streamlit demo
├── results.csv                       # comparison table written by the notebook
├── artifacts/                        # vocab.json, labels.json, one .keras file per model
├── Emotion_Classification_DL.pptx    # presentation slides
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
pip install tensorflow streamlit pandas numpy scikit-learn matplotlib seaborn datasets
```

## Usage

### 1. Train and evaluate (notebook)

Open `emotion_classifier.ipynb` (Google Colab or Jupyter) and run all cells in order.
This trains every model, writes `results.csv`, and shows the classification report,
confusion matrix and error analysis for the best model.

To compare all models in the app, add this cell at the end so each model is saved
(the notebook saves only the best one by default):

```python
os.makedirs("artifacts", exist_ok=True)
for name, m in trained.items():
    m.save(f"artifacts/{name}.keras")
```

Keep `vocab.json` and `labels.json` in the same `artifacts/` folder.

### 2. Predict on new text (notebook)

```python
predict_emotion(["i feel so happy and grateful today"])                 # best model
predict_emotion(["i feel so happy and grateful today"], name="LSTM")    # a specific model
```

Output example: `-> joy (99%), then love (0%)`

### 3. Run the demo app

Place `app.py`, `results.csv` and the `artifacts/` folder together, then:

```bash
streamlit run app.py
```

- **Live demo tab:** type a sentence and see every model's prediction and confidence, whether they agree, and a probability chart.
- **Model comparison tab:** the results table and a macro-F1 chart from `results.csv`.

## Limitations

- One dataset, English tweets only, with hashtag-based (noisy) labels
- Short "i feel ..." style texts: long or very different text can give unreliable answers, and there is no "neutral" class
- One random seed per model

## Future work

- Pretrained embeddings (for example GloVe)
- Fine-tuning a transformer such as DistilBERT
- Several seeds per model, with averaged results
- Testing on a human-labelled dataset

## Author

Anirudh, B.Tech CSE, IcfaiTech Hyderabad
