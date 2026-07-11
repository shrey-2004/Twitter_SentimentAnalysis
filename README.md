# Twitter Sentiment Analysis

End-to-end NLP pipeline that cleans, analyses, and classifies sentiment in Twitter data across 32 brands and gaming topics — with an interactive executive dashboard.

**Output:** Interactive HTML dashboard with KPI cards, word clouds, an entity × sentiment heatmap, and model comparison charts.

## Project structure

```
├── twitter_training.csv               # Raw dataset (you provide)
├── twitter_validation.csv             # Raw held-out set (you provide)
├── sentiment_analysis.py              # Full pipeline script
├── dashboard_template.html            # HTML/JS shell the script injects data into
├── requirements.txt
├── twitter_clean.csv                  # Cleaned & lemmatised output (generated)
├── stats.json                         # EDA output (generated)
├── ml_results.json                    # Model metrics (generated)
└── twitter_sentiment_dashboard.html   # Interactive dashboard (generated)
```

`sentiment_analysis.py` and `dashboard_template.html` must stay in the same folder — the script reads the template and writes the finished dashboard next to it.

## Setup

```bash
pip install -r requirements.txt
```

`nltk` is optional. If it's missing (or you're offline), the script automatically falls back to scikit-learn's stopword list and skips lemmatisation — it still runs end to end, just with slightly less normalised text.

## Run it

**Terminal / VS Code:**
```bash
cd twitter_sentiment
python sentiment_analysis.py
```

**Jupyter notebook:** put `twitter_training.csv`, `twitter_validation.csv`, `sentiment_analysis.py`, and `dashboard_template.html` in the same folder as the notebook, then run:
```python
%run sentiment_analysis.py
```
or, to work with the pieces interactively:
```python
import sentiment_analysis as sa

train_raw = sa.load_raw(sa.TRAIN_PATH)
val_raw   = sa.load_raw(sa.VAL_PATH)
train_clean = sa.clean_dataframe(train_raw)
val_clean   = sa.clean_dataframe(val_raw)

stats = sa.run_eda(pd.concat([train_clean, val_clean]))
ml_results = sa.run_ml(train_clean, val_clean)
```

Either way, open `twitter_sentiment_dashboard.html` in a browser when it's done — it's fully self-contained (data is embedded inline), so you can also just email or host the file directly.

## What the pipeline does

1. **Clean** — lowercases, strips URLs/mentions/hashtag symbols/punctuation/digits, drops stopwords, lemmatises (if NLTK is available), removes empty rows and duplicates.
2. **EDA** — sentiment distribution, top 15 entities by volume, entity × sentiment heatmap (row-normalised %), average token count by sentiment, top 25 terms per sentiment class.
3. **Model bake-off** — TF-IDF (unigrams + bigrams, 8k features) feeding **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM**, trained on `twitter_training.csv` and scored on `twitter_validation.csv` as the true held-out set. Accuracy, precision, recall, F1, and confusion matrices are saved for all three; the best-accuracy model is highlighted in the dashboard.
4. **Dashboard** — a single HTML file (dark "signal desk" theme, Chart.js via CDN, no server needed) with a scrolling brand-sentiment ticker, KPI cards, distribution charts, the entity heatmap, CSS-based word clouds per sentiment, and the model comparison + confusion matrix.

## Notes

- Input CSVs are expected with **no header row**, columns in order: `tweet_id, entity, sentiment, text` (matches the standard Kaggle Twitter Sentiment Analysis dataset).
- Valid sentiment labels: `Positive`, `Negative`, `Neutral`, `Irrelevant`. Anything else is dropped during cleaning.
- The dashboard needs internet access once, in the browser, to load Chart.js and Google Fonts from CDN — everything else is embedded.
