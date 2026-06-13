# 🔮 Crypto Fear & Sentiment Analyzer

A data science project that analyzes the relationship between cryptocurrency market sentiment and Bitcoin price movements. It combines real-time price data, the Fear & Greed Index, and news headline sentiment to generate custom risk metrics and train machine learning models for next-day price direction prediction.

---

## 📌 Problem Statement

> This project analyzes the relationship between cryptocurrency market sentiment and price movements.
> It combines historical crypto price data, Fear & Greed Index values, and social/news headline sentiment
> to generate custom risk metrics such as **Market Stress Index** and **Panic Score**.
> A machine learning model is trained to predict whether the next day's price movement will be negative.

Cryptocurrency markets are highly sentiment-driven. Fear, panic, and social media narratives can cause rapid price swings that are difficult to capture with traditional technical indicators alone. This project explores whether combining quantitative fear metrics with news sentiment can improve predictive power.

---

## 🗂️ Project Structure

```
crypto-fear-sentiment-analyzer/
│
├── data/                           # ⚠️ Not tracked by Git — regenerate via notebooks
│   ├── raw/                        # Raw API responses (gitignored)
│   │   ├── bitcoin_price_raw.csv   # CoinGecko: price, volume, market cap (365 days)
│   │   ├── fear_greed_raw.csv      # alternative.me: Fear & Greed Index (365 days)
│   │   └── headlines_raw.csv       # NewsAPI + CryptoCompare: news headlines
│   └── processed/                  # Processed data (gitignored)
│       └── bitcoin_processed.csv   # Merged & feature-engineered dataset (365 rows, 24 cols)
│
├── notebooks/
│   ├── 01_data_collection.ipynb    # API calls & raw data collection
│   ├── 02_feature_engineering.ipynb# Feature computation pipeline
│   ├── 03_visualization.ipynb      # Charts & exploratory analysis
│   └── 04_model_training.ipynb     # Model training & evaluation
│
├── src/
│   ├── coingecko_api.py            # Bitcoin price data (CoinGecko API)
│   ├── fear_greed_api.py           # Fear & Greed Index (alternative.me API)
│   ├── sentiment_scraper.py        # News headlines (NewsAPI + CryptoCompare)
│   ├── feature_engineering.py      # Feature & metric computation
│   └── model.py                    # ML model training & evaluation
│
├── outputs/
│   ├── model_results.csv           # Model performance metrics
│   └── charts/
│       ├── 1_price_vs_fear_greed.png
│       ├── 2_heatmap.png
│       ├── 3_panic_vs_daily_return.png
│       └── 4_sentiment_vs_return.png
│
├── .env                            # API keys (not committed to Git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Technologies Used

| Category | Library / Tool |
|---|---|
| **Language** | Python 3.10+ |
| **Data Collection** | `requests`, `python-dotenv` |
| **Data Processing** | `pandas`, `numpy` |
| **Feature Scaling** | `scikit-learn` (MinMaxScaler, StandardScaler) |
| **Machine Learning** | `scikit-learn` (LogisticRegression, RandomForestClassifier) |
| **Visualization** | `matplotlib`, `seaborn` |
| **Notebooks** | Jupyter Notebook |

---

## 📡 Data Sources

| Source | Description | API |
|---|---|---|
| **CoinGecko** | Bitcoin daily price, volume, market cap (365 days) | Free, no key required |
| **alternative.me** | Fear & Greed Index (0–100 daily score) | Free, no key required |
| **NewsAPI** | Bitcoin news headlines | Requires `NEWS_API_KEY` |
| **CryptoCompare** | Crypto-specific news headlines | Requires `CRYPTOCOMPARE_API_KEY` |

---

## ⚙️ Custom Metrics

### 📈 Market Stress Index
A composite metric combining volatility, volume spike, and sentiment negativity.

```
When news available:
  Stress = Volatility × 0.4 + Volume Spike × 0.3 + Sentiment Negativity × 0.3

When no news:
  Stress = Volatility × 0.571 + Volume Spike × 0.429
```

### 🚨 Panic Score
A daily risk score combining market stress, fear, and price drops.

```
Panic Score = Market Stress × 0.5 + Fear Component × 0.3 + Price Drop × 0.2

Panic Day = 1  if Panic Score > 0.60
          = 0  otherwise
```

### 📊 Sample Output

| date | price | fear_greed_value | sentiment_negativity | market_stress_index | panic_score | is_panic_day |
|---|---|---|---|---|---|---|
| 2026-02-06 | 62,853 | 9 | 0.00 | 0.688 | 0.817 | 1 |
| 2026-02-07 | 65,420 | 14 | 0.00 | 0.412 | 0.623 | 1 |
| 2026-03-01 | 71,200 | 38 | 0.00 | 0.287 | 0.441 | 0 |

> **21 panic days** detected over the 365-day analysis period.

---

## 🤖 Model Results

Binary classification task: **predict whether next-day Bitcoin price will drop (1) or not (0).**

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | 54.2% | 53.5% | 63.9% | 58.2% |
| Random Forest | 45.8% | 46.8% | 61.1% | 53.0% |

- **Train / Test split:** 80% / 20% (chronological — no data leakage)
- **Test set size:** 72 samples
- **Baseline (random):** 50.0%

> Logistic Regression outperforms Random Forest on this dataset size (~357 samples). Results are consistent with the Efficient Market Hypothesis — short-term crypto price direction is inherently noisy.

### Features Used
```
volatility · volume_spike · sentiment_negativity
fear_greed_value · market_stress_index · panic_score · momentum_score
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/crypto-fear-sentiment-analyzer.git
cd crypto-fear-sentiment-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the root directory:
```
NEWS_API_KEY=your_newsapi_key_here
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key_here
```

### 4. Run the notebooks in order
```
01_data_collection.ipynb      → Collect raw data
02_feature_engineering.ipynb  → Compute features & metrics
03_visualization.ipynb        → Generate charts
04_model_training.ipynb       → Train & evaluate models
```

---

## 📊 Visualizations

| Chart | Description |
|---|---|
| `1_price_vs_fear_greed.png` | Bitcoin price vs Fear & Greed Index over time |
| `2_heatmap.png` | Correlation matrix of all features |
| `3_panic_vs_daily_return.png` | Panic Score vs daily returns |
| `4_sentiment_vs_return.png` | News sentiment negativity vs price returns |

---

## ⚠️ Limitations

- **Sentiment coverage:** News headlines are available for only ~26 of 365 days due to API free-tier limits. Days without news use `sentiment_negativity = 0` (neutral).
- **Short horizon:** Next-day prediction is highly noisy in crypto markets.
- **Small dataset:** 365 days of data limits model complexity; ensemble models may overfit.

---

## 📄 License

This project is for academic and educational purposes.
