import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import string

# SENTIMENT SCORE

def calculate_sentiment_score(df_headlines):
    positive_words = [
        "pump", "bullish", "recover", "gain", "surge", "rally", "growth", "adoption", 
        "approval", "breakout", "bull", "high", "success", "rise", "green", "profit", 
        "win", "support", "investment", "buy", "up", "skyrocket", "positive"
    ]
    negative_words = [
        "crash", "dump", "fear", "panic", "selloff", "drop", "bearish", "liquidation", 
        "loss", "hack", "ban", "risk", "recession", "collapse", "unsafe", "scam", 
        "shut", "sell", "lose", "dystopian", "frozen", "decline", "red", "scammer", 
        "exploit", "stolen", "theft", "warn", "warning", "investigation", "sue", 
        "lawsuit", "regulation", "down", "negative", "bear"
    ]

    scores=[]

    for headline in df_headlines["headline"]:
        score=0
        # Clean punctuation to ensure accurate word matching
        headline_clean = str(headline).translate(str.maketrans('', '', string.punctuation))
        words = headline_clean.lower().split()
        for word in words:
            if word in positive_words:
                score+=1
            elif word in negative_words:
                score-=1
        scores.append(score)
    
    df_headlines["score"]=scores
    df_sentiment = df_headlines.groupby("date").agg(
       daily_sentiment_score=("score", "mean"),
       negative_headline_count=("score", lambda x: (x < 0).sum()),
       headline_count=("score", "count")
    ).reset_index()

    return df_sentiment

# MERGE DATAFRAMES 

def merge_dataframes(df_bitcoin_price,df_fear_greed,df_sentiment):
    df_merged=df_bitcoin_price.merge(df_fear_greed , on="date" , how="left")
    df_merged = df_merged.merge(df_sentiment, on="date", how="left")

    return df_merged

# FEATURE ENGİNEERİNG 

def calculate_features(df_merged):
    df=df_merged.copy()
    df["daily_return"] = df["price"].pct_change()
    df["volatility"] = df["daily_return"].rolling(7).std()
    df["volume_spike"] = df["volume"] / df["volume"].rolling(7).mean()
    
    # Calculate sentiment negativity only for days with news, otherwise NaN
    df["sentiment_negativity"] = df["negative_headline_count"] / df["headline_count"].replace(0, np.nan)
    
    df["momentum_score"] = df["price"] / df["price"].shift(7) - 1
    df["next_day_return"] = df["daily_return"].shift(-1)

    return df

# MARKET STRESS INDEX
  
def calculate_market_stress_index(df):
    scaler=MinMaxScaler()

    df["volatility_scaled"] = scaler.fit_transform(df[["volatility"]])
    df["volume_spike_scaled"] = scaler.fit_transform(df[["volume_spike"]])
    
    # Scale sentiment negativity only on rows that have news headlines
    has_news = df["headline_count"].notna() & (df["headline_count"] > 0)
    df["sentiment_negativity_scaled"] = np.nan
    
    sent_non_nan = df.loc[has_news, ["sentiment_negativity"]]
    if not sent_non_nan.empty:
        df.loc[has_news, "sentiment_negativity_scaled"] = scaler.fit_transform(sent_non_nan)
    
    # Dynamic Market Stress Index
    # When news is present: Volatility * 0.4 + Volume Spike * 0.3 + Sentiment Negativity * 0.3
    # When news is missing: Volatility * 0.57 + Volume Spike * 0.43 (re-normalized weights)
    df["market_stress_index"] = np.where(
        has_news,
        df["volatility_scaled"] * 0.4 + df["volume_spike_scaled"] * 0.3 + df["sentiment_negativity_scaled"].fillna(0) * 0.3,
        df["volatility_scaled"] * (0.4 / 0.7) + df["volume_spike_scaled"] * (0.3 / 0.7)
    )
    return df

# PANIC SCORE 

def calculate_panic_score(df):
    df["fear_component"] = 1 - df["fear_greed_value"] / 100

    # if (daily return < 0) -> price drop component=abs(daily return) else -> price drop component=0
    df["price_drop_component"] = np.where(
        df["daily_return"] < 0,
        df["daily_return"].abs(),
        0
    )
    
    # Use 98th percentile to prevent a single outlier crash from squashing normal price drops
    max_drop = df["price_drop_component"].quantile(0.98)
    if pd.isna(max_drop) or max_drop == 0:
        max_drop = 0.05  # fallback default of 5% drop if calculation is not possible
    df["price_drop_component_scaled"] = np.minimum(df["price_drop_component"] / max_drop, 1.0)

    df["panic_score"] = (
        df["market_stress_index"] * 0.5 +
        df["fear_component"] * 0.3 +
        df["price_drop_component_scaled"] * 0.2
    )

    # Panic threshold is set to 0.60 to yield a realistic frequency of panic days (~5-6% of the year)
    df["is_panic_day"] = np.where(df["panic_score"] > 0.60, 1, 0)

    return df
   
