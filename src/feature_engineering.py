import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# SENTIMENT SCORE

def calculate_sentiment_score(df_headlines):
    positive_words=["pump","bullish","recover","gain","surge","rally","growth","adoption","approval","breakout"]
    negative_words=["crash","dump","fear","panic","selloff","drop","bearish","liquidation","loss","hack","ban","risk","recession"]

    scores=[]

    for headline in df_headlines["headline"]:
        score=0
        words=headline.lower().split()
        for word in words:
            if word in positive_words:
                score+=1
            elif word in negative_words:
                score-=1
            else:
                score
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
    df["sentiment_negativity"] = df["negative_headline_count"] / df["headline_count"].replace(0, np.nan)
    df["momentum_score"] = df["price"] / df["price"].shift(7) - 1

    return df

# MARKET STRESS INDEX
  
def calculate_market_stress_index(df):
    scaler=MinMaxScaler()

    df["volatility_scaled"]=scaler.fit_transform(df[["volatility"]])
    df["volume_spike_scaled"]=scaler.fit_transform(df[["volume_spike"]])
    df["sentiment_negativity_scaled"]=scaler.fit_transform(df[["sentiment_negativity"]])

    df["market_stress_index"]=(
        df["volatility_scaled"]*0.4 +
        df["volume_spike_scaled"]*0.3 +
        df["sentiment_negativity_scaled"]*0.3
    )
    return df

# PANIC SCORE 

def calculate_panic_score(df):
    df["fear_component"]= 1- df["fear_greed_value"]/100

    # if (daily return < 0) -> price drop component=abs(daily return) else -> price drop component=0
    df["price_drop_component"]=np.where(
        df["daily_return"]<0,
        df["daily_return"].abs(),
        0
    )

    df["panic_score"]=(
        df["market_stress_index"]*0.5 +
        df["fear_component"]*0.3 +
        df["price_drop_component"]*0.2
    )

    # if (panic score > 0.70) -> panic day=1  else -> panic day=0
    df["is_panic_day"]=np.where(df["panic_score"]>0.70 ,1,0)

    return df
   
