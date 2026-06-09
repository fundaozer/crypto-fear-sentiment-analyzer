import requests
import pandas as pd, os,time,json

# FETCH PRICE DATA FROM COINGECKO

def fetch_bitcoin_data():
    base_url="https://api.coingecko.com/api/v3"
    endpoint="/coins/bitcoin/market_chart"
    params={
        "vs_currency":"usd",
        "days":365,
        "interval":"daily"
    }
    url=base_url+endpoint

    try:
        response=requests.get(url,params=params)
        response.raise_for_status() 
        data = response.json()  

        prices = data["prices"]
        volumes = data["total_volumes"]
        market_caps = data["market_caps"]

        df_price = pd.DataFrame(prices, columns=["date", "price"])
        df_volume = pd.DataFrame(volumes, columns=["date", "volume"])
        df_market_cap = pd.DataFrame(market_caps, columns=["date", "market_cap"])

        df = df_price.merge(df_volume, on="date").merge(df_market_cap, on="date")
        df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date

        return df 
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}") 
        return None

    






