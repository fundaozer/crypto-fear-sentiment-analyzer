from dotenv import load_dotenv
import pandas as pd,os,time,sys,requests

load_dotenv()
api_key = os.getenv("NEWS_API_KEY")
api_key2 = os.getenv("CRYPTOCOMPARE_API_KEY")

# FETCH NEWS HEADLİNES FROM NEWS

def fetch_news_headlines():
    url="https://newsapi.org/v2/everything"
    params={
        "q":"bitcoin",
        "language":"en",
        "pageSize":100,
        "apiKey":api_key
    }

    try:
        response=requests.get(url,params=params)
        response.raise_for_status() 
        data = response.json()

        records = []
        for article in data["articles"]:
            records.append({
                "date": article["publishedAt"],
                "source": "news",
                "headline": article["title"]
            })
        
        df=pd.DataFrame(records)
        df["date"]=pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["headline"])
        return df
      

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}") 
        return None
    
# FETCH NEWS HEADLINES FROM CRYPTOCOMPARE

def fetch_cryptocompare_headlines():
    url="https://min-api.cryptocompare.com/data/v2/news/"
    params={
        "lang":"EN",
        "apiKey":api_key2
    }

    try:
        response=requests.get(url,params=params)
        response.raise_for_status() 
        data = response.json()

        records = []
        for article in data["Data"]:
            records.append({
                "date": article["published_on"],
                "source": "cryptocompare",
                "headline": article["title"]
            })
        
        df=pd.DataFrame(records)
        df["date"]=pd.to_datetime(df["date"] , unit="s").dt.date
        df = df.dropna(subset=["headline"])
        return df
      

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}") 
        return None

    
