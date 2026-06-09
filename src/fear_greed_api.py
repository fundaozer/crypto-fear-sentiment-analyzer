import requests
import pandas as pd, os,time,json

# FETCH FEAR GREED İNDEX DATA

def fetch_fear_greed_data():
    base_url="https://api.alternative.me"
    endpoint="/fng/"
    params={
        "limit":365,
        "format":"json"
    }
    url=base_url+endpoint

    try:
        response=requests.get(url,params=params)
        response.raise_for_status() 
        data = response.json()

        df=pd.DataFrame(data["data"])
        df.drop("time_until_update",axis=1,inplace=True)
        df["value"] = df["value"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        df=df.rename(columns={"value":"fear_greed_value","value_classification":"fear_greed_label","timestamp":"date"})
        df =df[["date", "fear_greed_value", "fear_greed_label"]]
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}") 
        return None
