'''
This script downloads the data for
the stocks in ticker_industry_map.json.
It then calculates the p-value of each stock
w.r.t. the spread of COVID and stores it
as a csv in p_val_march.csv
'''

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from scipy.stats import pearsonr
import json

with open("data/ticker_industry_map.json","r") as fp:
    json_data = json.load(fp)
    
tickers=[]
for lst in list(json_data.values()):
    tickers += lst
    
query_str = str(tickers).replace("'", '').replace(",",'').replace("[","").replace("]","")
stockData = yf.download(tickers = query_str, 
                        start="2020-03-01", end="2020-04-30")

close_value = stockData.T.loc['Close']
tickers = close_value.index.unique()

# take only values from 1st March
time_series_confirmed = pd.read_csv('data/time_series_covid_19_confirmed.csv')
confirmed_count = time_series_confirmed.set_index("Country/Region").loc['US'][42:].to_list()

p_val = pd.DataFrame(columns = ["Ticker","Date","p_val","industry","p_val_cumsum"])
ticker_p_map = dict()
days_count = len(close_value.columns)
for ticker in tqdm(tickers):
    try:
        industry = yf.Ticker(ticker).info["industry"]
    except:
        print("Industry not found for ",ticker)
        industry = "Not found"
        
    for i,day in enumerate(close_value.columns.astype(str).to_list()):
        stat, p = pearsonr(confirmed_count[:i+1],close_value.loc[ticker].to_list()[:i+1])
        vals = {"Ticker":ticker,"Date":day,"p_val":10**p,
                    "price":close_value.loc[ticker][day],"industry":industry}
        p_val = p_val.append(vals,ignore_index=True)
    p_val.loc[p_val.Ticker==ticker,"p_val_cumsum"]=p_val[p_val.Ticker==ticker]["p_val"].cumsum()

p_val.fillna(0,inplace=True)
p_val.to_csv("data/p_val_march.csv")