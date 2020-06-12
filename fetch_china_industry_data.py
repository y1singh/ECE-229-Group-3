'''
This script downloads all the
Chinese stock market data and
stores the mapping of industry type
and stock in data/china_industry_map.p
and also stores the individual stock's
timeseries data in data/china_stocks_timeseries.csv.
'''

import pandas as pd
import pickle
from tqdm import tqdm
import yfinance as yf

ticker_industry_mapping = pd.DataFrame(columns=["ticker","longName","industry","country"])
china_stocks_timeseries = pd.DataFrame()

for i in tqdm(range(600100, 699999)):
    try:
        tick = str(i)+".SS"
        data = yf.Ticker(tick).info
        vals = {"ticker":tick,"longName":data['longName'],
                    "country":data["country"],"industry":data['industry']}
        ticker_industry_mapping = ticker_industry_mapping.append(vals,ignore_index=True)

        data2 = yf.download(
            tickers = str(i)+".SS",
            period = "6mo",
            group_by = 'ticker',
            auto_adjust = True,
            prepost = True,
            threads = True,
        )
        pd.concat([china_stocks_timeseries,data2])
    except Exception:
        pass
    finally:
        pass
    
china_stocks_timeseries.to_csv("data/china_stocks_timeseries.csv")

with open('data/china_industry_map.p', 'wb') as fp:
    pickle.dump(ticker_industry_mapping,fp,protocol=pickle.HIGHEST_PROTOCOL)