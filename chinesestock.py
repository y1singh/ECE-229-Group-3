import yfinance as yf

# 600000.SS 000000.SZ 300000.SZ
for i in range(600100, 699999):
    try:
        data = yf.Ticker(str(i)+".SS").info
        file = open("info.txt", "a")
        file.write("%d.SS %s %s\n"%(i, data['longName'], data['industry']))
        file.close()
        data2 = yf.download(
            tickers = str(i)+".SS",
            period = "6mo",
            group_by = 'ticker',
            auto_adjust = True,
            prepost = True,
            threads = True,
        )
        data2.to_csv(str(i)+".SS.csv")
        print(i)
    except Exception:
        pass
    finally:
        pass
