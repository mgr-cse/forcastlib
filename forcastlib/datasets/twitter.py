import pandas as pd
import numpy as np

def tweets_data():
    N = 360
    # replace it with url
    data = pd.read_csv('data/ethusd_tweets.csv') [::-1]
    highp = pd.to_numeric(data['High'].iloc[-N:])
    lowp = pd.to_numeric(data['Low'].iloc[-N:])
    openp = pd.to_numeric(data['Open'].iloc[-N:])
    closep = pd.to_numeric(data['Close'].iloc[-N:])
    tweets = pd.to_numeric(data['Tweets'].replace('null', 0).iloc[-N:])
    volume = pd.to_numeric(data['Volume'].iloc[-N:])
    marketcap = pd.to_numeric(data['Market Cap'].iloc[-N:])

    normal_close = closep

    highp = highp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    lowp = lowp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    openp = openp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    closep = closep.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    tweets = tweets.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    volume = volume.pct_change().replace(np.nan, 0).replace(np.inf, 0)
    marketcap = marketcap.pct_change().replace(np.nan, 0).replace(np.inf, 0)

    normal_close = np.array(normal_close)
    highp = np.array(highp)
    lowp = np.array(lowp)
    openp = np.array(openp)
    closep = np.array(closep)
    tweets = np.array(tweets)
    volume = np.array(volume)
    marketcap = np.array(marketcap)
    
    WINDOW = 7
    STEP = 1
    FORECAST = 1

    X, Y = [], []
    for i in range(0, len(openp), STEP): 
        try:
            o = openp[i:i+WINDOW]
            h = highp[i:i+WINDOW]
            l = lowp[i:i+WINDOW]
            c = closep[i:i+WINDOW]
            v = volume[i:i+WINDOW]
            t = tweets[i:i+WINDOW]
            m = marketcap[i:i+WINDOW]

    #         y_i = (normal_close[i+WINDOW+FORECAST] - normal_close[i+WINDOW]) / normal_close[i+WINDOW]
            y_i = closep[i+WINDOW+FORECAST]
            x_i = np.column_stack((o, h, l, c, v, t, m))
            x_i = x_i.flatten()

        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

    X, Y = np.array(X), np.array(Y)
    return (X, Y)