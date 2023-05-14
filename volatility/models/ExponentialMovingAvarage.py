import pandas as pd

import numpy as np
from typing import Tuple, List


def get_estimator(price_data, window=30, trading_periods=252, clean=True):
    result = getDailyVol(price_data['Close'], span0=window)
    if clean:
        return result.dropna()
    else:
        return result

def getDailyVol(close: pd.Series, span0: int = 100):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    except Exception as e:
        print(f"error: {e}\nplease confirm no duplicate indices")
    df0 = df0.ewm(span=span0).std().rename("dailyVol")
    return df0