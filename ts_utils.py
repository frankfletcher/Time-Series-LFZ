import logging
import numpy as np
import pandas as pd
from pandas.tseries import offsets

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime as dt


# utility - polynomial regression
def get_polyregr_pred(ts: pd.Series, degree=1, retmodel=False):
    """Return an array of Polynomial regression predictions from a DataFrame
    Alternatively, if retmodel==True, return the entire model
    """

    if type(ts) == pd.DataFrame:
        logging.info("type is DataFrame - converting to series")
        if ts.shape[1] > 1:
            logging.info(
                "DataFrame has more than one column, only selecting the first column"
            )
        ts = ts.iloc[:, 0]

    X = np.arange(len(ts)).reshape(-1, 1)

    y = ts.values.reshape(-1, 1) if type(ts) == pd.Series else ts.reshape(-1, 1)

    # build a polynomial regression pipeline
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # fit the model to our data
    polyreg.fit(X, y)

    # print(pd.Series(polyreg.predict(X).reshape(-1,)))
    return (
        polyreg
        if retmodel == True
        else polyreg.predict(X).reshape(
            -1,
        )
    )
