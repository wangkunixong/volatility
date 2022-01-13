#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author: wangkunxiong
@File: volatility_calculate.py
@Description: volatility calculation
'''
import numpy as np
import pandas as pd
from typing import Union, Optional


class volatility_method(object):
    """methods for calculate volatility

    """

    @staticmethod
    def close_to_close(close_ser: pd.Series):
        """Standard history volatility

        Args:
            close_ser: pandas.Series

        Returns:
            float

        References:
            https://portfolioslab.com/tools/close-to-close-volatility
        """
        log_ret = np.log(close_ser / close_ser.shift(1)).iloc[1:]
        return log_ret.std()

    @staticmethod
    def roger_satchell(ohlc_df: pd.DataFrame):
        """Roger Satchell volatility

        Args:
            ohlc_df: a pandas.DataFrame of datetime index and columns follow the order of OHLC

        Returns:
            a float

        References:
            https://portfolioslab.com/tools/rogers-satchell
        """
        open = ohlc_df.iloc[:, 0]
        high = ohlc_df.iloc[:, 1]
        low = ohlc_df.iloc[:, 2]
        close = ohlc_df.iloc[:, 3]

        return np.sqrt(np.mean((np.log(high / close) * np.log(high / open) + np.log(low / close) * np.log(low / open))))

    @staticmethod
    def yang_zhang(ohlc_df: pd.DataFrame, alpha: float = 0.34):
        """Yang Zhang volatility

        Args:
            ohlc_df: a pandas.DataFrame of datetime index and columns follow the order of OHLC
            alpha: float
        Returns:
            a float

        References:
            https://dynamiproject.files.wordpress.com/2016/01/measuring_historic_volatility.pdf
        """
        open = ohlc_df.iloc[:, 0]
        close = ohlc_df.iloc[:, 3]

        o2c = np.log(open / close.shift(1)).iloc[1:]
        o2c_var = o2c.var()

        c2o = np.log(close / open).iloc[1:]
        c2o_var = c2o.var()

        rsv = volatility_method.roger_satchell(ohlc_df.iloc[1:]) ** 2

        sample_num = len(ohlc_df.iloc[1:] - 1)

        k = alpha / ((1 + alpha) + (sample_num + 1) / (sample_num - 1))

        # print(f'o2c_var: {o2c_var * 242}, c2o_var: {c2o_var * 242}, rsv:{rsv * 242}, k: {k}')
        return np.sqrt(o2c_var + k * c2o_var + (1 - k) * rsv)


class volatility(volatility_method):

    def __init__(self,
                 period_factor: Optional[int] = None):
        self.period_factor = period_factor

    def calculate(self, method: str, data, *args, **kwargs):
        """volatility calculation

        Args:
            method: str, method in volatility_method
            data: pandas.Series or pandas.DataFrame

        Returns:
            float
        """
        try:
            res = getattr(self, method)(data, *args, **kwargs)
        except:
            raise Exception('check method and params of mathod')
        else:
            if self.period_factor is None:
                return res
            else:
                return res * np.sqrt(self.period_factor)


    def rolling_cal(self,
                    method: str,
                    rolling_windows: int,
                    data: Union[pd.Series, pd.DataFrame],
                    min_unnull_num: Optional[int] = None,
                    *args, **kwargs):
        """Calculate rolling volatility with specific method

        Args:
            method: str, method in volatility_method
            rolling_windows: int
            data: a pandas.Series or pandas.DataFrame depends on mehtod
            min_unnull_num: int, minimum number of unnull values in rolling subset
        Returns:
            a pandas.Series

        """
        res = {}
        if min_unnull_num is None:
            min_unnull_num = rolling_windows
        for idx_num in range(len(data)+1-rolling_windows):
            subset = data.iloc[idx_num:idx_num+rolling_windows]
            idx = subset.index[-1]
            if subset.isnull().values.sum() <= rolling_windows-min_unnull_num:
                res[idx] = self.calculate(method, subset, *args, **kwargs)
            else:
                res[idx] = np.nan
        return pd.Series(res)


if __name__ == '__main__':

    test = pd.DataFrame(np.random.random(size=(len(pd.date_range('20210101', '20210215')), 4))-0.5+10,
                        index=pd.date_range('20210101', '20210215'),
                        columns=['open', 'high', 'low', 'close'])

    models = ['close_to_close', 'roger_satchell', 'yang_zhang']


    for model in models:
        if model == "close_to_close":
            res = volatility(period_factor=252).calculate(model, test.close)
        else:
            res = volatility(period_factor=252).calculate(model, test)
        print(f'{model}: {res}')


    for model in models:

        if model == "close_to_close":
            rolling_res = volatility(period_factor=252).rolling_cal(model, 10, test.close)

        else:
            rolling_res = volatility(period_factor=252).rolling_cal(model, 10, test)
        print(f'{model}: \n{rolling_res}')