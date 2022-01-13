#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author: wangkunxiong
@File: AR_GARCH_model.py
@Description: AR_GARCH_model 拟合预测
'''

import numpy as np
import pandas as pd
import arch
from itertools import product


class ar_garch_model():
    '''
    AR-GARCH 模型， 给定均值模型和方差模型的参数取值范围，取最小BIC中的最小AIC作为最优模型进行预测
    @parm: mean_param_range: tuple or list, AR 模型的滞后阶数取值范围(上限值,下限值)
    @parm: vol_parms_range: tuple or list, GARCH 模型的参数取值范围(上限值,下限值)
    @example:

        tM = AR_GARCH_model((1, 5), (1, 3), test_ret_ser)
        tM.fits()
        tM.predict(1)
        out:
            {'mean': {'h.1': -0.24203111507167427},
             'variance': {'h.1': -0.24203111507167427}}
        tM.summary()
        out:
            optimized model is AR(5)-GARCH(1,1)
            AIC is 766.7286078301395, BIC is 798.2765916535308
            length of train dataset is 251

                                       AR - GARCH Model Results
            ==============================================================================
            Dep. Variable:                 CHGPct   R-squared:                       0.051
            Mean Model:                        AR   Adj. R-squared:                  0.031
            Vol Model:                      GARCH   Log-Likelihood:               -374.364
            Distribution:                  Normal   AIC:                           766.729
            Method:            Maximum Likelihood   BIC:                           798.277
                                                    No. Observations:                  246
            Date:                Thu, Dec 30 2021   Df Residuals:                      240
            Time:                        17:16:00   Df Model:                            6
                                             Mean Model
            ===========================================================================
                             coef    std err          t      P>|t|     95.0% Conf. Int.
            ---------------------------------------------------------------------------
            Const      7.1632e-03  6.322e-02      0.113      0.910    [ -0.117,  0.131]
            CHGPct[1]      0.0377  6.323e-02      0.596      0.551 [-8.622e-02,  0.162]
            CHGPct[2]     -0.0282  5.920e-02     -0.476      0.634  [ -0.144,8.784e-02]
            CHGPct[3]     -0.0190  6.291e-02     -0.303      0.762    [ -0.142,  0.104]
            CHGPct[4]      0.0897  5.808e-02      1.544      0.122 [-2.413e-02,  0.204]
            CHGPct[5]     -0.2030  6.413e-02     -3.165  1.549e-03 [ -0.329,-7.730e-02]
                                          Volatility Model
            ===========================================================================
                             coef    std err          t      P>|t|     95.0% Conf. Int.
            ---------------------------------------------------------------------------
            omega          0.0399  3.338e-02      1.196      0.232 [-2.550e-02,  0.105]
            alpha[1]       0.0640  3.452e-02      1.853  6.381e-02 [-3.676e-03,  0.132]
            beta[1]        0.9032  4.614e-02     19.576  2.487e-85    [  0.813,  0.994]
            ===========================================================================
            Covariance estimator: robust
            ARCHModelResult, id: 0x7fa2d4ae44c0
    '''


    def __init__(self,
                 mean_param_range: tuple or list,
                 vol_parms_range: tuple or list,
                 ret_ser: pd.Series,
                 ):

        self.mean_parms_list = [list(range(mean_param_range[0], x)) for x in
                                list(range(mean_param_range[0] + 1, mean_param_range[1] + 2))]
        self.vol_parms_list = list(product(list(range(vol_parms_range[0], vol_parms_range[1] + 1)),
                                           list(range(vol_parms_range[0], vol_parms_range[1] + 1))))

        nullnums = ret_ser.isnull().sum()
        if nullnums == 0:
            if isinstance(ret_ser.index, pd.DatetimeIndex):
                # 日频收益率需要整体放大来获取模型更好的拟合
                self.train_data = ret_ser * 100
            else:
                raise ValueError('trainning data must be a single index pd.Series with datetime as indexes')
        else:
            raise ValueError(f'There are {nullnums} Nan values')


    def AR_GARCH_model(self, mean_parms, vol_parms, ret_ser):
        ar = arch.univariate.ARX(ret_ser, lags=mean_parms)
        ar.volatility = arch.univariate.GARCH(p=vol_parms[0], q=vol_parms[1])
        # ar.distribution = arch.univariate.StudentsT()
        return ar


    def fits(self):
        score_df = pd.DataFrame()
        for mean_parms in self.mean_parms_list:
            for vol_parms in self.vol_parms_list:
                model = self.AR_GARCH_model(mean_parms, vol_parms, self.train_data)
                res = model.fit(disp='off')
                score_dict = {'mean_parms': mean_parms, 'vol_parms': vol_parms, 'AIC': res.aic, 'BIC': res.bic}
                score_df = score_df.append(score_dict, ignore_index=True)
        score_df.sort_values(['AIC'], inplace=True)
        self.opt_mean_parms = score_df.iloc[0, 0]
        self.opt_vol_parms = score_df.iloc[0, 1]
        self.aic = score_df.iloc[0, 2]
        self.bic = score_df.iloc[0, 3]
        opt_model = self.AR_GARCH_model(self.opt_mean_parms, self.opt_vol_parms, self.train_data)
        self.opt_model = opt_model
        # return opt_model.fit(disp='off')


    def predict(self, horizon=1):
        res_dict = {}
        frcst = self.opt_model.fit(disp='off').forecast(horizon=horizon, reindex=False)
        # 预测收益率和方差需要整体缩小，因为之前的收益率被放大100倍
        res_dict['mean'] = (frcst.mean.iloc[0, :] / 100).to_dict()
        res_dict['variance'] = (frcst.variance.iloc[0, :] / 10000).to_dict()
        return res_dict


    def summary(self):
        print(f'''
            \r optimized model is AR({self.opt_mean_parms[-1]})-GARCH({self.opt_vol_parms[0]},{self.opt_vol_parms[1]})
            \r AIC is {self.aic}, BIC is {self.bic}
            \r length of train dataset is {len(self.train_data)}
        ''')
        return self.opt_model.fit(disp='off')


    def rolling_predict(self, rolling_window, rolling_ret_ser, rescale=True):

        def updata_predictions(ser):
            model = self.AR_GARCH_model(self.opt_mean_parms, self.opt_vol_parms, ser).fit(disp='off')
            var_frcst = model.forecast(horizon=1, reindex=False).variance.iloc[0]
            mean_frcst = model.forecast(horizon=1, reindex=False).mean.iloc[0]
            var_forecasts[var_frcst.name] = var_frcst
            mean_forecasts[mean_frcst.name] = mean_frcst
            return True

        if rescale:
            scaled_ser = rolling_ret_ser * 100
        else:
            scaled_ser = rolling_ret_ser.copy()
        var_forecasts = {}
        mean_forecasts = {}

        scaled_ser.rolling(window=rolling_window).apply(updata_predictions)
        predict_df = pd.concat([pd.DataFrame(mean_forecasts).T, pd.DataFrame(var_forecasts).T], axis=1)
        predict_df.columns = ['mean_h1', 'var_h1']

        if rescale:
            predict_df['mean_h1'] = predict_df['mean_h1']/100
            predict_df['var_h1'] = predict_df['var_h1']/10000

        return predict_df




if __name__ == '__main__':
    # 生成测试数据
    train_ser = pd.Series((np.random.random(250) - 0.5) / 5, index=pd.date_range('20200101', '20201231')[:250])
    test_ser = pd.Series((np.random.random(250) - 0.5) / 5, index=pd.date_range('20200101', '20201231')[:250])
    # 训练模型
    tM = ar_garch_model((1, 5), (1, 3), train_ser)
    tM.fits()
    # 预测下一期收益率和波动率
    tM.predict(1)
    # 拟合结果
    tM.summary()
    predict_df = tM.rolling_predict(60, test_ser)
