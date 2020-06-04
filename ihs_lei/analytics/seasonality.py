import numpy as np
import scipy as sc
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from datetime import datetime
import scipy.stats as ss
from pprint import pprint


class SarimaxModelStructure(object):
    """
    Just holds the model structure for a particular sarimax model.
    Used when fitting exogenous variables to endogenous with a regressor.
    
    exogenous_regressors structure: {
        'name': Display Name,
        'internal_name': Internal name just for tracking (name of DF column)
        'data': Pandas series with the data for the exogenous regressor,
        'model': SarimaxModelStructure
    }
    """
    def __init__(self, periodicity=12,
                 fit_seasonality=True,
                 fit_trend=True,
                 fit_error=True,
                 exog_regressors=None,
                 trend_order=(1, 1, 0),
                 trend_seasonal_order=(2, 1, 1, 2),
                 season_order=(0, 0, 0),
                 season_seasonal_order=(1, 1, 1, 12),
                 error_order=(3, 0, 0),
                 error_seasonal_order=(0, 0, 1, 12)):
        
        self.fit_seasonality = fit_seasonality
        self.fit_trend = fit_trend
        self.fit_errors = fit_error
        self.exog_regressors = exog_regressors
        self.trend_order = trend_order
        self.trend_seasonal_order = trend_seasonal_order
        self.season_order = season_order
        self.season_seasonal_order = season_seasonal_order
        self.error_order = error_order
        self.error_seasonal_order = error_seasonal_order
        self.periodicity = periodicity

INDICATOR_TRANSFORMS = {
    "level": lambda series: series.dropna(how='any', axis=0),
    "change": lambda series: (series - series.shift(12)).dropna(how='any', axis=0),
    "original": lambda series: series
}

class PredictorSarimax(object):
    """
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    This class will fit a time series (Y) to a SARIMAX model of order (p, d, q)x(P, D, Q, S) with exogenous variables X.
    
    model
    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable or iterable of iterables, optional
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. `d` must be an integer
        indicating the integration order of the process, while
        `p` and `q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is
        an AR(1) model: (1,0,0).
    seasonal_order : iterable, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
        `d` must be an integer indicating the integration order of the process,
        while `p` and `q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. `s` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is to not
        include a trend component.
            
    """
    def __init__(self, endog, model_structure):
        # Model Parameters
        self.endog = endog
        self.model = model_structure
        self._fitted_model = None
        
    def export_data():
        """
        Returns all data for the model, including predictions and fitting data.
        """
        pass
        
    def fit(self, method='lbfgs', cov_type='opg'):
        """
        Fits *all* models. 
        """
        # The only reason for decomposition is to fit SARIMAX model to trend and to seasonality separately.
        # So let's do that.
        exog_dataset = self._get_exogenous_data()
        exog_dataset['endog'] = self.endog
        exog_dataset = exog_dataset.fillna(method='ffill').dropna(how='any')
        endog = exog_dataset['endog']
        del exog_dataset['endog']
        fitted_model = self._fit_sarimax(endog, exog_dataset, method, cov_type)
        self._fitted_model = fitted_model
        return fitted_model
        
    def summary(self):
        return self._fitted_model.summary()
        
    def get_data(self):
        """
        Just returns exogenous variable.
        """
        return self.endog
        
    def get_prediction(self):
        """
        Returns prediction
        """
        if self._fitted_model is None:
            raise NotImplementedError
            
        trend_prediction = self._fitted_model['trend'].get_prediction()
        seasonality_prediction = self._fitted_model['seasonality'].get_prediction()
        errors_prediction = self._fitted_model['errors'].get_prediction()
        final_prediction = trend_prediction.predicted_mean + seasonality_prediction.predicted_mean + errors_prediction.predicted_mean
        return final_prediction
        
    def _get_exogenous_data(self):
        """
        Just returns exogenous dataframe.
        """
        if self.model.exog_regressors is None:
            return None
        result = {}
        for exog_regressor, exog_regressor_model in self.model.exog_regressors.items():
            # Periodicity
            reg_periodicity = exog_regressor_model['model'].model.periodicity
            reg_trend, reg_seasonality = self.decompose(exog_regressor_model['model'].endog, reg_periodicity)
            if exog_regressor_model['transform_series'] is 'original':
                ex_reg = exog_regressor_model['model'].endog
            else:
                ex_reg = reg_trend
                
            
            result[exog_regressor] = INDICATOR_TRANSFORMS['level'](ex_reg)
            
        return pd.DataFrame(result).reindex(self.endog.index).fillna(method='ffill').dropna(how='all')

        
    def _fit_sarimax(self, endog, exog_dataset, method, cov_type):
        """
        Fits sarimax model given endogenous variable and exogenous structure
        """
        endog_trend, endog_seasonality = self.decompose(endog, self.model.periodicity)
        # Now we only fit IF we have to.
        trend_sarimax = None
        season_sarimax = None
        error_sarimax = None
        # Now.... what if I maintain the trend/cyclical info about exogenous variables and just use those?
        
        if self.model.fit_trend:
            trend_sarimax = sm.tsa.statespace.SARIMAX(
                endog_trend, trend='t', exog=exog_dataset,
                missing='drop',
                order=self.model.trend_order, seasonal_order=self.model.trend_seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False).fit(method=method)
        if self.model.fit_seasonality:
            season_sarimax = sm.tsa.statespace.SARIMAX(
                endog_seasonality,
                trend='n',
                missing='drop',
                order=self.model.season_order, seasonal_order=self.model.season_seasonal_order).fit(method=method)
        # Let's calculate the endogenous errors...
        
        trend_prediction = trend_sarimax.get_prediction().predicted_mean
        seasonality_prediction = season_sarimax.get_prediction().predicted_mean
        endog_errors = endog - trend_prediction - seasonality_prediction
        if self.model.fit_errors:
            error_sarimax = sm.tsa.statespace.SARIMAX(
                endog_errors, trend='n',
                missing='drop',
                order=self.model.error_order, seasonal_order=self.model.error_seasonal_order).fit(method=method)
        # Now let's calculate the errors
        return {
            'trend': trend_sarimax,
            'seasonality': season_sarimax,
            'errors': error_sarimax,
            'model': self
        }
        
    def decompose(self, series, periodicity, detrending_filter=(81, 3, 2), deseasoning_filter=(81 * 12, 3, 1), smoothing_trend_filter=(11, 3, 2)):
        """
        The decomposition for the series:
        Data = Seasonal Component + Trend Component + Errors
        
        The decomposition will follow the STL methodology.
        
        Inner Loop
            Step 1. Detrending.
            Step 2. Cycle Subseeries Smoothing.
            Step 3. Low Pass Filtering of Smoothed Cycle Subseeries.
            Step 4. Detrending of smoothed cycle subseries.
            Step 5. Deseasonalizing
            Step 6. Trend Smoothing
        
        Outer Loop:
            After doing all the above, we have the remainder.
            Calculate h = 6 * Median (Abs(Remainder))
            Then Robustness Weight at time point v is:
                Rho(v) = B(|R|/h)
            where
                B = (1 - u^2)^2 for 0<=u<1
                    0           otherwise
                    
        STL has 6 parameters:
            Number observations per cycle
            Number of passes through inner loop
            Number of robustness iterations in outer loop
            The smoothing parameter for the low-pass filter
            The smoothing parameter for the trend component
            The smoothing parameter for the seasonal component.
            
            The last one is the tough one, apparently. But with Savgol Filter It shouldn't be hard
        """
        deseasoning_savgol_window, deseasoning_savgol_poly, deseasoning_passes = deseasoning_filter
        deseasoning_savgol_window /= periodicity
        
        detrending_savgol_window, detrending_savgol_poly, detrending_passes = detrending_filter
        
        # Let's go now and build the decomposition algorithm
        # 1. Detrending
        trend_series = series
        for i in range(detrending_passes):
            trend_series = self.smoothing(trend_series,
                savgol_params=(detrending_savgol_window, detrending_savgol_poly))
        # 2. Smoothing Cycle Subseries
        smoothed_seasonality_series = series - trend_series
        for i in range(deseasoning_passes):
            smoothed_seasonality_series = self.smoothing(smoothed_seasonality_series,
                savgol_params=(deseasoning_savgol_window, deseasoning_savgol_poly))
        
        seasonality_series = series - trend_series - smoothed_seasonality_series
        
        trend_series = trend_series + smoothed_seasonality_series
        smoothing_trend_savgol_window, smoothing_trend_savgol_poly, smoothing_trend_passes = smoothing_trend_filter
        
        for i in range(smoothing_trend_passes):
            trend_series = self.smoothing(trend_series,
                savgol_params=(smoothing_trend_savgol_window, smoothing_trend_savgol_poly))
        
        error_series = series - trend_series - seasonality_series
        
        return trend_series, seasonality_series
        
    def smoothing(self, series, savgol_params=(81, 3)):
        """
        Just smooths series with savgol filter.
        """
        result = pd.Series(sc.signal.savgol_filter(series, *savgol_params), series.index)
        return result
