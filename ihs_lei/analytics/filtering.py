import numpy as np
import scipy as sc
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import warnings
warnings.simplefilter('ignore')
from statsmodels.nonparametric.smoothers_lowess import lowess


class SeasonalAdjustment(object):
    """
    This class will adjust the dataseries seasonality.
    Expecting monthly data.
    Now I have the exact methodology.
    Steps:
    1. Find trend with Savitzky-Golay filter
    2. Find the seasonality (residuals) of trend with residuals = data - trend
    3. Fit residuals with residuals(t) = residuals(t-1) + residuals(t-12)
    4. Fit residuals of residuals with SARIMAX
    5. Return everything
    """
    def __init__(self, data_series):
        self._data_series = data_series
        if isinstance(data_series, pd.DataFrame):
            raise NotImplementedError

    def _get_trend(self, series, window_length=81, polyorder=3):
        """
        Smoothes out the timeseries and returns the trend
        """
        series_np = series.values.reshape((len(series.values),))
        result = sc.signal.savgol_filter(series_np, window_length, polyorder)
        result = pd.Series(result, index=series.index)
        return result

    def _fit_seasonality_and_errors(self, seasonality, order, seasonal_order):
        """
        Fits seasonality first with a RLM and then with a SARIMAX model
        """
        seasonality_raw = seasonality.values.reshape((len(seasonality), ))
        seasonality_shifted_12 = seasonality.shift(12).values.reshape((len(seasonality), ))
        seasonality_shifted_1 = seasonality.shift(1).values.reshape((len(seasonality), ))

        seasonality_dataset = pd.DataFrame({
            'y': seasonality_raw,
            'x_12': seasonality_shifted_12
        }, index=seasonality.index)
        seasonality_dataset = seasonality_dataset.dropna(axis=0)

        # First we run the regression as y(t) = a * y(t-12) + eps
        rlm_model = sm.RLM(seasonality_dataset['y'], seasonality_dataset[['x_12']])
        rlm_results = rlm_model.fit()

        # Now let's fit the residuals of the regression above with SARIMAX
        rlm_seasonality_results = pd.Series(rlm_results.fittedvalues, index=seasonality_dataset.index)

        residuals = seasonality - rlm_seasonality_results
        residuals = residuals.dropna()
        residuals, residuals_result, residuals_model = self._fit_residuals(residuals, order, seasonal_order)
        pre_residual_fit = (seasonality - rlm_seasonality_results).dropna()
        fitted_seasonality = rlm_seasonality_results + residuals
        post_residual_fit = (seasonality - fitted_seasonality).dropna()
        # Now let's return everything we need.
        seasonality_final_result = {
            'residuals_model': residuals_model,
            'residuals': residuals,
            'residuals_result': residuals_result,
            'fitted_seasonality': fitted_seasonality,
            'seasonality_model': rlm_model,
            'seasonality_result': rlm_results,
            'pre_residual_fit_sse': sum((pre_residual_fit)**2),
            'post_residual_fit_sse': sum((post_residual_fit)**2),
            'seasonality_rlm_dataset': seasonality_dataset,
            'seasonality_raw': seasonality
        }
        return seasonality_final_result

    def _fit_residuals(self, residuals, order, seasonal_order):
        """ Let's fit the residuals to the SARIMAX model now."""
        residuals_model = sm.tsa.statespace.SARIMAX(residuals.dropna(), order=order, seasonal_order=seasonal_order, simple_differencing=True)
        residuals_result = residuals_model.fit()
        residuals_summary = residuals_result.summary()
        residuals_no_na = residuals.dropna()
        new_residuals = residuals_no_na - residuals_result.fittedvalues
        return new_residuals, residuals_result, residuals_model

    def _get_sa_timeseries(self, fit):
        """
            - Y = Seasonal + Cyclical + Trend + Error
            - Transform into Y = Cyclical + Trend + Error
        """
        sa_timeseries = fit['actual_series'] - fit['seasonality_and_errors']['fitted_seasonality']
        return sa_timeseries

    def _get_cycles_timeseries(self, series, window_length=81*2 + 1, polyorder=1):
        """
        Now we get the cycle timeseries from the trend series.
        Note that we use the savgol filter again, but with a bigger window and a smaller polyorder
        """
        series_np = series.values.reshape((len(series.values),))
        # Here's the problem.
        result = sc.signal.savgol_filter(series_np, window_length, polyorder)
        result = pd.Series(result, index=series.index)

        cycle_trends = result
        cycles = series - result
        # Now let's normalize the cycle series
        # 36 months for expanding window
        cycles = cycles.dropna()
        cycles_normalized = (cycles - cycles.expanding(min_periods=36).mean())/(cycles.expanding(min_periods=36).std()).fillna(0.0)
        cycles_normalized = cycles_normalized.dropna()
        # So that mean is zero and goes from -1 to 1
        cycles_normalized = cycles_normalized.rank(pct=True) - np.mean(cycles_normalized.rank(pct=True))
        cycles_normalized = (1.0 - (max(cycles_normalized) - cycles_normalized)/(max(cycles_normalized) - min(cycles_normalized)))
        cycles_normalized = cycles_normalized * 2.0 - 1.0
        return cycle_trends, cycles_normalized


    def fit(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), window_length=81, polyorder=3, diff=False):
        """
        Does all the fitting of the model.
        """
        trend_first = self._get_trend(self._data_series, window_length=window_length, polyorder=polyorder)
        seasonality = self._data_series.values.reshape((len(self._data_series), )) - trend_first.values.reshape((len(self._data_series), ))
        seasonality = pd.Series(seasonality, index=self._data_series.index)
        seasonality = self._fit_seasonality_and_errors(seasonality, order, seasonal_order)

        # Now let's remove seasonality from initial series
        seasonality_free_series = self._data_series - seasonality['fitted_seasonality']
        # Now let's calculate trend/cycles again. Have to do this because all I did was remove the 'errors' from the series.
        trend = self._get_trend(seasonality_free_series, window_length=window_length, polyorder=polyorder)
        
        try:
            if diff:
                trend_cycles, cycles = self._get_cycles_timeseries(trend.diff(1), window_length=window_length * 2 + 1, polyorder=3)
            else:
                trend_cycles, cycles = self._get_cycles_timeseries(trend, window_length=window_length * 2 + 1, polyorder=3)
        except:
            trend_cycles = trend
            cycles = trend
        fit = {
            'seasonality_and_errors': seasonality,
            'trend': trend,
            'first_trend': trend_first,
            'actual_series': self._data_series,
            'seasonality_adjusted_ts': seasonality_free_series,
            'original': self._data_series,
            'cycles': cycles
        }
        seasonally_adjusted_timeseries = self._get_sa_timeseries(fit)
        fit['sa_timeseries'] = seasonally_adjusted_timeseries
        self._fit = fit
        return self._fit






