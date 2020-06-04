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
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# Now let's find the perfect fit for the series.
def _optimization_function(reference_series, indicator_series, cutoff_above, cutoff_below):
    """
    This function will calculate the loss between the reference series and indicator series.
    Expecting pandas series.
    """
    min_valid_index = max(reference_series.index[0], indicator_series.index[0])
    max_valid_index = min(reference_series.index[-1], indicator_series.index[-1])

    reference_base = reference_series.where((reference_series < -0.85) | (reference_series > 0.85))
    indicator_base = indicator_series.where((indicator_series < -0.85) | (indicator_series > 0.85))

    reference_base = reference_base.where((reference_base.index > min_valid_index) & (reference_base.index < max_valid_index))
    indicator_base = indicator_base.where((indicator_base.index > min_valid_index) & (indicator_base.index < max_valid_index))

    # Just dropping nans before reindex
    reference_base = reference_base.dropna()
    indicator_base = indicator_base.dropna()

    # Now let's reindex both and set nans to max error (2)
    index_common = sorted(set(indicator_base.index).union(set(reference_base.index)))
    reference_base = reference_base.reindex(index_common)
    indicator_base = indicator_base.reindex(index_common)

    errors = sum((reference_base - indicator_base).apply(np.isnan))
    # Let's square it too
    return -np.square(errors)


class LeadingIndicatorAnalysis(object):
    """This object analyzes indicators against a reference and sees if it leads or lags.
    It basically does Series Evaluation:
        - Length of the lead
        - Cyclical conformity
        - Extra/missing cycles
        - Performance
    and returns the lag when fitted.
    """
    def __init__(self, reference_series, indicator_series):
        self.reference_series = reference_series
        self.indicator_series = indicator_series


    def fit(self, min_lead=2, max_lead=24, cutoff_above=0.75,
            cutoff_below=-0.75, expected_sign='+'):
        """
        This function fits the indicator to the reference with both cointegration and correlation.
        Picks the one which has better lead/lag and fit combination.
        Minimum lead is where we start to look, max_lead is where we end.
        I'll be minimizing the errors between peaks/throughs.
        That is, pick only dates that are above 0.75 and below -0.75
        Focus on making distance = 0 while changing the shift
        """
        lead_errors = []
        for lead in range(max_lead):
            if lead < min_lead:
                continue
            reference_series = self.reference_series.shift(-lead).dropna()
            indicator_series = self.indicator_series

            lead_errors.append({
                'error': _optimization_function(reference_series, indicator_series, cutoff_above, cutoff_below),
                'lead': lead
            })

        lead_errors = sorted(lead_errors, key=lambda x: -x['error'])
        best_lead = lead_errors[0]['lead']
        performance_data = self._performance(best_lead)
        reasoning = ''
        if expected_sign is '+':
            if performance_data['ols_model'].params[0] > 0:
                valid_model = True
            else:
                valid_model = False
        else:
            if performance_data['ols_model'].params[0] < 0:
                valid_model = True
            else:
                valid_model = False
        # Now let's see statistical validity of model
        # Looking for p-value < 0.01
        if not valid_model:
            reasoning = "Invalid model. Reason: Wrong Expected Sign\n"
        if performance_data['ols_model'].pvalues[0] > 0.001:
            valid_model = False
            reasoning += "Invalid model. Reason: Pvalues too big."

        res = {
            'fit': performance_data,
            'lead': best_lead,
            'valid_model': valid_model,
            'expected_sign': expected_sign,
            'analysis_model': self,
            'reasoning': reasoning
        }
        return res

    def _performance(self, lead):
        """
        Runs an ols and a generalized linear regression on indicator against the reference series to figure out how good it actually is.
        """
        y = self.reference_series.shift(-lead).dropna().copy()
        x = self.indicator_series.copy()

        index_common = sorted(set(y.index).union(set(x.index)))
        y_regression = y.reindex(index_common).fillna(0.0)
        x_regression = x.reindex(index_common).fillna(0.0)

        ols_model = sm.OLS(y_regression, x_regression).fit()
        rlm_model = sm.RLM(y_regression, x_regression).fit()
        ols_summary = ols_model.summary()
        rlm_summary = rlm_model.summary()
        x = x.reindex(index_common).dropna()
        y = y.reindex(index_common).dropna()
        return {
            'ols_model': ols_model,
            'rlm_model': rlm_model,
            'ols_summary': ols_summary,
            'rlm_summary': rlm_summary,
            'x': x,
            'y': y
        }








