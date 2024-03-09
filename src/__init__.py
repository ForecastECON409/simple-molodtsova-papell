# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import toolz as tz
import numpy as np
import statsmodels.tsa.api as tsa

from typing import Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from typing import Tuple

from icecream import ic
# %%
def pi_nowcast_(pi: pd.Series, steps: int) -> pd.Series:
    """
    Simulates a nowcast for a given pd.Series over a specified number of steps using a naive forecast approach,
    which assumes that the most recent observation will persist unchanged for the forecast horizon. This method
    effectively treats the series as if it follows a random walk model without drift, where the value at any future
    point is assumed to be equal to the last observed value.

    Parameters
    ----------
    pi : pd.Series
        A pandas Series object containing the data to be forecasted.
    steps : int
        The number of periods ahead for which the forecast is generated.

    Returns
    -------
    pd.Series
        A pandas Series where each value from the end of the series is projected forward for `steps` periods,
        creating a flat forecast. This method returns the original series unchanged, as it simulates a naive
        forecast approach without actually modifying the series based on the `steps` parameter.

    Notes
    -----
    Unlike typical forecasting models that adjust future values based on patterns observed in the data, this function assumes no change from the last observed value for the duration of the forecast period.
    The approach is a simplified version of forecasting, often used as a benchmark or placeholder in the absence of more complex models.
    """
    # Ensure the series is sorted by index
    pi_sorted = pi.sort_index()

    # This implementation does not alter the series; returns as-is for demonstration
    # In practice, you might extend the series based on `steps` with the last observed value
    return pi_sorted


# %%
def y_pot_drift_(y: pd.Series) -> pd.Series:
    """
    Calculates the drift of a stochastic trend from a pd.Series according to the following model:

    .. math::
        \mathbb{E}[\Delta \log Y_{t}] = \alpha_0

    Where :math:`\alpha_0` is the drift of the stochastic trend.

    Parameters
    ----------
    y : pd.Series
        A pandas Series object representing the time series data from which the drift of
        the stochastic trend is to be calculated.

    Returns
    -------
    pd.Series
        A pandas Series containing the calculated drift value, :math:`\alpha_0`, repeated
        across the entire index of the input Series `y`. The drift is calculated as the mean
        of the first differences of the logarithm of `y`.

    Notes
    -----
    The function first sorts the input Series `y` by its index to ensure chronological order.
    It then computes the logarithm of `y`, takes the first difference to obtain the growth
    rates, and finally calculates the mean of these differences to estimate the drift,
    :math:`\alpha_0`, of the stochastic trend.
    """
    drift = tz.pipe(y,
        lambda x: x.sort_index(),
        lambda x: np.log(x),
        lambda x: x.diff(),
        lambda x: x.mean()
    )
    drift = pd.Series(drift, index=y.index)
    return drift

# %%
def y_gap_(y: pd.Series) -> pd.Series:
    """
    Estimates the gap of a pd.Series according to the following model:

    .. math::
        y_t = \Delta \log Y_t - \text{\alpha}_0

    This model assumes that :math:`y_{t-1} = 0` and that :math:`\Delta \bar{y}_t = \alpha_0`.

    Parameters
    ----------
    y : pd.Series
        A pandas Series object containing the data from which the gap is estimated.

    Returns
    -------
    pd.Series
        A pandas Series containing the estimated gap for each period in the input series.

    Notes
    -----
    The gap is calculated as the difference between the first difference of the logarithm of the series
    and the drift (:math:`\alpha_0`). This reflects the deviation of the actual change in the series from
    the expected change, assuming the initial gap is zero.
    """
    gap = tz.pipe(y,
        lambda x: x.sort_index(),
        lambda x: pd.DataFrame(x),
        lambda x: x.assign(log_y=np.log(x.iloc[:,0])),
        lambda x: x.assign(drift=y_pot_drift_(x.iloc[:,0])),
        lambda x: x.assign(gap=x.log_y.diff() - x.drift),
        lambda x: x.gap
    )
    #gap = tz.pipe(y,
        #lambda x: x.sort_index(),
        #lambda x: pd.DataFrame(x),
        #lambda x: x.assign(log_y=np.log(x.iloc[:,0])),
        #lambda x: x.assign(drift=y_pot_drift_(x.iloc[:,0])),
        #lambda x: x.assign(cum_drift=x.drift.cumsum()),
        #lambda x: x.assign(y_pot=x.log_y.iloc[0] + x.cum_drift),
        #lambda x: x.assign(gap=x.log_y - x.y_pot),
        #lambda x: x.gap * 100
    #)
    return gap * 100
# %%
def gap_nowcast_(y: pd.Series, steps: int = 1, freq = 'M') -> pd.Series:
    """
    Calculates the nowcast for `steps` periods of the gap from a pd.Series,
    according to an AR(1) model's coefficient applied directly to the gap.

    Parameters
    ----------
    y : pd.Series
        A pandas Series object containing the data from which the gap is to be estimated.
    steps : int, optional
        The number of periods ahead for which the nowcast is generated. The default is 1.

    Returns
    -------
    pd.Series
        A pandas Series containing the nowcasted gap values for the specified number of steps.
        This approach applies the AR(1) model's coefficient in a direct multiplication to estimate
        the future gap values, with the assumption that the gap persists according to the AR(1)
        process over the specified number of steps.

    Notes
    -----
    Unlike traditional forecasting methods that predict future values based on a model fit,
    this function calculates the nowcast by applying the AR(1) model's coefficient to the
    last observed gap. This method assumes that the relationship captured by the AR(1) coefficient
    continues to hold over the forecast period. The gap is first estimated from the series, and
    then the AR(1) coefficient is used to project this gap forward for the specified number of steps.
    """
    y = y.sort_index()
    gap = y_gap_(y)
    
    gap_ts = gap.copy().dropna()
    gap_ts.index = pd.DatetimeIndex(gap_ts.index, freq=freq)
    ar1 = tsa.arima.ARIMA(gap_ts, order=(1, 0, 0)).fit()
    ar1_coef = ar1.params.iloc[1]
    gap_nowcast = pd.Series(
        np.vectorize(lambda x: ar1_coef**steps * x)(gap),
        index=gap.index
    )
    return gap_nowcast

# %%
def y_transform_(y: pd.Series) -> pd.Series:
    """
    Transforms the pd.Series `y` into :math:`t` to generate:

    .. math::
        \Delta \log Y_{t+1}

    Parameters
    ----------
    y : pd.Series
        A pandas Series object containing the data to be transformed.

    Returns
    -------
    pd.Series
        A pandas Series containing the transformed values, specifically the difference of the logarithm
        of `y` shifted by one period to represent :math:`\Delta \log Y_{t+1}`.

    Notes
    -----
    This function applies a log transformation to the series, computes the first difference to capture
    the change in the log values, and then shifts the resulting series by one period to align the change
    to period :math:`t+1`. This transformation is useful for modeling changes in the logarithmic scale of
    the data over time.
    """
    y = y.sort_index()  # Ensure the series is sorted by index before transformation
    transformed_y = tz.pipe(y,
        lambda x: np.log(x),
        lambda x: x.diff(),
        lambda x: x.shift(-1),
    )
    return transformed_y * 100
# %%
def fit_model_(X: pd.DataFrame, y: pd.Series, alpha: float, bootstrap:False, n_boot=10_000) -> Tuple[float, Pipeline]:
    """
    Fits the simple Papell model to the data and returns the Akaike Information Criterion (AIC) value
    along with the fitted pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        The explanatory variables as a pandas DataFrame.
    y : pd.Series
        The dependent variable as a pandas Series.
    alpha : float
        The regularization strength to be used by the Ridge regression model.

    Returns
    -------
    Tuple[float, Pipeline]
        A tuple containing the AIC value of the fitted model and the fitted Pipeline object.
        The Pipeline consists of a standard scaler followed by a Ridge regression model with the specified alpha.

    Notes
    -----
    The Akaike Information Criterion (AIC) is calculated using the formula:
    
    .. math::
        \text{AIC} = 2k + n\log(\hat{\sigma}^2)
    
    where :math:`k` is the number of estimated parameters in the model, :math:`n` is the number of observations,
    and :math:`\hat{\sigma}^2` is the estimated variance of the residuals. This criterion provides a measure of
    the quality of the model, taking into account the goodness of fit and the complexity of the model.
    """
    
    papell_pipe = Pipeline([ 
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    papell_pipe.fit(X, y)
    
    n_obs = len(y)
    k = len(papell_pipe.named_steps['ridge'].coef_)
    y_fitted = papell_pipe.predict(X)
    residuals = y - y_fitted
    sigma_squared = (residuals ** 2).mean()
    
    if bootstrap:
        residuals = np.random.choice(
            residuals.values,
            size=(n_boot, 1), 
            replace=True
        )
        sigma_squared = np.mean(residuals ** 2)
        aic = (2 * k) + np.log(sigma_squared)
    else:
        aic = (2 * k) + (n_obs * np.log(sigma_squared))
    
    return aic, papell_pipe
# %%
def preprocess_data_(X: pd.DataFrame, y: pd.Series, lags: dict = {}) -> pd.DataFrame:
    """
    Prepares the data for fitting the Papell model. This function is responsible for calculating the 
    Change in Logarithm of the exchange rate in `y` and returns a pd.DataFrame with the data in the 
    correct temporal window for model fitting. This function is designed as a preparatory step before 
    performing grid search.

    Parameters
    ----------
    X : pd.DataFrame
        A pandas DataFrame containing the explanatory variables.
    y : pd.Series
        A pandas Series containing the dependent variable, typically an exchange rate.
    lags : dict, optional
        A dictionary where keys are column names in `X` and values are the lag periods to apply. 
        Default is an empty dictionary.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the prepared data for model fitting. This includes the Change in 
        Logarithm of `y` shifted by one period and any specified lags for variables in `X`. The resulting 
        DataFrame is in the correct temporal window, with any rows containing NaN values due to lagging or 
        differencing removed.

    Notes
    -----
    This preprocessing includes calculating the Change in Logarithm of `y` for one period ahead, applying 
    lagged transformations based on the `lags` dictionary to the explanatory variables, and ensuring the 
    data is aligned in the correct temporal window by removing any resulting NaN values.
    """
    # Concatenate y and X, ensuring data is sorted by index
    data = pd.concat([y, X], axis=1).sort_index()
    original_columns = data.columns
    
    # Calculate Change in Logarithm of y for one period ahead
    data['Delta_log_S(+1)'] = y_transform_(data.iloc[:, 0]).shift(-1)
    
    # Apply specified lags to variables
    for key, value in lags.items():
        data[f"{key}({value['lag']})"] = data[key].shift(value['lag'])
        
    # Remove NaN values and drop original columns to retain only transformed/prepared data
    data = data.dropna().drop(columns=original_columns)

    return data
# %%
def nowcasting_(X: pd.DataFrame, y: pd.DataFrame, lags: dict) -> pd.DataFrame:
    """
    Applies nowcasting to a DataFrame `X` based on specified lag values and nowcasting functions provided in `lags`.
    This function preprocesses the data, applies nowcasting to the specified lagged variables, and returns a DataFrame
    with the nowcasted variables, ready for further analysis or modeling.

    Parameters
    ----------
    X : pd.DataFrame
        A pandas DataFrame containing the independent variables to be nowcasted.
    y : pd.DataFrame
        A pandas DataFrame containing the dependent variable(s) used in preprocessing data.
    lags : dict
        A dictionary specifying the variables to lag and the nowcasting function to apply. The keys should be
        the column names in `X` that need to be lagged, and the values should be dictionaries containing two keys:
        'lag' for the number of periods to lag the variable, and 'nowcast' for the function to apply for nowcasting.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the preprocessed and nowcasted independent variables. Original lagged variables
        are replaced with their nowcasted versions, and the DataFrame is cleaned of any NaN values resulting from
        the nowcasting process.

    Notes
    -----
    The `nowcasting_` function is designed to integrate preprocessing and nowcasting in a single step, streamlining
    the preparation of time series data for econometric modeling or forecasting. This function assumes that the
    preprocessing and nowcasting functions are defined externally and passed as arguments, allowing for flexible
    adaptation to different nowcasting methodologies.
    """
    X_pp = preprocess_data_(X, y, lags=lags)
    
    for key, value in lags.items():
        tmp_original_name = f"{key}({value['lag']})"
        tmp_new_name = f"{tmp_original_name}_nowcast"
        
        # Apply nowcasting function to the specified lagged variable
        X_pp[tmp_new_name] = value['nowcast'](X_pp[tmp_original_name], steps=value['lag'])
        # Drop the original lagged variable column
        X_pp = X_pp.drop(columns=tmp_original_name)
    
    # Drop any rows with NaN values that may result from the nowcasting process
    X_pp = X_pp.dropna()
    
    return X_pp
# %%
def rolling_fit_model_(data: pd.DataFrame, alpha: float, h: int, outsample=False, bootstrap=False, n_boot=10_000) -> pd.DataFrame:
    """
    Performs a rolling fit of a model over a dataset, using a specified alpha for regularization,
    a rolling window of size h, and optionally conducts out-of-sample forecasting.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to fit the model to, where the first column is the dependent variable
        and the remaining columns are independent variables.
    alpha : float
        The regularization strength used in the model fitting process.
    h : int
        The size of the rolling window over which to fit the model.
    outsample : bool, optional
        If True, the model performs out-of-sample forecasting for 1 period ahead after each fit.
        The default is False, indicating in-sample fitting without out-of-sample forecasting.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the rolling fits, including the Akaike Information Criterion (AIC) values,
        the fitted pipelines for each window, and optionally, the out-of-sample forecast errors if outsample is True.

    Notes
    -----
    This function iterates over the dataset, fitting a model to each rolling window of size `h` and optionally
    performing an out-of-sample forecast for the next period. It computes the Akaike Information Criterion (AIC)
    for each fit to assess model performance. For out-of-sample forecasting, it also computes the forecast error
    for the one period ahead forecast. This approach allows for assessing the model's predictive accuracy and
    performance over time, adjusting for changes in the data. The function returns a DataFrame with the AIC values,
    fitted model pipelines, and forecast errors (if applicable), providing a comprehensive overview of the model's
    rolling performance.
    """
    if outsample:
        h_estimation = h
        h_forecast = 1
    else:
        h_estimation = h
        h_forecast = 0

    def data_separator_(x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return x.iloc[:, 1:], x.iloc[:, 0]

    fit_model = lambda x: fit_model_(*data_separator_(x), alpha=alpha, bootstrap=bootstrap, n_boot=n_boot)

    results = {'aic': [], 'pipe': [], 'error': []}
    dates_results = []
    for i in range(len(data) - h_forecast):
        if (i+1) < (h_estimation + h_forecast):
            results['aic'].append(np.nan)
            results['pipe'].append(None)
            results['error'].append(np.nan)
            dates_results.append(data.index[i])
        else:
            w_forecast = (i+1) + h_forecast
            w_f = w_forecast - h_forecast
            w_0 = w_f - h_estimation
            
            tmp_aic, tmp_pipe = fit_model(data.iloc[w_0:w_f, :])
            if outsample:
                tmp_forecast = tmp_pipe.predict(data.iloc[w_forecast-1:w_forecast, 1:])
                tmp_error = data.iloc[w_forecast-1:w_forecast, 0].values - tmp_forecast
                results['error'].append(tmp_error[0])
            else:
                results['error'].append(np.nan)
            
            results['aic'].append(tmp_aic)
            results['pipe'].append(tmp_pipe)
            dates_results.append(data.index[i])
    
        
    if bootstrap and any(pd.Series(results['error']).notna()):
        # Bootstrap de `results['error']`
        error_boostrap = np.random.choice(
            pd.Series(results['error']).dropna().values,
            size=(n_boot, 1), 
            replace=True
        )
        return pd.DataFrame(results, index=dates_results), error_boostrap
    elif bootstrap and all(pd.Series(results['error']).notna()):
        results = pd.DataFrame(results, index=dates_results)
        results = results.drop(columns='error')
        return results, None
                                                  
    return pd.DataFrame(results, index=dates_results)

# %%
def rolling_aic_model_(data: pd.DataFrame, alpha: float, h: int, agg=np.mean, bootstrap=False, n_boot=10_000) -> float:
    """
    Calculates the aggregated Akaike Information Criterion (AIC) over a rolling window fit of a model
    to a dataset, using specified alpha for regularization, rolling window size, and aggregation function.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to fit the model to, where the first column is expected to be the dependent variable,
        and the remaining columns are independent variables.
    alpha : float
        The regularization strength used in the model fitting process.
    h : int
        The size of the rolling window over which the model is fitted.
    agg : function, optional
        An aggregation function (e.g., np.mean, np.median) to apply to the AIC values obtained from the
        rolling window fits. The default is np.mean.

    Returns
    -------
    float
        The aggregated AIC value, calculated as specified by the `agg` function, across all rolling window fits.

    Notes
    -----
    This function performs a rolling fit of a model over the provided dataset for each window of size `h`
    and computes the AIC for each fit. It then aggregates these AIC values using the specified aggregation
    function `agg` (e.g., mean, median). This approach provides a summary measure of the model's performance
    over the dataset, taking into account both the goodness of fit and the complexity of the model, while
    adjusting for changes in data over time.
    """
    if bootstrap:
        aic_values, _ = rolling_fit_model_(data, alpha, h, bootstrap=bootstrap, n_boot=n_boot)
        aic_values = aic_values['aic']
    else:
        aic_values = rolling_fit_model_(data, alpha, h, bootstrap=bootstrap, n_boot=n_boot)['aic']
    aggregated_aic = agg(aic_values.dropna())  # Ensure NaN values are excluded from aggregation
    return aggregated_aic
# %%
# %%
# %%
def rolling_msfe_model_(
        data: pd.DataFrame,
        alpha: float,
        h: int, 
        agg=np.mean,
        bootstrap=False,
        n_boot=10_000
    ) -> float:
    """
    Calculates the aggregated Akaike Information Criterion (AIC) over a rolling window fit of a model
    to a dataset, using specified alpha for regularization, rolling window size, and aggregation function.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to fit the model to, where the first column is expected to be the dependent variable,
        and the remaining columns are independent variables.
    alpha : float
        The regularization strength used in the model fitting process.
    h : int
        The size of the rolling window over which the model is fitted.
    agg : function, optional
        An aggregation function (e.g., np.mean, np.median) to apply to the AIC values obtained from the
        rolling window fits. The default is np.mean.

    Returns
    -------
    float
        The aggregated AIC value, calculated as specified by the `agg` function, across all rolling window fits.

    Notes
    -----
    This function performs a rolling fit of a model over the provided dataset for each window of size `h`
    and computes the AIC for each fit. It then aggregates these AIC values using the specified aggregation
    function `agg` (e.g., mean, median). This approach provides a summary measure of the model's performance
    over the dataset, taking into account both the goodness of fit and the complexity of the model, while
    adjusting for changes in data over time.
    """
    if bootstrap:
        _, msfe_values = rolling_fit_model_(data, alpha, h, outsample=True, bootstrap=bootstrap, n_boot=n_boot)
    else:
        msfe_values = rolling_fit_model_(data, alpha, h, outsample=True, bootstrap=bootstrap, n_boot=n_boot)['error']
        msfe_values = msfe_values.dropna()
    
    #if bootstrap:
        #msfe_values = np.array(msfe_values)
        #msfe_values = np.random.choice(
            #msfe_values,
            #size=(n_boot, len(msfe_values)), 
            #replace=True
        #)
        ##msfe_values = np.mean(msfe_values, axis=1)
    
    aggregated_msfe = agg(msfe_values)  # Ensure NaN values are excluded from aggregation
    return aggregated_msfe