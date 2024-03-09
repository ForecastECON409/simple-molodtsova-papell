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
    Estimates the gap for a pd.Series based on the model where the gap is the difference between the first
    difference of the logarithm of the series and its drift, scaled by 100 for percentage representation.

    Parameters
    ----------
    y : pd.Series
        A pandas Series object containing the data from which the gap is to be estimated.

    Returns
    -------
    pd.Series
        A pandas Series containing the estimated gap for each period in the input series, multiplied by 100
        to convert the gap into a percentage form.

    Notes
    -----
    The gap calculation involves the following steps:
    1. Sorting the input series `y` by its index to ensure chronological order.
    2. Converting `y` to a DataFrame for easier manipulation.
    3. Calculating the logarithm of `y`, storing it as `log_y`.
    4. Calculating the drift of the stochastic trend in `y`, denoted as `alpha_0`, using a separate function.
    5. Computing the first difference of `log_y` to get the change in logarithmic values.
    6. Subtracting the drift from the first difference of `log_y` to estimate the gap.
    7. Multiplying the gap by 100 to represent it in percentage terms.

    This process yields the estimated gap as a measure of deviation from the expected change in the series,
    assuming an initial gap of zero. The result is a series of gap estimates for each time period in the input series.
    """
    gap = tz.pipe(y,
        lambda x: x.sort_index(),
        lambda x: pd.DataFrame(x),
        lambda x: x.assign(log_y=np.log(x.iloc[:,0])),
        lambda x: x.assign(drift=y_pot_drift_(x.iloc[:,0])),
        lambda x: x.assign(gap=x.log_y.diff() - x.drift),
        lambda x: x.gap
    )
    return gap * 100

# %%
def gap_nowcast_(y: pd.Series, steps: int = 1, freq = 'M') -> pd.Series:
    """
    Projects the gap of a pd.Series forward for a specified number of periods using the coefficient
    from an AR(1) model fitted to the existing gap data. The projection assumes that the gap's evolution
    follows an AR(1) process, using the AR(1) coefficient to estimate future gap values.

    Parameters
    ----------
    y : pd.Series
        The input time series data from which the gap is estimated.
    steps : int, optional
        The number of future periods to project the gap for. Defaults to 1.
    freq : str, optional
        The frequency of the time series data, used for creating a DateTimeIndex when fitting the AR(1) model.
        Defaults to 'M' (monthly).

    Returns
    -------
    pd.Series
        A series containing the projected gap values for the specified number of future periods. The projection
        is based on applying the AR(1) model's coefficient directly to the last observed gap values, effectively
        extending the gap series forward while assuming the dynamic captured by the AR(1) process remains constant.

    Notes
    -----
    This method of nowcasting is distinct from traditional forecasting approaches that may use the model to generate
    new future values. Instead, it leverages the AR(1) model's insight into the data's temporal correlation structure,
    specifically the autoregressive nature of the gap, to project existing patterns into the near future. This is
    particularly useful for nowcasting, where the goal is to estimate current or very near-term values with minimal
    delay and maximum reliance on the most recent observations.
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
    Applies a transformation to a pd.Series to compute the expected change in the logarithm of the series
    values for the next period (:math:`\Delta \log Y_{t+1}`), multiplied by 100 for percentage representation.

    Parameters
    ----------
    y : pd.Series
        The input time series data to be transformed.

    Returns
    -------
    pd.Series
        A pandas Series containing the transformed values, representing the percentage change in the logarithm
        of `y` for the next period. The transformation includes taking the natural logarithm of `y`, computing
        the first difference to find the logarithmic change, and shifting the result by one period to align
        with :math:`t+1`. The final values are scaled by 100.

    Notes
    -----
    This transformation is designed to facilitate the analysis of time series data by focusing on the percentage
    change in logarithmic terms, which is often more interpretable for economic and financial data. The shift
    operation prepares the data for forecasting or nowcasting applications by aligning the change with the future
    period it predicts. Multiplying by 100 converts the log difference to a percentage change, making the output
    more intuitive to interpret.
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
    Fits a Ridge regression model to the data, optionally applying bootstrap for error estimation,
    and calculates the Akaike Information Criterion (AIC) for model evaluation.

    Parameters
    ----------
    X : pd.DataFrame
        The explanatory variables.
    y : pd.Series
        The dependent variable.
    alpha : float
        The regularization strength for the Ridge regression model.
    bootstrap : bool, optional
        If True, applies bootstrap to estimate the error variance. Default is False.
    n_boot : int, optional
        The number of bootstrap samples to use if bootstrap is True. Default is 10,000.

    Returns
    -------
    Tuple[float, Pipeline]
        The AIC value of the fitted model and the fitted pipeline, which includes a standard scaler
        and a Ridge regression model with the specified alpha.

    Notes
    -----
    The Akaike Information Criterion (AIC) is computed to assess the quality of the model, factoring
    in both the goodness of fit and the model complexity. The AIC is calculated as 2k + n*log(sigma^2),
    where k is the number of model parameters, n is the number of observations, and sigma^2 is the variance
    of the residuals. If bootstrap is enabled, sigma^2 is estimated using the bootstrapped residuals, providing
    a potentially more robust measure of model performance by incorporating variability from resampling.
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
    Prepares and transforms data for fitting a regression model, specifically designed for the Papell model,
    by calculating the change in the logarithm of the dependent variable and applying specified lags to
    explanatory variables. This function facilitates the alignment of data within the appropriate temporal window
    for model fitting and is particularly useful before performing grid search optimization.

    Parameters
    ----------
    X : pd.DataFrame
        The explanatory variables as a pandas DataFrame. These are the independent variables that will be used
        in the model fitting process.
    y : pd.Series
        The dependent variable as a pandas Series, typically representing an exchange rate or a similar financial
        time series that the model aims to predict.
    lags : dict, optional
        A dictionary specifying the lags to apply to the explanatory variables. Each key is a column name from `X`,
        and each value is the number of periods to lag that variable by. Defaults to an empty dictionary, indicating
        no lags will be applied unless specified.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prepared data for model fitting. This includes the calculated change in the logarithm
        of `y`, shifted by one period to reflect future change (:math:`\Delta \log Y_{t+1}`), and any lagged versions of
        the explanatory variables specified in the `lags` dictionary. The DataFrame is trimmed to ensure that no rows contain
        NaN values, which might result from the lagging and differencing operations, thus aligning the dataset within the
        correct temporal window for subsequent model fitting.

    Notes
    -----
    This function is a critical step in the data preparation process for time series modeling, particularly when the model
    incorporates lags of both the dependent and independent variables. It automates the process of calculating the expected
    future change in the dependent variable and adjusting the explanatory variables according to specified lags, thereby
    creating a dataset that is ready for rigorous analysis and model fitting.
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
    Processes and nowcasts a given DataFrame using specified lags and corresponding nowcasting functions. This method
    combines data preprocessing, including lagging of certain variables, with nowcasting to prepare the dataset for
    further econometric analysis or forecasting applications.

    Parameters
    ----------
    X : pd.DataFrame
        Independent variables in a DataFrame format, which will undergo preprocessing and nowcasting.
    y : pd.DataFrame
        Dependent variable(s) in a DataFrame format, used in the preprocessing step to ensure alignment and completeness of the dataset.
    lags : dict
        Specifies the variables for lagging and their corresponding nowcasting functions. Each entry in the dictionary
        should have a variable name from `X` as a key, and its value should be another dictionary with 'lag' indicating
        the number of periods to lag and 'nowcast' specifying the function for nowcasting that variable.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame containing the original and nowcasted variables, ready for analysis or modeling.
        The output DataFrame excludes any originally lagged variables, replacing them with their nowcasted versions,
        and ensures all rows with incomplete data due to lagging or nowcasting are removed.

    Notes
    -----
    The nowcasting_ function facilitates the integration of preprocessing steps, such as lagging, with the application
    of nowcasting techniques to selected variables. This streamlined approach allows for efficient preparation of
    time-series data, catering to specific modeling needs or forecasting objectives. The function's design supports
    flexible implementation of nowcasting methods, enabling its adaptation to various analytical frameworks and data
    specifications.
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
    Performs a rolling fit of a model over a dataset with optional out-of-sample forecasting and bootstrap error analysis.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset for model fitting, where the first column is the dependent variable,
        and the remaining columns are independent variables.
    alpha : float
        The regularization strength for the Ridge regression model.
    h : int
        The window size for the rolling fit.
    outsample : bool, optional
        If True, performs out-of-sample forecasting one period ahead after each fit. Default is False.
    bootstrap : bool, optional
        If True, performs bootstrap error analysis on the forecasting error. Default is False.
    n_boot : int, optional
        The number of bootstrap samples for the error analysis if bootstrap is True. Default is 10,000.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing results from the rolling fits, including AIC values, fitted model pipelines,
        and, if outsample is True, out-of-sample forecast errors. If bootstrap is True and outsample is True,
        it also returns bootstrap error analysis.

    Notes
    -----
    This function provides a framework for assessing model performance over time using a rolling window approach,
    with the flexibility to incorporate out-of-sample forecasting and robust error analysis through bootstrapping.
    It is particularly useful for time-series analysis where model stability and forecasting accuracy are of interest.
    The returned DataFrame offers a detailed account of model fits, allowing for further analysis of model behavior
    and prediction accuracy across different time periods.
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
    Aggregates the Akaike Information Criterion (AIC) values obtained from a series of rolling window fits of a model
    across a dataset, optionally incorporating bootstrap methodology for error variance estimation.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset on which the model is fitted, with the first column as the dependent variable and
        the subsequent columns as independent variables.
    alpha : float
        The regularization strength for the model.
    h : int
        The size of the rolling window for model fitting.
    agg : function
        The aggregation function (e.g., np.mean, np.median) applied to the series of AIC values to obtain a single
        summary statistic.
    bootstrap : bool, optional
        Indicates whether bootstrap methodology should be applied to estimate the error variance, affecting the
        calculation of AIC values. Default is False.
    n_boot : int, optional
        The number of bootstrap samples to use if bootstrap is enabled. Default is 10,000.

    Returns
    -------
    float
        The aggregated AIC value representing the overall quality of the model over the dataset, accounting for
        both model fit and complexity, and adjusted for bootstrap analysis if specified.

    Notes
    -----
    This function leverages a rolling window approach to fit a model repeatedly over the dataset, each time computing
    an AIC value to evaluate model performance. The aggregation of these AIC values provides a comprehensive measure
    of the model's quality across the dataset. When bootstrap is enabled, it introduces a robustness check by re-sampling
    the residuals to estimate error variance, potentially leading to a more accurate assessment of model performance.
    """
    if bootstrap:
        aic_values, _ = rolling_fit_model_(data, alpha, h, bootstrap=bootstrap, n_boot=n_boot)
        aic_values = aic_values['aic']
    else:
        aic_values = rolling_fit_model_(data, alpha, h, bootstrap=bootstrap, n_boot=n_boot)['aic']
    aggregated_aic = agg(aic_values.dropna())  # Ensure NaN values are excluded from aggregation
    return aggregated_aic
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
    Calculates the aggregated Mean Squared Forecast Error (MSFE) over a rolling window fit of a model
    to a dataset, using specified alpha for regularization, rolling window size, aggregation function,
    and optionally conducts bootstrap error analysis.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset for model fitting, where the first column is the dependent variable,
        and the remaining columns are independent variables.
    alpha : float
        The regularization strength for the model fitting process.
    h : int
        The size of the rolling window over which the model is fitted.
    agg : function
        An aggregation function, such as np.mean or np.median, to apply to the MSFE values
        obtained from the rolling window fits.
    bootstrap : bool, optional
        If True, performs bootstrap error analysis on the forecast errors. Defaults to False.
    n_boot : int, optional
        The number of bootstrap samples to use for error analysis if bootstrap is True. Defaults to 10,000.

    Returns
    -------
    float
        The aggregated MSFE value, calculated using the specified aggregation function across all rolling
        window fits, adjusted for bootstrap analysis if specified.

    Notes
    -----
    This function performs a rolling fit of a model over the dataset for each window of size `h`,
    optionally performing out-of-sample forecasting if `outsample` is True during the rolling fit process.
    It computes the Mean Squared Forecast Error (MSFE) for each forecast to assess the model's predictive accuracy.
    If `bootstrap` is True, it also performs a bootstrap error analysis on the forecast errors to provide
    a more robust measure of forecasting accuracy. The aggregated MSFE provides a summary measure of the
    model's forecasting performance over the dataset, considering both the fit quality and complexity of
    the model, as well as adjusting for changes in data over time.
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