# %%
import matplotlib.pyplot as plt
import pandas as pd
import toolz as tz
import numpy as np
import statsmodels.tsa.api as tsa
# %%
def pi_nowcast_(pi: pd.Series, steps: int) -> pd.Series:
    """
    Generate a nowcast forecast for a given pd.Series over a specified number of steps
    using a random walk model. The random walk model is defined as follows:

    .. math::
        \pi_{t} = \pi_{t-1} + \varepsilon_{t}, \quad \varepsilon_{t} \sim N(0, \sigma^2)

    The forecast at time :math:`t` for :math:`t+h` is:

    .. math::
        \mathbb{E}_{t}[\pi_{t+h}] = \pi_{t}

    where :math:`h` is the number of steps to forecast.

    Parameters
    ----------
    pi : pd.Series
        A pandas Series object containing the data to forecast.
    steps : int
        The number of periods ahead to forecast.

    Returns
    -------
    pd.Series
        A pandas Series containing the nowcasted values, where each value is the forecast
        for the corresponding period ahead specified by `steps`. The Series is shifted `steps`
        periods, resulting in NaN values for the last `steps` entries.

    Notes
    -----
    The function assumes that the input Series `pi` is ordered from the oldest to the most
    recent observation. It performs a shift operation to simulate the forecasting process,
    where the forecast for any future period `t+h` is equal to the value at period `t`.
    """
    # Order `pi` from the oldest to the most recent observation.
    pi = pi.sort_index()
    return pi.shift(steps)

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
        lambda x: np.log(x),
        lambda x: pd.DataFrame(x),
        lambda x: x.assign(drift=y_pot_drift_(x.iloc[:,0])),
        lambda x: x.assign(gap=x.iloc[:, 0].diff() - x.drift),
        lambda x: x.gap
    )
    return gap
# %%
def gap_nowcast_(y: pd.Series, steps: int = 1) -> pd.Series:
    """
    Calculates the nowcast forecast for `steps` periods of the gap from a pd.Series,
    according to an AR(1) model fitted to the gap.

    Parameters
    ----------
    y : pd.Series
        A pandas Series object containing the data from which the gap is estimated.
    steps : int, optional
        The number of periods ahead for which the forecast is to be generated. The default is 1.

    Returns
    -------
    pd.Series
        A pandas Series containing the nowcasted gap values for the specified number of steps.
        This Series combines the original gap values and the forecasted values, with forecasted
        values filling in after the last observed gap.

    Notes
    -----
    This function first estimates the gap of the given series using an AR(1) model. Then, it performs
    forecasting for the specified number of steps ahead based on the fitted model. The nowcast combines
    the historical gap data with the forecasted values, providing a complete series that extends `steps`
    periods beyond the original data.
    """
    y = tz.pipe(y,
        lambda x: x.sort_index()
    )
    gap = y_gap_(y)
    ar1 = tsa.arima.ARIMA(gap.dropna(), order=(1, 0, 0)).fit()
    ar1_nowcast = ar1.forecast(steps)
    gap_nowcast = gap.combine_first(ar1_nowcast)
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
    return transformed_y