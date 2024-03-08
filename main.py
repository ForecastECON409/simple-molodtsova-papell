# %% [markdown]
"""
# Procedimiento para la evaluación del modelo simple de Papell
"""
# %% [markdown]
"""
**Objetivo**: Ajustar los hiperparámetros del modelo simple de Papell propuesto con el fin de maximizar la precisión del modelo.
"""
# %% [markdown]
"""
- Métrica de evaluación: AIC
- Procedimiento de evaluación: Validación cruzada en una ventana temporal móvil tamaño $h$.
"""
# %% [markdown]
"""
Para un conjunto de parámetros $(h, \lambda)$:

- Información de $t - h$ a $t$. Calculada con un procedimiento simple de *Nowcasting*.
    - Nowcasting $\mathbb{E}_{t-1}[\pi_{t}]$.
    - Nowcasting $\mathbb{E}_{t-1}[\tilde{\pi}_{t}]$.
    - Nowcasting $\mathbb{E}_{t-1}[y_{t}]$.
    - Nowcasting $\mathbb{E}_{t-2}[\tilde{y}_{t}]$.
"""
# %%
import matplotlib.pyplot as plt
import pandas as pd
import toolz as tz
import numpy as np
import statsmodels.tsa.api as tsa
# %%
def pi_nowcast_(pi: pd.Series, steps) -> pd.Series:
    # Order `pi` from the oldest to the most recent observation.
    pi = pi.sort_index()
    return pi.shift(steps)
# %%
def y_pot_drift_(y: pd.Series) -> float:
    y = tz.pipe(y,
        lambda x: x.sort_index(),
        lambda x: np.log(x),
    )
    
    drift = tz.pipe(y,
        lambda x: x.diff(),
        lambda x: x.mean()
    )
    return drift 
# %%
def y_gap_(y: pd.Series, h: int) -> pd.Series:
    gap = tz.pipe(y,
        lambda x: x.sort_index(),
        lambda x: np.log(x),
        lambda x: pd.DataFrame(x),
        lambda x: x.assign(drift=x.rolling(h).apply(y_pot_drift_)),
        lambda x: x.assign(gap=x.iloc[:, 0].diff() - x.drift),
        lambda x: x.gap
    )
    return gap
# %%
def gap_nowcast_(gap: pd.Series, steps=1) -> pd.Series:
    gap = gap.sort_index()
    
    ar1 = tsa.arima.ARIMA(gap.dropna(), order=(1, 0, 0)).fit()
    
    ar1_nowcast = ar1.forecast(steps)
    
    gap_nowcast = tz.pipe(gap,
        lambda x: x.combine_first(ar1_nowcast)
    )
    return gap_nowcast 
# %%
data = pd.read_csv('data/processed_data.csv', index_col=0, parse_dates=True)