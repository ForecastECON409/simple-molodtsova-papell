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
def pi_nowcast_(pi):
    # Order `pi` from the oldest to the most recent observation.
    pi = pi.sort_index()
    return pi.shift(1)
# %%
def pi_tilde_nowcast_(pi_tilde):
    # Order `pi` from the oldest to the most recent observation.
    pi_tilde = pi_tilde.sort_index()
    return pi_tilde.shift(1)
# %%
def y_pot_drift_(y, h):
    y = y.sort_index()
    pass