---
marp: true
class: invert
---

# ECON 409 Project Proposal

*Idalee Vargas, Lora Yovcheva, Mauricio Vargas, Santiago Naranjo*

---
## Model

As suggested by Molodtsova and Papell, the model is outlined as follows:

$$
\Delta s_{t+1} = \beta_0 + 
    \beta_1(\tilde{\pi}_t - \pi_t) +
    \beta_2(\tilde{y}_t - y_t) +
    \epsilon_t, 
    \quad
    \epsilon_t \sim_{i.i.d.} N(0, \sigma^2)
$$
$$
\Delta s_{t+1} = s_{t+1} - s_t
$$
$$
s_t = \ln{S_t}
$$

This model is delineated as symmetric, non-smoothed, and homogeneous, with:

- $S_t$ representing the price of 1 pound sterling in US dollars.
- $\tilde{\pi}_t$ being the UK inflation rate.
- $\pi_t$ being the US inflation rate.
- $\tilde{y}_t$ indicating the UK output gap.
- $y_t$ indicating the US output gap.

---

The output gaps are determined through:

$$
\ln Y_t = y_t + \bar{y}_t
$$
$$ 
\bar{y}_t = \alpha_0 + \bar{y}_{t-1} + \nu_t,
\quad
\nu_t \sim_{i.i.d.} N(0, \sigma^2_{\nu})
$$
$$
y_t = \alpha_1 y_{t-1} + \eta_t,
\quad
\eta_t \sim_{i.i.d.} N(0, \sigma^2_{\eta})
$$

Introducing a stochastic trend for potential output and an autoregressive model for the output gap. It assumes that $\ln Y_t$ and $\bar{y}_t$ are I(1) variables, and $y_t$ is I(0).

---

For the UK output gap, we have:

$$
\ln \tilde{Y}_t = \tilde{y}_t + \bar{\tilde{y}}_t
$$
$$ 
\bar{\tilde{y}}_t = \gamma_0 + \bar{\tilde{y}}_{t-1} + \tilde{\nu}_t,
\quad
\tilde{\nu}_t \sim_{i.i.d.} N(0, \sigma^2_{\tilde{\nu}})
$$
$$
\tilde{y}_t = \gamma_1 \tilde{y}_{t-1} + \tilde{\eta}_t, 
\quad
\tilde{\eta}_t \sim_{i.i.d.} N(0, \sigma^2_{\tilde{\eta}})
$$

This approach assumes $\ln \tilde{Y}_t$ and $\bar{\tilde{y}}_t$ are I(1) variables and $\tilde{y}_t$ is I(0).

---

## Data

The model utilizes monthly data, with updates occurring on the last business day of each month:

- $\pi_t$: US CPI index, released with a one-month delay.
- $\tilde{\pi}_t$: UK CPI index, released with a one-month delay.
- $Y_t$: US Industrial Production Index, released with a one-month delay.
- $\tilde{Y}_t$: UK Industrial Production Index, released with a two-month delay.

---

The publication lag for these variables does not hinder the strategy's application, as the prediction for $\Delta s_{t+1}$ relies on data available up to month $t$.

---

Given the two-month delay in publishing $\tilde{Y}_t$, we propose an adjustment to the model:

$$
\Delta s_{t+1} = \beta_0 + 
    \beta_1(\tilde{\pi}_t - \pi_t) +
    \beta_2(\mathbb{E}_{t-1}[\tilde{y}_t] - y_t) +
    \epsilon_t
$$
$$
\Delta s_{t+1} = s_{t+1} - s_t
$$
$$
s_t = \ln{S_t}
$$

Predicting the UK output gap with:

$$
\mathbb{E}_{t-1}[{\tilde{y}_t}] = \gamma_1 \tilde{y}_{t-1}
$$

---

## Ridge Regression

We define the loss function as:

$$
L(\widehat{\Delta s}_{

t+1} \vert \beta_0, \beta_1, \beta_2) = 
    MSE(\widehat{\Delta s}_{t+1}) + 
    \lambda \sum_{i = 0}^{2} \beta_i^2
$$

where:

$$
\beta^* = (\beta^*_0, \beta^*_1, \beta^*_2)' = 
\text{arg min}_{\beta \in \mathbb{R}^3} L(\widehat{\Delta s}_{t+1} \vert \beta)
$$

---

## Tunning Parameters

1. $\lambda$: Regularization term in ***Ridge*** regression.
2. $h$ : Rolling window size for model estimation and the output gaps, to avoid including structural breaks that could affect forecasts.

---

## Estimating $y_t$ and $\tilde{y}_t$

To estimate $\Delta \ln Y_t$:

$$
\Delta \ln Y_t = \Delta y_t + \Delta \bar{y}_t
$$

Knowing $y_t$ is $I(0)$:

$$
\Delta y_t = y_t - y_{t-1}
$$
$$
\mathbb{E}[\Delta y_t] = \mathbb{E}[y_t - y_{t-1}]
$$
$$
\mathbb{E}[\Delta y_t] = 0
$$

Since:

$$
\mathbb{E}(y_{t+s}) = 0, \forall s \in \mathbb{Z}
$$

---
For $\Delta \bar{y}_t$:

$$
\Delta \bar{y}_t = \bar{y}_t - \bar{y}_{t-1} 
$$
$$
\Delta \bar{y}_t = \alpha_0 + \bar{y}_{t-1} + \nu_{t-1} - \bar{y}_{t-1} 
$$

$$
\mathbb{E}[\Delta \bar{y}_t] = 
\alpha_0 +
\mathbb{E}[\bar{y}_{t-1}] + 
\mathbb{E}[\nu_{t-1}] - 
\mathbb{E}[\bar{y}_{t-1}] 
$$
$$
\mathbb{E}[\Delta \bar{y}_t] = \alpha_0
$$

---

Thus:

$$
\mathbb{E}[\Delta \ln Y_t] =  
\mathbb{E}[\Delta y_t] +  
\mathbb{E}[\Delta \bar{y}_t]
$$

$$
\mathbb{E}[\Delta \ln Y_t] = \alpha_0
$$

We estimate the drift $\alpha_0$ of $\bar{y}_t$ as:

$$
\hat{\alpha}_0 = \frac{1}{h} \sum_{i = t - h}^{t} \Delta \ln Y_i
$$

$h$ is the previously mentioned second tuning parameter.

---

Similarly, for $\tilde{y}_t$:

$$
\hat{\gamma}_0 = \frac{1}{h} \sum_{i = t - h}^{t} \Delta \ln \tilde{Y}_i
$$