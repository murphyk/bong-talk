---
layout: cover
# background: ./images/bone-wall.png
class: text-center
title: BONE
mdc: true
---

# (B)ayesian (O)nline learning in (N)on-stationary (E)nvironments

**Gerardo Duran-Martin**, Leandro Sánchez-Betancourt, Alexander Shestopaloff, and Kevin Murphy

---

## What is BONE?

Framework that encompasses many Bayesian models that *tackle* online learning in non-stationary environments.

* By *Bayesian* we mean using Bayes' rule to update beliefs.
* By *online* we mean observing a stream of datapoints and making a decision about the next datapoint.
* By *non-stationary* we mean that the data generating process (DGP) changes through time.

---

# Online learning and prediction
Observe sequence of features $x_t$ and observations $y_t$:

$$
    {\cal D}_{1:t} = \{(x_1, y_1), \ldots, (x_t, y_t)\}.
$$


Given $x_{t+1}$, make a prediction

$$
    \hat{y}_{t+1} = \omega(x_{t+1}, {\cal D}_{1:t}).
$$


Variables $x_t$ and $y_t$ can live in different time frames.

---

## (Unsupervised) online sequential learning
Estimating unobservable quantities of interest from the data.

1. Filtering --- estimate value of *latent* process.
2. Segmentation --- partition the data into non-overlapping bins.
3. Switching state-space models (Switching SSMs) --- categorise current observation into a number of possible states.

---

## (Supervised) online sequential learning (and prediction)
Predicting a measurable outcome $y_t$
1. Online continual learning.
2. Bandits.
3. One-step-ahead (prequential) forecasting.
4. Reinforcement learning (TD learning).


---

## Bayes for sequential online learning


$$
    \hat{y}_{t+1}
    = \mathbb{E}[h(\theta_t; x_{t+1}) \vert {\cal D}_{1:t}]
    = \int
    \underbrace{h(\theta_t; x_{t+1})}_\text{measurement fn.}
    \overbrace{p(\theta_t; {\cal D}_{1:t})}^\text{posterior density}
    \,{\rm d}\theta_t.
$$


---

## Sequential classification
A running example

![linreg](./images/sequential-moons.gif)

---
layout: center
---

## Choice of measurement model (M.1)

Inductive bias on the observations given parameters $\theta$ and features $x_t$.

---

## Choice of measurement model (M.1)

In the linear setting:

$$
    h(\theta, x) =
    \begin{cases}
    \theta^\intercal\,x & \text{linear regression},\\
    \sigma(\theta^\intercal\,x) & \text{logistic regression}.
    \end{cases}
$$



In the non-linear classification setting,
$$
    h(\theta, x) = \sigma(\phi_\theta(x)),
$$
with $\phi_\theta: \reals^M \to \reals$ a neural network with real-valued output unit.

Then $p(y_t \vert x_t) = {\rm Bern}(y_t \vert h(\theta, x))$


---
layout: center
---

## Weight over model parameters (A.1) --- Bayes
Assume that observations $y_{1:t}$ are conditionally independent given $\theta$.


$$
    p(\theta | {\cal D}_{1:t}) \propto p(\theta)\,\prod_{k=1}^t p(y_t \vert \theta, x_t).
$$


In most cases, we cannot compute the posterior. We resort to either sample-based methods or approximations.

---

## Variational Bayes methods
Suppose $p(\theta) = {\cal N}(\theta \vert \mu_0, \Sigma_0)$

$$
    \mu_t, \Sigma_t = {\bf D}_\text{KL}\left(
        {\cal N}(\theta\,|\,\mu, \Sigma) \|
        {\cal N}(\theta \vert \mu_{0}, \Sigma_{0})\,\prod_{k=1}^t p(y_k | \theta, x_k).
    \right)
$$


---
layout: center
---

## Weight over model parameters (A.1) --- sequential Bayes


$$
    p(\theta | {\cal D}_{1:t}) \propto p(\theta | {\cal D}_{1:t-1}) p(y_t \vert \theta, x_t).
$$


---

## (Recursive) variational Bayes methods
Suppose
$p(\theta \vert {\cal D}_{1:t-1}) = {\cal N}(\theta \vert \mu_{t-1}, \Sigma_{t-1})$:

$$
    \mu_t, \Sigma_t = {\bf D}_\text{KL}
    \left(
        {\cal N}(\theta\,|\,\mu, \Sigma) \|
        {\cal N}(\theta \vert \mu_{t-1}, \Sigma_{t-1})\,p(y_t | \theta, x_t)
    \right).
$$


---

## Moment-matched linear Gaussian (LG)

First-order approximation of the choice of modelling function (M.1) around the previous mean $\mu_{t-1}$:

$$
    h(\theta, x) \approx \bar{h}_t(\theta, x) = H_t\,(\theta_t - \mu_{t-1}) +  h(\mu_{t-1}, x_t).
$$



$$
    p(y_t \vert \theta, x_t) \approx
    {\cal N}(y_t\,\vert\,\hat{y}_t, R_t),
$$

where 

$$
\begin{aligned}
    \hat{y}_t &= \mathbb{E}[\bar{h}(\theta_t, x_t) \vert {\cal D}_{1:t-1}] = h(\mu_{t-1}, x_t)\\
\end{aligned}
$$

and $R_t$ is the moment-matched variance of the observation process.

---

## Example: moment-matched linear Gaussian for classification


$$
\begin{aligned}
H_t &= \nabla_\theta h(\mu_{t-1}, x_t) & \text{(Jacobian)}\\
\hat{y}_t & = h(\mu_{t-1}, x_t) & \text{ (one-step-ahead prediction)} \\
R_t &= \hat{y}_t\,(1 - \hat{y}_t) & \text{ (moment-matched variance)}\\
\Sigma_t^{-1} &= \Sigma_{t-1}^{-1} + H_t^\intercal\,R_t^{-1}\,H_t & \text{(posterior precision)}\\
{\bf K}_t &= \Sigma_{t}\,H_t\,{R}_t^{-1} & \text{(Gain matrix)}\\
\hline
\mu_t &\gets \mu_{t-1} + {\bf K}_t\,(y_t - \hat{y}_t) & \text{(update)}
\end{aligned}
$$


---

## Sequential Online classification
* Online Bayesian learning using (M.1) a single hidden-layer neural network and (A.1) moment-matched LG.
![sequential classification with static dgp](./images/moons-c-static.gif)

---

## Online learning --- (M.1) and (A.1)
* The methods above (LG and VB) assume that the data-generating process is fixed / known.
* Mean estimate $\mu_T$ under LG converges to a point estimate as $T \to \infty$ regardless of error, i.e., $\Sigma_T \to 0 {\bf I}$.
* They are sequential, but do not have a *notion* of non-stationarity (model misspecification).

---

# Changes in the data-generating process
When more data does not lead to better performance.

Could be due to:
1. Not enough model capacity,
2. misspecified measurement model / lack of additional information.

---


## Non-stationary moons dataset
![non-stationary-moons-split](./images/mooons-dataset-split.png)

---

## The full dataset (without knowledge of the task boundaries)
![non-stationary-moons-full](./images/moons-dataset-full.png){style="max-width: 50%"}

---

## Constant moment-matched LG updates --- non-stationary moons

![sequential classification with varying dgp](./images/changes-moons-c-static.gif)


---

## Recap: tools for sequential learning

* (M.1) A model for observations (conditioned on features $x_t$) --- $h(\theta, x_t)$.
* (A.1) An algorithm to weight choices of $\theta$ (conditioned on data ${\cal D}_{1:t}$) ---
$p(\theta\vert{\cal D}_{1:t}) \propto p(\theta \vert {\cal D}_{1:t-1}) p(y_t \vert \theta, x_t)$.


In many situations, (M.1) and (A.1) are not enough to *adapt* to regime changes.

---

# Tackling non-stationarity
1. Keep track of _regimes_ (M.2).
2. Act on our beliefs based on the regime (M.3).
3. Assign probability (weights) over possible regimes (A.2).

---
layout: center
---

# Tracking regimes through an auxiliary variable (M.2)

1. Denote $\psi_t$ the auxiliary variable.
1. The value of $\psi_t$ depends on how we _track_ regimes.

---

## Runlenght (RL)
* number of timesteps since the last changepoint (lookback window).
* $\psi_t = r_t$.

![Runlength auxiliary variable](./images/auxvar-rl.png)

---

## Runlenght with changepoint count (RLCC)
* number of timesteps since the last changepoint (lookback window) and count of the number of changepoints.
* $\psi_t = (r_t, c_t)$.

![Runlength and changepoint count auxiliary variable](./images/auxvar-rlcc.png)

---

## Changepoint location (CPL)
* Binary sequence of changepoint locations.
* $\psi_t = s_{1:t} \in \{0,1\}^t$.

![Changepoint location auxiliary variable](./images/auxvar-cpl.png)

---

## Changepoint location (CPL) alt.
* Binary sequence of values belonging to the current regime.
* $\psi_t = s_{1:t} \in \{0,1\}^t$.

![Changepoint location auxiliary variable](./images/auxvar-cpl2.png)

---

## Changepoint probability (CPP)
* Changepoint probabilities.
* $\psi_t = \nu_t \in [0,1]$.

![Changepoint probability auxiliary variable](./images/auxvar-cpp.png)

---

## Mixture of experts (ME)
* Choices of over a fixed number of models.
* $\psi_t = \alpha_t \in \{1, \ldots, K\}$.

![Mixture of experts auxiliary variable](./images/auxvar-me.png)

---

## Constant \(C\)
* Single choice of model.
* $\psi_t = c$.

![Constant auxiliary variable](./images/auxvar-cst.png)

---

## Summary of auxiliary variables

![table of auxiliary variables](./images/auxvar-table.png)

---
layout: center
--- 

# (M.3) Modifying beliefs based on regime
* _Static_ recursive Bayes treats the prior as given and set at the begining of the experiment.
* Construct the posterior based on two modelling choices: a form for the prior and a form for the likelihood.


$$
    \underbrace{\lambda(\theta; \psi_t, {\cal D}_{1:t})}_{\text{posterior}}
     \propto
    \underbrace{\tau(\theta\,\vert\,\psi_t, {\cal D}_{1:t-1})}_\text{parametrised by $\psi_t$}\,
    \overbrace{p(y_t\,\vert\,\theta, x_t)}^\text{parametrised by $\theta$}
$$


---

## Choice of conditional prior (M.3) --- the Gaussian case


$$
    \tau(\theta_t \vert \psi_t,\, {\cal D}_{1:t-1}) =
    {\cal N}\big(\theta_t\vert g_{t}(\psi_t, {\cal D}_{1:t-1}), G_{t}(\psi_t, {\cal D}_{1:t-1})\big),
$$


* $g_t(\cdot, {\cal D}_{1:t-1}): \Psi_t \to \reals^m$ --- mean vector of model parameters.
* $G_t(\cdot, {\cal D}_{1:t-1}): \Psi_t \to \reals^{m\times m}$ --- covariance matrix of model parameters.

---

## `C-Static` --- constant update with static auxvar
* $\psi_t = c$.
* Classical (static) Bayesian update:

$$
\begin{aligned}
    g_t(c, {\cal D}_{1:t-1}) &= \mu_{t-1}\\
    G_t(c, {\cal D}_{1:t-1}) &= \Sigma_{t-1}\\
\end{aligned}
$$


---

## `RL-PR` --- runlength with prior reset
* $\psi_t = r_t$.
* Corresponds to the Bayesian online changepoint detection (BOCD) algorithm.


$$
    \begin{aligned}
        g_t(r_t, {\cal D}_{1:t-1}) &= \mu_0\,\mathbb{1}(r_t  = 0) + \mu_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
        G_t(r_t, {\cal D}_{1:t-1}) &= \Sigma_0\,\mathbb{1}(r_t  = 0) + \Sigma_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
    \end{aligned}
$$


where  $\mu_{(r_{t-1})}, \Sigma_{(r_{t-1})}$ denotes the posterior belief using observations
from indices $t - r_t$ to $t - 1$.
$\mu_0$ and $\Sigma_0$ are pre-defined prior mean and covariance.

---

## `CPP-OU` --- changepoint probability with Ornstein-Uhlenbeck process
* $\psi_t = \upsilon_t$.
* Mean reversion to the prior as a function of the probability of a changepoint:

$$
    \begin{aligned}
        g(\upsilon_t, {\cal D}_{1:t-1}) &= \upsilon_t \mu_{t-1} + (1 - \upsilon_t)  \mu_0 \,,\\
        G(\upsilon_t, {\cal D}_{1:t-1}) &=  \upsilon_t^2 \Sigma_{t-1} + (1 - \upsilon_t^2)  \Sigma_0\,.
    \end{aligned}
$$


---


## `CPL-sub` --- changepoint location with subset of data
* $\psi_t = s_{1:t}$.


$$
\begin{aligned}
    g_t(s_{1:t}, {\cal D}_{1:t-1}) &= \mu_{(s_{1:t})},\\
    G_t(s_{1:t}, {\cal D}_{1:t-1}) &= \Sigma_{(s_{1:t})},\\
\end{aligned}
$$

where  $\mu_{(s_{1:t})}, \Sigma_{(s_{1:t})}$ denote the posterior beliefs using observations that are 1-valued.

---

## `C-ACI` --- constant with additive covariance inflation

* $\psi_t = c$.
* Random-walk assumption. _Inject_ noise at every new timestep. Special case of a linear state-space-model (SSM).

$$
\begin{aligned}
    g_t(c, {\cal D}_{1:t-1}) &= \mu_{t-1},\\
    G_t(c, {\cal D}_{1:t-1}) &= \Sigma_{t-1} + Q_t.\\
\end{aligned}
$$

Here, $Q_t$ is a positive semi-definitive matrix. Typically, $Q_t = \alpha {\bf I}$ with $\alpha > 0$.

---

# Tools for sequential learning (conditioned on regime)
* (M.1) A model for observations (conditioned on features $x_t$) --- $h(\theta, x_t)$.
* (M.2) An auxiliary variable for regime changes --- $\psi_t$.
* (M.3) A model for prior beliefs (conditioned on $\psi_t$ and data ${\cal D}_{1:t-1}$) ---
$\tau(\theta \vert \psi_t, {\cal D}_{1:t-1})$.
* (A.1) An algorithm to weight over choices of $\theta$ (conditioned on data ${\cal D}_{1:t}$) ---

$\lambda(\theta;\,\psi_t, {\cal D}_{1:t}) \propto \tau(\theta \vert \psi_t, {\cal D}_{1:t-1}) p(y_t \vert \theta, x_t)$.


---

## A prediction conditioned on $\psi_t$


$$
    \hat{y}_{t+1}^{(\psi_t)}
    = \mathbb{E}_{\lambda_t}[h(\theta_t;\, x_{t+1}) \vert \psi_t]
    := \int h(\theta_t;\, x_{t+1})\,\lambda(\theta_t;\,\psi_t, {\cal D}_{1:t})d \theta_t\,.
$$


(A.2) An algorithm to weight over choices of $\psi_t \in \Psi_t$.

---

## Aggregate predictions

$$
\begin{aligned}
    \hat{y}_{t+1} 
    &= \int\left(\int h(\theta_t;\, x_{t+1})\,\lambda_t( \theta_t;\,\psi_t, {\cal D}_{1:t})\, d \theta_t\right)\,\nu_t(\psi_t)\,d\psi_t\\
    &= \int \hat{y}_{t+1}^{(\psi_t)}\,\nu_t(\psi_t)\,d\psi_t\\
    &=: \mathbb{E}_{\nu_t}\Big[\, \mathbb{E}_{\lambda_t}[h(\theta_t;\, x_{t+1}) \vert \psi_t ] \,\Big]\,.
\end{aligned}
$$


---
layout: center
---

## (A.2) Weighting function for auxiliary variables
Bayesian, loss-based, ad-hoc.

---

## (A.2) The recursive Bayesian choice

$$
\begin{aligned}
    \nu_t(\psi_t)
    &= p(\psi_t \vert {\cal D}_{1:t})\\
    &= 
    p(y_t \vert x_t, \psi_t, {\cal D}_{1:t-1})\,
    \int_{\psi_{t-1} \in \Psi_{t-1}}
    p(\psi_{t-1} \vert {\cal D}_{1:t-1})\,
    p(\psi_t \vert \psi_{t-1}, {\cal D}_{1:t-1}) d \psi_{t-1},
\end{aligned}
$$


For some $\psi_t$ and $p(\psi_t \vert \psi_{t-1})$, this method yields recursive update methods.

---

## (A.2) A loss-based approach
Suppose
$\psi_t = \alpha_t \in \{1, \ldots, K\}$, take

$$
    \nu_t(\alpha_t) = 1 - \frac{\ell(y_{t+1}, \hat{y}_{t+1}^{(\alpha_t)})}
    {\sum_{k=1}^K \ell(y_{t+1}, \hat{y}_{t+1}^{(k)})},
$$

with $\ell$ a loss function (lower is better).

---

## (A.2) An empirical-Bayes (point-estimate approach)
For $\psi_t = \upsilon_t \in [0,1]$,


$$
    \upsilon_t^* = \argmax_{\upsilon \in [0,1]} p(y_t \vert x_t, \upsilon, {\cal D}_{1:t-1}).
$$

Then,

$$
    \nu(\upsilon_t) = \delta(\upsilon_t = \upsilon_t^*).
$$


---
layout: center
---
    
# BONE --- Bayesian online learning in non-stationary environments
* (M.1) A model for observations (conditioned on features $x_t$) --- $h(\theta, x_t)$.
* (M.2) An auxiliary variable for regime changes --- $\psi_t$.
* (M.3) A model for prior beliefs (conditioned on $\psi_t$ and data ${\cal D}_{1:t-1}$) ---
$\tau(\theta \vert \psi_t, {\cal D}_{1:t-1})$.
* (A.1) An algorithm to weight over choices of $\theta$ (conditioned on data ${\cal D}_{1:t}$) ---
$\lambda(\theta;\,\psi_t, {\cal D}_{1:t}) \propto \tau(\theta \vert \psi_t, {\cal D}_{1:t-1}) p(y_t \vert \theta, x_t)$.
* (A.2) An algorithm to weight over choices of $\psi_t$ (conditioned on data ${\cal D}_{1:t}$).


---

# Back to the non-stationary moons example

Consider three combinations algorithms
1. Runlenght with prior reset and a single hypothesis --- `RL[1]-PR`.
2. Changepoint probability with OU dynamics --- `CPP-OU`.
3. Constant auxiliary variable with additive covariance inflation --- `C-ACI`.

Our choice of measurement model is a single hidden layer neural network
with linearised moment-matched Gaussian.

---

## RL[1]-PR
* When changepoint detected: reset back to initial weights

$$
    \begin{aligned}
        g_t(r_t, {\cal D}_{1:t-1}) &= \mu_0\,\mathbb{1}(r_t  = 0) + \mu_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
        G_t(r_t, {\cal D}_{1:t-1}) &= \Sigma_0\,\mathbb{1}(r_t  = 0) + \Sigma_{(r_{t-1})}\mathbb{1}(r_t > 0).\\
    \end{aligned}
$$

![rl-pr-sequential-classification](./images/changes-moons-rl-pr.gif){style="transform: translate(-50%, -20%)"}

---

## CPP-OU
* Revert to initial weights proportional to the probability of a changepoint

$$
    \begin{aligned}
        g(\upsilon_t, {\cal D}_{1:t-1}) &= \upsilon_t \mu_{t-1} + (1 - \upsilon_t)  \mu_0 \,,\\
        G(\upsilon_t, {\cal D}_{1:t-1}) &=  \upsilon_t^2 \Sigma_{t-1} + (1 - \upsilon_t^2)  \Sigma_0\,.
    \end{aligned}
$$

![cpp-ou-sequential-classification](./images/changes-moons-cpp-ou.gif){style="transform: translate(-50%, -20%)"}


---

## C-ACI
* Constantly forget past information.

$$
\begin{aligned}
    g_t(c, {\cal D}_{1:t-1}) &= \mu_{t-1},\\
    G_t(c, {\cal D}_{1:t-1}) &= \Sigma_{t-1} + Q_t.\\
\end{aligned}
$$

![c-aci-sequential-classification](./images/changes-moons-c-aci.gif){style="transform: translate(-50%, -20%)"}

---

# Sequential classification --- comparison
In-sample hyperparameter optimisation.

![comparison-sequential-classification](./images/changes-mooons-comparison.png)

---

## Unified view of examples in the literature


![BONE-methods-examples](./images/methods-bone.png)

---
layout: center
---

# Creating a new combination

---

## RL[1]-OUPR --- choice of (M.2) and (M.3)

Choice of (M.2)  
* Lookback windows are common in e.g., finance --- single runlength (`RL[1]`) as choice auxiliary variable.

Choice of (M.3)
* Reset if the hypothesis of a a changepoint is below some thresold $\varepsilon$.
* OU-like reversion rate otherwise.

$$
    g_t(r_t, {\cal D}_{1:t-1}) =
    \begin{cases}
        \mu_0\,(1 - \nu_t(r_t)) + \mu_{(r_t)}\,\nu_t(r_t) & \nu_t(r_t) > \varepsilon,\\
        \mu_0 & \nu_t(r_t) \leq \varepsilon,
    \end{cases}
$$



$$
   G_t(r_t, {\cal D}_{1:t-1}) =
    \begin{cases}
        \Sigma_0\,(1 - \nu_t(r_t)^2) + \Sigma_{(r_t)}\,\nu_t(r_t)^2 & \nu_t(r_t) > \varepsilon,\\
        \Sigma_0 & \nu_t(r_t) \leq \varepsilon.
    \end{cases}
$$


---

## RL[1]-OUPR --- choice of (A.2)
* Posterior predictive ratio test


$$
    \nu_t(r_t^{(1)}) =
    \frac{p(y_t \vert r_t^{(1)}, x_t, {\cal D}_{1:t-1})\,(1 - \pi)}
    {p(y_t \vert r_t^{(0)}, x_t, {\cal D}_{1:t-1})\,\pi + p(y_t \vert r_t^{(1)}, x_t, {\cal D}_{1:t-1})\,(1-\pi)}.
$$

Here, $r_{t}^{(1)} = r_{t-1} + 1$ and $r_{t}^{(0)} = 0$.


---

## Experiment: hourly electricity load

Seven features (lagged by one hour):
* pressure (kPa)
* cloud cover (\%)
* humidity (\%)
* temperature \(C\) 
* wind direction (deg), and
* wind speed (KmH).

One target variable:
* hour-ahead electricity load (kW).

---

## Experiment: hourly electricity load

![day-ahead electricity forecasting](./images/day-ahead-dataset.png){style="max-width: 70%"}

---

## Electricity forecasting during _normal_ times
![day ahead forecasting normal](./images/day-ahead-forecast-normal.png)


---

## Electricity forecasting before and after Covid _shock_
![day ahead forecasting shock](./images/day-ahead-forecast.png)

---

## Electricity forecasting
![day ahead forecasting shock](./images/day-ahead-forecast-rlpr.png)

---

## Electricity forecasting results (2018-2020)
![day ahead forecasting results](./images/day-ahead-results.png)


---


## Experiment --- heavy-tailed linear regression

![heavy-tailed-linear regression panel](./images/segments-tdist-lr.png)

---

## Heavy tailed linear regression (sample run)


![heavy-tailed-linear-regression](./images/outliers-all-panel.gif){style="max-width: 70%" .horizontal-center}

---

# Even more experiments!

* Bandits.
* Online continual learning.
* Segmentation and prediction with dependence between changepoints.

See paper for details.


---

# Conclusion
We introduce a framework for Bayesian online learning in non-stationary environments (BONE).

BONE methods are written as instances of:
## Three modelling choices
* (M.1) Measurement model
* (M.2) Auxiliary variable
* (M.3) Conditional prior

## Two algorithmic choices
* (A.1) weighting over model parameters (posterior computation)
* (A.2) weighting over choices of auxiliary function


---
layout: end
---

[gerdm.github.io/posts/bone-slides](https://gerdm.github.io/posts/bone-slides)  
[arxiv.org/abs/2411.10153](https://arxiv.org/abs/2411.10153)