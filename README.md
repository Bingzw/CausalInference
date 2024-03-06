# Today Insights Playbook - Root Cause Detection Through Causal Inference 

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [References](#references)
5. [Appendix](#appendix)
6. [FAQ](#faq)
7. [TODO](#todo)

## Introduction
This is the repo for training counterfactual causal inference models for detecting the root causes
of diagnosed campaigns from today's insights. It supports the ad-hoc/scheduled (planning in future) 
analysis of root cause detection. The job takes the input of the diagnosed campaigns and derives the optimal 
root causes for each diagnosed problem. The output root causes are coded in the form of 

<p align="center">
$f(\theta) < \mu$
</p>
<p align="center">
s.t.  $p(\eta) < \alpha$ and  $q(\zeta) > \beta$,
</p>

where $f(\theta)$ is the root cause, $\mu$ is the corresponding threshold, 
$p(\eta)$ and $q(\zeta)$ are the constraints on the root cause, $\alpha$ and $\beta$ are the corresponding thresholds.

In specific, the job will first conduct causal inference to estimate the average treatment effect (ATE) of each hypothesis, then searching 
for the optimal hypothesis that satisfies the constraints. 

## Quick Start
Run customized causal inference job (Jupyter Notebook)
```
>>> from core.causality import CausalModel
>>> from utils.causal_utils import random_data
>>> Y, T, X = random_data(N=10000, K=3)
>>> cm = CausalModel(Y, T, X)
>>> cm.fit(method="ip_weight")
>>> ate_ip = cm.ate_ip # average treatment effect through inverse probability weighting
>>> ate_ip_ci = cm.ate_ip_ci # confidence interval
>>> print("average treatment effect is: ", ate_ip))
>>> print("average treatment effect CI is: ", ate_ip_ci))
```

## References
- [Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- [Pearl, J., 2009. Causal inference in statistics: An overview](https://projecteuclid.org/journals/statistics-surveys/volume-3/issue-none/Causal-inference-in-statistics-An-overview/10.1214/09-SS057.pdf)
- [VAM Playbook Recommnedations By Causal Inference](https://docs.google.com/document/d/1psQWtxGtY9yp6CjvuZZgy30aFfNilJEogcoTeoYFWeI/edit)

## Appendix
### Causal Inference Approaches in the repo
| Method              | Description                                                 | Assumption                                        | Supported |
|---------------------|-------------------------------------------------------------|---------------------------------------------------|-----------|
| ip weighting        | Marginal structure model with inverse probability weighting | confounding variables observed                    | Y         |
| standardization     | standard outcome models with treatment and covariates       | confounding variables observed                    | Y         |
| stratification      | outcome model with treatment and propensity scores          | confounding variables observed                    | N         |
| propensity matching | match treatment and control via propensity scores           | confounding variables observed                    | N         |
| instrument variable | instrumental variable                                       | confounding variables not required to be observed | N         |

### Data requirements for causal inference
#### Data format needed for CausalModel:
- Outcome variable Y: 
  - binary outcome: 0/1
  - continuous outcome: float
- Treatment variable T: 
  - binary treatment: 0: control, 1: treatment
- Covariates X: 
  - only support numerical covariates
  - only support uncensored data (i.e. no missing or partially missing values)
#### Data format needed for RootCauseDetector:
- Dataframe that includes outcome variable, treatment variable, and covariates
- Condition expressions that support constraint optimizations. Examples:
  - `dataframe with columns: [outcome_variable, treatment_variable, covariate_1, covariate_2, covariate_3 ]`
  - `condition_expression = "covariate_1 and covariate_2"` would search for the optimal thresholds
    - `df[treatment_variable] < theta` that satisfies the condition expression `df[covariate_1] < mu_1 and df[covariate_2] < mu_2`

## FAQ
1. Missing package
   - Please check the requirement.txt and install corresponding packages
2. When does causal inference benefit my work?
   - when you would like to know conterfactual results (e.g. what would happen if I change the treatment from A to B)
   - when you would like to run quick **opportunity sizing** with existing historical data
   - when it's costly to run online A/B testing (limited data, long time to run, etc.), feed the limited online data to causal inference models to get the estimated effect
   - when your A/B testing is selected biased (e.g. users are not randomly assigned to treatment and control groups)
   - when you would like to conduct treatment hyper-parameter tuning (e.g. what is the optimal threshold for a treatment)

## TODO
- [ ] Add more causal inference methods: stratification, propensity matching, instrumental variable
- [ ] Add more optimization approaches: bayesian optimization, random search, MCMC etc
- [ ] Support censored data modeling
- [ ] Support categorical covariates
- [ ] Support pre-post time series causal inference

