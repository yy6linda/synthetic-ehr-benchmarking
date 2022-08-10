# Synthetic EHR benchmarking framework
---
## Overview
---
This document describes how to benchmark Generative Adversarial Networks(GANs)-based approaches for generating synthetic patient data. We provided evaluation from both utility and privacy prospectives.

The figure below describes the benchmarking framework, which involves three phases - Synthetic EHR data generation,  multifaceted assessment and model recommendation.

![image](./data/Figure1.png)

## Synthetic EHR data generation
---
We used this framework to evaluate five GAN-based models that were designed to synthesize EHR profiles of patients: 1) medGAN, 2) medBGAN, 3) EMR-WGAN, 4) WGAN, and 5) DPGAN. Additionally, we incorporated a baseline approach that randomly samples the values of features based on the marginal distributions of the real data  to complement the scope of benchmarking in terms of the variety of model behavior. We refer to this approach as the sampling baseline, or Baseline. Interestingly, as our results illustrate, this approach outperformed GAN-based models in practical use cases. 

## Multifaceted assessment
---
The Multifaceted assessment phase focuses on two perspectives — utility and privacy
### Utility
In earlier investigations, the term utility was defined in parallel with resemblance (i.e., the statistical similarity of two datasets), and was specifically used to refer to the value of real or synthetic data to support predictions. By contrast, in this work, data utility is defined to cover a general set of metrics, each measuring a factor to which the value of data is attributed. This is because numerous real-world use cases of synthetic data do not involve any prediction tasks, but still require the synthetic data to be useful (or has utility).

We included 7 metric for utility evaluation:
- **Feature-level statistics**
  - Dimension-wise distribution
  - Column-wise correlation
  - Latent cluster analysis
-  **Outcome prediction**
   - Train on real data test on synthetic data (TRTS) Model performance
   - Train on synthetic data test on real data (TSTR) Model performance
   - Feature selection
- **Record-level readability**
  -  Clinical knowledge violation
  
### Privacy
We focused on three types of privacy attacks that have targeted fully synthetic patient datasets: attribute inference, membership inference, and meaningful identity disclosure.
- **Attribute inference**
- **Membership inference**
-  **Meaningful identity disclosure**

## Model recommendation
---
We consider three use cases of synthetic data to demonstrate generative model selections in the context of specific needs. The benchmarking framework translates a use case into weights on the metric-level results. By default, a weight of 0.1 was assigned to each metric and all weights sum to 1. We adjusted the weights according to the needs of the use case. The following provides a summary of the use case, while the detailed weight profiles are provided below.

|Use case | Dimension-wide distribution | Column-wise correlation| Latent cluster analysis| Prediction performance| Feature selection| Clinical knowledge violation| Attribute inference| Membership inference| Meaningful identity disclosure|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|Education | 0.25|0.15| 0.1| 0.1| 0.1| 0.15| 0.05| 0.05| 0.05|
Medical AI development|0.05|0.05|0.05|0.35|0.15|0.05|0.1|0.1|0.1|
|System design|0.25|0.05|0.05|0.05|0.05|0.05|1/6|1/6|1/6

