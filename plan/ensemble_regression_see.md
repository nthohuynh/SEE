Based on the methodology and experimental evaluation details provided in the text, here is a reproduction plan formatted as a Markdown file.

***

```markdown
# Reproduction Plan: Ensemble Regression Models for Software Development Effort Estimation

**Paper:** Ensemble Regression Models for Software Development Effort Estimation: A Comparative Study  
**Source:** International Journal of Software Engineering & Applications (IJSEA), Vol.11, No.3, May 2020  
**Authors:** Halcyon D. P. Carvalho, Marília N. C. A. Lima, Wylliams B. Santos, Roberta A. de A. Fagundes

---

## 1. Environment and Prerequisites
To reproduce the study, the following environment is required (implied by the statistical and ML techniques mentioned):
*   **Language:** Python or R (Standard tools for ML/Statistical analysis).
*   **Libraries:** Libraries capable of handling:
    *   Linear Regression, Robust Regression, Ridge Regression, Lasso Regression.
    *   Bootstrap Aggregating (Bagging) and Stacking.
    *   Statistical tests: Kolmogorov-Smirnov and Wilcoxon.

## 2. Dataset Preparation
The study uses a dataset of software projects described in the literature (specifically referenced as the dataset used by Lopez-Martin).

### 2.1. Variables
*   **Independent Variables (Inputs):**
    *   `N&C` (New and Changed code): Physical lines of code added or modified.
    *   `R` (Reused code): Physical lines of reused code.
*   **Dependent Variable (Output):**
    *   `AE` (Actual Effort): Measured in minutes.

### 2.2. Pre-processing
*   **Normalization:** Apply the **Max-Min** method to normalize all data to a scale between 0 and 1.
    *   *Formula:* Use the maximum and minimum values of the variable to normalize to a uniform scale.

## 3. Model Implementation
Implement the following regression techniques grouped by ensemble method.

### 3.1. Bagging Models (Bootstrap Aggregating)
For each model, generate bootstrap samples (training sets with replacement) and average the individual forecasts.
*   **B-LR:** Bagging with Linear Regression.
*   **B-RR:** Bagging with Robust Regression.
*   **B-RI:** Bagging with Ridge Regression.
*   **B-LA:** Bagging with Lasso Regression.

### 3.2. Stacking Models
Implement a two-level learning process. Level-0 models are trained/tested on cross-validation examples; their outputs become inputs for the Level-1 (Meta-Predictor).

*   **ST-LR:**
    *   *Base Learners:* Robust, Ridge, Lasso.
    *   *Meta-Predictor:* Linear Regression.
*   **ST-RR:**
    *   *Base Learners:* Linear, Ridge, Lasso.
    *   *Meta-Predictor:* Robust Regression.
*   **ST-RI:**
    *   *Base Learners:* Robust, Linear, Lasso.
    *   *Meta-Predictor:* Ridge Regression.
*   **ST-LA:**
    *   *Base Learners:* Lasso, Linear, Ridge.
    *   *Meta-Predictor:* Lasso Regression.

### 3.3. Baseline Models (Literature Comparison)
*   **Linear Regression:** (Reference implementation or Equation 6: `Effort = 44.713 + (1.08 * N&C) - (0.145 * R)`).
*   **ELM (Extreme Learning Machine):**
    *   Variant 1: 2 hidden nodes.
    *   Variant 2: 5 hidden nodes.

## 4. Experimental Procedure
Execute the following algorithm (based on Algorithm 1) to evaluate the models.

**Parameters:**
*   **Iterations (Monte Carlo Simulation):** 1000.
*   **Data Split:** 70% Training, 30% Testing.

**Execution Loop:**
1.  **Shuffle:** Randomly shuffle the dataset.
2.  **Split:** Divide into Training (70%) and Test (30%) sets.
3.  **Train:** Apply all 8 ensemble models (B-LR, B-RR, B-RI, B-LA, ST-LR, ST-RR, ST-RI, ST-LA) to the training set.
4.  **Predict:** Generate estimates for the test set.
5.  **Calculate Error:** Compute Mean Absolute Residual (MAR) for each model.
6.  **Repeat:** Perform for 1000 iterations.
7.  **Aggregate:** Calculate the mean and standard deviation of the MAR for all iterations.

## 5. Evaluation Metrics
### 5.1. Performance Index
*   **Mean Absolute Residual (MAR):**
    $$ MAR = \frac{1}{n} \sum |y_i - \hat{y}_i| $$
    Where $y_i$ is the actual value, $\hat{y}_i$ is the estimated value, and $n$ is the number of cases.

### 5.2. Comparative Metric
*   **Relative Gain (RG):**
    $$ RG = 100 * \left( \frac{Error_a - Error_b}{Error_a} \right) $$
    Used to measure the percentage gain of the proposed models against literature models.

### 5.3. Statistical Analysis
*   **Normality Test:** Kolmogorov-Smirnov test to check distribution.
*   **Significance Test:** Wilcoxon hypothesis test (Significance level = 5%) to compare error distributions between models (specifically testing if B-LA or ST-LA have statistically smaller errors).

## 6. Expected Results
*   **Best Performers:** The reproduction should aim to confirm if **B-LA** (Bagging Lasso) and **ST-LA** (Stacking Lasso) produce the lowest mean MAR scores.
*   **Outliers:** Boxplots should indicate that Stacking models are generally less sensitive to outliers than Bagging models.
```