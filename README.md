
# Microsoft Stock Data - Comprehensive Probability Theory Analysis

## 1. Descriptive Statistics & Probability Measures
This section explores the **central tendency** and **dispersion** of stock returns using statistical concepts:
- **Mean & Median** represent the expected (average) outcome — the first moment of the probability distribution.
- **Variance & Standard Deviation** measure dispersion — the spread of returns around the mean.
- **Skewness** shows the asymmetry of return distribution.
- **Kurtosis** indicates tail heaviness — higher kurtosis means extreme events (outliers) are more likely.

Probability theory views these as characteristics of a random variable \( X \) where returns follow some distribution \( P(X) \).

---

## 2. Probability Distributions
The script fits several distributions to understand the probabilistic nature of returns:
- **Normal Distribution** \( N(μ, σ^2) \): assumes returns are symmetrically distributed around a mean.
- **Log-Normal Distribution** models stock prices (since prices can't be negative).
- **Student’s t-Distribution** accounts for fat tails — better for real financial data.

These fits estimate the **parameters (μ, σ, ν)** that define the likelihood of observing specific return values.

---

## 3. Probability Calculations
This section computes **empirical probabilities** based on observed frequencies:
- \( P(Return > 0) \): chance of gain on any day.
- \( P(Return < 0) \): chance of loss.
- \( P(μ - σ ≤ X ≤ μ + σ) \): probability of returns lying within one standard deviation (≈68% for normal distribution).

This applies the **frequentist definition of probability** — outcomes as long-run relative frequencies.

---

## 4. Conditional Probabilities
Conditional probability examines dependence between sequential returns:
\[ P(A | B) = \frac{P(A ∩ B)}{P(B)} \]
For example:
- \( P(Positive\ Today | Positive\ Yesterday) \) captures market momentum.
- \( P(Negative\ Today | Negative\ Yesterday) \) shows persistence of trends.

This reflects **Markovian behavior**, where current states depend probabilistically on prior states.

---

## 5. Bayes’ Theorem Application
Using **Bayes’ theorem**:
\[ P(H | E) = \frac{P(E | H) P(H)}{P(E)} \]
- \( H \): High volatility, \( E \): Large price move.
It helps reverse probabilities — e.g., finding the chance of high volatility given a large move.

This Bayesian perspective updates beliefs about market risk given new evidence.

---

## 6. Expected Value & Risk Metrics
Expected value \( E[X] \) represents the mean outcome of random variable \( X \):
\[ E[X] = \sum x_i P(x_i) \]
- **Variance (Var[X])** quantifies uncertainty.
- **Value at Risk (VaR)** estimates potential losses under confidence levels.
- **Conditional VaR (CVaR)** measures expected loss beyond the VaR threshold.
- **Sharpe Ratio** expresses return per unit of risk.

These are core **risk management measures** derived from probability distributions.

---

## 7. Markov Chain Analysis
Market states (Up, Down, Flat) are modeled as a **Markov process**:
\[ P(X_{t+1} = j | X_t = i) \]
Transition probabilities form a matrix, describing the likelihood of moving between states.
This embodies **stochastic process theory**, where future behavior depends only on current state.

---

## 8. Monte Carlo Simulation
Monte Carlo simulations apply **random sampling** to forecast possible future stock prices.
Given returns follow \( N(μ, σ) \), each iteration simulates potential paths over time.

This illustrates **law of large numbers** and **central limit theorem** — as simulations increase, average outcomes converge to theoretical expectations.

---

## Summary of Probabilistic Insights
- Stock returns follow near-normal but fat-tailed distributions.
- Probability of daily gain ≈50–55%.
- Volatility strongly influences the likelihood of large movements.
- Monte Carlo simulations reveal uncertainty bounds and expected price ranges.
