# 📈 Reinforcement Learning for Financial Trading

This project explores using a **Q-Learning agent** to develop an automated trading strategy based on financial time-series data.

---

## 🚀 Project Status and Q-Agent Performance

**Current Date:** October 23, 2025

I've implemented a **Q-Learning agent** which is currently outperforming several baseline strategies. While perfect trades are impossible, the goal is to consistently capture a significant percentage of the optimal profit.

### Performance Summary (35 Simulated Days)

| Metric | Q-Agent Performance | Optimal Strategy Benchmark | Target Goal |
| :--- | :--- | :--- | :--- |
| **Total Return** | **2.68%** | N/A | N/A |
| **Average Daily Return** | **0.08%** | Approx. **1.3%** | **0.25% to 0.50%** |
| **Agent Rank** | **6th** of tested strategies | **1st** (Benchmark) | N/A |

---

## 📊 Data Source and Current Challenges

### Training Data

The model was trained using historical financial data sourced from:
`https://github.com/FutureSharks/financial-data`

**Note:** The raw data files are not included in this repository due to their size.

### Overfitting and Trading Behavior

The current training data is a limited set (approx. 200 data points), primarily from **2011**. This has highlighted two key issues:

1.  **Limited Strategy:** The agent's current dominant behavior is a simplistic **"buy at the start of the day and hold."** This suggests it is over-simplifying the strategy for the specific, positive training window.
2.  **High Overfitting Risk:** Training on what appears to be a strong market year (2011) creates a model that is unlikely to generalize well to more volatile or negative market conditions.

### Next Steps

The immediate focus will be on **diversifying the training data** to include multiple years and varied market environments, aiming for a more robust and generalized trading model.

---

## 💡 Motivation

Finance may not be my specialty, but a computer's ability to recognize patterns and efficiently process large amounts of numerical data is unmatched. This project is driven by the belief that **Reinforcement Learning** can discover non-linear and profitable trading opportunities that traditional models often miss.