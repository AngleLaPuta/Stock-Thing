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


## 📅 October 24, 2025 Update

Trained the **Q-Learning agent** overnight with an extended runtime. Unfortunately, performance **deteriorated** on a longer simulation of **208 days**.

### Revised Performance Summary (208 Simulated Days)

| Metric | Q-Agent Performance | Optimal Strategy Benchmark | Target Goal |
| :--- | :--- | :--- | :--- |
| **Total Return** | **1.75%** | **1429.03%** | N/A |
| **Average Daily Return** | **0.01%** | **1.32%** | **0.30% to 0.40%** |
| **Agent Rank** | **8th** of tested strategies | **1st** (Benchmark) | N/A |

---

### Key Issues & Next Steps

* **Plateaued Accuracy:** The training accuracy has plateaued significantly, showing **no improvement** despite increased training time. This suggests the **Q-Learning model** may be fundamentally inadequate for the task.
* **Model Shift:** Given the poor generalization and plateauing, the immediate focus is shifting away from simple Q-Learning. I will research and pivot to more advanced techniques like **Deep Learning (DL)**.
* **Initial DL Experiment:** I have added an **LSTM (Long Short-Term Memory) predictive model** to the test pool, aiming to predict the stock curve and invest based on that prediction.
    * **LSTM Status:** The LSTM models currently rank **3rd, 4th, and 5th** among all strategies. They are the **most profitable** next to the two "perfect foresight" benchmarks, though still significantly underperforming the perfect benchmark's returns. Training will continue to improve their performance.
* **New Strategy Concept:** I plan to develop a new, conditional strategy:
    * Based on market data (graphs) from the **9:30 AM to 10:30 AM** window, the model will attempt to **predict (generate) the rest of the day's price plot**.
    * This predicted plot will then be fed to the "perfect fit" (buy at the lowest, sell at the highest) algorithm to determine the optimal entry/exit strategy for the rest of the day.


---

## 💡 Motivation

Finance may not be my specialty, but a computer's ability to recognize patterns and efficiently process large amounts of numerical data is unmatched. This project is driven by the belief that **Reinforcement Learning** can discover non-linear and profitable trading opportunities that traditional models often miss.