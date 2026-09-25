# Salary-prediction-linear-regression
A simple Machine Learning model built with Python and Scikit-Learn that predicts salary based on years of experience using Linear Regression.
# 📈 Salary Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A clean and well-structured Machine Learning project implementing **Linear Regression** from scratch using Python and Scikit-Learn. The model predicts an employee's salary based on their years of professional experience.

---

## 📌 Project Overview

The objective of this project is to model the linear relationship between years of experience (Independent Feature $X$) and salary (Dependent Target $y$). Using Supervised Learning, the algorithm finds the **Line of Best Fit** ($y = mx + c$) to make accurate future predictions.

### Key Objectives:
- Demonstrate Machine Learning workflow best practices.
- Implement data splitting into **Training** and **Testing** sets.
- Evaluate model accuracy using standard metrics ($R^2$ Score and Mean Squared Error).

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:**
  - `scikit-learn` — Model training, dataset splitting, evaluation metrics
  - `numpy` — Numerical operations and array manipulation

---

## ⚙️ Project Workflow

1. **Data Structuring:** Feature matrix ($X$) and target vector ($y$) definition.
2. **Train/Test Split:** Data partitioned into **80% Training** and **20% Testing** subsets with `random_state=42` for exact reproducibility.
3. **Model Training:** `LinearRegression()` model fitted on training data to learn the optimal slope ($m$) and intercept ($c$).
4. **Prediction & Evaluation:** Tested model performance on unseen test data (`X_test`) using **MSE** and **$R^2$ Score**.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. Install the necessary libraries using `pip`:

```bash
pip install numpy scikit-learn
