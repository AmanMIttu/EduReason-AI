# EduReason AI
### Explainable Hybrid AI for Student Success Prediction and Academic Decision Support

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

## 1. Overview

EduReason AI is an explainable hybrid artificial intelligence project designed to analyse student academic data and support early identification of students who may be at risk of dropping out.

The project combines machine learning-based prediction with symbolic, rule-based reasoning to produce predictions alongside interpretable explanations and potential academic support recommendations.

Rather than treating predictions as black-box outputs, EduReason AI aims to make the decision-making process more transparent and understandable.

## 2. Problem Statement

Educational institutions collect substantial amounts of student information, but identifying students who may struggle academically at an early stage remains challenging.

Traditional predictive models can identify patterns associated with academic outcomes, but their predictions may be difficult for educators to interpret.

EduReason AI explores how predictive machine learning and transparent reasoning rules can work together to support more understandable academic decision-making.

## 3. Project Objectives

- Predict student academic outcomes using machine learning.
- Explore early identification of students at risk of dropping out.
- Explain model predictions using Explainable AI techniques.
- Develop rule-based reasoning to complement predictive outputs.
- Investigate a hybrid approach combining statistical learning and symbolic reasoning.

## 4. Dataset

This project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

- Number of records: 4,424
- Input features: 36
- Target classes:
  - Dropout
  - Enrolled
  - Graduate

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## 5. Proposed Architecture

The project follows a modular pipeline:

1. **Data Processing:** Data loading, inspection, cleaning and preprocessing.
2. **Feature Engineering:** Selection and transformation of relevant student attributes.
3. **Machine Learning:** Training and evaluating predictive classification models.
4. **Explainability:** Using SHAP to investigate feature contributions to model predictions.
5. **Symbolic Reasoning:** Applying interpretable rules to derive additional insights.
6. **Hybrid Decision Support:** Combining model predictions and reasoning outputs.
7. **Dashboard:** Presenting predictions and explanations through an interactive interface.

## 6. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Explainable AI | SHAP |
| Interface | Streamlit |
| Version Control | Git, GitHub |

## 7. Current Progress

- [x] Repository and project structure created.
- [x] Dataset downloaded from the UCI repository.
- [x] Initial exploratory data analysis completed.
- [x] Target distribution and missing-value checks completed.
- [x] Age-based and academic-performance patterns explored.
- [ ] Data preprocessing and feature engineering.
- [ ] Baseline machine learning models.
- [ ] Model evaluation and comparison.
- [ ] SHAP-based model explanations.
- [ ] Symbolic reasoning implementation.
- [ ] Hybrid decision-support pipeline.
- [ ] Interactive dashboard.

## 8. Research Direction

An important aspect of this project is evaluating predictions using only information available at the intended prediction time.

For example, an early-warning model designed to operate at the end of the first semester should not use second-semester information. This helps avoid data leakage and makes the experimental setup more realistic.

The project will also investigate how explainability and symbolic reasoning can complement conventional predictive models.

## 9. Repository Structure

```text
EduReason-AI/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_initial_data_exploration.ipynb
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   └── reasoning/
├── tests/
├── models/
├── reports/
├── docs/
├── README.md
├── LICENSE
└── .gitignore
```

## 10. Project Status

**Currently under active development.**

This project is an exploration of explainable and hybrid AI methods for educational decision support. Model performance, experimental findings and dashboard capabilities will be documented as development progresses.

---

*Developed as an independent AI and Machine Learning project.*
