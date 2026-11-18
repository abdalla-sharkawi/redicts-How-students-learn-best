# 🎓 Predicting How Students Learn Best

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-8A2BE2?style=for-the-badge)
![ANN](https://img.shields.io/badge/Neural_Network-00BFA6?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **Predicting Student Academic Performance Using Artificial Neural Networks**

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [Project Objectives](#-project-objectives)
- [Dataset Description](#-dataset-description)
- [Learning Methods](#-learning-methods)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Data Preprocessing](#-data-preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Neural Network Architecture](#-neural-network-architecture)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Results and Findings](#-results-and-findings)
- [Visualizations](#-visualizations)
- [Installation and Usage](#-installation-and-usage)
- [Future Improvements](#-future-improvements)
- [Limitations](#-limitations)
- [Project Information](#-project-information)

---

## 📖 About The Project

**Predicting How Students Learn Best** is a Machine Learning project that uses an **Artificial Neural Network (ANN)** to predict students' academic marks based on their learning habits, academic background, and other related factors.

The project explores the relationship between students' study behaviors and their academic performance. It also investigates how different learning methods are associated with students' predicted marks.

A Neural Network regression model is developed using TensorFlow and Keras to learn patterns from the dataset and generate academic mark predictions.

The project demonstrates how Artificial Intelligence and Deep Learning can be applied to educational data analysis and predictive modeling.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Analyze the relationship between students' study habits and academic performance.
- Explore the impact of different learning-related features on academic marks.
- Compare average marks across different learning methods.
- Identify the features that have stronger correlations with students' marks.
- Develop an Artificial Neural Network regression model.
- Train the model using student-related features.
- Evaluate the model using appropriate regression metrics.
- Visualize actual and predicted values to understand model performance.

---

## 📊 Dataset Description

The project uses a **synthetic dataset** generated using NumPy.

The dataset contains 3,000 records representing simulated student information.

### Dataset Overview

| Property | Description |
|:---|:---|
| Dataset Type | Synthetic Dataset |
| Number of Records | 3,000 |
| Input Features | 6 |
| Target Variable | `marks` |
| Problem Type | Regression |

### Dataset Features

The dataset contains the following features:

| Feature | Type | Description |
|:---|:---|:---|
| `study_hours` | Numerical | Number of hours spent studying |
| `sleep_hours` | Numerical | Number of hours spent sleeping |
| `attendance` | Numerical | Student attendance percentage |
| `prev_gpa` | Numerical | Previous academic GPA |
| `distraction` | Numerical | Student distraction level |
| `method` | Categorical | Learning method used by the student |
| `marks` | Target | Student's academic marks |

### Target Variable

The target variable is:

```python
marks
