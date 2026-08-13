# Fetal Health Classification

## 1. Problem Statement:
Fetal health is an important task in prenatal care. Cardiotocography (CTG) is commonly used to monitor fetal heart rate and uterine contractions. The resulting measurements can be analyzed using machine learning techniques to classify fetal health conditions.

The objective of this project is to develop and compare multiple machine learning classification models for predicting fetal health based on Cardiotocography (CTG) features.

The fetal health classification consists of three target classes:

**1 - Normal**\
**2 - Suspect**\
**3 - Pathological**

We have implemented and evaluated the following models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

The above mentioned models are evaluated based on below metrics:

 - Accuracy
 - Precision
 - AUC Score
 - Recall
 - F1 Score
 - Matthews Correlation Coefficient (MCC)

 The objective is to identify the model that provides the best overall performance for the fetal health classification.

 ---

## 2. Data set description

The project uses **Fetal Health Classification dataset**.

The dataset contains Cardiotocography (CTG) measurements used to classify fetal health into three categories: Normal, Suspect, and Pathological.

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | Fetal Health Classification |
| Data Type | Numerical |
| Problem Type | Multiclass Classification |
| Target Variable | `fetal_health` |
| Number of Classes | 3 |
| Class 1 | Normal |
| Class 2 | Suspect |
| Class 3 | Pathological |

The dataset contains **2,126 observations** and **22 input features**, along with the target variable `fetal_health`.

### Input Features

The dataset includes CTG-related measurements such as:

- Baseline value
- Accelerations
- Fetal movement
- Uterine contractions
- Light decelerations
- Severe decelerations
- Prolonged decelerations
- Abnormal short-term variability
- Mean value of short-term variability
- Percentage of time with abnormal long-term variability
- Mean value of long-term variability
- Histogram width
- Histogram minimum
- Histogram maximum
- Histogram number of peaks
- Histogram number of zeroes
- Histogram mode
- Histogram mean
- Histogram median
- Histogram variance
- Histogram tendency

The target column is:

```text
fetal_health
```

## 3. Github Repository Link

```text
fetal_health
```