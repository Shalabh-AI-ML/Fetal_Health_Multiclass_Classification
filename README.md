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

Below is the link to github repository. 

[Fetal Health Multiclass Classification (click to open github URL)](https://github.com/Shalabh-AI-ML/Fetal_Health_Multiclass_Classification.git)

Repository contains:
- model - This is a folder where the code for all the 5 models implementation is available.
- saved_model - This folder contains the pretrained models which are also validated after training. It also contains the normalization technique used for training and validation data.
- app.py - This is a initialization of the streamlit app which holds the logic for choosing the models from the drop down list and displaying the results
- data_processing.py - This file is used for preprocessing fetal_health.csv file and dividing the dataset into train and validation.
- fetal_health.csv - This is the csv file contains the training and validation data.
- test_data.csv - This file is a test data file that needs to be uploaded from the streamlit app.
- README.md

## 4. Machine Learning Models Used
### 4.1 Logistic Regression:
Logistic Regression is a linear classification algorithm used to predict the probability of observations belonging to different classes.

For this project, multiclass Logistic Regression is used to classify fetal health into Normal, Suspect, and Pathological categories.

### 4.2 Decision Tree
Decision Tree is a tree-based classification algorithm that makes predictions by recursively splitting the dataset based on feature values.
The resulting tree structure consists of:
- Root node
- Decision nodes
- Branches
- Leaf nodes

Decision Trees are easy to interpret and can capture nonlinear relationships between features.

### 4.3 k-Nearest Neighbors (kNN)
kNN is a distance-based classification algorithm.

For a new observation, kNN:

1. Calculates the distance between the new observation and training observations.
2. Identifies the k nearest observations.
3. Uses the majority class among the neighbors as the prediction.

Feature scaling is important for kNN because it is based on distance calculations.

### 4.4 Naive Bayes
Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that the features follow a Gaussian (normal) distribution within each class.

The model calculates the probability of each fetal health class and selects the class with the highest probability.

### 4.5 Random Forest
Random Forest is an ensemble learning algorithm consisting of multiple Decision Trees.

Each tree is trained using different samples/features, and the final prediction is obtained by combining the predictions of the individual trees.

Random Forest is generally effective for datasets containing nonlinear relationships and interactions between features.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :---| :---| :--- | :---| :---|:--- |:---|
| **Logistic Regression**| 0.89| 0.96| 0.90| 0.89| 0.89| 0.71|
| **Descision Tree**| 0.98| 0.97| 0.98| 0.98| 0.98| 0.94|
| **kNN**| 0.90| 0.96| 0.89| 0.90| 0.89| 0.72|
| **Naive Bayes**| 0.70| 0.86| 0.86| 0.70| 0.74| 0.49|
| **Rambom Forest (Ensemble)**| 0.97| 0.99| 0.97| 0.97| 0.97| 0.93|
