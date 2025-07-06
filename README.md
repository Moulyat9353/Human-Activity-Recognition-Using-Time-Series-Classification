## Human-Activity-Recognition-Using-Time-Series-Classification
This project implements machine learning approaches to classify human activities based on time-series data collected from a wireless sensor network (AReM dataset).  

## 📂 Dataset

- Source: AReM dataset on UCI ML Repository: https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem

- 7 activity folders (bending1, bending2, cycling, lying, sitting, standing, walking)

- Each file = one recording instance

- Each file contains 6 time-series signals:

  - avg_rss12

  - var_rss12

  - avg_rss13

  - var_rss13

  - avg_rss23

  - var_rss23

Each time series has 480 data points

## 📝 Problem Statement
- Binary classification:

   Classify bending vs. other activities using time-domain features.

- Multi-class classification:

   Classify all 7 activities.

- The pipeline explores:

  - Time-domain feature extraction

  - Logistic Regression

  - L1-penalized Logistic Regression

  - Naive Bayes classification

  - Dimensionality reduction (PCA)
 
## 🔬 Project Workflow

# 1. Data Preparation

- Train-test split:

  - Test data:

    - Bending1 → datasets 1 & 2

    - Bending2 → datasets 1 & 2

  - Other activities → datasets 1, 2, and 3

  All other data used for training.
  

# 2. Time-Domain Feature Extraction
   
Extracted features from each time series:

- Minimum

- Maximum

- Mean

- Median

- Standard Deviation

- First Quartile

- Third Quartile

✅ Confidence Intervals:

Bootstrap-based 90% confidence intervals for the standard deviation of each feature.

# 3. Feature Selection

- Selected three key features (e.g. min, mean, max) based on domain insights and data analysis.

- Visualized scatterplots for bending vs. other activities:

  - Time series 1, 2, and 6

# 4. Binary Classification

# Logistic Regression

- Explored:

  - Splitting each time series into:

    - 2 segments

    - l = 12 … 20 segments

- Feature selection:

  - p-values or recursive feature elimination (RFE)

- Used 5-fold cross-validation

- Addressed class imbalance via:

  - Stratified cross-validation

  - Case-control sampling

- Metrics reported:

  - Confusion matrix

  - Accuracy

  - ROC curve

  - AUC

  - Model coefficients and p-values

# L1-Penalized Logistic Regression

- Cross-validated for:

  - l (number of segments)

  - C (inverse regularization strength)

- Compared L1 vs. classical feature selection:

  - Performance

  - Implementation complexity

# 5. Multi-class Classification

# L1-Multinomial Logistic Regression

- Cross-validated to find best l

- Reported:

  - Test error

  - Confusion matrix

  - ROC curves (if feasible for multiclass)

# Naive Bayes

- Gaussian and Multinomial variants compared.

# PCA + Naive Bayes

- Reduced feature space

- Visualized 1st and 2nd principal components

- Evaluated classification performance

## 💡 Key Insights

- Time-domain features carry strong signal for distinguishing activities.

- Logistic regression with proper feature selection performs well even for small segments.

- L1 regularization simplifies model while preserving accuracy.

- Naive Bayes is quick to implement but less robust with overlapping classes.

- PCA visualization helps identify separability among activities.

## Libraries

- pandas

- numpy

- scikit-learn

- matplotlib

- seaborn
