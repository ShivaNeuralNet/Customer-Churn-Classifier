# 🛍️ E-Commerce Customer Retention Classification

## 🔗 Objective
This project focuses on building a **supervised machine learning classification model** to predict whether a customer will be retained (subscribe) or churn (not subscribe) based on their demographic, transactional, and behavioral features. The insights derived from this model aim to help e-commerce businesses tailor loyalty programs and marketing strategies to improve retention and revenue.

---

##  Dataset Overview
The dataset includes features categorized as:

- **Demographics**: Age, Gender, Location  
- **Purchase Behavior**: Item Purchased, Category, Purchase Amount (USD), Previous Purchases  
- **Customer Engagement**: Review Rating, Subscription Status, Discount Applied, Promo Code Used  
- **Shipping & Payment**: Shipping Type, Payment Method  

---

## 📊 Exploratory Data Analysis (EDA)
- Checked for **duplicates**, **missing values**, and **outliers**
- Explored numerical statistics using `.describe()`
- Used **pairplots** and **correlation analysis** (Pearson and Spearman) to check linear and monotonic relationships

### Key Findings:
- No strong linear or monotonic relationship between numerical features
- Subscription status is **imbalanced**: 73% No, 27% Yes
- Categorical analysis showed:
  - Promo code usage and frequent purchases correlate with higher subscriptions
  - Discounts and free shipping are not strong incentives for subscription
  - Clothing and accessories had higher non-subscription rates

---

## Feature Engineering & Preprocessing
- Categorical features: One-hot encoded
- Ordinal features: Ordinal encoded with custom order (e.g., Size, Season, Frequency)
- Binary features: Ordinal encoded
- Numerical features: Standard scaled
- Built a **ColumnTransformer** pipeline for clean preprocessing

---

## Models Trained
Six classification models were evaluated using 5-fold cross-validation:

- Dummy Classifier (baseline)
- Logistic Regression ✅ Best Model
- SVM (RBF Kernel)
- Random Forest
- KNN
- Decision Tree

### Evaluation Metrics:
| Model                  | Accuracy | F1 Score | Recall | Precision | AUC     |
|------------------------|----------|----------|--------|-----------|---------|
| Logistic Regression    | 83.43%   | 76.52%   | 100%   | 61.97%    | **0.91** |
| SVM (RBF)              | 83.40%   | 76.48%   | 100%   | 61.92%    | -       |
| Random Forest          | 82.08%   | 70.92%   | 81.00% | 63.10%    | -       |
| Decision Tree          | 80.32%   | 62.65%   | 61.29% | 64.22%    | -       |
| KNN                    | 79.74%   | 62.23%   | 61.88% | 62.63%    | -       |
| Dummy Classifier       | 61.15%   | 29.89%   | 30.64% | 29.23%    | -       |

---

###  Handling Class Imbalance & Evaluation Strategies
Several resampling and adjustment techniques were applied to improve model performance on the imbalanced dataset (73% non-subscribers vs. 27% subscribers):

####  Evaluation of Techniques
- **Original Model**:
  - High accuracy and recall
  - Lower precision → Model predicts many false positives
  - AUC shows good class separation

- **Balanced Resampling**:
  - Perfect recall (100% of subscribers identified)
  - Slightly lower precision
  - Strong AUC and high F1 score → balanced and effective

- **Class Weight Adjustment**:
  - Poor recall and precision
  - Very low F1 score
  - AUC significantly lower than other methods → not suitable

- **SMOTE (Synthetic Minority Over-sampling Technique)**:
  - Perfect recall
  - High F1 score
  - AUC improves → model better distinguishes classes

- **Undersampling**:
  - Perfect recall
  - High F1 score
  - Strong AUC, similar to SMOTE

####  Conclusion:
- **SMOTE**, **undersampling**, and the **balanced strategy** are the most effective techniques.
- These methods achieved perfect recall and high F1 scores.
- **Class weight adjustment** underperformed and is not recommended for this dataset.

---

### 🧲 ROC & AUC
- **AUC = 0.91** for Logistic Regression
- Model performs significantly better than random guessing (baseline)

###  Confusion Matrix
Visualized to better understand the prediction errors.

---

##  Conclusion
- **Logistic Regression** is the best performing model in terms of accuracy, recall, and AUC.
- **Recall = 1.00**, which means the model catches all retained (subscribed) customers.
- Model is ideal for business deployment where **customer retention** is key.

---

##  Tools & Libraries
- Python (Pandas, NumPy, Seaborn, Matplotlib, Plotly)
- Scikit-learn
- imbalanced-learn (SMOTE, RandomUnderSampler)

---

##  Future Improvements
- Apply advanced models (XGBoost, Gradient Boosting)
- Analyze feature interactions


---

## 🤝 About the Author
**Shiva Dorri**  
Data Scientist | Background in Accounting & Finance | Passionate about solving business problems with machine learning and data analysis.

> "I believe that real impact comes from turning data into decisions."
