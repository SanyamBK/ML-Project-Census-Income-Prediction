
# Census Income Prediction

This project aims to predict whether an individual earns more than $50K annually based on demographic and employment-related attributes from the UCI Adult dataset. We compare and evaluate multiple machine learning models to determine the most effective approach.

## 📊 Dataset

The dataset used is the **Adult Income Dataset** from the UCI Machine Learning Repository:
- 32,561 training instances and 16,281 test instances
- Features include age, workclass, education, marital status, occupation, race, sex, hours-per-week, and more
- Target variable: `>50K` or `<=50K`

## 🛠️ Features Used
- Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- Numerical: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

## 📦 Models Implemented

| Model                | Description                              |
|---------------------|------------------------------------------|
| Logistic Regression | Baseline linear model                    |
| Random Forest       | Ensemble model with decision trees       |
| Multi-layer Perceptron (MLP) | Neural network for classification |
| Support Vector Machine (SVM) | Kernel-based classification       |

## ⚙️ Preprocessing
- Handled missing values (`?`) by removal or imputation
- Label encoding and one-hot encoding for categorical features
- Feature scaling for numerical values (where required)
- Train-test split (if not already provided)

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🔍 Results
| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 76.80%   |
| Random Forest       | 93.90%   |
| MLP                 | 82.80%   |
| SVM                 | 89.80%   |


## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/SanyamBK/ML-Project-Census-Income-Prediction.git
cd ML-Project-Census-Income-Prediction
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebooks to train and evaluate models.

## 📌 Future Work

* Hyperparameter tuning
* Feature selection and dimensionality reduction
* Handling class imbalance
* Deployment using FastAPI or Streamlit

## 📄 License

This project is for academic use and learning purposes.

---
