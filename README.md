# 🛒 Customer Conversion Prediction

An intelligent machine learning system to predict whether a user will complete a purchase based on their behavior.

---

## 📊 Project Overview

- **Goal**: Predict customer conversions (purchase = 1, no purchase = 0).
- **Dataset**: 400,000+ samples, behavior-based features.
- **Models Used**: Logistic Regression, Random Forest (with hyperparameter tuning).
- **Final Model**: Optimized Random Forest with ROC AUC ≈ **0.9974**.

---

## 🧠 Technologies

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- ipywidgets (optional dashboard)
- joblib (for saving models)

---

## 🗂 Project Structure
```
Customer_conversion_predictor/
├── data/
│   ├── training_sample.csv
│   └── testing_sample.csv
├── models/
│   └── best_random_forest_model.pkl
├── notebooks/
│   └── Customer_Conversion_Predictor.ipynb
├── plots/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── purchase_distribution.png
│   ...
├── README.md
└── requirements.txt

```

---

## 📈 Key Visualizations

### 📌 Correlation Heatmap
Shows the relationships between different behavioral features.

### 📌 Purchase Distribution
Unbalanced classes — significantly more non-purchases than purchases.

### 📌 Feature Importance
Key features influencing purchase prediction:
- `saw_checkout`
- `sign_in`
- `basket_add_detail`
- `returning_user`

---

## 🚀 Model Performance

| Metric        | Score         |
|---------------|---------------|
| Accuracy      | 99.24%         |
| Precision     | 85.26%         |
| Recall        | 99.07%         |
| F1 Score      | 91.65%         |
| ROC AUC       | 99.74%         |

✅ The optimized Random Forest model achieved very strong performance on an imbalanced dataset.

---

## 🏗 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Pashokkkk/customer_conversion_predictor.git
cd customer_conversion_predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook (`notebooks/Customer_Conversion_Predictor.ipynb`) or load the saved model:
```python
import joblib
model = joblib.load('models/best_random_forest_model.pkl')
```

---

## ✨ Future Improvements

- Apply SMOTE or ADASYN oversampling to improve minority class (purchase) prediction.
- Try advanced models like XGBoost or LightGBM.
- Deploy as a simple API using Flask/FastAPI.
- Create a web dashboard for real-time predictions.

---

## 📌 Author
**Pavlo Khomliuk**  
- GitHub: [Pashokkkk](https://github.com/Pashokkkk)  
- LinkedIn: [Pavlo Khomliuk](https://www.linkedin.com/in/pavlo-khomliuk-234799251/)

---
