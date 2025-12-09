# Heart Disease Prediction using Machine Learning

## 📌 Project Overview
Heart disease remains one of the leading causes of death worldwide. Early detection and risk assessment can significantly reduce fatality rates by enabling timely medical intervention. This project involves developing a machine learning-based predictive model capable of identifying whether a person is likely to suffer from heart disease based on clinical and physiological data.

Using historical medical data, the model learns patterns and relationships among different attributes—such as age, cholesterol level, and blood pressure—to predict the presence or absence of heart disease.

## 📂 Dataset Description
The dataset contains **303 rows** and **14 attributes** sourced from publicly available medical records.

### Key Attributes:
* **age**: Age of the patient.
* **sex**: Gender (0 = female, 1 = male).
* **cp**: Chest pain type (4 categories).
* **trestbps**: Resting blood pressure (mm Hg).
* **chol**: Serum cholesterol (mg/dl).
* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
* **restecg**: Resting electrocardiographic results.
* **thalach**: Maximum heart rate achieved.
* **exang**: Exercise-induced angina (1 = yes, 0 = no).
* **oldpeak**: ST depression induced by exercise.
* **slope**: Slope of the peak exercise ST segment.
* **ca**: Number of major vessels colored by fluoroscopy.
* **thal**: Defect type (normal, fixed, reversible).
* **target**: Presence of heart disease (1 = disease, 0 = no disease).

## ⚙️ Methodology
The project followed a standard machine learning pipeline:
1.  **Data Loading & Preprocessing:** * Checked for missing values.
    * Encoded categorical attributes (One-Hot Encoding).
    * Normalized numerical features using Standard Scaler.
    * Split dataset into 80% training and 20% testing.
2.  **Exploratory Data Analysis (EDA):** * Identified that `cp` (chest pain), `thalach` (max heart rate), and `oldpeak` show strong correlations with heart disease.
3.  **Model Training:** Trained 5 different classifiers to compare performance.
4.  **Evaluation:** Metrics included Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## 📊 Results & Performance
I implemented and compared five different algorithms. **Random Forest** achieved the best performance due to its ability to handle non-linear data and prevent overfitting.

| Model | Accuracy | Observations |
| :--- | :--- | :--- |
| **Random Forest** | **~90%** | **Best performance**, robust against overfitting. |
| SVM | ~88% | Effective for high-dimensional data. |
| Logistic Regression | ~85% | Good baseline for linear relationships. |
| KNN | ~82% | Performance varied significantly based on 'K' value. |
| Decision Tree | ~80% | Showed tendencies to overfit without pruning. |

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TomarBondhuRashed/-Heart-Disease-Prediction](https://github.com/TomarBondhuRashed/-Heart-Disease-Prediction)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```

## 🔮 Future Improvements
* Implement advanced ensemble models like **XGBoost** or **Gradient Boosting**.
* Perform extensive **hyperparameter tuning** (GridSearch/RandomSearch).
* Deploy the model as a web application using **Streamlit** or **Flask**.

---
**Repository:** [Heart Disease Prediction](https://github.com/TomarBondhuRashed/-Heart-Disease-Prediction)
