# Loan Approval Predictor

An end-to-end Machine Learning project that leverages demographic, financial, and credit history data to accurately predict whether a loan application will be approved or rejected. The pipeline includes extensive data cleaning, robust group-based mode/median imputation for missing values, feature engineering, and a scaled Logistic Regression classifier.

## Performance Highlights
* **Validation Accuracy:** 84.86%
* **Precision (Approved Loans - Class 1):** 0.84
* **Recall (Approved Loans - Class 1):** 0.97
* **F1-Score (Approved Loans - Class 1):** 0.90

---

## Project Structure

* [cite_start]`Train.csv` - The original training dataset containing historical applicant details and target loan statuses[cite: 3].
* [cite_start]`Test.csv` - The unseen test dataset used to evaluate the final deployment readiness of the model[cite: 2].
* [cite_start]`Final Loan Predictions.csv` - The generated output file containing the final predicted loan approval statuses (`Y`/`N`) mapped back to their respective `Loan_ID`s[cite: 1].
* `loan_prediction.ipynb` - The complete Jupyter Notebook documenting data exploration, custom data preprocessing, modeling, and inference pipelines.

---

## Features & Engineering Workflow

To maximize data integrity and optimize model stability, the following preprocessing steps are executed:

1. **Targeted Missing Value Imputation:**
   * **Gender/Marital Status:** Filled missing values dynamically using group-by modes based on correlated traits like education, income, or co-applicant presence.
   * **Loan Amount:** Imputed missing values using the group median calculated across `Is_Graduate` and `Is_Self_Employed` cohorts.
   * **General Defaults:** Replaced standard missing values with global modes for `Loan_Amount_Term`, `Credit_History`, and `Dependents`.
2. **Feature Engineering:**
   * Created a binary helper feature `Has_Coapplicant` derived from `CoapplicantIncome`.
   * Explicitly converted multi-class textual metrics like `Dependents` (`3+` parsed to `3`) into numeric segments.
3. **Encoding & Scaling:**
   * Maps textual categorical variables (`Gender`, `Married`, `Education`, `Self_Employed`, `Loan_Status`) to binary flags (`0` or `1`).
   * One-hot encodes geospatial features (`Property_Area_Semiurban`, `Property_Area_Urban`).
   * Applies a standard scaler to continuous variables (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`) to avoid scale bias.

---

## Evaluation Report

The Logistic Regression model demonstrates strong predictive capabilities on stratified validation data:

```text
               precision    recall  f1-score   support

           0       0.89      0.59      0.71        58
           1       0.84      0.97      0.90       127

    accuracy                           0.85       185
   macro avg       0.87      0.78      0.80       185
weighted avg       0.85      0.85      0.84       185
```
## Getting Started

### Prerequisites
Make sure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
### Run the Pipeline
1. Clone this repository to your local machine.
2. Ensure `Train.csv` and `Test.csv` are in your working directory.
3. Execute the script or notebook. The model will ingest `Train.csv`, train on it, map identical transformations onto `Test.csv`, and output your final predictions directly into `Final Loan Predictions.csv`.
