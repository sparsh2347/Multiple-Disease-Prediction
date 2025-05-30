# 🧠 Multiple Disease Prediction System

This project is a **Streamlit-based Machine Learning Web App** that allows users to predict the likelihood of three diseases — **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** — using trained machine learning models.

---

## 📌 Overview

Each prediction module is based on structured health data and utilizes supervised learning techniques to classify whether the individual is affected by the respective disease. The system provides a compact and interactive interface for users to input features and receive instant diagnostic results.

---

## 💾 Technologies Used

- Python  
- Pandas, NumPy, Scikit-learn  
- Streamlit  
- Pickle (for saving models)  
- Matplotlib/Seaborn (optional for data exploration)

---

## 🔬 Model Details

### 🩺 Diabetes Prediction

**Dataset**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Features Used**:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

**Model Used**: Support Vector Machine (SVM)

**Model Pipeline**:
1. **Data Preprocessing**  
   - Replaced 0 values in critical columns (Glucose, BloodPressure, etc.) with medians.  
   - Standardized features using `StandardScaler`.

2. **Train-Test Split**  
   - 80% training and 20% testing data.

3. **Model Training**  
   - SVM Classifier was used.  
   - The hyperplane separates data points with a maximum margin to distinguish diabetic and non-diabetic cases.

4. **Evaluation**  
   - Accuracy score and confusion matrix used.  
   - Achieved over 75% accuracy.

5. **Model Saving**  
   ```python
   pickle.dump(model, open('diabetes.sav', 'wb'))
   pickle.dump(scaler, open('diabetes_scaler.pkl', 'wb'))

### ❤️ Heart Disease Prediction

**Dataset**: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

**Features Used**:
- Age
- Sex
- Chest Pain Type (`cp`)
- Resting Blood Pressure (`trestbps`)
- Serum Cholesterol (`chol`)
- Fasting Blood Sugar (`fbs`)
- Resting Electrocardiographic Results (`restecg`)
- Maximum Heart Rate Achieved (`thalach`)
- Exercise Induced Angina (`exang`)
- ST Depression Induced by Exercise (`oldpeak`)
- Slope of the Peak Exercise ST Segment (`slope`)
- Number of Major Vessels Colored by Fluoroscopy (`ca`)
- Thalassemia (`thal`)
- Target (0 = No Disease, 1 = Disease)

**Model Used**: Logistic Regression

**Model Pipeline**:
1. **Data Preprocessing**
   - Checked and handled missing/null values
   - Converted categorical values to numerical using Label Encoding/One-Hot Encoding where necessary
   - Standardized numerical features using `StandardScaler` for consistency

2. **Train-Test Split**
   - Used an 80:20 split between training and testing data to evaluate generalization performance

3. **Model Training**
   - Logistic Regression was selected for its interpretability and effectiveness in binary classification tasks
   - Tuned hyperparameters like regularization (`C`) and solver method (`liblinear`)

4. **Model Evaluation**
   - Achieved accuracy of **X%**
   - Used classification report metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
   - ROC-AUC Curve to evaluate class separability

5. **Saving the Model**
   ```python
   import pickle
   pickle.dump(model, open('heart.sav', 'wb'))

---
### 🧠 Parkinson's Disease Detection

**Dataset**: [UCI Parkinson’s Disease Dataset](https://www.kaggle.com/datasets/karthickveerakumar/parkinsons-disease-data-set)

**Overview**:
Parkinson’s Disease is a progressive nervous system disorder that affects movement. This project focuses on early detection using vocal features extracted from biomedical voice measurements.

**Features Used (Excerpt)**:
- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- MDVP:Jitter(%) and related jitter measures
- MDVP:Shimmer and related shimmer measures
- NHR, HNR (noise-to-harmonics and harmonics-to-noise ratio)
- RPDE, DFA (nonlinear dynamical complexity measures)
- Spread1, Spread2, D2
- Status (Target: 1 = Parkinson’s, 0 = Healthy)

**Model Used**: Support Vector Machine (SVM)

**Model Pipeline**:
1. **Data Preprocessing**
   - Dropped the `name` column (non-numeric identifier)
   - Checked for null/missing values
   - Standardized all numerical features using `StandardScaler`

2. **Train-Test Split**
   - 80% training and 20% testing split for robust model validation

3. **Model Training**
   - Used **Support Vector Machine (SVM)** with a linear kernel
   - SVM was chosen for its performance on small, high-dimensional datasets

4. **Model Evaluation**
   - Achieved accuracy of **X%**
   - Evaluated with:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Confusion matrix and ROC curve used for visual diagnostics

5. **Saving the Model**
   ```python
   import pickle
   pickle.dump(model, open('parkinsons.sav', 'wb'))

---
## 🖥️ Web App Features

This Streamlit-based web application enables users to predict the presence of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using pre-trained Machine Learning models. Below are the key features of the application:

- 🎯 **Multi-Disease Prediction**  
  Predicts three different diseases based on relevant medical inputs.

- 🧾 **Clean and Dynamic Form Layout**  
  The user input forms are structured using Streamlit's column layout for an organized and intuitive UI.

- 💾 **In-App Prediction Logging**  
  Stores and displays previous predictions made during the current session using Streamlit’s `session_state`.

- 📊 **Interactive Dashboard**  
  View all previous predictions made in the session with input values, predicted disease, and result in a tabular format.

- 🧑‍💻 **User-Friendly Interface**  
  Simple, responsive, and beginner-friendly design suitable for educational or prototype-level use.

- ⚙️ **Modular Design**  
  Code is structured by disease modules (Diabetes, Heart, Parkinson), making it scalable and easy to add more models in the future.

---

## 📁 Project Structure

The project is organized into folders for each disease, storing trained models and scalers, along with the main application file.

```plaintext
Multiple Disease Prediction/
│
├── Diabetes/
│ ├── diabetes.sav # Trained SVM model for Diabetes
│ ├── diabetes_scaler.pkl # StandardScaler used for preprocessing
│
├── Heart/
│ └── heart.sav # Trained Logistic Regression model
│
├── Parkinson/
│ ├── parkinson.sav # Trained SVM model for Parkinson's
│ └── parkinson_scalar.pkl # StandardScaler used for preprocessing
│
├── app.py # Main Streamlit application file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
```
---

## 🚀 Run Locally

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multiple-disease-prediction.git
cd multiple-disease-prediction
```

### 2. Install Required Packages
Make sure you have Python 3.7+ installed, then install the required packages:

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App
Use the command below to launch the app in your browser:

```bash
streamlit run app.py
```
This will open a new browser window/tab with the web app running locally.

