# ❤️ Heart Disease Prediction Using Machine Learning

A Machine Learning project that predicts the likelihood of heart disease based on health-related input features. This project compares five classification algorithms and uses **Logistic Regression** as the final selected model, integrated into an interactive Streamlit web application.

> **Disclaimer:** This project is intended for educational purposes only. Its predictions are not a medical diagnosis and should not replace professional medical advice.

## 📌 Project Overview

Heart disease is a major health concern worldwide. Machine Learning can help identify patterns in health-related data and classify observations based on those patterns.

The goal of this project is to explore a heart disease dataset, train multiple classification models, compare their performance, select the best-performing model, and build a web application for interactive predictions.

## 🎯 Project Objectives

- Explore and preprocess the heart disease dataset.
- Train multiple Machine Learning classification algorithms.
- Compare models using accuracy and F1-score.
- Select the best-performing model based on experimental results.
- Save the trained model and preprocessing objects using Joblib.
- Build an interactive prediction interface using Streamlit.

## 🛠️ Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Matplotlib and Seaborn** — data visualization
- **Joblib** — model persistence
- **Streamlit** — interactive web application

## 📊 Machine Learning Workflow

1. Load and explore the dataset.
2. Perform data preprocessing and prepare the input features.
3. Split the dataset into training and testing sets.
4. Train five classification algorithms.
5. Evaluate and compare their performance.
6. Select Logistic Regression based on the observed results.
7. Save the trained model, scaler, and expected feature columns.
8. Integrate the saved files into the Streamlit application.

## 🤖 Models Compared

The following five classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

### Model Performance Comparison

| Model | Accuracy | F1-Score |
|---|---:|---:|
| **Logistic Regression** | **86.96%** | **88.57%** |
| K-Nearest Neighbors (KNN) | 86.41% | 88.15% |
| Naive Bayes | 85.33% | 86.83% |
| Support Vector Machine (SVM) | 84.78% | 86.79% |
| Decision Tree | 76.63% | 79.02% |

*The values above are based on the model evaluation results obtained in this project.*

## 🏆 Final Model Selection: Logistic Regression

After comparing the five classification algorithms, **Logistic Regression was selected as the final model** because it achieved the highest accuracy and F1-score in the evaluation.

- **Accuracy:** 86.96%
- **F1-Score:** 88.57%

KNN achieved the second-highest performance, with 86.41% accuracy and an 88.15% F1-score. Logistic Regression performed slightly better on both metrics in this experiment.

This comparison demonstrates the importance of evaluating multiple algorithms rather than choosing a model solely because it is more complex.

## 📈 Model Evaluation

The models were compared using the following metrics:

- **Accuracy:** The proportion of all predictions that were correct.
- **F1-Score:** The harmonic mean of precision and recall, balancing both metrics.
- **Confusion Matrix:** A breakdown of correct and incorrect positive and negative predictions.
- **Classification Report:** A summary of precision, recall, F1-score, and support for each class.

Accuracy and F1-score were used in the reported model comparison. The confusion matrix and classification report provide additional information about the selected model's performance.

## 🖥️ Streamlit Web Application

The project includes an interactive Streamlit application that allows users to enter health-related information and obtain a prediction from the saved Logistic Regression model.

The application loads the previously trained model and its preprocessing objects instead of retraining the model every time the app runs.

### Application Screenshot

Add a screenshot of your running Streamlit application here to help visitors understand the interface.

## 📁 Project Structure

```text
heart-disease-prediction/
├── app.py
├── heart.ipynb
├── heart.csv
├── LR_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application.
- `heart.ipynb` — notebook for data exploration, model training, and evaluation.
- `heart.csv` — dataset used in the project.
- `LR_heart.pkl` — saved Logistic Regression model.
- `scaler.pkl` — saved fitted scaler.
- `columns.pkl` — saved expected feature columns.
- `requirements.txt` — Python dependencies.
- `README.md` — project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/priyanaulakha1/heart-disease-prediction.git
```

### 2. Navigate to the project directory

```bash
cd heart-disease-prediction
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser, usually at `http://localhost:8501`.

## 🚀 Future Improvements

- Deploy the Streamlit application for public access.
- Add visualizations comparing model performance.
- Explore cross-validation and hyperparameter tuning.
- Improve input validation and application usability.
- Evaluate the model on suitable independent datasets.

## 👩‍💻 Author

**Priya Naulakha**

- GitHub: [priyanaulakha1](https://github.com/priyanaulakha1)
- LinkedIn: [Priya Naulakha](https://www.linkedin.com/in/priya-naulakha-89b0a331/)

---

*Built to practice Machine Learning classification, model evaluation, and deployment using Streamlit.*
