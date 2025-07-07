<div align="center">

# 🌦️ AI Weather Predictor 🤖

### An Interactive Machine Learning App with Python & Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.x-purple?style=for-the-badge&logo=pandas)

</div>

> A data-driven web application that forecasts tomorrow's maximum temperature using a Ridge Regression model. This project demonstrates a complete end-to-end machine learning workflow, from raw data cleaning and feature engineering to model training and deployment in a user-friendly, interactive web interface.

---

## 🚀 Live Demo

**Experience the live application here:**

### **[https://app-weather-predictor.streamlit.app/](https://app-weather-predictor.streamlit.app/)**

---

![image](https://github.com/user-attachments/assets/31614ffa-be75-47a0-8f5a-b6dd0b3627b0)


![image](https://github.com/user-attachments/assets/68ccc133-66c6-4994-9a5b-399efdad5c47)


![image](https://github.com/user-attachments/assets/bce0c7e4-545d-4fb6-88f1-b465919f49db)

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack & Architecture](#️-tech-stack--architecture)
- [🔧 The Machine Learning Workflow](#-the-machine-learning-workflow)
- [🎯 Model Performance](#-model-performance)
- [⚙️ How to Run Locally](#️-how-to-run-locally)
- [📞 Contact](#-contact)

---

## ✨ Key Features

-   **Interactive UI:** A clean and intuitive web interface created with Streamlit, allowing users to modify input values and receive instant predictions.
-   **Robust Data Processing:** Implements professional data cleaning techniques, including handling missing values (`NaN`) and infinity (`inf`) values.
-   **Advanced Feature Engineering:** Creates new predictive features from raw data, such as rolling 30-day averages and temperature ratios, to improve model accuracy.
-   **Time-Series Backtesting:** Utilizes a robust backtesting function to provide a realistic evaluation of the model's performance, preventing data leakage from future events.
-   **Optimized Performance:** Caches the expensive model training process using `@st.cache_data`, ensuring the app is fast and responsive for every user.
-   **Data Visualization:** Includes a simple bar chart to visually compare today's temperature with the model's prediction for tomorrow.

---

## 🛠️ Tech Stack & Architecture

This project leverages a modern, Python-based data science stack to deliver a seamless experience from data to deployment.

| Technology | Purpose |
| :--- | :--- |
| **Python** | The core programming language for all logic and model training. |
| **Pandas** | The primary library for efficient data loading, cleaning, and manipulation. |
| **Scikit-learn** | Used for training the `Ridge` regression model and evaluating its performance with Mean Absolute Error. |
| **NumPy** | For handling complex numerical operations and data transformations. |
| **Streamlit** | To rapidly build and deploy the interactive web application UI with Python alone. |

---

## 🔧 The Machine Learning Workflow

The project follows a structured machine learning pipeline to ensure robust and reliable predictions.

**`[weather.csv]` ➔ `[Data Cleaning]` ➔ `[Feature Engineering]` ➔ `[Model Training]` ➔ `[Streamlit UI]` ➔ `[Live Prediction]`**

1.  **Data Ingest:** The raw `weather.csv` dataset is loaded into a Pandas DataFrame.
2.  **Data Cleaning:** Key columns are selected, renamed for clarity, and missing or infinite values are handled to create a robust dataset.
3.  **Feature Engineering:** New, predictive features are created. This includes calculating rolling 30-day averages and key temperature ratios to give the model historical context.
4.  **Model Training:** A `Ridge` regression model is trained on the cleaned, feature-rich dataset. A backtesting function is used to validate the model for a realistic performance evaluation.
5.  **Prediction:** The trained model is served through the Streamlit interface, where it takes live user input and generates a prediction for tomorrow's maximum temperature.

---

## 🎯 Model Performance

After implementing feature engineering and robust backtesting, the final model achieves:

-   **Mean Absolute Error (MAE):** **4.98 degrees Fahrenheit**

This means that, on average, the model's prediction for tomorrow's maximum temperature is off by approximately 5 degrees. This score reflects a solid performance baseline for a simple model and demonstrates the effectiveness of the feature engineering process.

---

## ⚙️ How to Run Locally

To run this project on your own machine, please follow these steps:

#### **1. Prerequisites**
-   Python 3.8 or higher
-   `pip` and `venv` installed

#### **2. Clone the Repository**
```bash
git clone [https://github.com/your-username/streamlit-weather-predictor.git](https://github.com/your-username/streamlit-weather-predictor.git)
cd streamlit-weather-predictor
```
*(Replace `your-username` with your actual GitHub username)*

#### **3. Set Up the Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### **4. Install Dependencies**
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

#### **5. Run the Application**
Launch the Streamlit web server.
```bash
streamlit run predict.py
```
A new tab will automatically open in your browser at `http://localhost:8501`. You can now interact with the app!

---

## 📞 Contact

Your Name - `your.email@example.com`

Project Link: [https://github.com/your-username/streamlit-weather-predictor](https://github.com/your-username/streamlit-weather-predictor)

---
