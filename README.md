# 🌍 Delhi Air Pollution Analysis using Machine Learning

## 📌 Project Overview

This project focuses on analyzing Delhi Air Pollution data using Data Science and Machine Learning techniques. The dataset is cleaned, preprocessed, visualized, and used to build predictive models for PM2.5 concentration and AQI category classification.

The project demonstrates the complete Data Science workflow, including data preprocessing, exploratory data analysis, machine learning model development, evaluation, visualization, and pollution reduction simulation.

---

# 👨‍🎓 Student Details

- **Name:** Vineet Singh
- **Roll Number:** 25SCS1003005351
- **University:** IILM University Greater Noida
- **Department:** Computer Science & Engineering (CSE)

---

# 🎯 Project Objectives

- Analyze Delhi air pollution data.
- Clean and preprocess raw environmental data.
- Perform Exploratory Data Analysis (EDA).
- Predict PM2.5 concentration using Regression models.
- Classify AQI categories using Machine Learning.
- Visualize pollution trends using graphs.
- Simulate pollution reduction scenarios.

---

# 📂 Dataset

**Dataset Used**

- Delhi Pollution Combined Dataset (.xlsx)

**Features Used**

- Date
- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- SO2
- CO
- Ozone

---

# 🧹 Data Preprocessing

The dataset was processed using the following steps:

- Removed unwanted columns
- Renamed columns
- Converted Date column to datetime
- Converted numerical columns
- Removed duplicate records
- Filled missing values
- Removed outliers using IQR Method
- Generated AQI Categories

---

# 📊 Data Visualization

The project generates multiple visualizations including:

- Scatter Plot (NO₂ vs PM2.5)
- Monthly PM2.5 Analysis
- AQI Category Distribution
- Model Performance Comparison
- Feature Importance
- Classification Accuracy
- Pollution Reduction Simulation

---

# 🤖 Machine Learning Models

## Regression Models

- Linear Regression
- Random Forest Regressor

Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Cross Validation Score

---

## Classification Models

- Decision Tree Classifier
- Random Forest Classifier

Evaluation Metric

- Accuracy Score

---

# 📈 Project Results

### Regression

- Linear Regression R² Score: **0.4099**
- Random Forest R² Score: **0.6832**
- MAE: **25.11**
- RMSE: **41.27**

### Classification

- Decision Tree Accuracy: **57.78%**
- Random Forest Accuracy: **60.00%**

---

# 🔬 Pollution Reduction Simulation

The project also simulates the effect of reducing pollutant concentrations.

Simulation Results:

| Reduction in Pollutants | Estimated PM2.5 Reduction |
|--------------------------|---------------------------|
| 10% | 23.3% |
| 20% | 14.6% |
| 30% | 18.7% |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook
- VS Code

---

# 📁 Repository Structure

```
📦 Data Science Project
│
├── project.py
├── Delhi_Pollution_Combined.xlsx
├── Data Science Project Report.pdf
├── Methodology.pdf
├── PROJECT TITLE & SYNOPSIS.pdf
├── graph1.png
├── graph2.png
├── graph3.png
├── graph4.png
├── graph5.png
├── graph6.png
├── graph7.png
└── README.md
```

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/Vineets3119/VineetSingh_25SCS1003005351_DataScience.git
```

Move into the project directory

```bash
cd VineetSingh_25SCS1003005351_DataScience
```

Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

Run the project

```bash
python3 project.py
```

---

# 📌 Future Improvements

- Deep Learning based prediction
- Real-time Air Quality Prediction
- Interactive Dashboard
- Weather Data Integration
- Deployment using Streamlit

---

# 📜 License

This project is developed for academic and educational purposes.

---

# ⭐ Author

**Vineet Singh**

B.Tech CSE  
IILM University Greater Noida

---

⭐ If you found this project useful, consider giving it a Star on GitHub.
