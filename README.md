
# 🌧️ Rainfall Prediction using Random Forest  

## 📌 Project Overview  
This project focuses on **predicting rainfall** based on meteorological parameters using a **Random Forest Classifier**.  
The dataset is preprocessed, visualized, balanced, and trained using **GridSearchCV** for hyperparameter tuning.  

---

## 🚀 Features  
- Data preprocessing:  
  - Handling missing values.  
  - Encoding categorical variables.  
  - Dropping highly correlated features.  
- Data visualization with **Seaborn & Matplotlib**.  
- Handling class imbalance with **downsampling**.  
- Hyperparameter tuning with **GridSearchCV**.  
- Model evaluation with:  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  
  - Cross-validation scores  

---

## 🛠️ Tech Stack  
- **Python** 🐍  
- **Libraries:**  
  - `pandas`, `numpy` – data manipulation  
  - `matplotlib`, `seaborn` – visualization  
  - `scikit-learn` – ML models and evaluation  
  - `imblearn` – resampling techniques  

---

## 📂 Dataset  
The dataset used is **Rainfall.csv**, which contains weather-related features such as:  
- Pressure  
- Temperature  
- Dew point  
- Humidity  
- Cloud  
- Sunshine  
- Wind direction  
- Wind speed  
- Rainfall (target: yes/no)  

---

## 📊 Data Visualization  
- **Histograms** for feature distributions  
- **Boxplots** for detecting outliers  
- **Heatmap** for correlation analysis  
- **Countplot** for class distribution  

---

## ⚙️ Model Training  
- **Random Forest Classifier** was chosen due to its robustness.  
- Hyperparameters tuned with **GridSearchCV**:  
  ```python
  param_grid_rf = {
      "n_estimators": [50, 100, 200],
      "max_features": ["sqrt", "log2"],
      "max_depth": [None, 10, 20, 30],
      "min_samples_split": [2, 5, 10],
      "min_samples_leaf": [1, 2, 4]
  }
  ```
- Best parameters are displayed after tuning.  

---

## 📈 Model Evaluation  
- **Cross-validation scores**  
- **Test Accuracy**  
- **Confusion Matrix**  
- **Classification Report (Precision, Recall, F1-score)**  

---

## 🔮 Prediction Example  
You can test the model with custom input:  
```python
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=[
    'pressure', 'dewpoint', 'humidity', 'cloud',
    'sunshine', 'winddirection', 'windspeed'
])
prediction = best_rf_model.predict(input_df)
print("Predicted Rainfall:", prediction)
```

---

## 📌 Results  
- The tuned Random Forest model achieved **high accuracy** on the test dataset.  
- Demonstrates strong potential for **rainfall prediction** using meteorological data.  

---

## 📜 License  
This project is licensed under the **MIT License** – feel free to use and modify.  
