# 🏠 ML House Price Predictor

An end-to-end **Machine Learning web application** built with Python, Scikit-learn, Plotly, and Streamlit.  
Train multiple ML models on real housing data, compare their performance, and predict house prices in real time using interactive sliders — all in the browser, no setup required for the end user.

---

## 🌐 Live Demo

Experience the deployed Streamlit application here:

👉 https://ml-house-price-predictor-znkwaqsycsszwhfhsvappb9.streamlit.app/

---

## 🖼️ Screenshots
## 📸 Project Screenshots

![Dashboard](Screenshot%202026-05-15%20001349.png)

---

![Dataset](Screenshot%202026-05-15%20001400.png)

---

![Training](Screenshot%202026-05-15%20001411.png)

---

![Predictions](Screenshot%202026-05-15%20001425.png)

---

![Charts](Screenshot%202026-05-15%20001437.png)
> *(Add screenshots here after running the app)*
> - Dashboard overview
> - Live prediction with sliders
> - Model comparison chart

---

## ✨ Features

- 📊 **Dataset Overview** — Auto-loaded California Housing Dataset (20,640 records), price distribution histogram, summary statistics
- 🤖 **Model Training** — Train 3 ML algorithms with one click:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- 📐 **Model Evaluation** — R² Score, MAE (Mean Absolute Error), RMSE displayed as live metrics
- 📈 **Actual vs Predicted Chart** — Scatter plot with perfect-prediction reference line
- 🔍 **Feature Importance** — Horizontal bar chart showing which features drive price predictions
- 🔗 **Correlation Heatmap** — Full feature correlation matrix using Plotly
- 🎯 **Live Prediction** — Adjust 8 input sliders and get an instant price prediction from the trained model
- ⚡ **Model Comparison** — Run all 3 algorithms and compare R² scores side by side
- ⬇️ **Model Export** — Download the trained model as a `.pkl` file using Joblib
- ⚙️ **Sidebar Controls** — Switch algorithms, adjust train/test split, tune hyperparameters

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| Scikit-learn | ML model training, evaluation, dataset |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Plotly | Interactive charts and heatmaps |
| Streamlit | Web application framework |
| Joblib | Model serialisation and export |

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ml-house-price-predictor.git
cd ml-house-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run ml_predictor.py
```

**4. Open in browser**
```
http://localhost:8501
```

> No dataset download needed — the California Housing dataset loads automatically from Scikit-learn.

---

## 📁 Project Structure

```
ml-house-price-predictor/
│
├── ml_predictor.py       # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
joblib
```

---

## 🧠 ML Pipeline

```
Raw Data (California Housing Dataset)
        ↓
Data Loading & Exploration (Pandas)
        ↓
Train / Test Split (Scikit-learn)
        ↓
Model Training (RF / GB / Linear Regression)
        ↓
Evaluation (R², MAE, RMSE)
        ↓
Feature Importance Analysis
        ↓
Live Inference via Streamlit Sliders
        ↓
Model Export (.pkl via Joblib)
```

---

## 📊 Model Performance (Default Settings)

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Random Forest | ~0.81 | ~$32,000 | ~$51,000 |
| Gradient Boosting | ~0.79 | ~$35,000 | ~$53,000 |
| Linear Regression | ~0.60 | ~$52,000 | ~$74,000 |

> Results may vary slightly based on train/test split settings chosen in the sidebar.

---

## 🎯 About the Dataset

**California Housing Dataset** — sourced from the 1990 US Census, included in Scikit-learn's built-in datasets.

| Feature | Description |
|---|---|
| Median Income | Median income of households in the block (in $10k) |
| House Age | Median age of houses in the block |
| Avg Rooms | Average number of rooms per household |
| Avg Bedrooms | Average number of bedrooms per household |
| Population | Total population of the block |
| Avg Occupancy | Average number of people per household |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| **Price** | **Target variable — median house price (USD)** |

---

## 💡 What I Learned

- Full ML pipeline from data loading to model deployment
- Training and comparing multiple regression algorithms (Random Forest, Gradient Boosting, Linear Regression)
- Evaluating models using R², MAE, and RMSE metrics
- Feature importance analysis and correlation matrix visualisation
- Building real-time inference UIs using Streamlit sliders
- Saving and exporting trained models using Joblib
- Deploying ML applications on Streamlit Cloud

---

## 🔮 Future Improvements

- [ ] Support custom CSV dataset upload for training
- [ ] Add hyperparameter tuning with cross-validation
- [ ] Add SHAP values for explainable AI
- [ ] Integrate LLM to generate plain-English explanations of predictions
- [ ] Add neural network model option (TensorFlow/Keras)

---

## 👤 Author

**Vansh Rajdev**  
B.E. ECE — University Institute of Engineering & Technology, Panjab University, Chandigarh  
📧 vanshrajdev06@gmail.com  
🔗 [LinkedIn](https://linkedin.com)  
🐙 [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
