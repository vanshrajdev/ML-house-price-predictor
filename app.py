import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 ML House Price Predictor")
st.markdown("Train a machine learning model on real housing data and predict prices using interactive sliders.")
st.divider()

# ── Load Dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    data = fetch_california_housing(as_frame=True)
    df   = data.frame.copy()
    df.columns = [
        "Avg Rooms", "House Age", "Avg Bedrooms",
        "Population", "Avg Occupancy", "Latitude",
        "Longitude", "Median Income", "Price"
    ]
    # Price is in $100k units — convert to USD for clarity
    df["Price"] = (df["Price"] * 100000).round(0)
    return df

df = load_dataset()

# ── Sidebar — Model Settings ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose ML Algorithm",
    ["Random Forest", "Gradient Boosting", "Linear Regression"],
    index=0
)
test_size = st.sidebar.slider("Test Split Size (%)", 10, 40, 20) / 100
n_estimators = st.sidebar.slider("Number of Trees (RF/GB only)", 50, 300, 100, step=50)

st.sidebar.divider()
st.sidebar.markdown("**About the Dataset**")
st.sidebar.markdown(
    "California Housing Dataset from Scikit-learn.  \n"
    f"**{len(df):,} rows** | **{df.shape[1]} columns**  \n"
    "Sourced from 1990 US Census."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset Preview
# ══════════════════════════════════════════════════════════════════════════════
st.header("1️⃣  Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Total Rows",    f"{len(df):,}")
c2.metric("📌 Features",      f"{df.shape[1] - 1}")
c3.metric("💰 Avg Price",     f"${df['Price'].mean():,.0f}")
c4.metric("📈 Max Price",     f"${df['Price'].max():,.0f}")

st.subheader("Raw Data Sample")
st.dataframe(df.head(10), use_container_width=True)

# Price distribution
st.subheader("Price Distribution")
fig_dist = px.histogram(
    df, x="Price", nbins=50,
    title="Distribution of House Prices (USD)",
    color_discrete_sequence=["#4F8EF7"],
    labels={"Price": "Price (USD)"}
)
fig_dist.update_layout(bargap=0.05)
st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Train the Model
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("2️⃣  Train the Model")

features = [c for c in df.columns if c != "Price"]
X = df[features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

@st.cache_resource
def train_model(model_name, n_est, test_sz):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz, random_state=42)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    metrics = {
        "R² Score":  round(r2_score(y_te, preds), 4),
        "MAE":        round(mean_absolute_error(y_te, preds), 2),
        "RMSE":       round(np.sqrt(mean_squared_error(y_te, preds)), 2),
    }
    return model, preds, y_te, metrics

with st.spinner(f"Training {model_choice} model..."):
    model, preds, y_te, metrics = train_model(model_choice, n_estimators, test_size)

st.success(f"✅ {model_choice} trained successfully on {len(X_train):,} samples!")

# Metrics row
m1, m2, m3 = st.columns(3)
m1.metric("📐 R² Score",  f"{metrics['R² Score']}",  help="Closer to 1.0 = better. Explains variance in prices.")
m2.metric("💵 MAE",       f"${metrics['MAE']:,.0f}",  help="Average prediction error in USD.")
m3.metric("📉 RMSE",      f"${metrics['RMSE']:,.0f}", help="Root Mean Squared Error — penalises large errors more.")

# ── Actual vs Predicted chart ─────────────────────────────────────────────────
st.subheader("Actual vs Predicted Prices")

scatter_df = pd.DataFrame({"Actual": y_te.values, "Predicted": preds})
fig_sc = px.scatter(
    scatter_df, x="Actual", y="Predicted",
    title="Actual vs Predicted House Prices",
    labels={"Actual": "Actual Price (USD)", "Predicted": "Predicted Price (USD)"},
    opacity=0.5,
    color_discrete_sequence=["#4F8EF7"]
)
# Perfect prediction line
min_val = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
fig_sc.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", name="Perfect Prediction",
    line=dict(color="red", dash="dash", width=2)
))
st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
if model_choice in ["Random Forest", "Gradient Boosting"]:
    st.divider()
    st.header("3️⃣  Feature Importance")
    st.caption("Which features matter most when predicting price?")

    importance_df = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importance_df, x="Importance", y="Feature",
        orientation="h",
        title=f"Feature Importance — {model_choice}",
        color="Importance",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("4️⃣  Correlation Heatmap")
st.caption("How strongly are features related to each other and to price?")

corr = df.corr().round(2)
fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Feature Correlation Matrix"
)
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Live Prediction with Sliders
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("5️⃣  🎯 Predict a House Price")
st.markdown("Adjust the sliders below and get an **instant price prediction** from the trained model.")

col_l, col_r = st.columns(2)

with col_l:
    avg_rooms     = st.slider("Avg Rooms per Household",    1.0, 10.0, float(df["Avg Rooms"].median()),     0.1)
    house_age     = st.slider("House Age (years)",           1.0, 52.0, float(df["House Age"].median()),     1.0)
    avg_bedrooms  = st.slider("Avg Bedrooms per Household",  0.5, 5.0,  float(df["Avg Bedrooms"].median()),  0.1)
    population    = st.slider("Block Population",          100,  5000,  int(df["Population"].median()),      50)

with col_r:
    avg_occupancy = st.slider("Avg Occupancy",              1.0, 6.0,  float(df["Avg Occupancy"].median()),  0.1)
    latitude      = st.slider("Latitude",                  32.5, 42.0, float(df["Latitude"].median()),       0.1)
    longitude     = st.slider("Longitude",                -124.0,-114.0, float(df["Longitude"].median()),    0.1)
    median_income = st.slider("Median Income (in $10k)",    0.5, 15.0, float(df["Median Income"].median()),  0.1)

input_data = pd.DataFrame([[
    avg_rooms, house_age, avg_bedrooms,
    population, avg_occupancy, latitude,
    longitude, median_income
]], columns=features)

predicted_price = model.predict(input_data)[0]

st.markdown("---")
pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
with pred_col2:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 2rem;
             background: linear-gradient(135deg, #1a5276, #2980b9);
             border-radius: 16px; color: white;">
            <p style="font-size:18px; margin:0; opacity:0.85;">Predicted House Price</p>
            <p style="font-size:52px; font-weight:700; margin:0.2rem 0;">
                ${predicted_price:,.0f}
            </p>
            <p style="font-size:13px; opacity:0.7; margin:0;">
                Model: {model_choice} &nbsp;|&nbsp; R²: {metrics['R² Score']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

# Show input summary
st.subheader("Your Input Summary")
st.dataframe(input_data, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Compare All 3 Models
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("6️⃣  Model Comparison")
st.caption("See how all 3 algorithms perform on the same data.")

if st.button("⚡ Run All 3 Models & Compare"):
    results = []
    for name, n in [("Linear Regression", 100), ("Random Forest", 100), ("Gradient Boosting", 100)]:
        _, _, _, m = train_model(name, n, test_size)
        results.append({"Model": name, **m})
    comp_df = pd.DataFrame(results)
    st.dataframe(comp_df, use_container_width=True)

    fig_comp = px.bar(
        comp_df, x="Model", y="R² Score",
        title="R² Score Comparison Across Models",
        color="R² Score", color_continuous_scale="Blues",
        text="R² Score"
    )
    fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Download Trained Model
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("7️⃣  Download Trained Model")

buffer = io.BytesIO()
joblib.dump(model, buffer)
buffer.seek(0)

st.download_button(
    label=f"⬇️ Download {model_choice} Model (.pkl)",
    data=buffer,
    file_name=f"{model_choice.lower().replace(' ','_')}_house_price_model.pkl",
    mime="application/octet-stream"
)
st.caption("You can load this .pkl file later with `joblib.load('model.pkl')` to make predictions without retraining.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Python · Scikit-learn · Pandas · Plotly · Streamlit  |  Dataset: California Housing (Scikit-learn)")