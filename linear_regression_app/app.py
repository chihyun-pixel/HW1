# Step 1 & 2: 資料生成（Data Understanding & Preparation）
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Interactive Simple Linear Regression Demo")

# Streamlit sliders for user input
st.sidebar.header("Configuration")
a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0)
b = st.sidebar.slider("Intercept (b)", min_value=-10.0, max_value=10.0, value=1.0)
noise = st.sidebar.slider("Noise level", min_value=0.0, max_value=5.0, value=1.0)
n_points = st.sidebar.slider("Number of data points", min_value=5, max_value=100, value=20)

# Prompt / process explanation
st.write(f"Generating {n_points} points using y = {a}x + {b} + noise")

# Generate data
np.random.seed(42)  # reproducible
x = np.random.uniform(-10, 10, n_points).reshape(-1,1)
y = a * x.flatten() + b + np.random.normal(0, noise, n_points)
df = pd.DataFrame({'x': x.flatten(), 'y': y})

st.write("Sample data:")
st.dataframe(df.head())

# Step 3 & 4: Modeling
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Step 5: Evaluation
mse = np.mean((y - y_pred)**2)

# Model Coefficients
st.subheader("Model Coefficients")
st.write(f"**Slope (a):** {model.coef_[0]:.2f}")
st.write(f"**Intercept (b):** {model.intercept_:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**Noise Variance:** {noise**2:.2f}")

# Top 5 Outliers
st.subheader("Top 5 Outliers")
residuals = y - y_pred.flatten()
outlier_indices = np.argsort(np.abs(residuals))[-5:]
outliers = df.iloc[outlier_indices]
st.write("The following data points are identified as the top 5 outliers based on their distance from the fitted line:")
st.dataframe(outliers)


# Step 5: Visualization
fig, ax = plt.subplots()
ax.scatter(x, y, label="Data")
ax.plot(x, y_pred, color='red', label="Fitted line")
ax.scatter(outliers['x'], outliers['y'], color='orange', label="Outliers")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)