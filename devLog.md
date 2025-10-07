# Development Log: app.py

This log details the development process of the `app.py` Streamlit application.

## Step 1: Initial Creation

- Created the initial `app.py` file.
- Implemented a basic interactive linear regression demo using Streamlit.
- Included sliders for user to control slope (a), intercept (b), noise, and number of data points.
- Generated data based on user inputs.
- Fitted a linear regression model using scikit-learn.
- Displayed the fitted line on a plot.

## Step 2: Enhanced Features

- Added a "Model Coefficients" section to display the slope, intercept, and Mean Squared Error.
- Added a "Top 5 Outliers" section to identify and display the data points with the largest residuals.
- Highlighted the top 5 outliers on the scatter plot for better visualization.

## Step 3: Added Noise Variance

- Added the "Noise Variance" to the "Model Coefficients" section to provide more insight into the data generation process.
