# California Housing Price Prediction

This notebook demonstrates a machine learning workflow to predict housing prices in California using the California Housing dataset.

## Project Overview

The goal of this project is to build a regression model that can predict the median house value in a given district based on various features.

## Data

The dataset used in this notebook is the California Housing dataset, available through scikit-learn. It contains information about housing districts in California, including:

*   **MedInc**: Median income in the district
*   **HouseAge**: Median house age in the district
*   **AveRooms**: Average number of rooms
*   **AveBedrms**: Average number of bedrooms
*   **Population**: District population
*   **AveOccup**: Average household occupancy
*   **Latitude**: District latitude
*   **Longitude**: District longitude

## Steps Taken

1.  **Data Loading and Exploration**: The California Housing dataset was loaded and explored using pandas. Histograms were generated to visualize the distribution of features, and the relationship between house age and price was plotted.
2.  **Feature Engineering**: A new feature, `Lat_Lon`, was created by multiplying Latitude and Longitude. The original `AveBedrms`, `Latitude`, and `Longitude` columns were dropped.
3.  **Data Splitting**: The dataset was split into training and testing sets.
4.  **Model Training (LightGBM with GridSearchCV)**: A LightGBM Regressor model was trained using GridSearchCV to find the best hyperparameters.
5.  **Model Evaluation**: The trained model was evaluated on the test set using the R-squared score.
6.  **Model Saving**: The trained model with the best hyperparameters was saved using `joblib`.

## Results

The LightGBM model trained with the best hyperparameters achieved an R-squared score of approximately {{r2}} on the test set. The best hyperparameters found were: {{best_params}}.

## Files

*   `california_lgbm_model.pkl`: The saved trained LightGBM model.
