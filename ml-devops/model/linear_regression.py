
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

data = pd.read_csv("../data/housing_price_dataset.csv")

print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())


# --------------------------------------------------
# 2. Separate features (X) and target (y)
# --------------------------------------------------

# Price is what we want the model to predict.
X = data.drop(columns=["Price"]).copy()
y = pd.to_numeric(data["Price"], errors="coerce")

# Convert infinite values to missing values.
X = X.replace([np.inf, -np.inf], np.nan)

# Remove rows where the target Price is missing or invalid.
valid_rows = y.notna() & np.isfinite(y)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

numeric_features = [
    "SquareFeet",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt"
]

categorical_features = [
    "Neighborhood"
]


# --------------------------------------------------
# 4. Create preprocessing pipelines
# --------------------------------------------------

# Numerical features:
# Fill missing values, then standardize them.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features:
# Fill missing values, then convert categories into
# numerical columns using one-hot encoding.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine both preprocessing pipelines.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 5. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)


# --------------------------------------------------
# 6. Create the complete model pipeline
# --------------------------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", SGDRegressor(
        loss="squared_error",
        max_iter=5000,
        tol=1e-4,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42
    ))
])


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

# The pipeline automatically preprocesses X_train
# before passing it to SGDRegressor.
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

print(f"MAE:  ${mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# --------------------------------------------------
# 10. Save the complete pipeline
# --------------------------------------------------

# Saving the pipeline also saves the fitted
# preprocessing steps, not just the regressor.
joblib.dump(model, "linear_regression_model.pkl")

print("\nModel saved as linear_regression_model.pkl")


# --------------------------------------------------
# 11. Test a prediction for a new house
# --------------------------------------------------

new_house = pd.DataFrame([{
    "SquareFeet": 2000,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Neighborhood": "Suburb",
    "YearBuilt": 2000
}])

predicted_price = model.predict(new_house)[0]

print(f"\nPredicted price for the new house: ${predicted_price:,.2f}")
