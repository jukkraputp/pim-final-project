import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import joblib

# Load the data
file_path = './data/air_quality_health_impact_data.csv'
data = pd.read_csv(file_path)

# Define features and target
features = data[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
target = data['HealthImpactClass']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a pipeline with more complex preprocessing and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler()),                # Standardize the features
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),  # Feature selection
    ('classifier', RandomForestClassifier(random_state=42))  # Random Forest Classifier
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Save the best pipeline to a .pkl file
best_model_filename = 'best_air_quality_health_impact_model.pkl'
joblib.dump(grid_search.best_estimator_, best_model_filename)

print(f'Best model saved to {best_model_filename}')
