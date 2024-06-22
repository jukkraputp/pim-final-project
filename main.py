import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE,
    RFECV,
    VarianceThreshold,
    GenericUnivariateSelect,
    f_classif,
    chi2,
    mutual_info_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Load the data
@st.cache_data
def load_data():
    file_path = "./data/air_quality_health_impact_data.csv"
    return pd.read_csv(file_path)


data = load_data()


def get_features(data):
    return data[
        [
            "AQI",
            "PM10",
            "PM2_5",
            "NO2",
            "SO2",
            "O3",
            "Temperature",
            "Humidity",
            "WindSpeed",
        ]
    ]


features = get_features(data)


def get_target(data):
    return data["HealthImpactClass"]


target = get_target(data)


# Define features and target
# @st.cache_data
def split_data(data):
    return train_test_split(features, target, test_size=0.2, random_state=42)


# Split the data
X_train, X_test, y_train, y_test = split_data(data)


# Define the pipeline
def train_pipeline(X_train, y_train):
    selected_model = st.sidebar.selectbox(
        "Model",
        [
            "Random Forest Classifier",
            "Decision Tree Classifier",
            "K-Nearest Neighbors Classifier",
            "Support Vector Classifier",
        ],
    )
    model_mapper = {
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
    }
    selected_score_function = st.sidebar.selectbox(
        "Score Function", ["f_classif", "chi2", "mutual_info_classif"]
    )
    score_function_mapper = {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
    }
    selected_feature_selection = st.sidebar.selectbox(
        "Feature Selection",
        [
            "SelectKBest",
            "SelectPercentile",
            "VarianceThreshold",
            "GenericUnivariateSelect",
        ],
    )
    selected_k = 5
    if selected_feature_selection == "SelectKBest":
        selected_k = st.sidebar.slider("k", min_value=1, max_value=9, value=5)
    feature_selection_mapper = {
        "SelectKBest": SelectKBest(
            score_func=score_function_mapper[selected_score_function],
            k=selected_k,
        ),
        "SelectPercentile": SelectPercentile(
            score_func=score_function_mapper[selected_score_function]
        ),
        "VarianceThreshold": VarianceThreshold(),
        "GenericUnivariateSelect": GenericUnivariateSelect(
            score_func=score_function_mapper[selected_score_function]
        ),
    }
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("feature_selection", feature_selection_mapper[selected_feature_selection]),
            ("classifier", model_mapper[selected_model]),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# Train the pipeline
pipeline = train_pipeline(X_train=X_train, y_train=y_train)


# Define pages
def eda():
    st.title("Exploratory Data Analysis")
    st.write(data.describe())
    st.write(data.head())

    st.subheader("Feature Distributions")
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(features.columns):
        sns.histplot(data[col], ax=axs[i // 3, i % 3])
    st.pyplot(fig)


def feature_selection():
    selector = pipeline.named_steps["feature_selection"]
    try:
        scores = selector.scores_
        feature_scores = pd.DataFrame({"Feature": features.columns, "Score": scores})
        selected_feature = feature_scores.sort_values(by="Score", ascending=False)
        return [selected_feature, selected_feature[selected_feature["Score"] > 1]]
    except:
        # Fit the selector to your data (X_train is your feature matrix)
        selector.fit(X_train)
        return [selector.get_feature_names_out(), selector.get_feature_names_out()]


def model_score():
    X_train_selected = X_train[feature_selection()[1]["Feature"].tolist()]
    pipeline.fit(X_train_selected, y_train)
    X_test_selected = X_test[feature_selection()[1]["Feature"].tolist()]
    y_pred = pipeline.predict(X_test_selected)
    st.title("Model Score")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))


def predictor():
    st.title("Predictor")
    st.write("Enter new data to make a prediction:")
    selected_features = feature_selection()[1]["Feature"].tolist()
    X_train_selected = X_train[selected_features]

    if "AQI" in selected_features:
        AQI = st.slider(
            "AQI (Air Quality Index, a measure of how polluted the air currently is or how polluted it is forecast to become)",
            value=0.0,
            min_value=0.0,
            max_value=600.0,
        )
    if "PM10" in selected_features:
        PM10 = st.slider(
            "PM10 (Concentration of particulate matter less than 10 micrometers in diameter (μg/m³))",
            value=0.0,
            min_value=0.0,
            max_value=400.0,
        )
    if "PM2_5" in selected_features:
        PM2_5 = st.slider(
            "PM2_5 (Concentration of particulate matter less than 2.5 micrometers in diameter (μg/m³))",
            value=0.0,
            min_value=0.0,
            max_value=300.0,
        )
    if "NO2" in selected_features:
        NO2 = st.slider(
            "NO2 (Concentration of nitrogen dioxide (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=300.0,
        )
    if "SO2" in selected_features:
        SO2 = st.slider(
            "SO2 (Concentration of sulfur dioxide (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=200.0,
        )
    if "O3" in selected_features:
        O3 = st.slider(
            "O3 (Concentration of ozone (ppb))",
            value=0.0,
            min_value=0.0,
            max_value=400.0,
        )
    if "Temperature" in selected_features:
        Temperature = st.slider(
            "Temperature (Temperature in degrees Celsius (°C))",
            value=0.0,
            min_value=0.0,
            max_value=60.0,
        )
    if "Humidity" in selected_features:
        Humidity = st.slider(
            "Humidity (Humidity percentage (%))",
            value=0.0,
            min_value=0.0,
            max_value=200.0,
        )
    if "WindSpeed" in selected_features:
        WindSpeed = st.slider(
            "WindSpeed (Wind speed in meters per second (m/s))",
            value=0.0,
            min_value=0.0,
            max_value=40.0,
        )

    if st.button("Predict"):
        params = {}
        if "AQI" in selected_features:
            params["AQI"] = [AQI]
        if "PM10" in selected_features:
            params["PM10"] = [PM10]
        if "PM2_5" in selected_features:
            params["PM2_5"] = [PM2_5]
        if "NO2" in selected_features:
            params["NO2"] = [NO2]
        if "SO2" in selected_features:
            params["SO2"] = [SO2]
        if "O3" in selected_features:
            params["O3"] = [O3]
        if "Temperature" in selected_features:
            params["Temperature"] = [Temperature]
        if "Humidity" in selected_features:
            params["Humidity"] = [Humidity]
        if "WindSpeed" in selected_features:
            params["WindSpeed"] = [WindSpeed]
        st.write(params)
        new_data = pd.DataFrame(params)
        prediction = pipeline.predict(new_data)
        # prediction = model.predict(new_data)
        mapper = {0: "Very High", 1: "High", 2: "Moderate", 3: "Low", 4: "Very Low"}
        st.write("Predicted Health Impact Class:", mapper[prediction[0]])


# Streamlit navigation
page = st.sidebar.radio(
    "Pages", ["EDA", "Feature Selection", "Model Score", "Predictor"]
)

if page == "EDA":
    eda()
elif page == "Feature Selection":
    st.title("Feature Selection")
    st.write(feature_selection()[0])
elif page == "Model Score":
    model_score()
elif page == "Predictor":
    predictor()
