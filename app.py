import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    GenericUnivariateSelect,
    f_classif,
    mutual_info_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import _pages.EDA as EDA
import _pages.Feature_Selection as Feature_Selection
import _pages.Model_Score as Model_Score
import _pages.Predictor as Predictor


# Load the data
@st.cache_data
def load_data():
    file_path = "./data/air_quality_health_impact_data.csv"
    return pd.read_csv(file_path)


data = load_data()


def get_features():
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


features = get_features()


def get_target(data):
    return data["HealthImpactClass"]


target = get_target(data)


# st.title("Air Quality Health Impact Data")
# st.write(data)

# Streamlit navigation
page = st.sidebar.radio(
    "Pages", ["EDA", "Feature Selection", "Model Score", "Predictor"]
)


# Define features and target
def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)


selected_score_function = st.sidebar.selectbox(
    "Score Function", ["f_classif", "mutual_info_classif"]
)
score_function_mapper = {
    "f_classif": f_classif,
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
variance_threshold = 0.1
if selected_feature_selection == "VarianceThreshold":
    variance_threshold = st.sidebar.slider(
        "variance threshold", min_value=0.0, max_value=1.0, value=0.5
    )
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

selector = feature_selection_mapper[selected_feature_selection]
selector.fit_transform(features, target)


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

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("feature_selection", selector),
            ("classifier", model_mapper[selected_model]),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


feature_selection_result = pd.Series()

selected_features = feature_selection_result

X_train, X_test, y_train, y_test = split_data(features=features, target=target)

pipeline = train_pipeline(X_train, y_train)

if page == "EDA":
    st.title('Exploratory Data Analysis (EDA)')
    all_data = st.checkbox('Use All Data for EDA', value=True, key='eda_all_data')
    if (all_data):
        all_data = EDA.perform_eda(data=data)
    else:
        copy_data = features[selected_features].copy()
        copy_data['HealthImpactClass'] = target
        EDA.perform_eda(data=copy_data)
elif page == "Feature Selection":
    st.title("Feature Selection")
    method = st.selectbox("", ['select features automatically', 'select features manually'])
    if method == 'select features automatically':
        result = Feature_Selection.feature_selection(
            selector=selector, features=features
        )
    else:
        selected_features = st.multiselect("Select Features", features.columns)
        result = pd.Series(selected_features)
    feature_selection_result = result
    st.write(feature_selection_result)
elif page == "Model Score":
    Model_Score.model_score(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
elif page == "Predictor":
    Predictor.predictor(
        pipeline=pipeline,
        selected_features=features[selected_features],
    )
