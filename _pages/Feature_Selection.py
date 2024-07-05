import streamlit as st
import pandas as pd

@st.cache_data
def feature_selection(_selector, features, score_threshold):
    try:
        scores = _selector.scores_
        feature_scores = pd.DataFrame({"Feature": features.columns, "Score": scores})
        selected_feature = feature_scores.sort_values(by="Score", ascending=False)
        return selected_feature[selected_feature["Score"] >= score_threshold]['Feature']
    except Exception as e:
        # Fit the _selector to your data (X_train is your feature matrix)
        _selector.transform(features)
        return features[_selector.get_feature_names_out()].columns
    
