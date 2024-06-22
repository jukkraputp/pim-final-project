import pandas as pd


def feature_selection(selector, features):
    try:
        scores = selector.scores_
        feature_scores = pd.DataFrame({"Feature": features.columns, "Score": scores})
        selected_feature = feature_scores.sort_values(by="Score", ascending=False)
        return selected_feature[selected_feature["Score"] >= 0.01]['Feature']
    except Exception as e:
        # Fit the selector to your data (X_train is your feature matrix)
        selector.transform(features)
        return features[selector.get_feature_names_out()].columns
