import streamlit as st
from sklearn.metrics import classification_report, accuracy_score


def model_score(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    st.header("Model Score")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
