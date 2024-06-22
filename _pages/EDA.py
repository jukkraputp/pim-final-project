import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define pages
def eda(data, features):
    st.title("Exploratory Data Analysis")
    st.write(data.describe())
    st.write(data.head())

    st.subheader("Feature Distributions")
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(features.columns):
        sns.histplot(data[col], ax=axs[i // 3, i % 3])
    st.pyplot(fig)
