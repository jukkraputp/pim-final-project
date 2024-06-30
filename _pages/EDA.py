import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def perform_eda(data):
    st.write("## Summary Statistics")
    st.write(data.describe(include='all'))

    st.write("## Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    st.write("## Distribution of Numerical Features")
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    for feature in numerical_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        st.pyplot(plt)
        st.write(f"### Description of {feature} Distribution")
        st.write(f"The distribution of {feature} shows how values of {feature} are spread. If the distribution is skewed, it may indicate outliers or a non-normal distribution.")

    st.write("## Correlation Matrix")
    correlation_matrix = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Air Quality and Health Metrics')
    st.pyplot(plt)
    st.write("### Description of Correlation Matrix")
    st.write("The correlation matrix shows the relationship between different numerical features. High positive or negative values indicate strong correlations.")

    st.write("## Boxplots for Outliers Detection")
    for feature in numerical_features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data[feature])
        plt.title(f'Boxplot of {feature}')
        st.pyplot(plt)
        st.write(f"### Description of {feature} Boxplot")
        st.write(f"The boxplot of {feature} helps to identify outliers and understand the distribution's spread and central tendency. Any data points outside the whiskers are considered potential outliers.")