import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
#call the model and scaler from the pickle files and create scaler and kmeans objects
cluster_names = {
    0: "Average Income, Average Spender",
    1: "High Income, High Spender (Target)",
    2: "Low Income, High Spender (Impulsive)",
    3: "High Income, Low Spender (Careful)",
    4: "Low Income, Low Spender (Budget-Conscious)"
}# explain the cluster names and their characteristics
st.title(" Mall Customer Segmentation")
st.write("Enter a customer's details to predict which segment they belong to.")


annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
#take the input from the user for  annual income and spending score to predict the segment of the customer

if st.button("Predict Segment"):
    input_data = pd.DataFrame({
    'Annual Income (k$)': [annual_income],
    'Spending Score (1-100)': [spending_score]
})
    input_data_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_data_scaled)[0]
    segment = cluster_names[cluster]
    st.write(f"The customer belongs to the segment: **{segment}**")

# Visualization
    df = pd.read_csv('Mall_Customers.csv')
    df_scaled = scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    all_clusters = kmeans.predict(df_scaled)

    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                          c=all_clusters, cmap='viridis', s=50, alpha=0.6)
    ax.scatter(annual_income, spending_score, c='red', s=200, marker='*', label='This Customer')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Customer Segments')
    ax.legend()
    st.pyplot(fig) 

    
       