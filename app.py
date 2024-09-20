import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model from the pickle file
@st.cache_resource
def load_model():
    with open('propensity_model_bmw.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to make predictions based on predict_proba and user-defined percentage
def predict_top_percent(input_data, percent, car_model):
    # Convert the date columns to datetime
    input_data['Last_Service_Date'] = pd.to_datetime(input_data['Last_Service_Date'])
    input_data['Purchase_Date'] = pd.to_datetime(input_data['Purchase_Date'])
    
    # Calculate days since purchase
    input_data['Days_Since_Purchase'] = (input_data['Last_Service_Date'] - input_data['Purchase_Date']).dt.days  
    
    # Store the Customer_ID column separately to add it back later
    customer_ids = input_data['Customer_ID']
    
    # Drop unnecessary columns for prediction
    input_data = input_data.drop(columns=['Customer_ID', 'Purchase_Date', 'Last_Service_Date', 'Car_Model', 'Returned_for_Service'])  
    
    # Check if the data is empty after filtering
    if input_data.empty:
        return pd.DataFrame(), f"No data available for the car model: {car_model}"
    
    # Get predicted probabilities
    proba = model.predict_proba(input_data)

    # Assuming binary classification, pick the probability of the positive class
    proba_positive = proba[:, 1]
    
    # Calculate the threshold for the top X% based on the probability of the positive class
    threshold = np.percentile(proba_positive, 100 - percent)  
    
    # Get the indices of the top X% highest probabilities
    top_indices = np.where(proba_positive >= threshold)[0]
    
    # Create a DataFrame of top predictions
    top_predictions = input_data.iloc[top_indices].copy()
    
    # Add the Customer_ID and predicted probabilities to the top predictions
    top_predictions['Customer_ID'] = customer_ids.iloc[top_indices].values
    top_predictions['Probability'] = proba_positive[top_indices]
    
    return top_predictions, None


# Streamlit interface
st.title("Car Model Prediction App")

# Select car model
car_model = st.selectbox("Select the car model", options=['BMW', 'Nissan'])

# Upload a CSV file for prediction
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data.head(10))
    
    # Visualize data distribution
    st.subheader("Data Distribution")

    # Check for numeric columns
    numeric_columns = input_data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_columns:
        st.write("Numeric Columns Distribution:")
        
        # Plot each numeric column's distribution using Seaborn
        for col in numeric_columns:
            st.write(f"Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(input_data[col], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
    else:
        st.write("No numeric columns found for distribution plotting.")
    # Prompt for percentage input
    st.write("Enter the percentage of results you want to display (e.g., 30 for top 30%):")
    percentage_input = st.chat_input()

    if percentage_input:
        try:
            
            percent = float(percentage_input)
            
            if percent < 0 or percent > 100:
                st.error("Please enter a percentage between 0 and 100.")
            else:
                # Make predictions and get the top X% results
                
                top_predictions, error_message = predict_top_percent(input_data, percent, car_model)
                #st.error(f"Invalid input. Please enter a numeric {error_message}.")
                if error_message:
                    st.error(error_message)
                else:
                    # Show the top X% results
                    st.write(f"Top {percent}% Predictions for {car_model}:")
                    st.write(top_predictions)
                    
                    # Download the top X% predictions as a CSV file
                    csv = top_predictions.to_csv(index=False).encode('utf-8')
                    st.download_button(label=f"Download Top {int(percent)}% Predictions for {car_model} as CSV",
                                       data=csv,
                                       file_name=f'top_{int(percent)}_predictions_{car_model}.csv',
                                       mime='text/csv')
        except ValueError:
            st.error("Invalid input. Please enter a numeric value.")
