import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

# Load the dataset
@st.cache
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to plot average house price per region
def plot_average_price(df):
    plt.figure(figsize=(25, 16))
    plt.bar(df['Region'], df['Average current price'], color='lightblue')
    plt.xlabel('Region')
    plt.ylabel('Average Current Price')
    plt.title('Average House Price per Region')
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot()

# Function to plot distribution of property prices
def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Average current price'], bins=50, kde=True)
    plt.title('Distribution of Property Prices')
    plt.xlabel('Average Price')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot()

# Main function to run the Streamlit app
def main():
    st.title('Real Estate Insight Dashboard')

    # Load the data
    file_path = "C:\Users\OEM\Documents\Corelogic_NZ_HousePriceIndex_Residentialpricemovement_APR2024__1_.xlsx"
    df = load_data(file_path)

    # Sidebar selection
    option = st.sidebar.selectbox(
        'Choose an option:',
        ('Overview', 'Average Price per Region', 'Price Distribution')
    )

    # Display different components based on selection
    if option == 'Overview':
        st.write(df)
    elif option == 'Average Price per Region':
        plot_average_price(df)
    elif option == 'Price Distribution':
        plot_price_distribution(df)

if __name__ == '__main__':
    main()

