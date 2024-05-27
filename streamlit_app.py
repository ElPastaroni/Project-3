import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import StrMethodFormatter

# Configure Streamlit settings
st.set_page_config(layout="wide")

# Load the datasets
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(CoreLogic.xlsx)
    df.rename(columns={'Territorial authority': 'Region', 'Average current value': 'Average current price', '3 month change %': '3 month change%'}, inplace=True)
    df['Average value 3 months ago'] = df['Average current price'] / (1 + df['3 month change%'])
    df['Average value 12 months ago'] = df['Average current price'] / (1 + df['12 month change%'])
    return df

@st.cache_data
def load_data2(file_path):
    return pd.read_excel(NZrealestate.xlsx)

file_path = "CoreLogic.xlsx"
df = load_data(file_path)

file_path2 = "NZrealestate.xlsx"
df2 = load_data2(file_path2)

# Display DataFrame
st.title("NZ House Price Index - Residential Price Movement")
st.dataframe(df)

# Bar plot of average current price per region
st.subheader('Average House Price per Region')
fig, ax = plt.subplots(figsize=(25, 16))
ax.bar(df['Region'], df['Average current price'], color='lightblue')
ax.set_xlabel('Region')
ax.set_ylabel('Average Current Price')
ax.set_title('Average House Price per Region')
ax.set_xticklabels(df['Region'], rotation=90)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
st.pyplot(fig)

# Distribution of property prices
st.subheader('Distribution of Property Prices')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Average current price'], bins=50, kde=True, ax=ax)
ax.set_title('Distribution of Property Prices')
ax.set_xlabel('Average Price')
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
st.pyplot(fig)

# Line plots of price changes over time for each region
st.subheader('Change in Average House Prices for Each Region')
num_regions = len(df)
num_cols = 4
num_rows = (num_regions + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 6 * num_rows), sharey=True)
axes = axes.flatten()

for i, (index, row) in enumerate(df.iterrows()):
    x = ['12 months ago', '3 months ago', 'Current']
    y = [row['Average value 12 months ago'], row['Average value 3 months ago'], row['Average current price']]
    ax = axes[i]
    ax.plot(x, y, marker='o')
    ax.set_title(row['Region'])
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Change in Average House Prices for Each Region')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
st.pyplot(fig)

# Prepare the data for machine learning
st.subheader('Predicting House Prices with Linear Regression')

X = df[['3 month change%', '12 month change%']]  # Features
y = df['Average current price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
st.write(f"### Mean Squared Error: {mse:.2f}")
st.write(f"### R-squared: {r2:.2f}")

# Plot true vs predicted prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='b', s=100)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('True Prices')
ax.set_ylabel('Predicted Prices')
ax.set_title('True vs Predicted Prices')
st.pyplot(fig)

# Additional data analysis
st.subheader('Additional Data Analysis')

# Analysis based on number of bedrooms
mean_bedroomsNaN = df2.loc[df2.bedrooms.isna()].price.mean()
mean_bedrooms2 = df2.loc[df2.bedrooms == 2].price.mean()
mean_bedrooms3 = df2.loc[df2.bedrooms == 3].price.mean()
mean_bedrooms4 = df2.loc[df2.bedrooms == 4].price.mean()
mean_bedrooms5 = df2.loc[df2.bedrooms == 5].price.mean()
meanDataBed = [["NaN", mean_bedroomsNaN], ["2", mean_bedrooms2], ["3", mean_bedrooms3], ["4", mean_bedrooms4], ["5", mean_bedrooms5]]
dfMeanBed = pd.DataFrame(meanDataBed, columns=["bedrooms", "AVG price"])

fig, ax = plt.subplots()
dfMeanBed.sort_values(by="bedrooms").plot.bar(x="bedrooms", y="AVG price", ax=ax)
ax.set_xlabel("Number of bedrooms")
ax.set_ylabel("Price of house (NZD, millions)")
ax.set_title("Average House Price by Number of Bedrooms")
st.pyplot(fig)

# Analysis based on number of bathrooms
mean_bathroomsNaN = df2.loc[df2.bathrooms.isna()].price.mean()
mean_bathrooms1 = df2.loc[df2.bathrooms == "1"].price.mean()
mean_bathrooms2 = df2.loc[df2.bathrooms == "2"].price.mean()
mean_bathrooms3 = df2.loc[df2.bathrooms == "3"].price.mean()
mean_bathrooms4 = df2.loc[df2.bathrooms == "4"].price.mean()
mean_bathrooms7 = df2.loc[df2.bathrooms == "7"].price.mean()
meanDataBath = [["NaN", mean_bathroomsNaN], ["1", mean_bathrooms1], ["2", mean_bathrooms2], ["3", mean_bathrooms3], ["4", mean_bathrooms4], ["7", mean_bathrooms7]]
dfMeanBath = pd.DataFrame(meanDataBath, columns=["bathrooms", "AVG price"])

fig, ax = plt.subplots()
dfMeanBath.sort_values(by="bathrooms").plot.bar(x="bathrooms", y="AVG price", ax=ax)
ax.set_xlabel("Number of bathrooms")
ax.set_ylabel("Price of house (NZD, millions)")
ax.set_title("Average House Price by Number of Bathrooms")
st.pyplot(fig)

# Analysis based on land size
df3 = df2.drop_duplicates(subset=["address"])
df3["land_sizem2"] = df3["land_size"]
x = 0
while x < len(df3):
    if isinstance(df3.land_size.iloc[x], float):
        pass
    elif "m2" in df3.land_size.iloc[x]:
        df3.at[x, "land_sizem2"] = int(df3.land_size.iloc[x].replace("m2", ""))
    elif "ha" in df3.land_size.iloc[x]:
        df3.drop(df3.index[x], inplace=True)
    x += 1

fig, ax = plt.subplots()
ax.scatter(df3.land_sizem2, df3)
