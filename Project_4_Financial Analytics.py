#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Loading Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


fdf = pd.read_csv("Financial Analytics data.csv")


# # Data Exploration and Understanding

# In[3]:


fdf.head()


# In[4]:


fdf.info()


# In[5]:


fdf.describe()


# In[6]:


fdf.duplicated().sum()


# In[7]:


fdf.isnull().sum()


# In[8]:


nan_mask = fdf['Sales Qtr - Crore'].isna()

# Filter rows where 'Unnamed: 4' has numerical values corresponding to NaN in 'Sales Qtr - Crore'
verification_df = fdf[nan_mask & pd.to_numeric(fdf['Unnamed: 4'], errors='coerce').notna()]

# Display the verification DataFrame
print(verification_df)


# # Data Cleaning and Preprocessing

# In[9]:


fdf.fillna(0,inplace=True)
fdf


# In[10]:


fdf["Sale Qtr (in Cr)"] = fdf['Sales Qtr - Crore'] + fdf['Unnamed: 4']
fdf


# In[11]:


#Dropping unnecessary columns for data efficiency
fdf.drop(['Sales Qtr - Crore','Unnamed: 4'],axis=1,inplace=True)
fdf


# In[12]:


#Indexes with both zero Mar Cap - Crore and Sale Qtr (in Cr)
fdf[(fdf['Mar Cap - Crore'] == 0) & (fdf['Sale Qtr (in Cr)'] == 0)].index


# In[13]:


#dropping Company data having both zero Mar Cap - Crore and Sale Qtr (in Cr)
fdf = fdf.drop(fdf[(fdf['Mar Cap - Crore'] == 0) & (fdf['Sale Qtr (in Cr)'] == 0)].index)
fdf


# In[14]:


feature1 = 'Mar Cap - Crore'
feature2 = 'Sale Qtr (in Cr)'

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(fdf[feature1].dropna(), bins=30, color='blue', alpha=0.7)
plt.title(f'Distribution of {feature1}')
plt.xlabel(feature1)
plt.ylabel('Frequency')
plt.axvline(fdf[feature1].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(fdf[feature1].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(fdf[feature2].dropna(), bins=30, color='orange', alpha=0.7)
plt.title(f'Distribution of {feature2}')
plt.xlabel(feature2)
plt.ylabel('Frequency')
plt.axvline(fdf[feature2].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(fdf[feature2].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

plt.tight_layout()
plt.show()


# In[15]:


fdf_d = fdf[(fdf['Mar Cap - Crore'] == 0) & (fdf['Sale Qtr (in Cr)'] == 0)]
fdf_d


# In[16]:


fdf


# In[17]:


# Count zero entries in each column
zero_counts = (fdf == 0).sum()
print(zero_counts)


# In[18]:


fdf.info()


# In[19]:


fdf.describe()


# In[20]:


# Create a boxplot for Mar Cap - Crore
column_name = 'Mar Cap - Crore'
plt.figure(figsize=(12, 6))
plt.boxplot(fdf[column_name], vert=False, sym='b.')
plt.title(f'Boxplot for {column_name}')
plt.xlabel(column_name)
plt.ylabel('Distribution')
plt.show()


# In[21]:


# Create a boxplot for Sale Qtr (in Cr)
column_name = 'Sale Qtr (in Cr)'
plt.figure(figsize=(12, 6))
plt.boxplot(fdf[column_name], vert=False, sym='b.')
plt.title(f'Boxplot for {column_name}')
plt.xlabel(column_name)
plt.ylabel('Distribution')
plt.show()


# In[22]:


fdf_d = fdf[(fdf['Mar Cap - Crore'] != 0) & (fdf['Sale Qtr (in Cr)'] == 0)]
fdf_d


# # Feature Engineering

# In[23]:


# Define a function to calculate Market Cap-to-Sales Ratio
def market_cap_to_sales_ratio(row):
    if row['Sale Qtr (in Cr)'] != 0:
        return row['Mar Cap - Crore'] / row['Sale Qtr (in Cr)']
    else:
        return 0  # Handle division by zero scenario

# Apply the function to create the new column
fdf['Market Cap-to-Sales Ratio'] = fdf.apply(market_cap_to_sales_ratio, axis=1)
fdf


# In[24]:


fdf_d = fdf[(fdf['Sale Qtr (in Cr)'] == 0)]
fdf_d


# In[25]:


fdf.sort_values('Market Cap-to-Sales Ratio',ascending=False)
fdf


# In[26]:


# Specify the top N companies
top_n = 50  

# Select the top N companies based on Market Cap-to-Sales Ratio
top_companies = fdf.nlargest(top_n, 'Market Cap-to-Sales Ratio')

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.bar(top_companies['Name'], top_companies['Market Cap-to-Sales Ratio'], color='skyblue')
plt.title(f'Top {top_n} Companies by Market Cap-to-Sales Ratio')
plt.xlabel('Name')
plt.ylabel('Market Cap-to-Sales Ratio')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[27]:


feature = 'Market Cap-to-Sales Ratio'

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(fdf[feature].dropna(), bins=30, color='blue', alpha=0.7)
plt.title(f'Distribution of {feature}')
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.axvline(fdf[feature].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(fdf[feature].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

plt.tight_layout()
plt.show()


# In[28]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(fdf['Mar Cap - Crore'], fdf['Sale Qtr (in Cr)'], color='blue', alpha=0.7)
sns.regplot(x='Mar Cap - Crore', y='Sale Qtr (in Cr)', data=fdf, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

# Set plot title and labels
plt.title('Scatterplot with Trend Line - Market Cap vs. Qtr Sales')
plt.xlabel('Market Cap (in Crore)')
plt.ylabel('Quarterly Sales (in Crore)')
plt.grid(True)
plt.show()


# In[29]:


correlation_data = fdf[['Mar Cap - Crore', 'Sale Qtr (in Cr)','Market Cap-to-Sales Ratio']]

correlation_matrix = correlation_data.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title('Correlation Heatmap - Mid Cap vs. Qtr Sales vs. Market Cap-to-Sales Ratio',fontweight='bold',fontsize = 14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()


# Observation: The correlation coefficient of 0.63 indicates a positive relationship between Market Capitalization and Quarterly Sales. However, there is minimal or negative correlation between these parameters and the ratio.

# In[30]:


# Create a boxplot for  Market Cap-to-Sales Ratio
column_name = 'Market Cap-to-Sales Ratio'
plt.figure(figsize=(12, 6))
plt.boxplot(fdf[column_name], vert=False, sym='b.')
plt.title(f'Boxplot for {column_name}')
plt.xlabel(column_name)
plt.ylabel('Distribution')
plt.show()


# In[31]:


# Step 1: Calculate IQR
q1 = fdf['Market Cap-to-Sales Ratio'].quantile(0.25)
q3 = fdf['Market Cap-to-Sales Ratio'].quantile(0.75)
iqr = q3 - q1

# Step 2: Define lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Step 3: Remove outliers
fdf_no_outliers = fdf[(fdf['Market Cap-to-Sales Ratio'] >= lower_bound) & (fdf['Market Cap-to-Sales Ratio'] <= upper_bound)]

# Display the updated DataFrame
print("DataFrame after removing outliers:")
print(fdf_no_outliers)


# In[32]:


fdf_no_outliers.describe()


# In[33]:


feature1 = 'Mar Cap - Crore'
feature2 = 'Sale Qtr (in Cr)'

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(fdf_no_outliers[feature1].dropna(), bins=30, color='blue', alpha=0.7)
plt.title(f'Distribution of {feature1}')
plt.xlabel(feature1)
plt.ylabel('Frequency')
plt.axvline(fdf_no_outliers[feature1].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(fdf_no_outliers[feature1].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(fdf_no_outliers[feature2].dropna(), bins=30, color='orange', alpha=0.7)
plt.title(f'Distribution of {feature2}')
plt.xlabel(feature2)
plt.ylabel('Frequency')
plt.axvline(fdf_no_outliers[feature2].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(fdf_no_outliers[feature2].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

plt.tight_layout()
plt.show()


# In[34]:


correlation_data = fdf_no_outliers[['Mar Cap - Crore', 'Sale Qtr (in Cr)','Market Cap-to-Sales Ratio']]

correlation_matrix = correlation_data.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title('Correlation Heatmap - Mid Cap vs. Qtr Sales vs. Market Cap-to-Sales Ratio',fontweight='bold',fontsize = 14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()


# Observation: The correlation coefficient of 0.62 indicates a positive relationship between Market Capitalization and Quarterly Sales. However, there is minimal or negative correlation between these parameters and the ratio.

# In[35]:


top_n = 10  

top_companies = fdf_no_outliers.nlargest(top_n, 'Market Cap-to-Sales Ratio')

plt.figure(figsize=(12, 6))
plt.bar(top_companies['Name'], top_companies['Market Cap-to-Sales Ratio'], color='skyblue')
plt.title(f'Top {top_n} Companies by Market Cap-to-Sales Ratio')
plt.xlabel('Name')
plt.ylabel('Market Cap-to-Sales Ratio')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[36]:


# Create a scatter plot with a trend line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mar Cap - Crore', y='Sale Qtr (in Cr)', data=fdf_no_outliers, color='blue', alpha=0.7)
sns.regplot(x='Mar Cap - Crore', y='Sale Qtr (in Cr)', data=fdf_no_outliers, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

# Set plot title and labels
plt.title('Scatterplot with Trend Line - Market Cap vs. Qtr Sales')
plt.xlabel('Market Cap (in Crore)')
plt.ylabel('Quarterly Sales (in Crore)')
plt.grid(True)
plt.show()


# In[37]:


# Create a boxplot for Mar Cap - Crore
column_name = 'Market Cap-to-Sales Ratio'
plt.figure(figsize=(12, 6))
plt.boxplot(fdf_no_outliers[column_name], vert=False, sym='b.')
plt.title(f'Boxplot for {column_name}')
plt.xlabel(column_name)
plt.ylabel('Distribution')
plt.show()


# In[38]:


fdf_no_outliers


# # Exporting cleaned and preprocessed data

# In[39]:


fdf_no_outliers.to_csv('Project_4_Financial Analytics_(Cleaned and Preprocessed Dataset).csv')


# In[ ]:




