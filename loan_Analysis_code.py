#!/usr/bin/env python
# coding: utf-8

# ## 1- Import Libraries and Load Data

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
pd.get_option("display.max_column")
pd.get_option("display.max_rows",None)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Loan_Data.csv')
df.head(5)


# ## 2- Data Exploration

# In[5]:


df.sample(n=5)


# In[6]:


df.info()


# In[7]:


df.describe(include= 'all')


# In[8]:


df.isnull().sum()


# In[9]:


#Lets chcek the value counts for categorical data
for i in df.columns:
    if df[i].dtypes == 'object':
        print(df[i].value_counts(dropna=False))
        print('---------'*10)


# In[10]:


df.Loan_Amount_Term.value_counts(dropna=False)


# In[11]:


df.Credit_History.value_counts(dropna=False)


# ## 3- Data Cleaning

# In[12]:


columns_to_fill_with_mode = ["Gender", "Married", "Self_Employed", "Dependents"]
columns_to_fill_with_mean = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]

for col in columns_to_fill_with_mode:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in columns_to_fill_with_mean:
    df[col].fillna(df[col].mean(), inplace=True)


# In[13]:


column = ['Dependents']
for i in column:
    df[i] = df[i].replace({'3+':'3'})

df.Dependents.unique()    


# In[14]:


df.isnull().sum()


# ### -Treatment outliers 

# In[15]:


"""

# Identify numerical columns for outlier removal (excluding Credit_History)
numerical_columns = df.select_dtypes(include=[np.number]).columns.difference(['Credit_History'])

numerical_data = df[numerical_columns]

# Calculate quartiles and IQR 
Q1 = numerical_data.quantile(0.25)
Q3 = numerical_data.quantile(0.75)
IQR = Q3 - Q1

# Apply outlier removal condition correctly
df = df[~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

"""


# In[16]:


"""
# Removing skewness in LoanAmount variable by log transformation
df['LoanAmount_log'] = np.log(df['LoanAmount'])

ax1 = plt.subplot(121)
df['LoanAmount_log'].hist(bins=20, figsize=(12,4))
ax1.set_title("Train")

"""


# In[17]:


for col in df.columns:
    print(f"{col}: {df[col].unique()}")


# In[18]:


df.describe(include="all")


# ### - Feature Engineering: 

# In[19]:


df["Total Income"]= df["ApplicantIncome"] + df["CoapplicantIncome"]
df.head()


# ## 4-Visualization and Answer Questions

# ### 4.1 Data Overview and Summary:

# In[20]:


# How many loan applications are included in the dataset?
num_applications = df.shape[0]
print("Number of loan applications:", num_applications)


# In[21]:


# What is the overall approval rate for loan applications in the dataset?
approval_rate = df['Loan_Status'].value_counts(normalize=True)['Y'] * 100
print("Overall loan approval rate:", approval_rate, "%")


# In[22]:


# What is the distribution of loan applicants based on gender and marital status?
# Create a crosstab to count applicants by gender and marital status
gender_marital_crosstab = pd.crosstab(df['Gender'], df['Married'])

# Print the crosstab to see the numerical distribution
print("Distribution of Loan Applicants by Gender and Marital Status:\n", gender_marital_crosstab)

# Create a stacked bar chart to visualize the distribution
gender_marital_crosstab.plot(kind='bar', stacked=True)
plt.xlabel('Gender')
plt.ylabel('Number of Applicants')
plt.title('Distribution of Loan Applicants by Gender and Marital Status')
plt.legend(title='Married')
plt.show()


# ### 4.1 Key Finding: 
# ##### In our dataset comprising 614 loan applications, the overall loan approval rate stands at 68.73%. Exploring the distribution based on gender and marital status, we find diverse demographics seeking financial assistance.

# ###### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### 4.2. Income and Loan Amount:

# In[23]:


#What is the range of applicant income and coapplicant income?
# Calculate income ranges for each column
income_ranges = {
    "ApplicantIncome": df["ApplicantIncome"].max() - df["ApplicantIncome"].min(),
    "CoapplicantIncome": df["CoapplicantIncome"].max() - df["CoapplicantIncome"].min()
}

# Print the income ranges
print("Income Ranges:")
for income, range_value in income_ranges.items():
    print(f"{income}: {range_value}")        


# In[24]:


#What is the range of applicant income and coapplicant income?
income_data = df[["ApplicantIncome", "CoapplicantIncome"]]

# Create histograms for each income column
plt.figure(figsize=(10, 5)) 

plt.hist(income_data["ApplicantIncome"], bins=15, alpha=0.7, label="Applicant Income")
plt.hist(income_data["CoapplicantIncome"], bins=15, alpha=0.7, label="Coapplicant Income")

plt.title("Distribution of Applicant and Coapplicant Incomes")
plt.xlabel("Income (in relevant currency)")  # Specify currency if applicable
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[25]:


#How does the distribution of applicant income differ based on gender and education?
genders = df["Gender"].unique()
education_levels = df["Education"].unique()

# Define colors for each gender
gender_colors = {"Male": "blue", "Female": "orange"}  # Adjust colors as needed

# Create grid of subplots
fig, axes = plt.subplots(nrows=len(genders), ncols=len(education_levels), figsize=(10, 6))

# Iterate through subplots and create histograms with descriptive statistics
for i, gender in enumerate(genders):
    for j, education_level in enumerate(education_levels):
        subset = df[(df["Gender"] == gender) & (df["Education"] == education_level)]

        # Calculate descriptive statistics
        mean_income = subset["ApplicantIncome"].mean()
        std_dev = subset["ApplicantIncome"].std()

        # Create histogram with descriptive labels
        axes[i, j].hist(subset["ApplicantIncome"], bins=15, alpha=0.7, color=gender_colors[gender])
        axes[i, j].set_title(f"{gender} - {education_level}\nMean: {mean_income:.2f}\nStd Dev: {std_dev:.2f}")
        axes[i, j].set_xlabel("Applicant Income")  # Specify currency if applicable
        axes[i, j].set_ylabel("Frequency")

plt.tight_layout()  # Adjust spacing
plt.show()


# In[26]:


#Is there a correlation between applicant income and the requested loan amount?
# Calculate correlation coefficient
correlation = df["ApplicantIncome"].corr(df["LoanAmount"])

# Print result
print("Correlation coefficient between applicant income and loan amount:", correlation)

# Interpret the correlation
if correlation > 0.7:
    print("Strong positive correlation")
elif correlation > 0.3:
    print("Moderate positive correlation")
elif correlation > 0:
    print("Weak positive correlation")
elif correlation < -0.3:
    print("Moderate negative correlation")
elif correlation < 0:
    print("Weak negative correlation")
else:
    print("No significant correlation")

# Create scatter plot to visualize the relationship
plt.figure(figsize=(8, 6))
plt.scatter(df["ApplicantIncome"], df["LoanAmount"], alpha=0.7)

plt.title("Correlation between Applicant Income and Loan Amount")
plt.xlabel("Applicant Income (in relevant currency)")  # Specify currency if applicable
plt.ylabel("Loan Amount (in relevant currency)")
plt.show()


# ### 4.2 Key Finding: 
# ##### The income range for applicants and co-applicants spans from $0 to $80,850 and $0 to $41,667, respectively. Delving into gender and education-based income disparities, we observe variations among different groups. Additionally, there exists a moderate positive correlation (0.57) between applicant income and the requested loan amount.

# ###### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### 4.3. Loan Approval and Credit History:

# In[27]:


# Calculate approval rates directly within crosstab
approval_rates = ( pd.crosstab(df["Credit_History"], df["Loan_Status"], normalize="index") * 100).astype(int)

# Print approval rates (integers)
print(approval_rates)

# Create stacked bar chart with integer labels
approval_rates.plot(kind="bar", stacked=True, figsize=(6, 4))
plt.xlabel("Credit History")
plt.ylabel("Approval Rate (%)")
plt.title("Loan Approval Rates by Credit History")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}%"))  # Format y-axis labels as integers
plt.show()


# In[28]:


# Create crosstab with all three variables
crosstab = pd.crosstab([df['Credit_History'], df['Property_Area']], df['Loan_Status'])

# Calculate approval rates within each combination
approval_rates = crosstab.div(crosstab.sum(axis=1), axis=0)["Y"]

# Calculate percentage of approved loans for good credit history
good_credit_approval_rate = approval_rates.loc[1].mean() * 100

# Print results
print(f"Percentage of loan applicants with good credit history approved: {good_credit_approval_rate:.2f}%")

# Analyze differences in approval rates by property area
print("\nLoan approval rates by property area:")
print(approval_rates.loc[1])  # Show rates for good credit history

# Visualize approval rates for good credit history
plt.figure(figsize=(8, 6))
plt.bar(approval_rates.loc[1].index, approval_rates.loc[1].values, color='skyblue')
plt.title("Loan Approval Rates for Good Credit History by Property Area")
plt.xlabel("Property Area")
plt.ylabel("Approval Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### 4.3 Key Finding:
# ##### Credit history significantly influences loan approval rates, with 78.68% of applicants possessing good credit history securing loans. Furthermore, property area impacts approval rates, with semiurban areas leading at 87.70%.

# ###### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### 4.4.Dependents and Education:

# In[29]:


dependents_crosstab= pd.crosstab(df["Dependents"], df["Loan_Status"])

# Print the original crosstab
print("Distribution of Loan Statuses by Number of Dependents:")
print(dependents_crosstab)

# Calculate approval rates
approval_rates = dependents_crosstab["Y"] / dependents_crosstab.sum(axis=1) * 100

# Print the calculated approval rates
print("\nLoan approval rates by number of dependents:")
print(approval_rates)


# In[30]:


# Visualize the original crosstab (unchanged)
dependents_crosstab.div(dependents_crosstab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Dependents')
plt.ylabel('Percentage')
plt.title("Distribution of Loan Statuses by Number of Dependents")
plt.show()

# Visualize approval rates with percentages above columns (corrected)
plt.figure(figsize=(8, 6))
bars = plt.bar(approval_rates.index, approval_rates.values, color='skyblue')  # Create bars for accurate positioning

# Use `get_width` and `get_x` methods for accurate positioning
for bar, height in zip(bars, approval_rates):
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center")

plt.title("Loan Approval Rates by Number of Dependents")
plt.xlabel("Number of Dependents")
plt.ylabel("Approval Rate (%)")
plt.xticks(rotation=0)  
plt.tight_layout()
plt.show()


# In[31]:


print(pd.crosstab(df['Education'],df['Loan_Status']))

Education=pd.crosstab(df['Education'],df['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Education')
p = plt.ylabel('Percentage')


# In[32]:


approval_rates = (pd.crosstab(df['Education'], df['Loan_Status'])["Y"] / pd.crosstab(df['Education'], df['Loan_Status']).sum(axis=1) * 100)

# Print approval rates
print("Loan approval rates by education level:")
print(approval_rates)

fig, ax = plt.subplots(figsize=(8, 6))  
approval_rates.plot(kind="bar", color='skyblue', ax=ax)  
plt.bar_label(ax.containers[0])  


plt.title("Loan Approval Rates by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Approval Rate (%)")
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()


# ### 4.4 Key Finding:
# ##### The number of dependents does influence loan approval, revealing a trend towards decreasing approval rates as the number of dependents increases. Educational background is also a factor, with graduates enjoying a higher approval rate (70.83%) compared to non-graduates (61.19%).

# ###### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### 4.5.Loan Term and Amount:

# In[33]:


# Convert the categorical variables to numeric using pandas factorize method
df['Gender'], _ = pd.factorize(df['Gender'])
df['Married'], _ = pd.factorize(df['Married'])
df['Education'], _ = pd.factorize(df['Education'])
df['Self_Employed'], _ = pd.factorize(df['Self_Employed'])
df['Property_Area'], _ = pd.factorize(df['Property_Area'])
df['Loan_Status'], _ = pd.factorize(df['Loan_Status'])


# In[34]:


# Print mean applicant income by loan status
print(df.groupby('Loan_Status')['ApplicantIncome'].mean())  

# Create the bar chart
df.groupby('Loan_Status')['ApplicantIncome'].mean().plot(kind='bar')
plt.xlabel('Loan Status')
plt.ylabel('Average Applicant Income')
plt.title('Average Applicant Income by Loan Status')
plt.show()


# In[35]:


# Create a box plot of loan amount vs property area
plt.figure(figsize=(10, 6))
plt.boxplot([df[df['Property_Area'] == 0]['LoanAmount'],
             df[df['Property_Area'] == 1]['LoanAmount'],
             df[df['Property_Area'] == 2]['LoanAmount']],
            labels=['Rural', 'Semiurban', 'Urban'])
plt.xlabel('Property Area')
plt.ylabel('Loan Amount (Thousands)')
plt.title('Loan Amount vs Property Area')
plt.show()


# In[36]:


# making bins for Total Income variable
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High', 'Very high']
df['Total_Income_bin'] = pd.cut(df['Total Income'],bins,labels=group)


# In[37]:


# plot the chart
Total_Income_bin = pd.crosstab(df['Total_Income_bin'],df['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')


# #### 
# While applicants across all income levels experience some loan approvals, the data clearly demonstrates a decreasing trend in approval rates as total income decreases. This finding suggests a systematic preference for applicants with higher incomes, warranting further investigation into potential fairness concerns.

# In[38]:


# making bins for LoanAmount variable
bins = [0,100,200,700]
group = ['Low','Average','High']
df['LoanAmount_bin'] = pd.cut(df['LoanAmount'],bins,labels=group)


# In[39]:


# plot the chart 
LoanAmount_bin = pd.crosstab(df['LoanAmount_bin'],df['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')


# #### 
# The data reveals a significant decrease in loan approval rates as loan amounts increase. This finding validates our hypothesis that borrowers seeking smaller loans face a higher chance of approval, potentially highlighting a risk-averse lending strategy.

# ###### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### 4.5 Key Finding.. Loan Term and Amount:
#   ##### Analyzing loan amounts across different property areas, we find a slight variation, with semiurban areas having marginally higher loan amounts on average.

# ### 4.6 Key Finding... Summary and Insights:
# ##### 1. Across all income levels, there's a discernible decrease in loan approval rates as total income decreases, indicating a preference for higher-income applicants.
# ##### 2. Notably, there's a significant drop in loan approval rates with increasing loan amounts, underscoring a risk-averse lending strategy favoring smaller loans.

# In[ ]:




