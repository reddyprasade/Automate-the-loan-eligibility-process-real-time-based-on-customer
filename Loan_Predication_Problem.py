#!/usr/bin/env python
# coding: utf-8

# <h1 align=center>Problem Statement</h1>
# 
# **About Company**
# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.
# 
# **Problem**
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.
# ________________________________________
# #### Data
# 
# |Variable|Description|
# |--------|-----------|
# |Loan_ID|Unique Loan ID|
# |Gender|Male/ Female|
# |Married|Applicant married (Y/N)|
# |Dependents|Number of dependents|
# |Education|Applicant Education (Graduate/ Under Graduate)|
# |Self_Employed|Self employed (Y/N)|
# |ApplicantIncome|Applicant income|
# |CoapplicantIncome|Coapplicant income|
# |LoanAmount|Loan amount in thousands|
# |Loan_Amount_Term|Term of loan in months|
# |Credit_History|credit history meets guidelines|
# |Property_Area|Urban/ Semi Urban/ Rural|
# |Loan_Status|Loan approved (Y/N)|
# 
# <hr> 
# 
# Note: 
# 1.	Evaluation Metric is accuracy i.e. percentage of loan approval you correctly predict[Accuracy Metric,Precision Metric,Classification Report ,Confustion Matrix].
# 2.	You are expected to upload the solution in the format of "sample_submission.csv"
# 
# #### Loading Packages

# In[109]:


import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling                # Report Generation
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")


# #### Data
# 
# For this practice problem, we have been given three CSV files: train, test and sample submission.
# 
# * Train file will be used for training the model, i.e. our model will learn from this file. It contains all the independent variables and the target variable.
# * Test file contains all the independent variables, but not the target variable. We will apply the model to predict the target variable for the test data.
# * Sample submission file contains the format in which we have to submit our predictions.

# In[110]:


train=pd.read_csv("Data/train_ctrUa4K.csv") 
test=pd.read_csv("Data/test_lAUu6dG.csv")


# The file name should be replaced with the name of the train and test file that you have downloaded from the github
# 
# Let’s make a copy of train and test data so that even if we have to make any changes in these datasets we would not lose the original datasets.

# In[111]:


train_original=train.copy() 
test_original=test.copy()


# In[112]:


train_original.shape


# In[113]:


test_original.shape


# In this section, we will look at the structure of the train and test datasets. Firstly, we will check the features present in our data and then we will look at their data types.

# In[114]:


train.columns


# We have 12 independent variables and 1 target variable, i.e. Loan_Status in the train dataset. Let’s also have a look at the columns of test dataset.

# In[115]:


test.columns


# We have similar features in the test dataset as the train dataset except the Loan_Status. We will predict the Loan_Status using the model built using the train data.
# 
# Given below is the description for each variable.

# |Variable|Description|
# |--------|-----------|
# |Loan_ID|Unique Loan ID|
# |Gender|Male/ Female|
# |Married|Applicant married (Y/N)|
# |Dependents|Number of dependents|
# |Education|Applicant Education (Graduate/Under Graduate)|
# |Self_Employed|Self employed (Y/N)|
# |ApplicantIncome|Applicant income|
# |CoapplicantIncome|Coapplicant income|
# |LoanAmount|Loan amount in thousands|
# |Loan_Amount_Term|Term of loan in months|
# |Credit_History|Credit history meets guidelines|
# |Property_Area|Urban/ Semi Urban/ Rural|
# |Loan_Status|Loan approved (Y/N)|

# In[116]:


pd.DataFrame({"Train_DataTypes":train.dtypes,
              "Test_DataTypes":test.dtypes})


# We can see there are three format of data types:
# 
# * `object:` Object format means variables are categorical. Categorical variables in our dataset are: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
# 
# * `int64:` It represents the integer variables. ApplicantIncome is of this format.
# 
# * `float64:` It represents the variable which have some decimal values involved. They are also numerical variables. Numerical variables in our dataset are: CoapplicantIncome, LoanAmount, Loan_Amount_Term, and Credit_History
# Let’s look at the shape of the dataset.

# In[117]:


train.shape


# In[118]:


test.shape


# We have 614 rows and 13 columns in the train dataset and 367 rows and 12 columns in test dataset.
# 
# 
# In this section, we will do univariate analysis. It is the simplest form of analyzing data where we examine each variable individually. For categorical features we can use frequency table or bar plots which will calculate the number of each category in a particular variable. For numerical features, probability density plots can be used to look at the distribution of the variable.
# 
# #### Target Variable
# 
# We will first look at the target variable, i.e., Loan_Status. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot.
# 
# Frequency table of a variable will give us the count of each category in that variable.

# In[119]:


train['Loan_Status'].value_counts()


# In[120]:


# Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)


# In[121]:


train['Loan_Status'].value_counts().plot.bar()


# The loan of 422(around 69%) people out of 614 was approved.
# 
# Now lets visualize each variable separately. Different types of variables are Categorical, ordinal and numerical.
# 
# * Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
# Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
# * Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)
# Let’s visualize the categorical and ordinal features first.

# In[122]:


plt.figure(1) 
plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()


# It can be inferred from the above bar plots that:
# 
# * 80% applicants in the dataset are male.
# * Around 65% of the applicants in the dataset are married.
# * Around 15% applicants in the dataset are self employed.
# * Around 85% applicants have repaid their debts.
# 
# Now let’s visualize the ordinal variables.

# #### Independent Variable (Ordinal)

# In[123]:



plt.figure(1) 
plt.subplot(131) 
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
plt.subplot(132) 
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133) 
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()


# Following inferences can be made from the above bar plots:
# 
# * Most of the applicants don’t have any dependents.
# * Around 80% of the applicants are Graduate.
# * Most of the applicants are from Semiurban area.
# 
# #### Independent Variable (Numerical)
# Till now we have seen the categorical and ordinal variables and now lets visualize the numerical variables. Lets look at the distribution of Applicant income first.

# In[124]:


plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['ApplicantIncome']); 
plt.subplot(122) 
train['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()


# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed. We will try to make it normal in later sections as algorithms works better if the data is normally distributed.
# 
# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels. Let us segregate them by Education:

# In[125]:


train.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("")


# We can see that there are a higher number of graduates with very high incomes, which are appearing to be the outliers.
# 
# Let’s look at the Coapplicant income distribution.

# In[126]:


plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['CoapplicantIncome']); 
plt.subplot(122) 
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()


# We see a similar distribution as that of the applicant income. Majority of coapplicant’s income ranges from 0 to 5000. We also see a lot of outliers in the coapplicant income and it is not normally distributed.
# 
# Let’s look at the distribution of LoanAmount variable.

# In[127]:


plt.figure(1) 
plt.subplot(121) 
df=train.dropna(inplace=True) 
sns.distplot(train['LoanAmount']); 
plt.subplot(122) 
train['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()


# We see a lot of outliers in this variable and the distribution is fairly normal. We will treat the outliers in later sections.
# 
# Now we would like to know how well each feature correlate with Loan Status. So, in the next section we will look at bivariate analysis.
# 
# 
# Lets recall some of the hypotheses that we generated earlier:
# 
# * Applicants with high income should have more chances of loan approval.
# * Applicants who have repaid their previous debts should have higher chances of loan approval.
# * Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# * Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.
# 
# Lets try to test the above mentioned hypotheses using bivariate analysis
# 
# After looking at every variable individually in univariate analysis, we will now explore them again with respect to the target variable.
# 
# #### Categorical Independent Variable vs Target Variable
# 
# First of all we will find the relation between target variable and categorical independent variables. Let us look at the stacked bar plot now which will give us the proportion of approved and unapproved loans.

# In[128]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(16,9))


# It can be inferred that the proportion of male and female applicants is more or less same for both approved and unapproved loans.
# 
# Now let us visualize the remaining categorical variables vs target variable.

# In[129]:


Married=pd.crosstab(train['Married'],train['Loan_Status']) 
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
Education=pd.crosstab(train['Education'],train['Loan_Status']) 
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 


# In[130]:


Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(16,9)) 
plt.show() 


# In[131]:


Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(16,9)) 
plt.show() 


# In[132]:


Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(16,9)) 
plt.show() 


# In[133]:


Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(16,9)) 
plt.show()


# * Proportion of married applicants is higher for the approved loans.
# * Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.
# * There is nothing significant we can infer from Self_Employed vs Loan_Status plot.
# 
# Now we will look at the relationship between remaining categorical independent variables and Loan_Status.

# In[134]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 


# In[135]:


Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(16,9))
plt.show() 


# In[136]:


Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(16,9)) 
plt.show()


# * It seems people with credit history as 1 are more likely to get their loans approved.
# * Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.
# 
# Now let’s visualize numerical independent variables with respect to target variable.
# 
# ### Numerical Independent Variable vs Target Variable
# We will try to find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.

# In[137]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar(figsize=(16,9))


# Here the y-axis represents the mean applicant income. We don’t see any change in the mean income. So, let’s make bins for the applicant income variable based on the values in it and analyze the corresponding loan status for each bin.

# In[138]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 


# In[139]:


Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(16,9)) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')


# It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed that if the applicant income is high the chances of loan approval will also be high.
# 
# We will analyze the coapplicant income and loan amount variable in similar manner.

# In[140]:


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 


# In[141]:


Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(16,9)) 
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')


# It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.
# 
# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

# In[142]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 


# In[143]:


Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20,9)) 
plt.xlabel('Total_Income') 
plt.ylabel('Percentage')


# We can see that Proportion of loans getting approved for applicants having low Total_Income is very less as compared to that of applicants with Average, High and Very High Income.
# 
# Let’s visualize the Loan amount variable.

# In[144]:


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])


# In[145]:


LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount')
plt.ylabel('Percentage')


# It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount which supports our hypothesis in which we considered that the chances of loan approval will be high when the loan amount is less.
# 
# Let’s drop the bins which we created for the exploration part. We will change the 3+ in dependents variable to 3 to make it a numerical variable.We will also convert the target variable’s categories into 0 and 1 so that we can find its correlation with numerical variables. One more reason to do so is few models like logistic regression takes only numeric values as input. We will replace N with 0 and Y with 1.

# In[146]:


train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True) 
train['Loan_Status'].replace('N', 0,inplace=True) 
train['Loan_Status'].replace('Y', 1,inplace=True)


# Now lets look at the correlation between all the numerical variables. We will use the heat map to visualize the correlation. Heatmaps visualize data through variations in coloring. The variables with darker color means their correlation is more.

# In[149]:


matrix = train.corr() 
f, ax = plt.subplots(figsize=(16, 9)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# We see that the most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status). LoanAmount is also correlated with CoapplicantIncome.

# After exploring all the variables in our data, we can now impute the missing values and treat the outliers because missing data and outliers can have adverse effect on the model performance.
# 
# ### Missing value imputation
# Let’s list out feature-wise count of missing values.

# In[151]:


train.isnull().sum()


# There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term and Credit_History features.
# 
# We will treat the missing values in all the features one by one.
# 
# We can consider these methods to fill the missing values:
# 
# * For numerical variables: imputation using mean or median
# * For categorical variables: imputation using mode
# There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features so we can fill them using the mode of the features.

# In[154]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# Now let’s try to find a way to fill the missing values in Loan_Amount_Term. We will look at the value count of the Loan amount term variable.

# In[156]:


train['Loan_Amount_Term'].value_counts()


# It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the missing values in this variable using the mode of this variable.

# In[158]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# Now we will see the LoanAmount variable. As it is a numerical variable, we can use mean or median to impute the missing values. We will use median to fill the null values as earlier we saw that loan amount have outliers so the mean will not be the proper approach as it is highly affected by the presence of outliers.

# In[160]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# Now lets check whether all the missing values are filled in the dataset.

# In[162]:


train.isnull().sum()


# As we can see that all the missing values have been filled in the test dataset. Let’s fill all the missing values in the test dataset too with the same approach.

# In[164]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# ### Outlier Treatment
# 
# As we saw earlier in univariate analysis, LoanAmount contains outliers so we have to treat them as the presence of outliers affects the distribution of the data. Let's examine what can happen to a data set with outliers. For the sample data set:
# 
# 
# 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4
# 
# We find the following: mean, median, mode, and standard deviation
# 
# Mean = 2.58
# 
# Median = 2.5
# 
# Mode = 2
# 
# Standard Deviation = 1.08
# 
# If we add an outlier to the data set:
# 
# 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 400
# 
# The new values of our statistics are:
# 
# Mean = 35.38
# 
# Median = 2.5
# 
# Mode = 2
# 
# Standard Deviation = 114.74
# 
# It can be seen that having outliers often has a significant effect on the mean and standard deviation and hence affecting the distribution. We must take steps to remove outliers from our data sets.
# 
# Due to these outliers bulk of the data in the loan amount is at the left and the right tail is longer. This is called right skewness. One way to remove the skewness is by doing the log transformation. As we take the log transformation, it does not affect the smaller values much, but reduces the larger values. So, we get a distribution similar to normal distribution.
# 
# Let’s visualize the effect of log transformation. We will do the similar changes to the test file simultaneously.

# In[167]:


train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 
test['LoanAmount_log'] = np.log(test['LoanAmount'])


# Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided. Let’s build a logistic regression model and make predictions for the test dataset.
# 
# The process of model building is not complete without evaluation of model’s performance. Suppose we have the predictions from the model, how can we decide whether the predictions are accurate? We can plot the results and compare them with the actual values, i.e. calculate the distance between the predictions and actual values. Lesser this distance more accurate will be the predictions. Since this is a classification problem, we can evaluate our models using any one of the following evaluation metrics:
# 
# Accuracy: Let us understand it using the confusion matrix which is a tabular representation of Actual vs Predicted values. This is how a confusion matrix looks like:
# 
# <img src='https://lh3.googleusercontent.com/-uT9iaVy0uPI/XhMomA7OsqI/AAAAAAAAlwo/X0ikk8YWrzs48no_Wt0ScRU1rX34bldXgCK8BGAsYHg/s0/2020-01-06.png' />
# 
# * True Positive - Targets which are actually true(Y) and we have predicted them true(Y)
# * True Negative - Targets which are actually false(N) and we have predicted them false(N)
# * False Positive - Targets which are actually false(N) but we have predicted them true(T)
# * False Negative - Targets which are actually true(T) but we have predicted them false(N)
# 
# Using these values, we can calculate the accuracy of the model. The accuracy is given by:

# <img src="https://lh3.googleusercontent.com/-FySFkWnPvO0/XhMsAN-hepI/AAAAAAAAlw0/eYNkU-Wxj78RvlqbGU3jmXP320b20L35wCK8BGAsYHg/s0/2020-01-06.png"/>

# **Precision:** It is a measure of correctness achieved in true prediction i.e. of observations labeled as true, how many are actually labeled true.
# Precision = TP / (TP + FP)
# 
# **Recall(Sensitivity)** - It is a measure of actual observations which are predicted correctly i.e. how many observations of true class are labeled correctly. It is also known as ‘Sensitivity’.
# Recall = TP / (TP + FN)
# 
# **Specificity** - It is a measure of how many observations of false class are labeled correctly.
# Specificity = TN / (TN + FP)
# 
# Specificity and Sensitivity plays a crucial role in deriving ROC curve.
# 
# **ROC curve**
# Receiver Operating Characteristic(ROC) summarizes the model’s performance by evaluating the trade offs between true positive rate (sensitivity) and false positive rate(1- specificity).
# The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction power of the model.
# This is how a ROC curve looks like:
# 
# <img src = "https://lh3.googleusercontent.com/-gMM5YSC_b5Y/XhMtBPogmyI/AAAAAAAAlw8/2ZEaJ0gxAsMNY_9QHdemzSdy08yA4ax-wCK8BGAsYHg/s0/2020-01-06.png"/>
# 
# *  The area of this curve measures the ability of the model to correctly classify true positives and true negatives. We want our model to predict the true classes as true and false classes as false.
# * So it can be said that we want the true positive rate to be 1. But we are not concerned with the true positive rate only but the false positive rate too. For example in our problem, we are not only concerned about predicting the Y classes as Y but we also want N classes to be predicted as N.
# * We want to increase the area of the curve which will be maximum for class 2,3,4 and 5 in the above example.
# * For class 1 when the false positive rate is 0.2, the true positive rate is around 0.6. But for class 2 the true positive rate is 1 at the same false positive rate. So, the AUC for class 2 will be much more as compared to the AUC for class 1. So, the model for class 2 will be better.
# * The class 2,3,4 and 5 model will predict more accurately as compared to the class 0 and 1 model as the AUC is more for those classes.
# At the competition’s page, it has been mentioned that our submission data would be evaluated based on the accuracy. Hence, we will use accuracy as our evaluation metric.
# 
# 
# <h3 align=center >Model Evaluation Metrics for Machine Learning </h3>

# 1. Confusion Matrix
# 2. F1 Score
# 3. Gain and Lift Charts
# 4. Kolmogorov Smirnov Chart
# 5. AUC – ROC
# 6. Log Loss
# 7. Gini Coefficient
# 8. Concordant – Discordant Ratio
# 9. Root Mean Squared Error

# #### Model Building

# Let us make our first model to predict the target variable. We will start with Logistic Regression which is used for predicting binary outcome.
# 
# * Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
# * Logistic regression is an estimation of Logit function. Logit function is simply a log of odds in favor of the event.
# * This function creates a s-shaped curve with the probability estimate, which is very similar to the required step wise function
# 
# Lets drop the Loan_ID variable as it do not have any effect on the loan status. We will do the same changes to the test dataset which we did for the training dataset.
# 

# In[ ]:


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)


# We will use scikit-learn (sklearn) for making different models which is an open source library for Python. It is one of the most efficient tool which contains many inbuilt functions that can be used for modeling in Python.

# Sklearn requires the target variable in a separate dataset. So, we will drop our target variable from the train dataset and save it in another dataset.

# In[172]:


X = train.drop('Loan_Status',1) 
y = train.Loan_Status


# Now we will make dummy variables for the categorical variables. Dummy variable turns categorical variables into a series of 0 and 1, making them lot easier to quantify and compare. Let us understand the process of dummies first:
# 
# * Consider the “Gender” variable. It has two classes, Male and Female.
# * As logistic regression takes only the numerical values as input, we have to change male and female into numerical value.
# * Once we apply dummies to this variable, it will convert the “Gender” variable into two variables(Gender_Male and Gender_Female), one for each class, i.e. Male and Female.
# * Gender_Male will have a value of 0 if the gender is Female and a value of 1 if the gender is Male.

# In[174]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# Now we will train the model on training dataset and make predictions for the test dataset. But can we validate these predictions? One way of doing this is we can divide our train dataset into two parts: train and validation. We can train the model on this train part and using that make predictions for the validation part. In this way we can validate our predictions as we have the true predictions for the validation part (which we do not have for the test dataset).
# 
# We will use the train_test_split function from sklearn to divide our train dataset. So, first let us import train_test_split.

# In[176]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


# The dataset has been divided into training and validation part. Let us import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.

# In[178]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, 
                   fit_intercept=True,          
                   intercept_scaling=1, 
                   max_iter=100,
                   multi_class='ovr', 
                   n_jobs=1,          
                   penalty='l2', 
                   random_state=1, 
                   solver='liblinear', 
                   tol=0.0001,          
                   verbose=0, 
                   warm_start=False)


# Here the C parameter represents inverse of regularization strength. Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values of C specify stronger regularization

# Let’s predict the Loan_Status for validation set and calculate its accuracy.

# In[179]:


pred_cv = model.predict(x_cv)


# In[181]:


accuracy_score(y_cv,pred_cv)


# So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly.
# 

# In[184]:


# Let’s make predictions for the test dataset.

pred_test = model.predict(test)


# 
# Lets import the submission file which we have to submit on the solution checker.

# In[187]:


submission=pd.read_csv("Data/sample_submission_49d68Cx.csv")


# We only need the Loan_ID and the corresponding Loan_Status for the final submission. we will fill these columns with the Loan_ID of test dataset and the predictions that we made, i.e., pred_test respectively.

# In[188]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

# In[190]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[191]:


df = pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Data/logistic.csv')
df


# In[192]:


df

