# Linear-Regression-
Analysing a retail outlet sales data and predicting the results of the given data consecutively using machine learning techniques of Linear Regression and SKlearn


# # Simple Linear Regression
# 
# In this notebook, we'll build a linear regression model to predict `Sales` using an appropriate predictor variable.

# ## Step 1: Reading and Understanding the Data
# 
# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[74]:


# Import the numpy and pandas package

import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[11]:


# Read the given CSV file, and view some sample records

advertising = pd.read_csv("advertising.csv")
advertising.head()


# Let's inspect the various aspects of our dataframe

# In[12]:


advertising.shape


# In[13]:


advertising.info()


# ## Step 2: Visualising the Data
# 
# Let's now visualise our data using seaborn. We'll first make a pairplot of all the variables present to visualise which variables are most correlated to `Sales`.

# In[14]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[17]:


#visualize the data 
sns.regplot(x='Radio', y='Sales', data = advertising)


# In[15]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales',size=4, aspect=1, kind='scatter')
plt.show()


# In[18]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# As is visible from the pairplot and the heatmap, the variable `TV` seems to be most correlated with `Sales`. So let's go ahead and perform simple linear regression using `TV` as our feature variable.

# ---
# ## Step 3: Performing Simple Linear Regression
# 
# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 
# ---

# # Steps:
# 
# Create X and Y 
# Create traun ad test sets (70-30,80-20)
# Train your model on te training test (ie. learn the coeffecients)
# Evaluuate the midel (training set, test set) 

# ### Generic Steps in model building using `statsmodels`
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[43]:


X = advertising['TV']
y = advertising['Sales']


# #### Train-Test Split
# 
# You now need to split our variable into training and testing sets. You'll perform this by importing `train_test_split` from the `sklearn.model_selection` library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset

# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[45]:


# Let's now take a look at the train dataset

X_train.head()


# In[46]:


X_train.shape

y_train.head()
# #### Building a Linear Model
# 
# You first need to import the `statsmodel.api` library using which you'll perform the linear regression.

# In[26]:


import statsmodels.api as sm


# By default, the `statsmodels` library fits a line on the dataset which passes through the origin. But in order to have an intercept, you need to manually use the `add_constant` attribute of `statsmodels`. And once you've added the constant to your `X_train` dataset, you can go ahead and fit a regression line using the `OLS` (Ordinary Least Squares) attribute of `statsmodels` as shown below

# In[34]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[35]:


lr_model = lr.fit()
lr_model.params


# In[36]:


X_train_sm.head()


# In[31]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[32]:


Lr_model = lr.fit()


# In[37]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# ####  Looking at some key statistics from the summary

# The values we are concerned with are - 
# 1. The coefficients and significance (p-values)
# 2. R-squared
# 3. F statistic and its significance

# ##### 1. The coefficient for TV is 0.054, with a very low p value
# The coefficient is statistically significant. So the association is not purely by chance. 

# ##### 2. R - squared is 0.816
# Meaning that 81.6% of the variance in `Sales` is explained by `TV`
# 
# This is a decent R-squared value.

# ###### 3. F statistic has a very low p value (practically low)
# Meaning that the model fit is statistically significant, and the explained variance isn't purely by chance.

# ---
# The fit is significant. Let's visualize how well the model fit the data.
# 
# From the parameters that we get, our linear regression equation becomes:
# 
# $ Sales = 6.948 + 0.054 \times TV $

# In[40]:


plt.scatter(X_train, y_train)


# In[38]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# ## Step 4: Residual analysis 
# To validate assumptions of the model, and hence the reliability for inference

# #### Distribution of the error terms
# We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


# predicted values (reiduals) 
#error = f(y_train -  y_train_pred)


# In[50]:


# since we have already prepared a lr model we can use the trained model with trained data set to predict y_train
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[51]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# The residuals are following the normally distributed with a mean 0. All good!

# #### Looking for patterns in the residuals

# In[52]:


plt.scatter(X_train,res)   # where res is y values
plt.show() 

# we see residuals are normally distributed


# We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.
# 
# Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

# As you can see, the regression line is a pretty good fit to the data

# ## Step 5: Predictions on the Test Set
# 
# Now that you have fitted a regression line on your train dataset, it's time to make some predictions on the test data. For this, you first need to add a constant to the `X_test` data like you did for `X_train` and then you can simply go on and predict the y values corresponding to `X_test` using the `predict` attribute of the fitted regression line.

# In[55]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[56]:


y_pred.head()


# In[57]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ## Evaluating all the models 

# ##### Looking at the RMSE

# In[59]:


#r-squared
#2 = r2_score(y_true = y_test, y_pred = y_test_pred)


# In[61]:


#r2 on train 
#r2 = r2_score(y_true = y_test , y_pred = y_test_pred)


# In[ ]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:





# ###### Checking the R-squared on the test set

# In[60]:


r_squared = r2_score(y_test, y_pred)
r_squared


# ##### Visualizing the fit on the test set

# In[62]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# #mean squared error 
# mean_squared_error(y_true = y_test, y_pred = y_test_pred)

#  

#  

#  

#  

# ### Linear Regression using `linear_model` in `sklearn`
# 
# Apart from `statsmodels`, there is another package namely `sklearn` that can be used to perform linear regression. We will use the `linear_model` library from `sklearn` to build the model. Since, we hae already performed a train-test split, we don't need to do it again.
# 
# There's one small step that we need to add, though. When there's only a single feature, we need to add an additional column in order for the linear regression fit to be performed successfully.

# In[75]:


from sklearn.model_selection import train_test_split
X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[76]:


X_train_lm.shape


# In[80]:


print(X_train_lm.shape)
print(y_train_lm.shape)
print(X_test_lm.shape)
print(y_test_lm.shape)


# In[82]:


X_train_lm = X_train_lm.values.reshape(-1,1)
X_test_lm = X_test_lm.values.reshape(-1,1)


# In[83]:


print(X_train_lm.shape)
print(y_train_lm.shape)
print(X_test_lm.shape)
print(y_test_lm.shape)


# In[93]:


from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# Fit the model using lr.fit()
lm.fit(X_train_lm, y_train_lm)


# In[85]:


help(LinearRegression)


# In[94]:


print(lm.intercept_)
print(lm.coef_)


# In[97]:


# make predictions 
y_train_pred = lm.predict(X_train_lm)
y_test_pred = lm.predict(X_test_lm)


# In[96]:


### Evaluate the model 
print(r2_score(y_true = y_train, y_pred = y_train_pred))
print(r2_score(y_true = y_test, y_pred = y_test_pred))


# The equationwe get is the same as what we got before!
# 
# $ Sales = 6.948 + 0.054* TV $

# In[98]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[99]:


fig2 = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms using Sklearn', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[100]:


plt.scatter(X_train,res)   # where res is y values
plt.show() 
