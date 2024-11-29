#!/usr/bin/env python
# coding: utf-8

# <a id='intro'></a>
# # Introduction
# **The aim of the project** is to employ several supervised algorithms to accurately model individuals' income, whether he makes more than 50,000 or not, using data collected from the 1994 U.S. Census.
# 
# The dataset that will be used is the **census income dataset**, which was extracted from the machine learning repository (UCI), which contains about 32561 rows and 15 features.
# 
# ## Dataset Description
# <table>
#     <tr>
#         <th><center> Name </center></th>
#         <th><center> Type </center></th>
#         <th><center> Values </center></th>
#         <th><center> Description </center></th>
#     </tr>
#     <tr>
#         <td><center> age </center></td>
#         <td><center> Continuous </center></td>
#         <td><center> From 17 to 90 </center></td>
#         <td><center> The age of an individual </center></td>
#     </tr>
#     <tr>
#         <td><center> workclass </center></td>
#         <td><center> Nominal </center></td>
#         <td><center> Private, Federal-Government, etc </center></td>
#         <td><center> A general term to represent the employment status of an individual</center></td>
#     </tr>
#     <tr>
#         <td><center> fnlwgt </center></td>
#         <td><center> Continuous </center></td>
#         <td><center> Integer greater than 0 </center></td>
#         <td><center> Final weight: is the number of people the census believes the entry represents </center></td>
#     </tr>
#     <tr>
#         <td><center> education </center></td>
#         <td><center> Ordinal  </center></td>
#         <td><center> Some-college, Prof-school, etc </center></td>
#         <td><center> The highest level of education achieved by an individual </center></td>
#     </tr>
#     <tr>
#         <td><center> education-num </center></td>
#         <td><center> Discrete  </center></td>
#         <td><center> From 1 to 16 </center></td>
#         <td><center> The highest level of education achieved in numerical form </center></td>
#     </tr>
#     <tr>
#         <td><center> marital-status </center></td>
#         <td><center> Nominal  </center></td>
#         <td><center> Married, Divorced, etc. </center></td>
#         <td><center> Marital status of an individual </center></td>
#     </tr>
#     <tr>
#         <td><center> occupation </center></td>
#         <td><center> Nominal </center></td>
#         <td><center> Transport-Moving, Craft-Repair, etc </center></td>
#         <td><center> The general type of occupation of an individual </center></td>
#     </tr>
#     <tr>
#         <td><center> relationship </center></td>
#         <td><center> Nominal  </center></td>
#         <td><center> Unmarried, not in the family, etc </center></td>
#         <td><center> Represents what this individual is relative to others  </center></td>
#     </tr>
#     <tr>
#         <td><center> race </center></td>
#         <td><center> Nominal  </center></td>
#         <td><center> White, Black, Hispanic, etc. </center></td>
#         <td><center> Descriptions of an individual’s race </center></td>
#     </tr>
#     <tr>
#         <td><center> sex </center></td>
#         <td><center> Nominal  </center></td>
#         <td><center> Male, Female </center></td>
#         <td><center> The biological sex of the individual </center></td>
#     </tr>
#     <tr>
#         <td><center> capital-gain </center></td>
#         <td><center> Continous </center></td>
#         <td><center> Integer greater than or equal to 0 </center></td>
#         <td><center> Capital gains for an individual  </center></td>
#     </tr>
#     <tr>
#         <td><center> capital-loss </center></td>
#         <td><center> Continous </center></td>
#         <td><center> Integer greater than or equal to 0 </center></td>
#         <td><center> Capital loss for an individual  </center></td>
#     </tr>
#     <tr>
#         <td><center> hours-per-week </center></td>
#         <td><center> Continous </center></td>
#         <td><center> From 1 to 99 </center></td>
#         <td><center> The hours an individual has reported to work per week </center></td>
#     </tr>
#     <tr>
#         <td><center> native-country </center></td>
#         <td><center> Nominal  </center></td>
#         <td><center> United-States, Cambodia, England, Puerto-Rico, Canada and more </center></td>
#         <td><center> Country of origin for an individual  </center></td>
#     </tr>
#     <tr>
#         <td><center> income </center></td>
#         <td><center> Discrete  </center></td>
#         <td><center> (≤50k USD, >50k USD) </center></td>
#         <td><center> The label whether or not an individual  </center></td>
#     </tr>
# 
# </table>

# In[1]:


import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore


# In[2]:


# Clean the income column
import pandas as pd # type: ignore
from ucimlrepo import fetch_ucirepo  # type: ignore
df = fetch_ucirepo(id=20) 


# In[3]:


# metadata 
print(df.metadata) 
  
# variable information 
print(df.variables) 


# In[4]:


X_str = df.data.features 
y_str = df.data.targets 
data = pd.concat([X_str, y_str], axis=1) 

print(data.info())


# In[5]:


print(data.head())


# In[6]:


# Display column names
print("\nColumn names in the DataFrame:")
print(data.columns)


# In[7]:


# Display the structure of the fetched data_ss
print("Features (X) Shape: ", X_str.shape)
print("Target (y) Shape: ", y_str.shape)

# Display the first few rows of features and target
print("\nFirst 5 rows of Features (X):")
print(X_str.head())

print("\nFirst 5 rows of Target (y):")
print(y_str.head())


# ## Pre-processing

# In[8]:


print(data['income'].value_counts())    # Check the distribution of the target variable


# - We found that there is an error with income column, s.t., there exists more than the two classes mentioned (<=50K, >50K)

# In[9]:


# Remove trailing periods in the 'income' column to standardize the labels
data['income'] = data['income'].str.strip().replace({'<=50K.': '<=50K', '>50K.': '>50K'})


# In[10]:


print(data["workclass"].value_counts())
print("\n")
print(data["occupation"].value_counts())
print("\n")
print(data["native-country"].value_counts())
print("\n")


# - We found "?" symbol in the dataset, so we changed it to "Unknown", for better interpretation and cleaner representation.

# In[11]:


# changing "?" to Unknown
change_columns = ['workclass', 'occupation', 'native-country']
for column in change_columns:
        data[column] = data[column].replace({'?': 'Unknown'})


# In[12]:


print(data["education"].value_counts())
print("\n")
print(data["education-num"].value_counts())


# - We found that "education" is a duplicate feature of "education-num" so we drop it "education" column.

# In[13]:


# drop education column
data.drop(columns=['education'], inplace=True)
data.columns.tolist()


# In[14]:


# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())


# In[15]:


# Drop rows with missing values (if necessary)
data.dropna(inplace=True)


# In[16]:


# Basic statistics for numerical features
print("\nDescriptive Statistics for Numerical Features:")
data.describe()


# #### Label Encoding

# In[17]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# List of categorical features to encode
categorical_features = [
    'workclass', 'marital-status', 
    'occupation', 'relationship', 'race', 
    'sex', 'native-country', 'income'
]

# Apply Label Encoding to each categorical column
for feature in categorical_features:
    data[feature] = le.fit_transform(data[feature])

# Display the first few rows of the encoded dataset
print("Dataset after Label Encoding:")
print(data.head())


# In[18]:


data.describe()


# #### Standard Scaling

# In[19]:


from sklearn.preprocessing import StandardScaler

# Define the numerical features with continuous values
numerical_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

data_ss = data.copy()
# Standardizing the numerical features
scaler = StandardScaler()
data_ss[numerical_features] = scaler.fit_transform(data_ss[numerical_features])


# In[20]:


data_ss.describe()


# ## EDA

# In[21]:


# Total number of records
n_records = data.shape[0]

# Total number of features
n_features = data.shape[1]

# Number of records where individual's income is more than $50,000
n_greater_50k = data[data['income'] == '>50K'].shape[0]

# Number of records where individual's income is at most $50,000
n_at_most_50k = data[data['income'] == '<=50K'].shape[0]

# Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k / n_records) * 100  

# Print the results
print("Total number of records: {}".format(n_records))
print("Total number of features: {}".format(n_features))
print("Individuals making more than $50k: {}".format(n_greater_50k))
print("Individuals making at most $50k: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50k: {:.2f}%".format(greater_percent))


# ##### No. of indiviudals with income <=50K & >50K denoted by 0 and 1 respectively

# In[22]:


import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Corrected countplot
sns.countplot(x='income', data=data, hue='income', palette='Set2', legend=False)
plt.title('Count of Individuals by Income Level')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()


# #### Correlation Heatmap

# In[23]:


# Correlation matrix for numerical features
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# In[24]:


import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore

# Assuming `data` is your DataFrame containing the dataset

# Select only numerical features from the dataset
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

# Define the number of subplots based on the number of numerical features
num_features = len(numerical_features)
fig, axes = plt.subplots(nrows=(num_features + 1) // 2, ncols=2, figsize=(14, 4 * ((num_features + 1) // 2)))

# Flatten the axes array for easy iteration if it exists (depends on the number of features)
axes = axes.flatten()

# Plot each numerical feature
for i, feature in enumerate(numerical_features):
    sns.histplot(data[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[25]:


# Heat map
plt.figure(figsize=[10,10])
 
ct_counts = data.groupby(['education-num', 'income']).size()
ct_counts = ct_counts.reset_index(name = 'count')
ct_counts = ct_counts.pivot(index = 'education-num', columns = 'income', values = 'count').fillna(0)

sns.heatmap(ct_counts, annot = True, fmt = '.0f', cbar_kws = {'label' : 'Number of Individuals'})
plt.title('Number of People for Education Class relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Education Class')


# - In the graph above, we can see that people with education classes of 9 & 10 make up the highest portion in the dataset. Also, we notice that people with education class of 14 to 16 proportionally usually make >50k as income in the statistics we have in the dataset, unlike lesser education classes where they usually make <=50k as income.

# In[26]:


# Clustered Bar Chart 
plt.figure(figsize=[8,6])
ax = sns.barplot(data = data, x = 'income', y = 'age', hue = 'sex')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'Sex')
plt.title('Average of Age for Sex relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Average of Age')


# - The figure shows in general that the people with >50K has a higher average age than the ones with <=50K. And in both cases of income, we see that the male category has a little bit greater age average than the female category.

# In[27]:


# Bar Chart 
plt.figure(figsize=[8,6])
sns.barplot(data=data, x='income', hue="income", y='hours-per-week', palette='YlGnBu', legend=False)
plt.title('Average of Hours per Week relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Average of Hours per Week')


# - We notice here that the income grows directly with the average of work hours per week, which is a pretty reasonable and logical result.

# In[28]:


import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Set figure size
fig, axes = plt.subplots(4, 4, figsize=(20, 18))
fig.suptitle('Feature Distribution w.r.t. Income', fontsize=20)
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Numeric features with respect to income (Boxplots)
for idx, feature in enumerate(numerical_features):
    row, col = divmod(idx, 4)
    sns.boxplot(x='income', hue="income", y=feature, data=data, ax=axes[row][col], palette='Set3', legend=False)
    axes[row][col].set_title(f'{feature} vs Income')
    axes[row][col].set_xlabel('Income')
    axes[row][col].set_ylabel(feature)

# Categorical features with respect to income (Violin plots)
for idx, feature in enumerate(categorical_features):
    row, col = divmod(idx + len(numerical_features), 4)
    sns.violinplot(x='income', hue="income", y=feature, data=data, ax=axes[row][col], palette='Set2', inner='quartile', legend=False)
    axes[row][col].set_title(f'{feature} vs Income')
    axes[row][col].set_xlabel('Income')
    axes[row][col].set_ylabel(feature)

# Hide any unused subplots
for i in range(len(numerical_features) + len(categorical_features), 16):
    fig.delaxes(axes[i // 4][i % 4])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ##### Descriptive Analysis
# 1. Age vs Income:
# Observation: Individuals with higher income (>50K) tend to be older compared to those with lower income (<=50K), as the median age for higher income is clearly greater.
# 2. Fnlwgt vs Income:
# Observation: The distribution of fnlwgt (final weight) appears to have no clear relationship with income. Both classes have a broad range of weights.
# 3. Education-num vs Income:
# Observation: Higher education levels (as represented by education-num) are associated with higher income, indicated by a higher median education level for individuals earning >50K.
# 4. Capital-gain vs Income:
# Observation: Capital gain shows a strong correlation with higher income. Those with high income (>50K) tend to have significantly higher capital gains compared to the lower-income group, where most values are near zero.
# 5. Capital-loss vs Income:
# Observation: Similar to capital gains, capital loss is also higher for individuals earning >50K, but the distribution has many zero values for both income groups.
# 6. Hours-per-week vs Income:
# Observation: Individuals with higher income tend to work more hours per week, with the median hours worked for those earning >50K being higher than for those earning <=50K.
# 7. Workclass vs Income:
# Observation: The work class distribution shows some variation between income groups, but the difference in distribution is not very significant.
# 8. Marital-status vs Income:
# Observation: Marital status seems to have a noticeable impact on income, with a distinct difference in the distribution between the two income groups, suggesting that certain marital statuses may correlate with higher earnings.
# 9. Occupation vs Income:
# Observation: The occupation feature shows a difference in distribution between income groups, suggesting certain occupations are more common in higher-income individuals.
# 10. Relationship vs Income:
# Observation: Relationship status has a strong influence on income. Certain relationship categories (such as married individuals) seem to be more prevalent in the higher income group.
# 11. Race vs Income:
# Observation: The distribution of race across income groups does not show a significant disparity, suggesting race might not be a strong determinant of income in this dataset.
# 12. Sex vs Income:
# Observation: There is a noticeable difference between genders, with more men represented in the higher income group (>50K).
# 13. Native-country vs Income:
# Observation: The distribution for native country shows some differences, but there isn’t a clear, strong pattern distinguishing income groups.
# ##### General Observations:
# - Features like age, education-num, capital-gain, and hours-per-week have clearer, more distinguishable patterns between income groups.
# - Categorical features such as occupation, marital-status, and relationship also show distinct patterns across income groups.

# In[29]:


# Pairplot of numerical features colored by income
sns.pairplot(data, hue='income', palette='Set2')
plt.suptitle('Pairplot of Numerical Features colored by Income', y=1.02)
plt.show()


# ## Initial Model Training

# In[30]:


X = data_ss.drop('income', axis=1)
y = data_ss['income']


# In[31]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}


# #### Logistic regression

# In[32]:


# Train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Make predictions with Logistic Regression
y_pred_lr = logistic_model.predict(X_test)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='binary')
recall_lr = recall_score(y_test, y_pred_lr, average='binary')
f1_lr = f1_score(y_test, y_pred_lr, average='binary')

print(f"Logistic Regression:- \nAccuracy: {accuracy_lr} | Precision: {precision_lr} | Recall: {recall_lr} | F1: {f1_lr}")

models["Logistic Regression"] = logistic_model


# #### Decision Tree

# In[33]:


# Train the Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions with Decision Tree
y_pred_tree = tree_model.predict(X_test)

# Evaluate Decision Tree model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, average='binary')
recall_tree = recall_score(y_test, y_pred_tree, average='binary')
f1_tree = f1_score(y_test, y_pred_tree, average='binary')

print(f"Decision Tree:- \nAccuracy: {accuracy_tree}, Precision: {precision_tree}, Recall: {recall_tree}, F1: {f1_tree}")

models["Decision Tree"] = tree_model


# #### Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore

# Train the Random Forest model
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)


# Predict on the test data
y_pred_forest = forest_model.predict(X_test)

# Evaluate Decision Tree model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
precision_forest = precision_score(y_test, y_pred_forest, average='binary')
recall_forest = recall_score(y_test, y_pred_forest, average='binary')
f1_forest = f1_score(y_test, y_pred_forest, average='binary')

print(f"Random Forest:- \nAccuracy: {accuracy_forest} | Precision: {precision_forest} | Recall: {recall_forest} | F1: {f1_forest}")

models["Random Forest"] = forest_model


# In[35]:


from sklearn.model_selection import cross_validate # type: ignore
# Cross validation
for model_name in models:
    model = models[model_name]
    results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1'], return_train_score=True)
    
    print(model_name + ":")
    print("Accuracy:" , 'train: ', results['train_accuracy'].mean(), '| test: ', results['test_accuracy'].mean())
    print("F1-score:" , 'train: ', results['train_f1'].mean(), '| test: ', results['test_f1'].mean())
    print("---------------------------------------------------------")


# - As it appears from the exploration in our dataset that there is an imbalance between the classes of classifications. Since the individuals making more than 50k as income represent 75% of the data. So, we would try to make oversampling.

# In[36]:


import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import learning_curve # type: ignore

# Function to plot learning curves for bias-variance tradeoff in subplots
def plot_learning_curves(estimator, X_train, y_train, ax, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    
    # Calculate mean and std of training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Plot learning curves
    ax.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
    ax.plot(train_sizes, test_scores_mean, label='Validation Score', color='red')
    ax.set_title(title)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    ax.grid(True)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot learning curves for Logistic Regression
plot_learning_curves(logistic_model, X_train, y_train, axes[0], "Logistic Regression")

# Plot learning curves for Decision Tree
plot_learning_curves(tree_model, X_train, y_train, axes[1], "Decision Tree")

# Plot learning curves for Random Forest
plot_learning_curves(forest_model, X_train, y_train, axes[2], "Random Forest")

# Adjust layout for better readability
plt.tight_layout()
plt.show()


# 1. Logistic Regression:
# - Shows moderate performance with scores around 0.82-0.824
# - Training score (blue) gradually improves with more data
# - Validation score (red) shows some fluctuation but generally maintains a slight edge over the training score
# - Appears to be slightly underfitting as the training score is lower than validation score
# 
# 2. Decision Tree:
# 
# - Shows a significant gap between training and validation performance
# - Training score (blue) is perfect (1.0) across all training set sizes
# - Validation score (red) is much lower (~0.8-0.81)
# - Clear signs of overfitting as the model performs perfectly on training data but poorly on validation data
# 
# 
# 3. Random Forest:
# 
# - Similar pattern to Decision Tree but with better validation performance
# - Training score (blue) remains perfect (1.0)
# - Validation score (red) is higher than Decision Tree (~0.85-0.86)
# - Still shows overfitting, but less severe than the Decision Tree
# - Slight improvement in validation score as training data increases

# ### Using Random Over Sampling

# In[37]:


from imblearn.over_sampling import RandomOverSampler # type: ignore

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

models_resampled = {}


# #### Logistic Regression

# In[38]:


# Train the Logistic Regression model
logistic_model_ros = LogisticRegression(max_iter=5000)
logistic_model_ros.fit(Xr_train, yr_train)

# Make predictions with Logistic Regression
yr_pred_lr = logistic_model_ros.predict(Xr_test)

# Evaluate Logistic Regression model
accuracy_lr1 = accuracy_score(yr_test, yr_pred_lr)
precision_lr1 = precision_score(yr_test, yr_pred_lr, average='binary')
recall_lr1 = recall_score(yr_test, yr_pred_lr, average='binary')
f1_lr1 = f1_score(yr_test, yr_pred_lr, average='binary')

print(f"Logistic Regression:- \nAccuracy: {accuracy_lr1}, Precision: {precision_lr1}, Recall: {recall_lr1}, F1: {f1_lr1}")

models_resampled["Logistic Regression"] = logistic_model_ros


# #### Decision Tree

# In[39]:


# Train the Decision Tree model
tree_model_ros = DecisionTreeClassifier(random_state=42)
tree_model_ros.fit(Xr_train, yr_train)

# Make predictions with Decision Tree
yr_pred_tree = tree_model_ros.predict(Xr_test)

# Evaluate Decision Tree model
accuracy_tree1 = accuracy_score(yr_test, yr_pred_tree)
precision_tree1 = precision_score(yr_test, yr_pred_tree, average='binary')
recall_tree1 = recall_score(yr_test, yr_pred_tree, average='binary')
f1_tree1 = f1_score(yr_test, yr_pred_tree, average='binary')

print(f"Decision Tree:- \nAccuracy: {accuracy_tree1}, Precision: {precision_tree1}, Recall: {recall_tree1}, F1: {f1_tree1}")

models_resampled["Decision Tree"] = tree_model_ros


# #### Random Forest

# In[40]:


# Train the Random Forest model
forest_model_ros = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model_ros.fit(Xr_train, yr_train)


# Predict on the test data
yr_pred_forest = forest_model_ros.predict(Xr_test)

# Evaluate Decision Tree model
accuracy_forest1 = accuracy_score(yr_test, yr_pred_forest)
precision_forest1 = precision_score(yr_test, yr_pred_forest, average='binary')
recall_forest1 = recall_score(yr_test, yr_pred_forest, average='binary')
f1_forest1 = f1_score(yr_test, yr_pred_forest, average='binary')

print(f"Random Forest:- \nAccuracy: {accuracy_forest1} | Precision: {precision_forest1} | Recall: {recall_forest1} | F1: {f1_forest1}")

models_resampled["Random Forest"] = forest_model_ros


# In[41]:


from sklearn.metrics import roc_curve, auc

# Function to plot ROC curve
def plot_roc_curve(model, Xr_test, yr_test, label, ax):
    yr_pred_prob = model.predict_proba(Xr_test)[:, 1]
    fpr, tpr, _ = roc_curve(yr_test, yr_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

# Create a plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot ROC curve for Logistic Regression
plot_roc_curve(logistic_model_ros, Xr_test, yr_test, "Logistic Regression", ax)

# Plot ROC curve for Decision Tree
plot_roc_curve(tree_model_ros, Xr_test, yr_test, "Decision Tree", ax)

# Plot ROC curve for Random Forest
plot_roc_curve(forest_model_ros, Xr_test, yr_test, "Random Forest", ax)

# Plot diagonal line (no skill model)
ax.plot([0, 1], [0, 1], 'k--')

# Customize the plot
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves for Logistic Regression, Decision Tree, and Random Forest")
ax.legend(loc="lower right")

# Show plot
plt.show()


# In[42]:


from sklearn.model_selection import cross_validate # type: ignore
# Cross validation
for model_name in models_resampled:
    model = models[model_name]
    results = cross_validate(model, X_resampled, y_resampled, cv=5, scoring=['accuracy', 'f1'], return_train_score=True)
    
    print(model_name + ":")
    print("Accuracy:" , 'train: ', results['train_accuracy'].mean(), '| test: ', results['test_accuracy'].mean())
    print("F1-score:" , 'train: ', results['train_f1'].mean(), '| test: ', results['test_f1'].mean())
    print("---------------------------------------------------------")


# In[43]:


# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot learning curves for Logistic Regression
plot_learning_curves(logistic_model_ros, Xr_train, yr_train, axes[0], "Logistic Regression")

# Plot learning curves for Decision Tree
plot_learning_curves(tree_model_ros, Xr_train, yr_train, axes[1], "Decision Tree")

# Plot learning curves for Random Forest
plot_learning_curves(forest_model_ros, Xr_train, yr_train, axes[2], "Random Forest")

# Adjust layout for better readability
plt.tight_layout()
plt.show()


# Descriptive Analysis:
# 1. Logistic Regression:
# - Training score (blue) steadily decreases as training set size increases (from ~0.769 to ~0.766)
# - Validation score (red) initially increases and then plateaus around 0.765
# - The curves converge as training size increases, suggesting good generalization
# - Shows signs of high bias (underfitting) as both scores are relatively low
# 
# 2. Decision Tree:
# - Training score (blue) remains perfect at 1.0 across all training sizes
# - Validation score (red) shows steady improvement from ~0.77 to ~0.90
# - Large gap between training and validation scores indicates overfitting
# - However, validation performance continues to improve with more data
# - Could potentially benefit from further increasing training data
# 
# 3. Random Forest:
# - Training score (blue) stays at 1.0, similar to Decision Tree
# - Validation score (red) shows consistent improvement from ~0.83 to ~0.92
# - Better validation performance than both Decision Tree and Logistic Regression
# - Still shows overfitting but with better generalization than Decision Tree
# - Validation score hasn't plateaued, suggesting potential benefit from more training data
# 
# Comparative Analysis:
# - Random Forest shows the best overall performance
# - Decision Tree and Random Forest both suffer from overfitting but show promising validation improvements
# - Logistic Regression shows underfitting, suggesting it might be too simple for this problem
# - More training data appears beneficial for tree-based models but doesn't help Logistic Regression much
# - The learning curves suggest tree-based models (especially Random Forest) are more suitable for this particular problem

# In[44]:


# View a list of the features and their importance scores
print('\nFeatures Importance:')
feat_imp = pd.DataFrame(zip(X_resampled.columns.tolist(), forest_model_ros.feature_importances_ * 100), columns=['feature', 'importance'])
feat_imp


# In[49]:


# Features importance plot
plt.figure(figsize=[20,6])
sns.barplot(data=feat_imp, x='feature', hue='feature', y='importance', palette='viridis', legend=False)
plt.title('Features Importance', weight='bold', fontsize=20)
plt.xlabel('Feature', weight='bold', fontsize=13)
plt.ylabel('Importance (%)', weight='bold', fontsize=13)


# add annotations
impo = feat_imp['importance']
locs, labels = plt.xticks()

for loc, label in zip(locs, labels):
    count = impo[loc]
    pct_string = '{:0.2f}%'.format(count)

    plt.text(loc, count-0.8, pct_string, ha = 'center', color = 'w', weight='bold')


# - Since we have highest accuracy on Random Forest, here is a visual representation for feature importance.

# In[ ]:




