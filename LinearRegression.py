# %% [markdown]
# <h1> <strong> <center> Problem Background: The Great Migration </center> </strong> </h1>
# 
# **Year: 3050**
# 
# The world as we knew it has changed drastically. After decades of conflict, disease, and a relentless zombie apocalypse, the human race faces extinction. The relentless hordes of the undead have ravaged cities, reduced populations, and devastated the planet's ecosystems. 
# 
# In a desperate bid for survival, the remaining factions of humanity have united to embark on a monumental journey to a new home: **Earth Junior**, a distant planet believed to be free from the scourge of the undead. 
# 
# As they prepare for this monumental journey, the survival of the human race hinges on their ability to screen potential passengers at the spaceship station. It is crucial that no zombies are allowed to enter the spacecraft, as even a single infected individual could jeopardize the entire mission and the future of humanity.
# 
# In this dire situation, you find yourself as the last surviving machine learning engineer, tasked with developing a screening model to predict the zombie-ness of a person. A high human-zombie score would mean the person is likely to be a zombie. The fate of the human race depends on your expertise in machine learning, and you must create a reliable system to ensure the success of this migration to 'Earth Junior'.
# 

# %% [markdown]
# <h2> <strong> <center> Linear Regression </center> </strong> </h2>
# 
# 
# ### Introduction
# 
# In this assignment, you will be implementing a Linear Regression model for the provided dataset from scratch, and will be familiarizing yourself with the corresponding scikit-learn APIs. A description of the problem statement is given at the start of each part.
# 
# After this notebook you should be able to:
# 
# - Set up simple regression tasks.
# 
# - Understand the working of Linear Regression models and simple data preprocessing.
# 
# - Implement Linear Regression models using simple `numpy`.

# %% [markdown]
# # Human-Zombie Dataset
# 
# This dataset simulates human and zombie characteristics based on various lifestyle and physical traits. The dataset contains 1,000 entries, each with features that correlate with a continuous "Human-Zombie Score" ranging from 0 (complete human) to 100 (complete zombie).
# 
# This generation of human race has smart-chips embedded in their bloodstream that can keep track of and record all these features.
# 
# ## Features
# 
# - **Height (cm):** The height of the individual measured in centimeters, it decreases with zombie score because zombies are known to shrink in height.
# 
# - **Weight (kg):** The weight of the individual measured in kilograms. Zombies tend to have a lower weight because of loss of muscle mass, tissue, organs (and soul??).
# 
# - **Screen Time (hrs):** The average number of hours spent in front of screens daily. This feature increases with the human-zombie score, reflecting a more sedentary lifestyle.
# 
# - **Junk Food (days/week):** The average number of days per week the individual consumes junk food. This feature also increases with the human-zombie score, indicating poorer dietary habits.
# 
# - **Physical Activity (hrs/week):** The total hours spent on physical activities per week. This feature decreases as the human-zombie score increases, suggesting a decline in physical activity.
# 
# - **Task Completion (scale):** Scale from 0 to 10 representing how often tasks are completed on time (0 = always on time, 10 = never on time). This score decreases with a higher human-zombie score, indicating declining productivity.
# 
# - **Human-Zombie Score:** A continuous score from 0 to 100 representing the degree of "zombie-ness" of the individual, where 0 is fully human and 100 is completely zombie-like.

# %% [markdown]
# ## Multivariate Linear Regression
# 
# In this part, you will implement multivariate linear regression (from scratch) to predict the the human-zombie score during screening before the person can be allowed to enter the spaceship.
# 
# To do this, you have the human-zombie-datset.csv containing 1000 examples of the features described above and their scores.
# 
# Each one of these input features is stored using a different scale. The features include ranges 0-10, 17-100 and some between 130-200.  This is often the case with real-world data, and understanding how to explore and clean such data is an important skill to develop.
# 
# A common way to normalize features that use different scales and ranges is:
# 
# - Subtract the mean value of each feature from the dataset.
# 
# - After subtracting the mean, additionally scale (divide) the feature values by their respective standard deviations.
# 
# Note: We only use examples of the train set to estimate the mean and standard deviation.
# 
# You have to follow exactly the same steps as above i.e. implement hypothesis, cost function and gradient descent for multivariate linear regression to learn parameters $\theta$ using train set. Finally, report the cost (error) using your learned parameters $\theta$ on test set.

# %% [markdown]
# ### Part A: Implementation from Scratch
# 
# #### Imports
# 
# Note that you are **only** allowed to use `sklearn`'s train_test_split in this part.

# %%
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% [markdown]
# #### Preparing Data
# 
# Load in your dataset and perform train-test split. Apply preprocessing steps to cater to missing values, categorical variables etc. if necessary.

# %%
df = pd.read_csv("human_zombie_dataset_v5.csv")

for column in df.columns:
    if df[column].isnull().any():
        df[column].fillna(df[column].mean(), inplace=True)

# %%
X = df.drop(columns=["Human-Zombie Score"])  
y = df["Human-Zombie Score"]                

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of original dataframe: ", df.shape)
print("Shape of features matrix: ", X.shape)
print("Shape of labels matrix: ", y.shape)
print("Shape of X train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

# %% [markdown]
# #### Normalizing Data 
# 
# For models like Linear/Logistic Regression, and even for Neural Networks, Standardization/Normalization is a non-negotiable step in the preprocessing stage. You would find your model **broken** if you do not normalize your data.
# 
# One nice way to implement this is to create a class, `StandardScaler` that can be used to store the mean and standard deviation of each feature of the training set. The `StandardScaler` class also has two functions, `fit` and `transform`.
# 
# - The `fit` function is used to calculate the mean and standard deviation of each feature of the training set.
# 
# - The `transform` function is used to transform all the features using the corresponding mean and standard deviation, i.e. subtracting the mean and dividing by the standard deviation. 
# 
# A very subtle and important point to note here is that the mean and standard deviation should be calculated only on the training set, and then applied to the test set. This is because in real-world scenarios, we do not have access to the test set, and hence we cannot calculate the mean and standard deviation of the test set.

# %%
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.stdev = None

    def fit(self, X):
        X = np.array(X)
        
        self.mean = np.mean(X, axis=0)
        self.stdev = np.std(X, axis=0)
        
    def transform(self, X):
        X = np.array(X)

        X_normalized = (X - self.mean) / (self.stdev)
        return X_normalized

scaler = StandardScaler()
scaler.fit(X_train)

X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)

X_train_normalized = np.hstack((np.ones((X_train_normalized.shape[0], 1)), X_train_normalized))
X_test_normalized = np.hstack((np.ones((X_test_normalized.shape[0], 1)), X_test_normalized))

# %% [markdown]
# #### Gradient Descent
# 
# Now that your data is ready, you can start implementing the gradient descent algorithm. 
# 
# You should be creating a class `LinearRegression`. This means implementing the following methods:
# 
# - `__init__`: The constructor of the class. You can initialize any variables, like the learning rate and the number of iterations, here. [5 points]
# 
# - `fit`: This method will be used to train your model. It should take in the training data and labels as parameters, and learn the parameters using gradient descent. Save the loss values after every iteration in a list, and return it (for plotting later). 
# 
# - `predict`: This method will be used to predict the labels for a given set of test data. It should take in the test data as a parameter, and return the predicted labels.
# 
# - `score`: This method will be used to calculate the mean square error for the test data. It should take in the test data and labels as parameters, and return the mean square error. Note that this is a unique case where the loss function and the final reported metric are the same. 
# 
# Plot the cost function, and print your mean square error for both the train and test set. 

# %%
class LinearRegression:
    def __init__(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y_actual):
        X = np.array(X)
        y_actual = np.array(y_actual)
        
        m = X.shape[0]
        n = X.shape[1] 
        
        self.theta = np.zeros(n) 
        loss_values = [] 

        for i in range(self.epochs):
            y_predicted = X.dot(self.theta) 
            errors = y_predicted - y_actual
            gradients = np.zeros(n)
            for j in range(m):
                gradients = gradients + (errors[j] * X[j, :])   
            self.theta = self.theta - (self.alpha * gradients) 

            cost = (1 / (2 * m)) * np.sum(np.power(errors, 2))
            loss_values.append(cost)

        return loss_values

    def predict(self, X):
        return X.dot(self.theta)

    def score(self, X, y_actual):
        y_predicted = self.predict(X)
        error = np.mean(np.power((y_predicted - y_actual), 2))
        return error/2

# %%
alpha = 0.0001
epochs = 200

LR = LinearRegression(alpha, epochs)
loss_values = LR.fit(X_train_normalized, y_train)

train_mse = LR.score(X_train_normalized, y_train)
test_mse = LR.score(X_test_normalized, y_test)
print("Mean Squared Error on Training Set using Manual Implementation:", train_mse)
print("Mean Squared Error on Test Set using Manual Implementation:", test_mse)

plt.figure(figsize=(16, 6)) 
plt.plot(loss_values, label='Cost', color='red')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ### Part B: Regularized Linear Regression Using Scikit-learn
# 
# Now, you'll use the [scikit-learn](https://scikit-learn.org/stable/index.html) to implement [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso), [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet) and apply them to the human-zombie dataset.
# 
# Try out different values of regularization coefficient (known as `alpha` in `sklearn`) and use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to report loss with each regression.
# 
# Finally, plot the regularization coefficients alpha (x-axis) with learned parameters $\theta$ (y-axis) for Ridge and Lasso. Please read [this blog](https://scienceloft.com/technical/understanding-lasso-and-ridge-regression/) to get better understanding of the desired plots.
# 

# %% [markdown]
# #### Importing Libraries
# 
# You have to use scikit-learn for this task

# %%
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# %% [markdown]
# #### Linear Regression (using `sklearn`)
# 
# Use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to find loss and print it. 

# %%
simpleLR_sklearn = LinearRegression()
simpleLR_sklearn.fit(X_train_normalized, y_train)

y_train_predicted = simpleLR_sklearn.predict(X_train_normalized)
y_test_predicted = simpleLR_sklearn.predict(X_test_normalized)

train_error = mean_squared_error(y_train, y_train_predicted)/2
test_error = mean_squared_error(y_test, y_test_predicted)/2

print("Mean Squared Error on Training Set using Sklearn:", train_error)
print("Mean Squared Error on Test Set using Sklearn:", test_error)

# %% [markdown]
# #### Ridge
# 
# Use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to find loss and print it. Also plot the regularization coefficients alpha (x-axis) with learned parameters $\theta$  (y-axis) for Ridge. 

# %%
feature_list = ["Height (cm)", "Weight (kg)", "Screen Time (hrs)", "Junk Food (days/week)", "Physical Activity (hrs/week)", "Task Completion (scale)"]

alpha_list_ridge = np.logspace(-2, 8, 100)
ridge_error = []
ridge_thetas = []

min_test_error_ridge = float('inf')
min_alpha_ridge = None

for alphaRidge in alpha_list_ridge:
    ridgeLR_sklearn = Ridge(alpha= alphaRidge)
    ridgeLR_sklearn.fit(X_train_normalized, y_train)
    
    y_train_predicted_ridge = ridgeLR_sklearn.predict(X_train_normalized)
    y_test_predicted_ridge = ridgeLR_sklearn.predict(X_test_normalized)
    
    train_error_ridge = mean_squared_error(y_train, y_train_predicted_ridge)/2
    test_error_ridge = mean_squared_error(y_test, y_test_predicted_ridge)/2
    
    ridge_error.append((alphaRidge, train_error_ridge, test_error_ridge))
    ridge_thetas.append(ridgeLR_sklearn.coef_)
    
    if test_error_ridge < min_test_error_ridge:
        min_test_error_ridge = test_error_ridge
        min_alpha_ridge = alphaRidge

print(f"Lowest Test MSE: {min_test_error_ridge:.3f} at Alpha: {min_alpha_ridge:.3f}")

ridge_thetas = np.array(ridge_thetas)

plt.figure(figsize=(20, 8))
log_alpha = np.log10(alpha_list_ridge) 

for i in range(ridge_thetas.shape[1] - 1):
    plt.plot(log_alpha, ridge_thetas[:, i], label=feature_list[i])

plt.xlabel('log(Alpha)')
plt.ylabel('Thetas')
plt.title('Ridge Regression Coefficients vs log(Alpha)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# %% [markdown]
# #### Lasso
# 
# Use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to find loss and print it. Also plot the regularization coefficients alpha (x-axis) with learned parameters $\theta$  (y-axis) for Lasso. 

# %%
alpha_list_lasso = np.logspace(-4, 4, 100)
lasso_error = []
lasso_thetas = []

min_test_error_lasso = float('inf')
min_alpha_lasso = None

for alphaLasso in alpha_list_lasso:
    lassoLR_sklearn = Lasso(alpha= alphaLasso)  
    lassoLR_sklearn.fit(X_train_normalized, y_train)
    
    y_train_predicted_lasso = lassoLR_sklearn.predict(X_train_normalized)
    y_test_predicted_lasso = lassoLR_sklearn.predict(X_test_normalized)
    
    train_error_lasso = mean_squared_error(y_train, y_train_predicted_lasso) / 2
    test_error_lasso = mean_squared_error(y_test, y_test_predicted_lasso) / 2
    
    lasso_error.append((alphaLasso, train_error_lasso, test_error_lasso))
    lasso_thetas.append(lassoLR_sklearn.coef_)
    
    if test_error_lasso < min_test_error_lasso:
        min_test_error_lasso = test_error_lasso
        min_alpha_lasso = alphaLasso

print(f"Lowest Test MSE: {min_test_error_lasso:.3f} at Alpha: {min_alpha_lasso:.3f}")

lasso_thetas = np.array(lasso_thetas)

plt.figure(figsize=(20, 8))
log_alpha = np.log10(alpha_list_lasso)

for i in range(lasso_thetas.shape[1] - 1):
    plt.plot(log_alpha, lasso_thetas[:, i], label=feature_list[i])

plt.xlabel('log(Alpha)')
plt.ylabel('Thetas')
plt.title('Lasso Regression Coefficients vs log(Alpha)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# %% [markdown]
# #### Elastic Net
# 
# Use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to find loss and print it. 

# %%
alpha_list_EN = np.logspace(-2, 4, 100)
EN_error = []
EN_thetas = []

l1_ratio = 0.5

min_test_error_EN = float('inf')
min_alpha_EN = None

for alphaEN in alpha_list_EN:
    EN_sklearn = ElasticNet(alpha= alphaEN, l1_ratio= l1_ratio)
    EN_sklearn.fit(X_train_normalized, y_train)
    
    y_train_predicted_EN = EN_sklearn.predict(X_train_normalized)
    y_test_predicted_EN = EN_sklearn.predict(X_test_normalized)
    
    train_error_EN = mean_squared_error(y_train, y_train_predicted_EN) / 2
    test_error_EN = mean_squared_error(y_test, y_test_predicted_EN) / 2
    
    EN_error.append((alphaEN, train_error_EN, test_error_EN))
    EN_thetas.append(EN_sklearn.coef_)
    
    if test_error_EN < min_test_error_EN:
        min_test_error_EN = test_error_EN
        min_alpha_EN = alphaEN

print(f"Lowest Test MSE: {min_test_error_EN:.3f} at Alpha: {min_alpha_EN:.3f}")
 
EN_thetas = np.array(EN_thetas)
plt.figure(figsize=(20, 8))
log_alpha = np.log10(alpha_list_EN)

for i in range(EN_thetas.shape[1] - 1):
    plt.plot(log_alpha, EN_thetas[:, i], label= feature_list[i])

plt.xlabel('log(Alpha)')
plt.ylabel('Thetas')
plt.title('Elastic Net Regression Coefficients vs log(Alpha)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


