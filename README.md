# Hi, I'm Haris! 👋

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 

# Linear Regression

The repository contains the following files:

* **LinearRegression.ipynb** – contains code for training and evaluating the linear regression model
* **LinearRegression.py** – corresponding .py file for the Linear Regression Jupyter Notebook
* **human_zombie_dataset_v5.csv** – contains the dataset required for the model
* **human_zombie_dataset_v5.xlsx** – corresponding Excel file for easier visualization <br>

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Assignment Overview](#assignment-overview)
4. [Data](#data)
5. [Training and Evaluation](#training-and-visualization)
6. [Screenshots](#screenshots)
   
## Introduction

This assignment focuses on implementing and analyzing **Linear Regression**, one of the most fundamental algorithms in supervised machine learning. Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a straight line to observed data. Its primary goal is to predict a continuous output based on input features by minimizing the error between the predicted and actual values, typically using the **least squares method**.

In this assignment, a **custom dataset named `human_zombie_dataset_v5`** is used to explore how linear regression can model and predict trends within synthetic or scenario-driven data.  

The `.csv` file contains the actual dataset used for model training and evaluation, while the `.xlsx` version provides a convenient format for manual inspection and visualization.


## Installation Requirements

To run the notebooks in this repository, you will need the following packages:


!pip install idx2numpy

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`


You can install these packages using pip:

```bash
pip install numpy
```

```bash
pip install pandas
```

```bash
pip install matplotlib
```

```bash
pip install scikit-learn
```

After installing the required libraries, run the **"Imports"** cell in the notebook to begin.

Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
Also, you will need a GPU to run the notebooks. It is recommended to have a Google Colab Account (perhaps multiple accounts) for this purpose.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Assignment Overview

The `LinearRegression.ipynb` notebook is the core component of this assignment, where various forms of **linear regression** are implemented and evaluated on the `human_zombie_dataset_v5` dataset. The notebook is structured in a step-by-step manner, moving from foundational concepts to more advanced regularized models. Each section is designed to deepen understanding of regression techniques and their real-world applicability.

### 1. **Linear Regression from Scratch**

The assignment begins by implementing linear regression **manually**, using only NumPy and basic Python functions. <br>

This section helps reinforce mathematical intuition and the inner workings of linear regression without relying on external libraries.

### 2. **Linear Regression using Scikit-learn**

Next, the same task is approached using `scikit-learn`'s `LinearRegression` class. This includes:

This section provides a fast and scalable way to train and test models, ideal for practical usage. It also allows us to compare the manual implementation results with state-of-the-art results.

### 3. **Ridge Regression**

This part introduces **Ridge Regression**, a regularized variant of linear regression that adds an L2 penalty term to the cost function. It is useful in reducing overfitting by penalizing large weights. The notebook demonstrates Ridge Regression using `Ridge` from `scikit-learn`,

### 4. **Lasso Regression**

In this section, **Lasso Regression** is implemented using `scikit-learn`’s `Lasso` class. Unlike Ridge, Lasso applies **L1 regularization**, which can reduce some weights to zero, effectively performing **feature selection**. 

### 5. **Elastic Net Regression**

The final model is **Elastic Net**, which combines both L1 and L2 penalties. It is implemented using `ElasticNet` from `scikit-learn`, and is especially useful when there are multiple correlated features, and we want a balance between feature selection and generalization.



## Data

The dataset used in this assignment is titled human_zombie_dataset_v5.csv, a creatively themed, synthetic dataset designed to explore linear relationships between variables in a fictional scenario involving humans and zombies. While humorous in concept, the dataset serves as a meaningful resource for applying regression techniques and understanding the impact of regularization.

This dataset simulates human and zombie characteristics based on various lifestyle and physical traits. The dataset contains 1,000 entries, each with features that correlate with a continuous "Human-Zombie Score" ranging from 0 (complete human) to 100 (complete zombie).

**Features**

- **Height (cm):** The height of the individual measured in centimeters, it decreases with zombie score because zombies are known to shrink in height.

- **Weight (kg):** The weight of the individual measured in kilograms. Zombies tend to have a lower weight because of loss of muscle mass, tissue, organs (and soul??).

- **Screen Time (hrs):** The average number of hours spent in front of screens daily. This feature increases with the human-zombie score, reflecting a more sedentary lifestyle.

- **Junk Food (days/week):** The average number of days per week the individual consumes junk food. This feature also increases with the human-zombie score, indicating poorer dietary habits.

- **Physical Activity (hrs/week):** The total hours spent on physical activities per week. This feature decreases as the human-zombie score increases, suggesting a decline in physical activity.

- **Task Completion (scale):** Scale from 0 to 10 representing how often tasks are completed on time (0 = always on time, 10 = never on time). This score decreases with a higher human-zombie score, indicating declining productivity.

- **Human-Zombie Score:** A continuous score from 0 to 100 representing the degree of "zombie-ness" of the individual, where 0 is fully human and 100 is completely zombie-like.

## Training and Visualization

The entire training process alongside the relevant evaluations and visualizations are explained in detail in the jupyter notebook. 

## Screenshots

<h4> 1. This image shows how Ridge Regression shrinks the coefficients of all features as regularization strength increases. In Ridge regression, all coefficients shrink smoothly toward zero as log(Alpha) increases. Unlike Lasso, Ridge never eliminates features entirely — it reduces their influence proportionally.  </h4>
<img src="pic1.png" width="1000px" height="500px"> <br> 


<h4> 2. This image shows how Lasso Regression shrinks the coefficients of all features as regularization strength increases. Lasso regression causes some coefficients to drop sharply to zero as log(Alpha) increases. This behavior is due to the L1 penalty, which promotes **feature selection** by eliminating less important features. Only a few dominant features survive high regularization, making the model simpler and sparser. </h4>
<img src="pic2.png" width="1000px" height="500px"> <br> 

<h4> 3. This plot shows how Elastic Net shrinks the coefficients of all features as regularization strength increases. Some features (e.g., Physical Activity and Junk Food) reduce significantly but not as abruptly as in Lasso. Elastic Net combines L1 and L2 penalties, leading to both shrinkage and sparsity in a balanced manner. </h4>
<img src="pic3.png" width="1000px" height="500px">
 <br> 
 
## License

[MIT](https://choosealicense.com/licenses/mit/)


