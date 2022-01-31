# Predicting IMDB Movie Rating by Regression & XGB Classifier 

##  Table of contents

	Technologies
	Setup
	Installation
	Problem
	Data 
	Model - Regression
	Model validation
		Lasso method
		XGBoost model
	Algorithm Evaluation
		Data Preparation and Cleaning
		Handling duplicate, unique, and missing values
		Confirming data cleanliness and value types
	EDA
	Data Preprocseeing: Encoding
	Model
	Model validation
	Hyperparameter Tunning
		Tuning Tree-based Tunning
		Tuning Regularization parameters
		Tuning using grid search
		tuning using random search
	Credits
	More resources

## Technologies
## Installation
## Setup

## Problem

The objective of this project was to learn and implement the Machine learning algorithm using Python and the IMDB Rating dataset. We will be using the basic regression model, XGBoost regression, XGBoost Classification model. XGBoost is a powerful approach for Supervised regression models. 

The goal of this project was: **“To predict the quality of new contents added to streaming websites based on the movie, genres, cast, votes, and directors”**. This kind of approach is necessary to save a huge amount of time and money before promoting and telecasting any content on streaming websites. It is also crucial to know about the audience’s likes and dislikes before its availability for the audience. 

Through this project, we are going to predict the success of the movie based on the rating already given to the movie's contents and features which are important to keep in mind before producing and telecasting such content. 

The other goal of this project was: **“To compare and figure the most applicable algorithm for predicting movie ratings?”**
The proposed models intend to predict perfect accuracy, but ratings come from the complicated human nature we can see some assumptions here. Predictions that lie within +/-1 rating range can be considered for the linear regression model. 

So, later we will check the accuracy of the Supervised Machine Learning technique and how accurately it can predict the ratings of movies. 


## Data 

Data has been scraped from the publicly available website https://www.imdb.com. Kaggle web scraping project and datasets are of sufficient size to develop a good predictive model for movie ratings. To view the original data and related information click below link.

[Kaggle Dataset](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset)\
https://www.imdb.com   

## Method 

### Regression model: 
We are going to start our modeling with linear regression because of its wide usability, fast run time, easy to use, and high interpretability. It is the basis for many other methods. [See here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report)

## Model Validation

### Lasso method: l1 regularization 

We will be starting model validation with Lasso. Lasso regression are some of the simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression. So, Lasso regression not only helps in reducing over-fitting, but it can help us in feature selection. [See here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report)

### XGBoost Method

The main benefits of using XGBoost are high execution speed and model performance. In both classification and regression predictive modeling, XGBoost dominates structured or tabular datasets. 

XG boost (Extreme Gradient Boosting) is widely used for classification and regression problems and gives better performance than other algorithms. It is the execution of gradient boosted decision trees. It is good for the small to medium tabular or structured data. [See here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report)
                                                                                                                                                                               The default metric for the classification type of problem is an error and for regression, the metric is RMSE. 

### Linear Regression Vs XGBoost

Linear regression is a parametric model. It assumes the target variable can be expressed as a linear combination of the independent variables (plus error). Gradient boosted trees are nonparametric: they will approximate any function.

XGboost deprecated the objective Reg: Linear. It has been replaced by reg: squared error, and has always meant minimizing the squared error, just as in linear regression.
So, boost will generally fit training data much better than linear regression, but that also means it is prone to overfitting, and it is less easily interpreted. Either one may end up being better, depending on your data and your needs.

We have started from the Regression model and later moved from lasso to XGboost Classifier. 

## Algorithms Evaluation 

To implement and compare the algorithms there must be a metric to compare. Mean Absolute Error (MAE), Mean Square Error (MSE), and Root-Mean Squared Error (RMSE) are some of the most common for regression algorithms.

MAE gets all the delta between the predicted value and the actual value, then adds them up, and then divides it by the number of predicted values, it is a very basic but efficient way of measuring error. 
MSE error works similarly but squares the delta, then adds them up and divides them, making it more practical to differentiate between smaller numbers. 
RMSE is just the square root of MSE, which is used for determining accuracy. 

## Data Preparation and Cleaning

[Data Cleaning report](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/1%20Cleaning%20and%20merging%20Data%20part%20-1%20.ipynb), [Link2] (https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/2%20IMDB%20data%20wrangling%20and%20Cleaning%20Part%202.ipynb)

### Merging files 

The data sets are pulled in seven small tsv files (see a full list here). We have loaded and merged them all as a single Data Frame named Final_DF with 14999145 rows and 17 columns. [Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report) [link2](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/1%20Cleaning%20and%20merging%20Data%20part%20-1%20.ipynb)

### Handling duplicate, unique, and missing values
After getting our final Data Frame, we have checked for the Duplicates, Index setting, datatypes, columns names, null/missing, and unique values. (for more details here) To make our data Frame tidy, we have removed all duplicate values, renamed columns name as appropriate, checked and filled null values. We have also performed data type conversion as per the nature of values.[Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report) [link2](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/1%20Cleaning%20and%20merging%20Data%20part%20-1%20.ipynb)

### Confirming data cleanliness and value types

There are a few more things to check column by column. This process is to make sure our data is all set for further processing. We found a few columns like the year, birthyear_Director, death_year were as object types. We have converted it to numerical as per the types of the variables. (See full details and report here)  

We have calculated the age of the director by subtracting the death_year from the birth_year. We have also derived the age of the movie by subtracting the release year from the current year. Later, We have divided movies into decades based on the age of the movie. 
 
One final step we have performed before moving further was checking any null/missing values and datatypes. To make sure we have required values as needed.[Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report) [link2] (https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/1%20Cleaning%20and%20merging%20Data%20part%20-1%20.ipynb)

## EDA

[EDA report](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/3%20data%20handling%20and%20EDA%20.ipynb)

We have performed Exploratory data Analysis to perform initial investigations on data with the help of summary statistics and graphical representations.[See here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/3%20data%20handling%20and%20EDA%20.ipynb)

## Data preprocessing: Encoding

[Data preprocessing Report link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/4%20Data%20handling%20for%20%20Preprocessing%20and%20%20Scaling%20.ipynb)[link2](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

Before fitting our data to any model, we must make sure all our categorical features areas are in numerical form. Here we have categorical columns like titleId, title, region, titleType, directors, writers, primaryName_Director, primaryProfession_director, Dir_knownForTitles, Decade. We have used oneHotencoder to convert it into numerical columns. Columns like genres, Dir_knownForTitles contain more than comma-separated values in single columns. For that, we have used the multilabelBinarizer which can easily deal with multi values.
 
 We have concatenated both encoded data frames and now our data is ready to fit in the model [Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/4%20Data%20handling%20for%20%20Preprocessing%20and%20%20Scaling%20.ipynb)[link2](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb) 
 
## Modeling

[Modeling report](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

We have started with the regression model. We have fit the model on to train the data set and predict the value of the test data set as y_pred. The model performance was evaluated from the r_squared values, which was 0.32 in our case. [Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report)[link2] (https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb) 

## Model Validation 

[Model validation](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

We have used the Lasso method, XGBoost Regressor, and XGBClassifier for the model validation. We have got 0.12, 0.31, and 0.14 rmse respectively. It seems XGBoost Regressor is giving better performance in all other methods. [see here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

## Hyperparameter Tunning 

[Hyperparameter tunning report](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)
 
## Tuning of Tree-based parameters

Learning rate (prevent overfitting), max_depth (more complex and overfit model with high value), gamma (conservative algorithm with large value), min_child_weigh (conservative algorithm with larger min_child_weight), colsample_bytree, min_child_weight, colsample_bytree, and Subsample are a few parameters which we have tuned here 
[Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report),[link2] (https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb) 

## Tuning regularization parameters

We have tuned Lambda: L2 regularization and  Alpha: L1 regularization term on weights. Increasing this value will make the model more conservative. The default value is 1 and the best lambda was 1. The default value for Alpha is 0. Our best alpha was 0.1 [Link1](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/tree/main/IMDB%20movie%20Ratings/Project%20report), [link2] (https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

After feeding all the best tree-based and regularization parameters, we have found the five best features which are playing an important role in increasing the movie ratings. These 5 features are f12135 score 97, f12137 score 95, f8741 score 60, f12136 score 60, f7980 score 49

## Hyper-parameter tuning by grid search

We have also run the gird search to find the best parameter values for the model. After running grid search the best values were colsample_bytree -0.5, learning_rate - 0.01, max_depth - 7, min_child_weight - 1, n_estimators - 200, subsample - 0.6. 
The best features we found were f7979 score 150, f12090 score 73,f8697 score 65, f11259 score 63, f12092 score 56. [see here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb) 

## Hyper-parameter tuning by Random Search

We have also tried the Random search to find the best parameter values for the model. After running random search, the best values were subsample - 0.6, n_estimators - 25, min_child_weight - 10, max_depth - 11, learning_rate - 1.0, colsample_bytree - 0.9 
The best features we found h were f7976 score 542, f7979 score 484, f12136 score 477, f12137 score 432, f8741 score 384 [see here for more details](https://github.com/ReetuData/Predicting-IMDB-Movie-Rating-by-Regression-XGB-Classifier/blob/main/IMDB%20movie%20Ratings/5%20Data%20encoding%20and%20Modeling.ipynb)

## Credits

Thanks, Vivek Kumar for being an amazing Springboard mentor throughout my course work. 

## References

https://www.diva-portal.org/smash/get/diva2:1574293/FULLTEXT01.pdf

https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

https://machinelearningmastery.com/xgboost-for-regression/

https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
















