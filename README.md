```
Working on project related regression model, which is related to predict model. In this machine learning project, we are going to use all regression models to get best score/ accuracy.

```

# About Data
```
In this dataset, we are having total 4 columns. In 4 columns, 3 columns (TV,Radio,Newspaper) are belongs to independent variables and remaining column (Sales) is related to dependent variable.
```
# Observations
```
- In this dataset, having total 200 rows and 4 columns.
- Newspaper column having 2 outliers.
- TV and Radio features are correlated with dependent feature compare to Newspaper feature.
- Compare to all features, TV is highly correlated with sales.
```

# Descriptive Statistic
```
- Applyed IRQ method to identify the outliers in each column. Only Newspaper feature having 2 outliers.
```

# Inferential Statistic
```
- In this predict model, we are not going to apply Hypothesis test.
```

# Data Cleaning
```
- We find 2 outliers in Newspaper feature. So, applyed IQR method and replaced with upper boundry value.
```

# Data Wrangling
```
- Descritisation, In this project we are not going cloub the features from multiple variable into single variable based on sertain range.
- Data Transfermation, we did not find any skewness inside feature. which means data is symentrically distributed.
- Encoding, In this project no need to apply encoding. why because, all features are having continuous values only.
- Scaling method, for continuous, which mean regression model no need to apply scaling technique. why because it will not show any effect on result for regression model.
```

# Feature Selection
```
- For Feature Selection, we applyed OLS method to identify the which feature is going to fit with my model. In this method, which variable having value lessthan alpha (p<alpha) that variable to consider for build the model and other variable we are going to delete.
- For Feature Selection, we can aplly Lasso regression and can use Hypothesis test as well.
```

# Linear Regression
```
- With 3 features, Applyed Linear Regression model on 3 features along with random_state=86 and got Train score - 0.89%, Test score - 0.79%, Cross validation score - 0.87%.
- With 2 features, Applyed Linear Regression model on 2 features along with random_state=86 and got Train score - 0.92%, Test score - 0.81%, Cross validation score - 0.87%.
```

# Polynomial Regression
```
- Applyed Polynomial Regression on 2 features along with random_state=86 and degree=3 got Train score - 0.994%,Test score - 0.978%, Cross validation score - 0.876%.
```

# Lasso Regression 
```
- Applyed Lasso Regression on 3 and 2 features along with its hyperparameters (alpha = 85, random_state = 6, selection = 'random', tol = 0.003) we got Train score - 0.921%, Test score - 0.810%, Cross validation score - 0.876% for both trailers.
```


# Ridge Regression
```
- Applyed Ridge Regression on 2 features along with its hyperparameters ({'alpha': 1, 'random_state': 11, 'solver': 'sag', 'tol': 0.003}) we got Train score - 0.921%, Test score - 0.811%, Cross validation score - 0.876% for both 

```


# Elastic Regression
```
- Applyed Elastic Regression on 2 features and we got Train score - 0.921%, Test score - 0.810%, Cross validation score - 0.876% for both 

```



















