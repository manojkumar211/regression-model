import numpy as np
import pandas as pd
from concat import ds
from concat import X,y
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,ElasticNetCV,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures


''' X_train_lasso,X_test_lasso,y_train_lasso,y_test_lasso=train_test_split(X,y,test_size=0.2,random_state=86)

lasso_regression_grid=Lasso()

lasso_param={'alpha':range(1,100), 'tol':[0.0001,0.00001,0.0002,0.002,0.0003,0.003], 'random_state':range(1,100), 'selection':['cyclic','random']}

grid_lasso=GridSearchCV(lasso_regression_grid,param_grid=lasso_param,cv=5,verbose=5)
grid_lasso.fit(X_train_lasso,y_train_lasso)
print(grid_lasso.best_params_) '''


Xr=ds[['TV','radio']]
yr=ds['sales']


X_train_ridge,X_test_ridge,y_train_ridge,y_test_ridge=train_test_split(Xr,yr,test_size=0.2,random_state=86)

ridge_regression_grid=Ridge()

ridge_param={'alpha':range(1,100), 'tol':[0.0001,0.00001,0.0002,0.002,0.0003,0.003], 'random_state':range(1,100), 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}


grid_lassocv=GridSearchCV(ridge_regression_grid,param_grid=ridge_param,cv=5,verbose=5)
grid_lassocv.fit(X_train_ridge,y_train_ridge)
print(grid_lassocv.best_params_)
