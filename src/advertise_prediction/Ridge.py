import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concat import ds
from concat import X,y
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,ElasticNetCV,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_validate,GridSearchCV,train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures



Xr=ds[['TV','radio']]
yr=ds['sales']




class ridge_regression:

    X_train_ridge,X_test_ridge,y_train_ridge,y_test_ridge=train_test_split(Xr,yr,test_size=0.2,random_state=86)

    ridge_reg=Ridge(alpha= 1, random_state= 11, solver= 'sag', tol= 0.003)
    ridge_reg.fit(X_train_ridge,y_train_ridge)
    ridge_intercept=ridge_reg.intercept_
    ridge_coefficient=ridge_reg.coef_
    train_pred_ridge=ridge_reg.predict(X_train_ridge)
    test_pred_ridge=ridge_reg.predict(X_test_ridge)
    train_ridge_score=ridge_reg.score(X_train_ridge,y_train_ridge)
    test_ridge_score=ridge_reg.score(X_test_ridge,y_test_ridge)
    train_ridge_r2=r2_score(y_train_ridge,train_pred_ridge)
    test_ridge_r2=r2_score(y_test_ridge,test_pred_ridge)
    cross_val_ridge=cross_val_score(ridge_reg,Xr,yr,cv=5)
    train_ridge_residuals=y_train_ridge-train_pred_ridge
    test_ridge_residuals=y_test_ridge-test_pred_ridge


    def __init__(self,ridge_intercept,ridge_coefficient,train_pred_ridge,test_pred_ridge,train_ridge_score,test_ridge_score,
                 train_ridge_r2,test_ridge_r2,cross_val_ridge,train_ridge_residuals,test_ridge_residuals):
        
        self.ridge_intercept=ridge_intercept
        self.ridge_coefficient=ridge_coefficient
        self.train_pred_ridge=train_pred_ridge
        self.test_pred_ridge=test_pred_ridge
        self.train_ridge_score=train_ridge_score
        self.test_ridge_score=test_ridge_score
        self.train_ridge_r2=train_ridge_r2
        self.test_ridge_r2=test_ridge_r2
        self.cross_val_ridge=cross_val_ridge
        self.train_ridge_residuals=train_ridge_residuals
        self.test_ridge_residuals=test_ridge_residuals

    def rid_inter(self):
        return self.ridge_intercept
    
    def rid_coef(self):
        return self.ridge_coefficient
    
    def rid_train_pred(self):
        return self.train_pred_ridge
    
    def rid_test_pred(self):
        return self.test_pred_ridge
    
    def rid_train_sc(self):
        return self.train_ridge_score
    
    def rid_test_sc(self):
        return self.test_ridge_score
    
    def rid_train_r2(self):
        return self.train_ridge_r2
    
    def rid_test_r2(self):
        return self.test_ridge_r2
    
    def rid_cross(self):
        return self.cross_val_ridge
    
    def rid_train_resu(self):
        return self.train_ridge_residuals
    
    def rid_test_resu(self):
        return self.test_ridge_residuals
    

plt.scatter(ridge_regression.train_pred_ridge,ridge_regression.train_ridge_residuals,c='r') # type: ignore
plt.axhline(y=0,color='b')
plt.xlabel('Fitted Vallues')
plt.ylabel('Residuals')
plt.savefig('E:/NareshiTech/Advertise_predition/visualization/Ridge_train_scatter.png')

plt.scatter(ridge_regression.test_pred_ridge,ridge_regression.test_ridge_residuals,c='r') # type: ignore
plt.axhline(y=0,color='b')
plt.xlabel('Fitted Vallues')
plt.ylabel('Residuals')
plt.savefig('E:/NareshiTech/Advertise_predition/visualization/Ridge_test_scatter.png')