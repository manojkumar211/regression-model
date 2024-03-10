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


Xl=ds[['TV','radio']]
yl=ds['sales']



class lasso_regression:
    
    X_train_lasso,X_test_lasso,y_train_lasso,y_test_lasso=train_test_split(Xl,yl,test_size=0.2,random_state=86)

    lasso_reg=Lasso(alpha = 85, random_state = 6, selection = 'random', tol = 0.003)
    lasso_reg.fit(X_train_lasso,y_train_lasso)
    lasso_intercept=lasso_reg.intercept_
    lasso_coefficient=lasso_reg.coef_
    train_pred_lasso=lasso_reg.predict(X_train_lasso)
    test_pred_lasso=lasso_reg.predict(X_test_lasso)
    train_score_lasso=lasso_reg.score(X_train_lasso,y_train_lasso)
    test_score_lasso=lasso_reg.score(X_test_lasso,y_test_lasso)
    train_R2_lasso=r2_score(y_train_lasso,train_pred_lasso)
    test_R2_lasso=r2_score(y_test_lasso,test_pred_lasso)
    cross_valid_lasso=cross_val_score(lasso_reg,X,y,cv=5)

    def __init__(self,lasso_reg,lasso_intercept,lasso_coefficient,train_pred_lasso,test_pred_lasso,train_score_lasso,
                 test_score_lasso,train_R2_lasso,test_R2_lasso,cross_valid_lasso):
        
        self.lasso_reg=lasso_reg
        self.lasso_intercept=lasso_intercept
        self.lasso_coefficient=lasso_coefficient
        self.train_pred_lasso=train_pred_lasso
        self.test_pred_lasso=test_pred_lasso
        self.train_score_lasso=train_score_lasso
        self.test_score_lasso=test_score_lasso
        self.train_R2_lasso=train_R2_lasso
        self.test_R2_lasso=test_R2_lasso
        self.cross_valid_lasso=cross_valid_lasso

    def las_reg(self):
        return self.lasso_reg
    
    def las_inter(self):
        return self.lasso_intercept
    
    def las_coe(self):
        return self.lasso_coefficient
    
    def las_train_pred(self):
        return self.train_pred_lasso
    
    def las_test_pred(self):
        return self.test_pred_lasso
    
    def las_train_score(self):
        return self.train_score_lasso
    
    def las_test_score(self):
        return self.test_score_lasso
    
    def las_r2_train(self):
        return self.train_R2_lasso
    
    def las_r2_test(self):
        return self.test_R2_lasso
    
    def las_cross_val(self):
        return self.cross_valid_lasso
    
