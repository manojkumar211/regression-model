import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concat import ds
from concat import X,y
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_validate,GridSearchCV,train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures






Xe=ds[['TV','radio']]
ye=ds['sales']



class elastic_regression:

    X_train_elastic,X_test_elastic,y_train_elastic,y_test_elastic=train_test_split(Xe,ye,test_size=0.2,random_state=86)

    elastic_regression=ElasticNet()
    elastic_regression.fit(X_train_elastic,y_train_elastic)
    elastic_intercept=elastic_regression.intercept_
    elastic_coefficient=elastic_regression.coef_
    train_elastic_pred=elastic_regression.predict(X_train_elastic)
    test_elastic_pred=elastic_regression.predict(X_test_elastic)
    train_elastic_score=elastic_regression.score(X_train_elastic,y_train_elastic)
    train_elastic_r2=r2_score(y_train_elastic,train_elastic_pred)
    test_elastic_score=elastic_regression.score(X_test_elastic,y_test_elastic)
    test_elstic_r2=r2_score(y_test_elastic,test_elastic_pred)
    cross_valid_elastic=cross_val_score(elastic_regression,Xe,ye,cv=5)
    train_elastic_residuals=y_train_elastic-train_elastic_pred
    test_elastic_residuals=y_test_elastic-test_elastic_pred

    def __init__(self,elastic_intercept,elastic_coefficient,train_elastic_pred,test_elastic_pred,train_elastic_score,
                 train_elastic_r2,test_elastic_score,test_elstic_r2,cross_valid_elastic,train_elastic_residuals,test_elastic_residuals):
        
        self.elastic_intercept=elastic_intercept
        self.elastic_coefficient=elastic_coefficient
        self.train_elastic_pred=train_elastic_pred
        self.test_elastic_pred=test_elastic_pred
        self.train_elastic_score=train_elastic_score
        self.train_elastic_r2=train_elastic_r2
        self.test_elastic_score=test_elastic_score
        self.test_elstic_r2=test_elstic_r2
        self.cross_valid_elastic=cross_valid_elastic
        self.train_elastic_residuals=train_elastic_residuals
        self.test_elastic_residuals=test_elastic_residuals

    def elastic_inter(self):
        return self.elastic_intercept
    
    def elastic_coe(self):
        return self.elastic_coefficient
    
    def train_elas_pred(self):
        return self.train_elastic_pred
    
    def test_elas_pred(self):
        return self.test_elastic_pred
    
    def train_elas_scor(self):
        return self.train_elastic_score
    
    def train_elas_r2(self):
        return self.train_elastic_r2
    
    def test_elas_scor(self):
        return self.test_elastic_score
    
    def test_els_r2(self):
        return self.test_elstic_r2
    
    def cross_val_elas(self):
        return self.cross_valid_elastic
    
    def train_elas_res(self):
        return self.train_elastic_residuals
    
    def test_elas_res(self):
        return self.test_elastic_residuals
    

        
        

