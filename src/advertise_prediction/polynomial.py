import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concat import ds
from concat import X,y
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_validate,GridSearchCV,train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures



X_poly=ds[['TV','radio']]
y_poly=ds['sales']


class Linear_Regression_poly:

    Xp_train,Xp_test,yp_train,yp_test=train_test_split(X_poly,y_poly,test_size=0.2,random_state=86)

    poly_regression=PolynomialFeatures(degree=3)
    train_poly=pd.DataFrame(poly_regression.fit_transform(Xp_train))
    test_poly=pd.DataFrame(poly_regression.transform(Xp_test)) # type: ignore

    lr_poly=LinearRegression()
    lr_poly.fit(train_poly,yp_train)
    lr_poly_intercept=lr_poly.intercept_
    lr_poly_coefficient=lr_poly.coef_
    train_pred_poly=lr_poly.predict(train_poly)
    train_score_poly=lr_poly.score(train_poly,yp_train)
    R2_train_score_poly=r2_score(yp_train,train_pred_poly)
    test_pred_poly=lr_poly.predict(test_poly)
    test_score_poly=lr_poly.score(test_poly,yp_test)
    R2_test_score_poly=r2_score(yp_test,test_pred_poly)
    cross_score_poly=cross_val_score(lr_poly,X,y,cv=5)
    train_residual_poly=yp_train-train_pred_poly
    test_residual_poly=yp_test-test_pred_poly
    

    def __init__(self,lr_poly,lr_poly_intercept,lr_poly_coefficient,train_pred_poly,train_score_poly,R2_train_score_poly,test_pred_poly,test_score_poly,R2_test_score_poly,
                 cross_score_poly,train_residual_poly,test_residual_poly):
        
        self.lr_poly=lr_poly
        self.lr_poly_intercept=lr_poly_intercept
        self.lr_poly_coefficient=lr_poly_coefficient
        self.train_pred_poly=train_pred_poly
        self.train_score_poly=train_score_poly
        self.R2_train_score_poly=R2_train_score_poly
        self.test_pred_poly=test_pred_poly
        self.test_score_poly=test_score_poly
        self.R2_test_score_poly=R2_test_score_poly
        self.cross_score_poly=cross_score_poly
        self.train_residual_poly=train_residual_poly
        self.test_residual_poly=test_residual_poly

    def linear_model_poly(self):
        return self.lr_poly
    
    def linear_intercept_poly(self):
        return self.lr_poly_intercept
    
    def linear_coefficient_poly(self):
        return self.lr_poly_coefficient
    
    def train_prediction_poly(self):
        return self.train_pred_poly
    
    def train_scoring_poly(self):
        return self.train_score_poly
    
    def R2_training_scoring_poly(self):
        return self.R2_train_score_poly
    
    def test_prediction_poly(self):
        return self.test_pred_poly
    
    def test_scoring_poly(self):
        return self.test_score_poly
    
    def R2_testing_scoring_poly(self):
        return self.R2_test_score_poly
    
    def cross_validation_scoring_poly(self):
        return self.cross_score_poly
    
    def training_residual_values_poly(self):
        return self.train_residual_poly
    
    def testing_residual_values_poly(self):
        return self.test_residual_poly
    
    ''' By applying polynomial method got train score is: 0.9941
    test score is: 0.9783 and cross validation score is: 0.8767
    '''




ols_method_poly=smf.ols(formula='y_poly~X_poly',data=ds).fit()


sm.graphics.plot_partregress_grid(ols_method_poly)
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/ols_poly_pair_plot.png")


sm.graphics.influence_plot(ols_method_poly)
plt.title("How elements are infuencing the data")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/ols_poly_influence_plot.png")
        


plt.scatter(Linear_Regression_poly.train_pred_poly,Linear_Regression_poly.train_residual_poly,c='r') # type: ignore
plt.axhline(y=0,color='b')
plt.xlabel('Fitted Vallues')
plt.ylabel('Residuals')
plt.savefig('E:/NareshiTech/Advertise_predition/visualization/poly_train_scatter.png')

plt.scatter(Linear_Regression_poly.train_pred_poly,Linear_Regression_poly.train_residual_poly,c='r') # type: ignore
plt.axhline(y=0,color='b')
plt.xlabel('Fitted Vallues')
plt.ylabel('Residuals')
plt.savefig('E:/NareshiTech/Advertise_predition/visualization/poly_test_scatter.png')

