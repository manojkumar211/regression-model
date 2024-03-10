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



class BestRandomState:

    best_random_state_train=[]
    best_random_state_test=[]

    for i in range(0,100):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i) # type: ignore
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        train_pred=lr.predict(X_train)
        test_pred=lr.predict(X_test)
        lr.score(X_train,y_train)
        lr.score(X_test,y_test)
        best_random_state_train.append(r2_score(y_train,train_pred))
        best_random_state_test.append(r2_score(y_test,test_pred))

    def __ini__(self,best_random_state_train,best_random_state_test):

        self.best_random_state_train=best_random_state_train
        self.best_random_state_test=best_random_state_test

    def bestrand_train(self):
        return self.best_random_state_train
    
    def bestrand_test(self):
        return self.best_random_state_test
    





class BestDegreeValue:

    X_poly=ds[['TV','radio']]
    y_poly=ds['sales']

    Xp_train,Xp_test,yp_train,yp_test=train_test_split(X_poly,y_poly,test_size=0.2,random_state=86)

    lis_poly_train=[]
    lis_poly_test=[]

    for i in range(0,10):
        poly_degree=PolynomialFeatures(degree=i)

        Xp_train_poly=pd.DataFrame(poly_degree.fit_transform(Xp_train))

        lr1=LinearRegression()
        lr1.fit(Xp_train_poly,yp_train)

        train_poly_pred=lr1.predict(Xp_train_poly)
        lis_poly_train.append(lr1.score(Xp_train_poly,yp_train))


        Xp_test_poly=pd.DataFrame(poly_degree.transform(Xp_test)) # type: ignore

        test_poly_pred=lr1.predict(Xp_test_poly)
        lis_poly_test.append(lr1.score(Xp_test_poly,yp_test))

    def __ini__(self,lis_poly_train,lis_poly_test):

        self.lis_poly_train=lis_poly_train
        self.lis_poly_test=lis_poly_test

    def bestdegree_train(self):
        return self.lis_poly_train
    
    def bestdegree_test(self):
        return self.lis_poly_test
    











