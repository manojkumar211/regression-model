from data import df,detail,TV
from eda import Outliers
from concat import df1,ds1,ds,X,y
from best_rand_degree_value import BestRandomState,BestDegreeValue
from Linear_model import ols_method,Linear_Regression
from polynomial import Linear_Regression_poly,ols_method_poly
from Lasso import lasso_regression
from Ridge import ridge_regression
from ElasticNet import elastic_regression


print('Lasso Regression Accuracy')
print('--'*20)
print('Train Accuracy : ',lasso_regression.train_R2_lasso) # type: ignore
print('Test Accuracy : ',lasso_regression.test_R2_lasso) # type: ignore
print('Cross validation score : ',lasso_regression.cross_valid_lasso.mean()) # type: ignore
print("##"*20)
print('Linear Regression Accuracy')
print('--'*20)
print('Train Accuracy : ',Linear_Regression.R2_train_score) # type: ignore
print('Test Accuracy : ',Linear_Regression.R2_test_score) # type: ignore
print('Cross validation score : ',Linear_Regression.cross_score.mean()) # type: ignore
print("##"*20)
print('Polynomial Regression Accuracy')
print('--'*20)
print('Train Accuracy : ',Linear_Regression_poly.R2_train_score_poly) # type: ignore
print('Test Accuracy : ',Linear_Regression_poly.R2_test_score_poly) # type: ignore
print('Cross validation score : ',Linear_Regression_poly.cross_score_poly.mean()) # type: ignore
print("##"*20)
print('Ridge Regression Accuracy')
print("--"*20)
print('Train Accuracy : ',ridge_regression.train_ridge_r2) # type: ignore
print('Test Accuracy : ',ridge_regression.test_ridge_r2) # type: ignore
print('Cross validation score : ',ridge_regression.cross_val_ridge.mean()) # type: ignore
print("##"*20)
print('Elastic Regression Accuracy')
print("--"*20)
print('Train Accuracy : ',elastic_regression.train_elastic_r2) # type: ignore
print('Test Accuracy : ',elastic_regression.test_elstic_r2) # type: ignore
print('Cross validation score : ',elastic_regression.cross_valid_elastic.mean()) # type: ignore


