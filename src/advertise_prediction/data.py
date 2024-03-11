import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


""" Loading the Advertising data to predict the sales """


df=pd.read_csv("E:/NareshiTech/Advertising.csv")


""" Analysis of the advertising data"""

class detail:
    head_5 = df.head()
    column = df.columns.to_list()
    inf = df.info()
    desc = df.describe()
    shap = df.shape
    dtyp = df.dtypes
    duplic = df.duplicated().sum()
    cor = df.corr()

    def __init__(self,head_5,column,inf,desc,shap,dtyp,duplic,cor):

        self.head_5=head_5
        self.column=column
        self.inf=inf
        self.desc=desc
        self.shap=shap
        self.dtyp=dtyp
        self.duplic=duplic
        self.cor=cor

    def heading(self):
        return self.head_5
    def col(self):
        return self.column
    def infer(self):
        return self.inf
    def des(self):
        return self.desc
    def shapp(self):
        return self.shap
    def dtypp(self):
        return self.dtyp
    def dupl(self):
        return self.duplic
    def co(self):
        return self.cor
    

"""" Heatmap for how data is correlated with other variables."""
    
fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(data=df.corr(),annot=True,cmap='tab20',ax=ax)
plt.title("Data Correlation Analysis")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/data_corr.png")
    


"""" Analysis by columns wise """

""" Analysis of TV column """

class TV:
    tv_describe=df['TV'].describe()
    tv_duplicates=df['TV'].duplicated().sum()
    tv_nullvalues=df['TV'].isnull().sum()
    tv_skewness=df['TV'].skew()

    def __init__(self,tv_describe,tv_duplicates,tv_nullvalues,tv_skewness):
        self.tv_describe=tv_describe
        self.tv_duplicates=tv_duplicates
        self.tv_nullvalues=tv_nullvalues
        self.tv_skewness=tv_skewness

    def describing_for_tv(self):
        return self.tv_describe
    def duplicates_for_tv(self):
        return self.tv_duplicates
    def nullvalues_for_tv(self):
        return self.tv_nullvalues
    def skewness_for_tv(self):
        return self.tv_skewness
    

""" Distribution plot for TV column """
    
fig,ax=plt.subplots(figsize=(10,5))
sns.distplot(df['TV'],color='r',ax=ax)
plt.title("TV Density Distribution")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/tv_distplot.png")


""" Analysis of Radio column """

class radio:
    radio_describe=df['radio'].describe()
    radio_duplicates=df['radio'].duplicated().sum()
    radio_nullvalues=df['radio'].isnull().sum()
    radio_skewness=df['radio'].skew()

    def __init__(self,radio_describe,radio_duplicates,radio_nullvalues,radio_skewness):
        self.radio_describe=radio_describe
        self.radio_duplicates=radio_duplicates
        self.radio_nullvalues=radio_nullvalues
        self.radio_skewness=radio_skewness

    def describing_for_radio(self):
        return self.radio_describe
    def duplicates_for_radio(self):
        return self.radio_duplicates
    def nullvalues_for_radio(self):
        return self.radio_nullvalues
    def skewness_for_radio(self):
        return self.radio_skewness
    

""" Distribution plot for radio column """
    
fig,ax=plt.subplots(figsize=(10,5))
sns.distplot(df['radio'],color='r',ax=ax)
plt.title("Radio Density Distribution")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/radio_distplot.png")





""" Analysis of Newspaper column """


class newspaper:
    newspaper_describe=df['newspaper'].describe()
    newspaper_duplicates=df['newspaper'].duplicated().sum()
    newspaper_nullvalues=df['newspaper'].isnull().sum()
    newspaper_skewness=df['newspaper'].skew()

    def __init__(self,newspaper_describe,newspaper_duplicates,newspaper_nullvalues,newspaper_skewness):
        self.newspaper_describe=newspaper_describe
        self.newspaper_duplicates=newspaper_duplicates
        self.newspaper_nullvalues=newspaper_nullvalues
        self.newspaper_skewness=newspaper_skewness

    def describing_for_newspaper(self):
        return self.newspaper_describe
    def duplicates_for_newspaper(self):
        return self.newspaper_duplicates
    def nullvalues_for_newspaper(self):
        return self.newspaper_nullvalues
    def skewness_for_newspaper(self):
        return self.newspaper_skewness
    

""" Distribution plot for newspaper column """
    
fig,ax=plt.subplots(figsize=(10,5))
sns.distplot(df['newspaper'],color='r',ax=ax)
plt.title("newspaper Density Distribution")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/newspaper_distplot.png")



""" Analysis of sales column """

class sales:
    sales_describe=df['sales'].describe()
    sales_duplicates=df['sales'].duplicated().sum()
    sales_nullvalues=df['sales'].isnull().sum()
    sales_skewness=df['sales'].skew()

    def __init__(self,sales_describe,sales_duplicates,sales_nullvalues,sales_skewness):
        self.sales_describe=sales_describe
        self.sales_duplicates=sales_duplicates
        self.sales_nullvalues=sales_nullvalues
        self.sales_skewness=sales_skewness

    def describing_for_sales(self):
        return self.sales_describe
    def duplicates_for_sales(self):
        return self.sales_duplicates
    def nullvalues_for_sales(self):
        return self.sales_nullvalues
    def skewness_for_sales(self):
        return self.sales_skewness
    

""" Distribution plot for sales column """
    
fig,ax=plt.subplots(figsize=(10,5))
sns.distplot(df['sales'],color='r',ax=ax)
plt.title("sales Density Distribution")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/sales_distplot.png")

    
