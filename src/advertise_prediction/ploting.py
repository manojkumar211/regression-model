import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import df
from concat import X

""" Making box-plot for each column to identify the outliers """

""" TV box-plot for identify the outliers """

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['TV'],ax=ax) # type: ignore
plt.title("TV column box-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/tv_boxplot.png")

""" Radio box-plot for identify the outliers """

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['radio'],ax=ax) # type: ignore
plt.title("Radio column box-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/radio_boxplot.png")


""" Newspaper box-plot for identify the outliers """

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['newspaper'],ax=ax) # type: ignore
plt.title("Newspaper column box-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/newspaper_boxplot.png")

# Newspaper column having the outlies.

""" Sales box-plot for identify the outliers """
fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=df['sales'],ax=ax) # type: ignore
plt.title("Sales column box-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/sales_boxplot.png")


""" Pair plot for all columns which columns are having inside our dataset """
fig=plt.subplots(figsize=(10,5))
sns.pairplot(data=df)
plt.title("Pair plot for all columns")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/sales_boxplot.png")


# Individual scatter-ploting based on columns

""" TV scatter-plot """
fig,ax=plt.subplots(figsize=(10,5))
sns.scatterplot(data=df,x='TV',y='sales',ax=ax)
plt.title("TV scatter-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/tv_scatter.png")



""" Radio scatter-plot """
fig,ax=plt.subplots(figsize=(10,5))
sns.scatterplot(data=df,x='radio',y='sales',ax=ax)
plt.title("Radio scatter-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/radio_scatter.png")




""" Newspaper scatter-plot """
fig,ax=plt.subplots(figsize=(10,5))
sns.scatterplot(data=df,x='newspaper',y='sales',ax=ax)
plt.title("Newspaper scatter-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/newspaper_scatter.png")




""" Sales scatter-plot """
fig,ax=plt.subplots(figsize=(10,5))
sns.scatterplot(data=df,x='sales',y='sales',ax=ax)
plt.title("Sales scatter-plot")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/sales_scatter.png")



""" Box-plotting on new data frame after remove outliers """

fig,ax=plt.subplots(figsize=(10,5))
sns.boxplot(data=X)
plt.title("Box-plot on New data frame")
plt.savefig("E:/NareshiTech/Advertise_predition/visualization/all_columns_boxplot.png")



