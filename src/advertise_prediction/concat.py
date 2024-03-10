import numpy as np
import pandas as pd
from data import df
from eda import Outliers

''' Creating new data frame based on new list after removing outliers '''

df1=pd.DataFrame(Outliers.after_list,columns=['Newspaper_new']) # type: ignore

''' Sorting the new data frame by values '''

df1=df1.sort_values('Newspaper_new',axis=0,ignore_index=True)

''' Sorting main data frame by values as well '''

ds1=df.sort_values("newspaper",axis=0,ignore_index=True)

''' Combining the new data frame with main data frame '''

ds=pd.concat([ds1,df1],axis=1)


''' Need to replace the null values with upper boundry values '''

ds.fillna(Outliers.upper_bound,axis=0,inplace=True) # type: ignore



X=ds.drop(columns=['newspaper','sales','Newspaper_new'])
y=ds['sales']



