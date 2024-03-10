import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import df




class Outliers:

    after_list=[]
    outlies_list=[]

    q1=df['newspaper'].quantile(q=0.25)
    q2=df['newspaper'].quantile(q=0.50)
    q3=df['newspaper'].quantile(q=0.75)
    

    iqr=q3-q1

    lower_bound=q1-(1.5*iqr)

    upper_bound=q3+(1.5*iqr)

    for val in df['newspaper']:
            if (val>upper_bound) or (val<lower_bound):
                outlies_list.append(val)
            if (val<upper_bound) and (val>lower_bound):
                after_list.append(val)

    def __init__(self,q1,q2,q3,iqr,lower_bound,upper_bound,after_list,outlies_list):
        
        self.q1=q1
        self.q2=q2
        self.q3=q3
        self.iqr=iqr
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.after_list=after_list
        self.outlies_list=outlies_list
        
        
    def quantile_one(self):
        return self.q1
    def quantile_two(self):
        return self.q2
    def quantile_three(self):
        return self.q3
    def inter_quantile_range(self):
        return self.iqr
    def lower_fen(self):
        return self.lower_bound
    def upper_fen(self):
        return self.upper_bound
    def aft_out(self):
        return self.after_list
    def with_out(self):
        return self.outlies_list
    
    

            
