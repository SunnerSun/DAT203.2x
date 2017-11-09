# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:05:40 2017

@author: Sunner
"""
#import pandas as pd


def azureml_main(df, quantile):
    """
    .merge:
    left: A DataFrame object, right: Another DataFrame object
    left_on: Columns from the left DataFrame to use as keys.
    right_on: Columns from the right DataFrame to use as keys.
    INNER JOIN: Use intersection of keys from both frames.
    .drop:
    axis -> Whether to drop labels from the index (0 / ‘index’) or 
    columns (1 / ‘columns’).
    """
    import pandas as pd
    
    ## Save the original names of the Dataframe.
    in_names = list(df)
    df = pd.merge(df, quantile,
                  left_on = ['monthCount', 'workHr'],
                  right_on = ['monthCount', 'workHr'],
                  how = 'inner')
    
    ## Filter rows where the count of bikes is less than the quantile
    ## Note ---
    df = df.ix[df['cnt'] > df['quantile']]
    
    ## Remove the unneeded column and restore the original column
    df.drop('quantile', axis = 1, inplace = True)
    df.colums = in_names
    
    ##Sort the data frame based on the dayCount
    df.sort(['days', 'workHr'], axis = 0, inplace = True)
    
    return df