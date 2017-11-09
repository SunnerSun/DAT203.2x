# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:38:52 2017

@author: Sunner
"""

def azureml_main(df):
    """
    Alpha-trimmed:
    The basic idea here is to order elements, discard elements at the beginning 
    and at the end of the got ordered set and then calculate average value using 
    the rest.
    """
    import pandas as pd
    ## Compute the lower quantile of the number of biked grouped by and 
    ## time values.
    out = df.groupby(['monthCount', 'workHr']).cnt.quantile(q = 0.2)
    out = pd.DataFrame(out)
    out.reset_index(inplace = True)
    out.columns = ['monthCount', 'workHr', 'quantile']
    return out