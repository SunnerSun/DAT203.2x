# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:25:38 2017

@author: Sunner
"""

def dup_label(df, label, val, n):
    temp = df[df[label] == val]
    for _ in range(n):
        df = df.append(temp, ignore_index = True)
    return df

def azureml_main(df):
    return dup_label(df, 'CreditStatus', val = 1, n = 9)