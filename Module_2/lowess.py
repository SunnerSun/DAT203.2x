# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:37:22 2017

@author: sunner
"""

import statsmodels.nonparametric.smoothers_lowess as lo
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(500)
nobs = 250
sig_fac = 0.5
#x = np.random.normal(size=nobs)
x = np.random.uniform(-2, 2, size=nobs)
#print (x)
#y = np.array([np.sin(i*5)/i + 2*i + (3+i)*np.random.normal() for i in x])
y = np.sin(x*5)/x + 2*x + sig_fac * (3+x)*np.random.normal(size=nobs)

fig2 = plt.figure()
ax5 = fig2.add_subplot(111)

ys = lo.lowess(y, x)
#print (ys)
ax5.plot(ys[:,0], ys[:,1], 'b-')
ys2 = lo.lowess(y, x, frac=0.25)
ax5.plot(ys2[:,0], ys2[:,1], 'b--', lw=2)