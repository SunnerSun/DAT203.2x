"""
Introduction
This notes book uses a simple python simulation to illustrate some basic 
principles of regression models. Specifically, this notebook proceeds through
following steps.
Create a test data set.
Plot the data set.
Compute and plot the results of a linear regression model on these data.
Compute and plot the results of linear regression models for data sets with 
increasing dispursion of the data.
Compute and plot the results of linear regression models for data sets with
outliers.
"""
import numpy as np
import pandas as pd
np.random.seed(0)

def sim_reg_data(xmin, xmax, ymin, ymax, n, sd):
    """
    Create the data set
    The code in the cell below computes the data set. The data are along a 
    straight line with intercept of 0 and a slope of 1, with normally 
    distributed noise added.
    Run this code and examine the first few lines of the data frame.
    """
    import pandas as pd
    import numpy.random as nr
    
    w = nr.normal(loc = 0, scale = sd, size = n)
    
    xstep = float(xmax - xmin)/float(n - 1)
    ystep = float(ymax - ymin)/float(n - 1)
    
    x = []
    xcur = xmin
    y = []
    ycur = ymin
    for i in range(n):
        x.append(xcur)
        xcur += xstep
        y.append(ycur + w[i])
        ycur += ystep
        
    out = pd.DataFrame([x, y]).transpose()
#    print (out)
    out.columns = ['x', 'y']
#    print (out)
    return out
sim_data = sim_reg_data(0 ,10, 0, 10, 50, 1)
#print (sim_data.head())

def plot_2D(df):
    """
    Plot the data set
    Run the code in the cell below to plot the data set. Notice that the data 
    fall close to a stright line.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    fig.clf()
    #Get the current Axes instance on the current figure matching the given 
    #keyword args, or create one.
    ax = fig.gca()
    df.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('X vs. Y')
    return 'Done'
#plot_2D(sim_data)

def plot_reg(df):
    """
    The code in the cell below dose the following:
        The linear regression model is computed and scored using scikit-learn
        The regression line is ploted along with a histogram of the residuals
        Some model performance statistics are computed
    one of the performance statistics used in adjusted R**2
    R**2 adjusted = 1 - (n-1)*Sum square error/(n-2)*Sum square residual
    Run this code and examine the results. Notice that the intercept and slope
    are close the exact values. The regression line is shown on the left plot.
    The histogram of the residuals is shown on the right. the residuals are 
    approximately normally distributed.
    
    y = mx + b, where m is the slope of the line and b is the y-intercept.
    """
    import matplotlib.pyplot as plt
    from sklearn import linear_model
#    import math
    
    ## Prepare data for model, shape[0] is the number of row
    nrow = df.shape[0]
    ## In order to use regression model, we need to set dataframe to martix
    X = df.x.as_matrix().reshape((nrow,1))
#    print (df.x.as_matrix(), X)
    Y = df.y.as_matrix()
#    print (Y)
    ## Compute the linear model
    clf = linear_model.LinearRegression()
    lm = clf.fit(X,Y)
    ## Compute the y values
    df['lm_y'] = lm.predict(X)
#    print (df['lm_y'])
    ##Sort by the values along either axis, inplace change the order of df
    df.sort_values(by='x', ascending=True, inplace =True)
    
    fig, ax = plt.subplots(1, 2, figsize = (12,6))
    df.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax[0], alpha = 0.5)
    df.plot(kind = 'line', x = 'x', y = 'lm_y', style = ['r'], ax = ax[0])
#    df.plot(kind = 'line', x = 'x', y = 'y', style = ['y'], ax = ax[0], alpha = 0.5)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('X vs. Y')
    
    df['resids'] = (df.lm_y - df.y)
#    print (df['resids'])
    ax[1].hist(df['resids'], bins = 30, alpha = 0.7)
    ax[1].set_xlabel('Residual')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Histogram of Residuals')
    
    SSE = sum(df.resids * df.resids)
#    print (SSE)
    SSR = sum(df.y * df.y)
    R2_adj = 1.0 - (SSE/(SSR+SSE)) *((nrow - 1)/(nrow - 2))
    print ('Intercept = ' + str(lm.intercept_))
    print ('Slope = ' + str(lm.coef_[0]))
    print ('Adjusted R^2 = ', str(R2_adj))
    return ' '
#plot_reg(sim_data)
    
def sim_reg():
    """
    Regression with increasing data dispursion
    The code in the cell below computes data sets with increasing dispursion
    (standard deviation); 1, 5 and 10. The regression model is plotted and
    evaluated for each data set.
    Run this code and examine the results. Notice that the accuracy of the 
    model and dispursion of the residuals of the models increases with 
    increasing dispursion of the data.
    """
    sds = [1, 5, 10]
    for sd in sds:
        reg_data = sim_reg_data(1, 10, 1, 10, 50, sd)
        plot_reg(reg_data)
    return 'Done'
#sim_reg()
    
def sim_reg_outlier(xmin, xmax, ymin, ymax, n, sd, olx, oly):
    """
    The code cell below does the following:
        First the data set, including single outlier, is included, the 
        outliner is placed in three different locations.
        The function then calls the functions to compute the model and 
        evaluate the results.
    Run the code and examine the effects of the single outliner on the 
    regression results. Notice how each position of the outlier has a distinct 
    effect on the result.
    """
    import pandas as pd
    import numpy.random as nr
    
    w = nr.normal(loc = 0, scale = sd, size = n)
    
    xstep = float(xmax - xmin)/float(n - 1)
    ystep = float(ymax - ymin)/float(n - 1)
    
    x = []
    xcur = xmin
    y = []
    ycur = ymin
    for i in range(n):
        x.append(xcur)
        xcur += xstep
        y.append(ycur + w[i])
        ycur += ystep
        
    x.append(olx)
    y.append(oly)
    
    out = pd.DataFrame([x, y]).transpose()
    out.columns = ['x', 'y']
#    print (out)
    return out

def sim_outlier():
    ox = [0, 0, 5]
    oy = [10, -10, 10]
    for x, y in zip(ox, oy):
        reg_data = sim_reg_outlier(1, 10, 1, 10, 50, 1, x, y)
        plot_reg(reg_data)
    return 'Done'
sim_outlier()