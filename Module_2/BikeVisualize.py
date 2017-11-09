num_cols = ["temp", "hum", "windspeed", "hr"] 
           
def bike_scatter(df, cols):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    # It's using local data to generator a regression model
    import statsmodels.nonparametric.smoothers_lowess as lw
    
    print ('Columns = ' + str(df.columns))
    print ('Clos = ' + str(cols))
    ## Loop over the columns and create the scatter plots
    for col in cols:
        ## first compute a lowess fit to the data
        ## cnt is the total number of bikes in use for that hour of a day
        los = lw.lowess(df['cnt'], df[col], frac = 0.3)
    
        ## Now make the plots
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.plot(kind = 'scatter', x = col, y = 'cnt', ax = ax, alpha = 0.05)
        plt.plot(los[:, 0], los[:, 1], axes = ax, color = 'red')
        ax.set_xlabel(col)
        ax.set_ylabel('Number of bikes')
        ax.set_title('Number of bikes vs. ' + col)
        fig.savefig('scatter_' + col + '.png')
    return 'Done'        

cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday',
           'workingday', 'weathersit', 'dayOfWeek']

def bike_box(df, cols):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    
    ## Loop over the columns and create the box plots
    for col in cols:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.boxplot(column = 'cnt', by = col, ax = ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Number of bikes')
        ax.set_title('Number of bikes vs. ' + col)
        fig.savefig('box_' + col + '.png')
    return 'Done'
 
plt_times = [6, 8, 10, 12, 14, 16, 18, 20]
def bike_series(df, tms):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
        
    lims = (min(df.cnt), max(df.cnt))   
    for t in tms:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df[df.hr == t].plot(kind = 'line', x = 'days', y = 'cnt',
                            ylim = lims, ax = ax)    
        plt.xlabel("Days from start")
        plt.ylabel("Bikes rented")
        plt.title("Bikes rented by day for hour = " + str(t))
        fig.savefig('series_' + str(t) + '.png')
    return 'Done'
  
hist_cols = ["cnt", "temp", "hum", "windspeed"] 
def bike_hist(df, cols):    
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    
    ## Loop over columns and plot histograms
    for col in cols:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df[col].hist(bins = 30, ax = ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Density of ' + col)
        ax.set_title('Density of ' + col) 
        fig.savefig('hist_' + col + '.png')
    return 'Done'
   
def bike_hist_cond(df, col, by):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    
    ##isin:Return boolean DataFrame showing whether each element in the 
    ##DataFrame is contained in values.
    df = df[df.hr.isin(by)]
    ## Plot conditioned histograms
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    df[[col, 'hr']].hist(bins = 30, by = ['hr'], ax = ax)
    fig.savefig('hist_cod' + '.png')
    return 'Done'

def azureml_main(df):
    bike_scatter(df, num_cols)
    bike_box(df, cat_cols)
    bike_series(df, plt_times)
    bike_hist(df, hist_cols)
    bike_hist_cond(df, 'cnt', plt_times)
    return df