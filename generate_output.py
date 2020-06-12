#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly.graph_objects as go
import requests, re, pandas as pd
import matplotlib.pyplot as mp
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import matplotlib
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import column
import datetime
from bokeh.palettes import Dark2_5 as palette
import itertools
import panel as pn
import panel.widgets as pnw
import json
import plotly.express as px
from bokeh.models import NumeralTickFormatter
import plotly as py
colors = itertools.cycle(palette)
import pdb
import datetime
from scipy.stats import pearsonr
import yfinance as yf
import random
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline
import matplotlib.pyplot as plt
from hvplot import hvPlot
import holoviews as hv
from ipywidgets import interact
import hvplot.pandas  # noqa
hv.extension('bokeh')
# import panel as pn
pn.extension(comms='ipywidgets')
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import display, HTML


# In[ ]:


def drop_null_vals(df,axis='both',subset=[]):
    '''
    Drops columns with all
    nan values from a given 
    data frame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame for which
        columns are to be
        dropped.
        
    axis : str
        Drops all rows with
        nan if axis=rows,
        all columns if axis=columns,
        and both if axis=both.
        
    subset : list of str
        For all columns in
        subset, remove the
        NaN rows.
    '''
    assert(isinstance(df,pd.DataFrame))
    assert(isinstance(axis,str))
    assert(isinstance(subset,list))
    assert(isinstance(col,str) for col in subset)
    
    df = df.dropna(subset=subset)
    
    if(axis=='rows'):
        df = df.dropna(how='all',axis=0)
    elif(axis=='columns'):
        df = df.dropna(how='all',axis=1)
    elif(axis=='both'):
        df = df.dropna(how='all',axis=0).dropna(how='all',axis=1)
    
    return df

def covid_spread_animation(data,mode="north america",world=True):
    '''
    animation of how it grows over time around the US
    '''
    assert(isinstance(world,bool))
    assert(isinstance(mode,str))
    
    if(world==False):
        data.iloc[:,11:] = data.iloc[:,11:]+1
        tidy_df = data.iloc[:,8:].melt(id_vars=["Combined_Key","Lat","Long"],var_name="Date",value_name="value")
        log_value=(tidy_df.value).apply(np.log10)
        tidy_df['log_value']=log_value

        fig_growth = px.scatter_geo(tidy_df, 
                                    lat=tidy_df['Lat'],
                                    lon=tidy_df['Long'],
                                    hover_name="Combined_Key", 
                                    size="log_value",
                                    animation_frame='Date',
                                    projection="natural earth",
                                    title="Outbreak Across The US",
                                    scope=mode,
                                   width=1000,height=700
                                   )
#         py.offline.plot(fig_growth,filename='outputs/html/growth_america.html');
    else:
        tidy_df = data.melt(id_vars=["Province/State","Country/Region","Lat","Long"],var_name="Date",value_name="value")
        log_value=(tidy_df.value+1).apply(np.log10)
        tidy_df['log_value']=log_value

        fig_growth = px.scatter_geo(tidy_df, 
                                    lat=tidy_df['Lat'],
                                    lon=tidy_df['Long'],
                                    hover_name="Country/Region", 
                                    size="log_value",
                                    animation_frame='Date',
                                    projection="natural earth",
                                    title="Outbreak Across The World")

        fig_growth.show()
#         py.offline.plot(fig_growth,filename='outputs/html/world_spread.html');

    return fig_growth

def industry_visualization(dict_industries):
    '''
    Description: Converts the dict of industries to holoviews time series plot
    Input: Dict of industries where each key value is a dataframe
    Output: Panel app which updates the plot based on menu selection
    
    '''

    #Finding the top 6 companies with max diff in stock prices over time
    for key in list(dict_industries.keys()):
        data_tr = dict_industries[key].T
        data_tr.loc['max_diff']=dict_industries[key].T.max()-dict_industries[key].T.min()
        dict_industries[key] = data_tr.T

        dict_industries[key] = dict_industries[key].sort_values('max_diff' , ascending=False)
        dict_industries[key] = dict_industries[key].head(6)
        dict_industries[key] = dict_industries[key].drop(['max_diff'], axis = 1)
        
        pulldown = pn.widgets.Select(name='Industry',options=list(dict_industries.keys()))

    @pn.depends(pulldown.param.value)
    def load_symbol(symbol, **kwargs):
        return hvPlot(dict_industries[symbol].T)()


    app=pn.Column(pulldown,load_symbol)
    app.servable() # % panel serve
    return app

def str_to_datetime_index(data):
    '''
    Converts the string date index
    to datetime index.
    '''
    return data.set_index(pd.to_datetime(data.iloc[:,2:].set_index(data.Date).index),inplace=True)


# In[ ]:


try:
    confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    us_confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    
    print('Fetching confirmed data from git...')
    time_series_confirmed = drop_null_vals(pd.read_csv(confirmed_url,error_bad_lines=False))
    print('Fetched confirmed data from git')
    
    time_series_confirmed.to_csv('data/time_series_covid_19_confirmed.csv')
    
    print('Fetching US confirmed data from git...')
    us_time_series_confirmed = drop_null_vals(pd.read_csv(us_confirmed_url,error_bad_lines=False))
    print('Fetched US confirmed data from git')
    
    us_time_series_confirmed.to_csv('data/time_series_covid_19_confirmed_us.csv')
    
    print('Fetching deaths data from git...')
    time_series_deaths = drop_null_vals(pd.read_csv(deaths_url,error_bad_lines=False))
    time_series_deaths.to_csv('data/time_series_covid_19_deaths.csv')
    print('Fetched deaths data from git')
    
    print('Fetching recovered data from git...')
    time_series_recovered = drop_null_vals(pd.read_csv(recovered_url,error_bad_lines=False))
    time_series_recovered.to_csv('data/time_series_covid_19_recovered.csv')
    print('Fetched recovered data from git')
    
except:
    # data not able to be fetched from git, fetching from local system
    print("Data not able to fetch from git. Using local filesystem.")
    time_series_confirmed = drop_null_vals(pd.read_csv('data/time_series_covid_19_confirmed.csv'))
    time_series_deaths = drop_null_vals(pd.read_csv('data/time_series_covid_19_deaths.csv'))
    time_series_recovered = drop_null_vals(pd.read_csv('data/time_series_covid_19_recovered.csv'))


# In[ ]:


covid_spread_animation(time_series_confirmed)


# In[ ]:


us_time_series_confirmed = drop_null_vals(us_time_series_confirmed)
us_time_series_confirmed = us_time_series_confirmed.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region', 'Long_': 'Long'})
covid_spread_animation(us_time_series_confirmed,"north america",world=False).show()


# # Stock prices visualization

# In[ ]:


# with open('data/save.p', 'rb') as handle:
with open('data/stocks_data.p', 'rb') as handle:
    dict_industries_pickle = pickle.load(handle)


# In[ ]:


result_industry = industry_visualization(dict_industries_pickle)
result_industry


# # p-value Visualization

# In[ ]:


start_dt = datetime.datetime.strptime("3/01/20",'%m/%d/%y')
end_dt = datetime.datetime.strptime("4/30/20",'%m/%d/%y')


# In[ ]:


finalpval = pd.read_csv('data/p_val_march.csv')
finalpval


# In[ ]:


industry_pval_dict = {}
for industry_val in finalpval.industry.unique().tolist():
    industry_pval_dict[industry_val] = finalpval.loc[finalpval['industry'] == industry_val]


# In[ ]:


def my_function(symbol):
    
    figure = px.scatter(industry_pval_dict[symbol], 'Ticker' , 'price'  , size = 'p_val' , 
                        animation_frame = 'Date', range_x = [start_dt,end_dt], size_max=50,
                        range_y = [industry_pval_dict[symbol]['price'].min()-50,industry_pval_dict[symbol]['price'].max()+50],
                        hover_data= {"price":True, "p_val_cumsum":False,"p_val":False})

    return (figure.show())


industries = list(industry_pval_dict.keys())
interact(my_function, symbol=industries)


# # Prediction

# #### Fitting the spline

# In[ ]:


china_industry_data = pd.read_csv("data/china_industry_mean.csv")


# In[ ]:



# In[ ]:


china_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="China"].iloc[:,4:].sum()
china_spread=pd.DataFrame(china_spread,columns=["Confirmed"])
china_spread.index.rename("Date",inplace=True)
china_spread.set_index(pd.to_datetime(china_spread.index),inplace=True)

us_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="US"].iloc[:,4:].sum()
us_spread=pd.DataFrame(us_spread,columns=["Confirmed"])
us_spread.index.rename("Date",inplace=True)
us_spread.set_index(pd.to_datetime(us_spread.index),inplace=True)


# In[ ]:


str_to_datetime_index(china_industry_data)
china_industry_data=china_industry_data.iloc[:,2:]
joined_table = pd.concat([china_spread,china_industry_data],axis=1,join="inner")
joined_table = joined_table[joined_table.Entertainment.notna()]
joined_table = joined_table[joined_table.Confirmed.notna()]
joined_table.drop("Unnamed: 118",axis=1,inplace=True)
joined_table.fillna(method="ffill",inplace=True)


# In[ ]:


# industries = list(plot_dict.keys())
industries = list(china_industry_data.columns[2:])
pulldown = pn.widgets.Select(name='Industry',options=industries)

@pn.depends(pulldown.param.value)
def predict_and_plot(symbol):
    try:
        tick = random.choices(industry_pval_dict[symbol].Ticker.unique())[0]
        joined_table[["Confirmed",symbol]]
        
        # Leave out the last value as stock price is incorrectly filled as 0
        confirmed_cases = joined_table.Confirmed.to_list()[1:-1]
        prev_stock_val = joined_table[symbol].to_list()[:-2]
        curr_stock_val = joined_table[symbol].to_list()[1:-1]
        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        scaler3 = MinMaxScaler(feature_range=(-1, 1))

        x1 = scaler1.fit_transform(np.array(confirmed_cases).reshape(-1, 1))
        x2 = scaler2.fit_transform(np.array(prev_stock_val).reshape(-1, 1))
        y = scaler3.fit_transform(np.array(curr_stock_val).reshape(-1, 1))

        test_data = finalpval[finalpval["Ticker"]==tick]
        test_data.set_index(pd.to_datetime(test_data.Date),inplace=True)
        concat_test_data = pd.concat([us_spread,test_data.price],axis=1,join="inner")

        test_scaler1 = MinMaxScaler(feature_range=(-1, 1))
        test_scaler2 = MinMaxScaler(feature_range=(-1, 1))

        test_confirmed_cases = concat_test_data.Confirmed.to_numpy()
        test_confirmed_cases = test_scaler1.fit_transform(test_confirmed_cases[:-2].reshape(-1,1))
        test_stock_price = concat_test_data.price.to_numpy()
        test_stock_price = test_scaler2.fit_transform(test_stock_price[:-2].reshape(-1,1))

        spline = SmoothBivariateSpline(x1,x2,y,s=5)

        prediction=[]
        for i,x in enumerate(test_confirmed_cases):
            # If first iteration, take the actual stock price
            # Else take the predicted value from previous iteration
            if(i==0):
                prediction.append(spline(test_confirmed_cases[i],test_stock_price[i]))
            else:
                prediction.append(spline(test_confirmed_cases[i],prediction[i-1]))

        predicted_y = test_scaler2.inverse_transform(np.array(prediction).reshape(-1,1))

        figure = plt.figure(figsize=(15,8))
        plt.plot(predicted_y)
        plt.plot(concat_test_data.index.strftime('%m/%d').to_list()[1:-1],concat_test_data.price.to_list()[1:-1])
        plt.legend(["Predicted Value","Actual Value"]);
        plt.title("Predicted vs. actual stock values for "+tick,fontsize=20);
        plt.ylabel("Stock price",fontsize=15)
        plt.xlabel("Date",fontsize=15)
        plt.xticks(rotation=45)

    except:
        print("Data not found for ",symbol)
        figure = plt.figure(figsize=(10,6))
        
    return figure.show()


interact(predict_and_plot, symbol=industries)


# In[ ]:




