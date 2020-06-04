import pytest
import plotly.graph_objects as go
import requests, re, pandas as pd
import matplotlib.pyplot as mp
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import matplotlib
from bokeh.plotting import figure, output_file, show, output_notebook
import datetime
import panel as pn
import panel.widgets as pnw
import json
import plotly.express as px
import plotly as py
import datetime
from scipy.stats import pearsonr
import yfinance as yf
import random
from tqdm import tqdm
import pickle
# from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from hvplot import hvPlot
import holoviews as hv
import hvplot.pandas  # noqa
hv.extension('bokeh')
pn.extension(comms='ipywidgets')
pn.extension('plotly')
# %matplotlib inline
from IPython.core.display import display, HTML, Image
from ipywidgets import interact

# panel
template = """
{% extends base %}

<!-- goes in body -->
{% block postamble %}
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
{% endblock %}

<!-- goes in body -->
{% block contents %}
<main role="main">
<div class="carousel-inner">
  <div class="carousel-item active">
    <img class="first-slide" 
    src='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTHEZgYN6qQ-nvq0QpFA65PmyEKyvxqSh8YQDnEoqwLIsaTVeba&usqp=CAU'
    style='width: inherit;'
    alt="First slide">
    <div class="container">
      <div class="carousel-caption text-left">
        <h1 style="
                color: goldenrod;
                font-size: xxx-large;
                font-weight: bold;
                text-align: left;
            ">Economic Impacts of <br>COVID-19</h1>
        <p style="
            text-align: left;
            font-size: larger;
            color: gray;
            ">Analysis and prediction of stocks with respect to spread of COVID.</p>
      </div>
    </div>
  </div>
</div>
<br>



<div class="container marketing">
  <div class="row featurette">
    <div class="col-md-14" style="margin-top: 20px;">
        <h2 class="featurette-heading">Outbreak of COVID across the US <span class="text-muted"></span></h2>
        <p class="lead">The outbreak in US took place in mid February. From March
        onwards, the cases expanded exponentially.</p>
    </div>
    <div class="col-md-8">
      {{ embed(roots.us_spread) }}
    </div>
  </div>
  
    <div class="col-md-15">
      <div class="row featurette">
        <div class="row-md-5" style="margin-top: 20px;">
            <h2 class="featurette-heading">Industry wise Stock Prices<span class="text-muted"></span></h2>
            <p class="lead">This graph gives an idea of stock prices as compared with time. Notice that
            there is a huge fall in the market overall.</p>
        </div>
      </div>
      <div class="row-md-6" style="margin-left: 90px;">
        {{ embed(roots.stocks) }}
      </div>
    </div>
    
    <br>
    
    <div class="row featurette">
        <div class="col-md-10">
            <h2 class="featurette-heading">Analyzing correlation of stocks with COVID
            <span class="text-muted">(p-value analysis)</span></h2>
            <p class="lead">Note:- If you select one of the industries, it will open up 
            a time series graph.<br> In this graph, you can see stock prices of companies of the selected
            industry. The size of bubbles is the p-value. This means that larger the bubble, lesser
            is the impact of spread of COVID. On the other hand, smaller the bubble, the impact of
            COVID is more. Thus, you might notice that a lot companies have small bubble sizes(large correlation with 
            COVID) during the early March phase.</p>
        </div>
      <div class="row-md-2" style='margin-top: 50px; margin-left: 90px;'>
        {{ embed(roots.p_val) }}
      </div>
    </div>
    
    <br>
    
    <div class="col-md-15">
      <div class="row featurette">
        <div class="row-md-5" style="margin-top: 20px;">
            <h2 class="featurette-heading">Prediction of Stock Prices<span class="text-muted"></span></h2>
            <p class="lead">We fit in a Spline to the correlation between COVID and stock prices
            and predicted stock prices. The results were as follows.</p>
        </div>
      </div>
      <div class="row-md-6">
        {{ embed(roots.prediction) }}
      </div>
    </div>
    
</div>


<footer class="pt-4 my-md-5 pt-md-5 border-top">
        <div class="row">
          <div class="col-6 col-md">
          </div>
          <div class="col-12 col-md">
            <small class="d-block mb-3 text-muted">ECE 229 Spring 2020</small>
          </div>
          <div class="col-6 col-md">
            <h5>Group 3 Members</h5>
            <ul class="list-unstyled text-small">
              <li><a class="text-muted" href="#">Aditi Tyagi</a></li>
              <li><a class="text-muted" href="#">Chang Zhou</a></li>
              <li><a class="text-muted" href="#">Shangzhe Zhang</a></li>
              <li><a class="text-muted" href="#">Yuance Li</a></li>
              <li><a class="text-muted" href="#">Yue Yang</a></li>
              <li><a class="text-muted" href="#">Yashdeep Singh</a></li>
            </ul>
          </div>
          <div class="col-6 col-md">
            <h5>Resources</h5>
            <ul class="list-unstyled text-small">
              <li><a class="text-muted" href="https://github.com/y1singh/ECE-229-Group-3">GitHub</a></li>
              <li><a class="text-muted" href="https://coronavirus.jhu.edu/us-map">John Hopkins University Website</a></li>
            </ul>
          </div>
          <div class="col-6 col-md">
          </div>
        </div>
</footer>
</main>


{% endblock %}
"""

nb_template = """
{% extends base %}

{% block contents %}
<main role="main">
<div class="carousel-inner">
  <div class="carousel-item active">
    <img class="first-slide" 
    src='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTHEZgYN6qQ-nvq0QpFA65PmyEKyvxqSh8YQDnEoqwLIsaTVeba&usqp=CAU'
    style='width: inherit;'
    alt="First slide">
    <div class="container">
      <div class="carousel-caption text-left">
        <h1 style="
            color: goldenrod;
            font-size: xxx-large;
            font-weight: bold;
            text-align: left;
        ">Economic Impacts of COVID-19</h1>
        <p style="
            text-align: left;
            font-size: larger;
            color: gray;
        ">Analysis and prediction of stocks with respect to spread of COVID.</p>
      </div>
    </div>
  </div>
</div>
<br>

<div class="container marketing">
  <div class="row featurette">
    <div class="col-md-5" style="margin-top: 20px;">
        <h2 class="featurette-heading">Outbreak of COVID across the US <span class="text-muted"></span></h2>
        <p class="lead">The outbreak in US took place in mid February. From March
        onwards, the cases expanded exponentially.</p>
    </div>
    <div class="col-md-6">
      {{ embed(roots.us_spread) }}
    </div>
  </div>
  
    <div class="col-md-15">
      <div class="row featurette">
        <div class="row-md-5" style="margin-top: 20px;">
            <h2 class="featurette-heading">Industry wise Stock Prices<span class="text-muted"></span></h2>
            <p class="lead">This graph gives an idea of stock prices as compared with time. Notice that
            there is a huge fall in the market overall.</p>
        </div>
      </div>
      <div class="row-md-6">
        {{ embed(roots.stocks) }}
      </div>
    </div>
    
    <br>
    
    <div class="row featurette">
      <div class="row-md-2" style='margin-top: 50px;'>
        {{ embed(roots.p_val) }}
      </div>
        <div class="col-md-7" style="margin-left: 60px; margin-top: 20px;">
            <h2 class="featurette-heading">Analyzing correlation of stocks with COVID
            <span class="text-muted">(p-value analysis)</span></h2>
            <p class="lead">Note:- If you select one of the industries, it will open up 
            a time series graph.<br> In this graph, you can see stock prices of companies of the selected
            industry. The size of bubbles is the p-value. This means that larger the bubble, lesser
            is the impact of spread of COVID. On the other hand, smaller the bubble, the impact of
            COVID is more. Thus, you might notice that a lot companies have small bubble sizes(large correlation with 
            COVID) during the early March phase.</p>
        </div>
    </div>
    
    <br>
    
    <div class="col-md-15">
      <div class="row featurette">
        <div class="row-md-5" style="margin-top: 20px;">
            <h2 class="featurette-heading">Prediction of Stock Prices<span class="text-muted"></span></h2>
            <p class="lead">We fit in a Spline to the correlation between COVID and stock prices
            and predicted stock prices and the results are as shown in the graph.</p>
        </div>
      </div>
      <div class="row-md-6">
        {{ embed(roots.prediction) }}
      </div>
    </div>
    
</div>


<footer class="pt-4 my-md-5 pt-md-5 border-top">
        <div class="row">
          <div class="col-6 col-md">
          </div>
          <div class="col-12 col-md">
            <small class="d-block mb-3 text-muted">ECE 229 Spring 2020</small>
          </div>
          <div class="col-6 col-md">
            <h5>Group 3 Members</h5>
            <ul class="list-unstyled text-small">
              <li><a class="text-muted" href="#">Aditi Tyagi</a></li>
              <li><a class="text-muted" href="#">Chang Zhou</a></li>
              <li><a class="text-muted" href="#">Shangzhe Zhang</a></li>
              <li><a class="text-muted" href="#">Yuance Li</a></li>
              <li><a class="text-muted" href="#">Yue Yang</a></li>
              <li><a class="text-muted" href="#">Yashdeep Singh</a></li>
            </ul>
          </div>
          <div class="col-6 col-md">
            <h5>Resources</h5>
            <ul class="list-unstyled text-small">
              <li><a class="text-muted" href="https://github.com/y1singh/ECE-229-Group-3">GitHub</a></li>
              <li><a class="text-muted" href="https://coronavirus.jhu.edu/us-map">John Hopkins University Website</a></li>
            </ul>
          </div>
          <div class="col-6 col-md">
          </div>
        </div>
</footer>
</main>

{% endblock %}
"""

tmpl = pn.Template(template, nb_template=nb_template)

# spread
def covid_spread_animation(data,mode="north america",world=True):
    '''
    animation of how it grows over time around the US
    '''
    assert(isinstance(world,bool))
    assert(isinstance(mode,str))
    assert(isinstance(data,pd.DataFrame))
           
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
                                    scope=mode
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
#                                     color = "Country/Region",
                                    title="Outbreak Across The World")
#         py.offline.plot(fig_growth,filename='outputs/html/world_spread.html');

    return fig_growth

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

# industry
def industry_visualization(dict_industries):
    '''
    Description: Converts the dict of industries to holoviews time series plot
    Input: Dict of industries where each key value is a dataframe
    Output: Panel app which updates the plot based on menu selection
    
    '''
    assert isinstance(dict_industries,dict)
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
    def load_symbol(Industry, **kwargs):
        return hvPlot(dict_industries[Industry].T, width=800, height=400)()

    app=pn.Column(pulldown,load_symbol)
#     app.servable() # % panel serve
    return app

# p value
with open('data/save.p', 'rb') as handle:
    dict_industries_pickle = pickle.load(handle)
    
result_industry = industry_visualization(dict_industries_pickle)

tmpl.add_panel('stocks', result_industry)

tmpl.add_panel('us_spread', pn.pane.HTML(HTML("outputs/html/growth_america.html")))

start_dt = datetime.datetime.strptime("3/01/20",'%m/%d/%y')
end_dt = datetime.datetime.strptime("4/30/20",'%m/%d/%y')
finalpval = pd.read_csv('data/p_val_march.csv')
industry_pval_dict = {}

for industry_val in finalpval.industry.unique().tolist():
    industry_pval_dict[industry_val] = finalpval.loc[finalpval['industry'] == industry_val]

def calc_pval(industries,industry_pval_dict):
    assert isinstance(industry_pval_dict,dict)
    pulldown = pn.widgets.Select(name='Industry',options=industries)
    
    @pn.depends(pulldown.param.value)
    def get_plot(industry):
        figure = px.scatter(industry_pval_dict[industry], 'Ticker' , 'price'  , size = 'p_val' , 
                        animation_frame = 'Date', range_x = [start_dt,end_dt], size_max=50,
                        range_y = [industry_pval_dict[industry]['price'].min()-50,industry_pval_dict[industry]['price'].max()+50],
                        hover_data= {"price":True, "p_val_cumsum":False,"p_val":False})
        return figure
    
    app = pn.WidgetBox(pulldown,get_plot)
    return app

industries = list(industry_pval_dict.keys())
tmpl.add_panel('p_val', calc_pval(industries,industry_pval_dict))

# prediction
confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
time_series_confirmed = drop_null_vals(pd.read_csv(confirmed_url,error_bad_lines=False))

china_industry_data = pd.read_csv("data/china_industry_mean.csv")
china_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="China"].iloc[:,4:].sum()
china_spread=pd.DataFrame(china_spread,columns=["Confirmed"])
china_spread.index.rename("Date",inplace=True)
china_spread.set_index(pd.to_datetime(china_spread.index),inplace=True)

us_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="US"].iloc[:,4:].sum()
us_spread=pd.DataFrame(us_spread,columns=["Confirmed"])
us_spread.index.rename("Date",inplace=True)
us_spread.set_index(pd.to_datetime(us_spread.index),inplace=True)

def str_to_datetime_index(data):
    '''
    Converts the string date index
    to datetime index.
    '''
    assert isinstance(data,pd.DataFrame)
    return data.set_index(pd.to_datetime(data.iloc[:,2:].set_index(data.index).index),inplace=True)


str_to_datetime_index(china_industry_data)
china_industry_data=china_industry_data.iloc[:,2:]
joined_table = pd.concat([china_spread,china_industry_data],axis=1,join="inner")
joined_table = joined_table[joined_table.Entertainment.notna()]
joined_table = joined_table[joined_table.Confirmed.notna()]
joined_table.drop("Unnamed: 118",axis=1,inplace=True)
joined_table.fillna(method="ffill",inplace=True)

def predict_and_plot(industries):
    
    pulldown = pn.widgets.Select(name='Industry',options=industries)

    @pn.depends(pulldown.param.value)
    def get_plot(symbol):
        try:
            tick = random.choices(industry_pval_dict[symbol].Ticker.unique())[0]
            joined_table[["Confirmed",symbol]]
            train_x = joined_table.Confirmed.to_list()
            train_y = joined_table[symbol].to_list()
            scaler1 = MinMaxScaler(feature_range=(-1, 1))
            scaler2 = MinMaxScaler(feature_range=(-1, 1))
            scaler3 = MinMaxScaler(feature_range=(-1, 1))

            x = scaler1.fit_transform(np.array(train_x).reshape(-1, 1))
            y = scaler2.fit_transform(np.array(train_y).reshape(-1, 1))


            test_data = finalpval[finalpval["Ticker"]==tick]
            test_data.set_index(pd.to_datetime(test_data.Date),inplace=True)
            concat_test_data = pd.concat([us_spread,test_data.price],axis=1,join="inner")

            test_x = concat_test_data.Confirmed.to_numpy()
            test_x = scaler1.fit_transform(test_x.reshape(-1,1))
            test_y = concat_test_data.price.to_numpy()

            spline = UnivariateSpline(x,y,k=3)

            prediction=[]
            for i in test_x:
                prediction.append(spline(i))

            temp = scaler3.fit_transform(np.array(test_y).reshape(-1, 1))

            predicted_y = scaler3.inverse_transform(np.array(prediction).reshape(-1,1))

            figure = plt.figure(figsize=(13,8));
            plt.plot(predicted_y[:-1]);
            plt.plot(concat_test_data.price.to_list()[:-1]);
            plt.legend(["Predicted Value","Actual Value"]);
            plt.title("Predicted vs. actual stock values for "+tick);
            plt.ylabel("Stock price");
            plt.xlabel("Date");

        except:
            print("Data not found for ",symbol)
            figure = plt.figure(figsize=(10,6));

        return figure;
    
    app=pn.Column(pulldown,get_plot)
#     app.servable()
    return app

industries = list(china_industry_data.columns[2:])
tmpl.add_panel('prediction', predict_and_plot(industries))
