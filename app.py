import dash
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from IPython.core.display import display, HTML, Image
import pickle
import matplotlib.pyplot as plt
import random
from scipy.interpolate import SmoothBivariateSpline
from sklearn.preprocessing import MinMaxScaler

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

def pick_top_six_stocks(dict_industries):
    
    for key in list(dict_industries.keys()):
            data_tr = dict_industries[key].T
            data_tr.loc['max_diff']=dict_industries[key].T.max()-dict_industries[key].T.min()
            dict_industries[key] = data_tr.T

            dict_industries[key] = dict_industries[key].sort_values('max_diff' , ascending=False)
            dict_industries[key] = dict_industries[key].head(6)
            dict_industries[key] = dict_industries[key].drop(['max_diff'], axis = 1)
            
    return dict_industries

def plot_stock_timeseries(industry):
    '''
    Given the industry type,
    returns timeseries
    plot of the stock
    from the dict_industries
    dictionary.
    '''
    assert(isinstance(industry,str))
    
    return px.line(dict_industries[industry].T,width=800,height=500)

def plot_pval_animation(industry):
    '''
    Given the industry type,
    returns the pvalue
    animation for each
    stock.
    '''
    assert(isinstance(industry,str))
    
    
    plot =  px.scatter(industry_pval_dict[industry], 'Ticker' , 'price'  , color = 'p_val' ,
             color_continuous_scale=[(0.00, "red"),   (0.05, "red"),(0.33, "blue"), (1.00, "blue")],
             size = 'p_val' , 
             range_color=[0.0, 1.0],
             animation_frame = 'Date',
             range_x = [start_dt,end_dt], size_max=50,
             range_y = [industry_pval_dict[industry]['price'].min()-50,industry_pval_dict[industry]['price'].max()+50],
             hover_data= {"price":True, "p_val_cumsum":False,"p_val":False}, width=800, height=500,
             title = "P-Value plots for stocks of "+industry+" industry")
    
    return plot

def predict_and_plot(symbol):
#     try:
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

    result = np.hstack((np.array(concat_test_data.price.to_list()[1:-1]).reshape(41,1),predicted_y))
    fig = px.line(result,width=800,height=500,title="Plotting predicted vs actual values of "+tick)
        
    return fig

def str_to_datetime_index(data):
    '''
    Converts the string date index
    to datetime index.
    '''
    return data.set_index(pd.to_datetime(data.iloc[:,2:].set_index(data.Date).index),inplace=True)

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
    
    
    
# Step 1. Launch the application
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
bgd_img_url = "https://ewscripps.brightspotcdn.com/dims4/default/7671677/2147483647/strip/true/crop/1303x733+15+0/resize/1280x720!/quality/90/?url=https%3A%2F%2Fewscripps.brightspotcdn.com%2F0a%2Ff2%2F72b1b4d94794992a0772cb593ce5%2Fscreen-shot-2020-02-25-at-10.49.27%20AM.png"

with open('data/stocks_data.p', 'rb') as handle:
    dict_industries_pickle = pickle.load(handle)
    handle.close()
    
# Get all the data
start_dt = datetime.datetime.strptime("3/01/20",'%m/%d/%y')
end_dt = datetime.datetime.strptime("4/30/20",'%m/%d/%y')
china_industry_data = pd.read_csv("data/china_industry_mean.csv")
china_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="China"].iloc[:,4:].sum()
china_spread=pd.DataFrame(china_spread,columns=["Confirmed"])
china_spread.index.rename("Date",inplace=True)
china_spread.set_index(pd.to_datetime(china_spread.index),inplace=True)
str_to_datetime_index(china_industry_data)
china_industry_data=china_industry_data.iloc[:,2:]
joined_table = pd.concat([china_spread,china_industry_data],axis=1,join="inner")
joined_table = joined_table[joined_table.Entertainment.notna()]
joined_table = joined_table[joined_table.Confirmed.notna()]
joined_table.drop("Unnamed: 118",axis=1,inplace=True)
joined_table.fillna(method="ffill",inplace=True)

us_spread = time_series_confirmed[time_series_confirmed["Country/Region"]=="US"].iloc[:,4:].sum()
us_spread=pd.DataFrame(us_spread,columns=["Confirmed"])
us_spread.index.rename("Date",inplace=True)
us_spread.set_index(pd.to_datetime(us_spread.index),inplace=True)
dict_industries = pick_top_six_stocks(dict_industries_pickle)
    

industry_pval_dict = dict()
finalpval = pd.read_csv('data/p_val_march.csv')

for industry_val in finalpval.industry.unique().tolist():
    industry_pval_dict[industry_val] = finalpval.loc[finalpval['industry'] == industry_val]
industries = list(industry_pval_dict.keys())

# dropdown options
industry_opts = [{'label' : i, 'value' : i} for i in industries]
stock_opts = [{'label' : i, 'value' : i} for i in list(dict_industries.keys())]
predict_opts = [{'label' : i, 'value' : i} for i in list(china_industry_data.columns[2:])]

time_stock_layout = go.Layout(title = 'Time Series Plot',
                   hovermode = 'closest')

# Initialize the plots with "Airlines" industry
time_stock_plot = go.Figure(data=plot_stock_timeseries("Airlines"),layout=time_stock_layout)
pval_plot = go.Figure(data=plot_pval_animation("Airlines"),layout=time_stock_layout)
prediction_plot = go.Figure(data=predict_and_plot("Airlines"))


app.layout = dbc.Container([
#     dbc("Hello Bootstrap!", color="success"),
#     className="p-5",
                # a header and a paragraph
    html.Main(role="main",children = [
        
        # Code for top-panel
                html.Div(className="carousel-inner",children=[
                    html.Div(className = "carousel-item active",children=[
                        html.Img(src=bgd_img_url,
                                style = {"width": "inherit"}),
                        html.Div(className="container",children=[
                          html.Div(className = "carousel-caption text-left",children=[
                              html.H1(style={"text-align": "left"},
                                      children="Economic impacts of"),
                              html.H1(style={"text-align": "left"},
                                      children="COVID-19"),
                              html.P(style={"text-align":"left","font-size": "larger",
                                            "color": "lightgrey"},
                                    children="Analysis and prediction of stocks with respect to spread of COVID.")
                           ])  
                        ])
                    ])
                ]),
        
                html.Br(),
        
                html.Div(className="container marketing",
                        children=[
                            
                            # Code for US-Spread animation
                            html.Div(className="row featurette",
                              children = [
                                  html.Div(className="col-md-14",
                                      style={"margin-top":"20px"},
                                      children=[
                                          html.H2(className="featurette-heading",
                                              children = ["Outbreak of COVID across the US"
                                                  ]),
                                          html.P(className="lead",children="The outbreak in US took place in mid February. From March onwards, the cases expanded exponentially.")
                                               ]),
                                  html.Div(className="col-md-8",
#                                       style = {"left":"45px"},
                                      children = [
                                          html.Iframe(id = 'World',
                                                      srcDoc=open('outputs/html/growth_america.html','r').read(),
                                                      width=1050, height=720)
                                      ])
                                 ]),
                         
                            # Stock market data
                            html.Div(className="row-md-15",
                                children=[
                                    html.Div(className="row featurette",
                                        children=[
                                            html.Div(className="col-md-5",
                                                     style={"margin-top":"20px"},
                                                 children=[
                                                    html.H2(className="featurette-heading",
                                                       children="Industry wise Stock Prices"),
                                                     html.P(className='lead',
                                                           children="This graph gives an idea of stock prices\
                                                           as compared with time. Notice that\
                                                           there is a huge fall in the market overall.")
                                             ])
                                    ]),html.Br(),
                                    html.Div([
                                        html.Div(style={"width":"300px","padding-right":"20px","padding-top":"100px"},
                                                children=[
                                                     # dropdown
                                                    html.P([
                                                        html.Label("Select Industry"),
                                                        dcc.Dropdown(id = 'stock_opt',
                                                                     options = stock_opts,
                                                                    value = "Airlines")
                                                        ])
                                        ]),dcc.Graph(id="stock_plot", figure=time_stock_plot)
                                    ],className = "row")
                            ]),html.Br(),
                            
                            
                            # P-value plots
                            html.Div(className="row featurette",
                                 children=[
                                     html.Div(className="col-md-10",
                                         children=[
                                             html.H2([
                                                 "Analyzing correlation of stocks with COVID",
                                                 html.Span([" (p-value analysis)"],className="text-muted")
                                             ],className="featurette-heading"),
                                             html.P(["In this graph, you can see stock prices of companies of the selected\
                                                    industry. The size of bubbles and its color as well is the p-value. Blue signifies lesser correlation with spread\
                                                     and Red signifies high correlation with the spread. Also, larger the bubble, lesser\
                                                    is the impact of spread of COVID. On the other hand, smaller the bubble, the impact of\
                                                    COVID is more. Thus, you might notice that a lot companies have small red bubbles (large correlation with \
                                                    COVID) during the mid March phase."],className="lead")
                                         ]),
                                    html.Div([
                                        html.Div(style={"width":"300px","padding-right":"20px","padding-top":"100px"},
                                                children=[
                                                     # dropdown
                                                    html.P([
                                                        html.Label("Select Industry"),
                                                        dcc.Dropdown(id = 'industry_opt',
                                                                     options = industry_opts,
                                                                    value = "Airlines")
                                                        ])
                                        ]),dcc.Graph(id="industry_plot", figure=pval_plot)
                                    ],className = "row")
                             ]),html.Br(),
                            
                            # Training and prediction
                            html.Div(className="col-md-15",
                                children=[
                                    html.Div(className="row featurette",
                                        children=[
                                            html.Div(className="row-md-5",
                                                children=[
                                                    html.H2(className="featurette-heading",
                                                       children="Prediction of Stock Prices"
                                                       ),
                                                    html.P(className="lead",
                                                          children="We fit in a Spline to the correlation between COVID and stock prices\
                                                            and predicted stock prices. The results were as follows.")
                                                ])
                                        ]),
                                    html.Div([
                                        html.Div(style={"width":"300px","padding-right":"20px","padding-top":"100px"},
                                                children=[
                                                     # dropdown
                                                    html.P([
                                                        html.Label("Select Industry"),
                                                        dcc.Dropdown(id = 'prediction_opt',
                                                                     options = predict_opts,
                                                                    value = "Airlines")
                                                        ])
                                        ]),dcc.Graph(id="prediction_plot", figure=prediction_plot)
                                    ],className = "row")
                                ])
                     ])
            ])
        ])


@app.callback(Output('stock_plot', 'figure'),
             [Input('stock_opt', 'value')])
def update_stock_figure(industry):
    # updating the plot
    print("Showing stocks for industry: ",industry)
    fig = go.Figure(data = plot_stock_timeseries(industry) , layout = time_stock_layout)
    return fig

@app.callback(Output('industry_plot', 'figure'),
             [Input('industry_opt', 'value')])
def update_pval_figure(industry):
    # updating the plot
    print("Plotting animation for industry: ",industry)
    fig = go.Figure(data = plot_pval_animation(industry) , layout = time_stock_layout)
    return fig

@app.callback(Output('prediction_plot', 'figure'),
             [Input('prediction_opt', 'value')])
def update_prediction_figure(industry):
    # updating the plot
    print("Predicting value for industry: ",industry)
    fig = go.Figure(data = predict_and_plot(industry))
    return fig
  
# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server(debug = False)