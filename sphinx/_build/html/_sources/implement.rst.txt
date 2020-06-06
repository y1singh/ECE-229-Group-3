Function Descriptions
========================

app.py
------------------------

.. py:function:: drop_null_vals(df,axis='both',subset=[])

    Drops columns with all nan values from a given data frame.

    :param pandas.DataFrame df: DataFrame for which columns are to be dropped.
    :param str axis: Drops all rows with nan if axis=rows, all columns if axis=columns, and both if axis=both. 
    :param list subset:
        For all columns in
        subset, remove the
        NaN rows.

    :return: DataFrame without NaN
    :rtype: pandas.DataFrame

.. py:function:: pick_top_six_stocks(dict_industries)

    Picks top six stocks with the large ``max_diff``
   
    :param dict dict_industries: Dictionary of dataframes that contains stock market prices with each key value as the industry type.
    :return: Dictionary of dataframes of selected stocks.
    :rtype: dict

.. py:function:: plot_stock_timeseries(industry)

    Given the industry type,
    returns timeseries
    plot of the stock
    from the dict_industries
    dictionary.
    
    :param str industry: Industry type
    :return: timeseries plot of the stock
    :rtype: plotly.express.line

.. py:function:: plot_pval_animation(industry)

    Given the industry type,
    returns the pvalue
    animation for each
    stock.
    
    :param str industry: Industry type
    :return: pvalue animation
    :rtype: plotly.express.scatter

.. py:function:: predict_and_plot(symbol)

    Given the industry
    of the market, trains
    a spline on the data
    from China and selects
    a random ticker from
    the selected industry
    and predicts the plot
    for that ticker based
    on the spread of COVID
    on that day and
    the stock value on
    previous day.

    :param str symbol: Industry type
    :return: prediction line of stock value
    :rtype: plotly.express.line

.. py:function:: str_to_datetime_index(data)

    Converts the string date index
    to datetime index.
    
    :param pandas.DataFrame data: DataFrame before conversion
    :return: DataFrame after conversion
    :rtype: pandas.DataFrame

.. py:function:: update_stock_figure(industry)

    Updates the stock figure.

    :param str industry: Industry type
    :return: Updated stock figure
    :rtype: plotly.graph_objects.Figure

.. py:function:: update_pval_figure(industry)

    Updates the pvalue figure.

    :param str industry: Industry type
    :return: Updated pvalue figure
    :rtype: plotly.graph_objects.Figure

.. py:function:: update_prediction_figure(industry)

    Updates the prediction figure.

    :param str industry: Industry type
    :return: Updated prediction figure
    :rtype: plotly.graph_objects.Figure

cal_pval.py
------------------------

This script downloads the data for
the stocks in ticker_industry_map.json.
It then calculates the p-value of each stock
w.r.t. the spread of COVID and stores it
as a csv in p_val_march.csv

China_industry_data.py
------------------------

This script downloads all the
Chinese stock market data and
stores the mapping of industry type
and stock in data/china_industry_map.p
and also stores the individual stock's
timeseries data in data/china_stocks_timeseries.csv.

generate_output.ipynb
------------------------

.. py:function:: covid_spread_animation(data,mode="north america",world=True)

    The function generates an html output file to store the increase in confirmed cases in United States on Data axis animation. 

    :param pandas.DataFrame data: DataFrame containing the number of confirmed covid cases across US.
    :param bool world: Boolean value to confirm if it is world map or not.
    :param str mode: String that tells which scatter geo map to plot on.
    :return: Plotly scatter geo plot with increase of confirmed cases visualization. 
    :rtype: plotly.express.scatter_geo

.. py:function:: industry_visualization(dict_industries)

    The function generates a panel servable app to plot holoview time series analysis of stock market prices within different industries. 

    :param dict dict_industries: Dictionary of dataframes that contains stock market prices with each key value as the industry type.
    :return: Panel app which updates the plot based on menu selection. 
    :rtype: panel.Column

.. py:function:: calc_pval(industries, industy_pval_dict)

    The function generates a plotly scatter animation plot with bubble size representing the p value to compare the correlation of stock prices with confirmed cases over time. 
    
    :param list industries: List of unique keys of the the industries dictionary for drop down selection. 
    :param dict industy_pval_dict: Dictionary of stock prices and p values with keys are industry types. 
    :return: Panel app which updates the plotly plot based on menu selection and provides p value animation. 
    :rtype: matplotlib.pyplot.figure.show()

.. py:function:: predict_and_plot(industries)

    The function trains a spline regression model based on the China Industries data and predicts the stock prices for US industries. 

    :param list industries: List of china industries to train the model on. 
    :return: Panel app which updates the matplotlib plot based on menu selection and provides stock prices prediction plot. 
    :rtype: matplotlib.pyplot.figure.show()