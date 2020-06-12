from generate_output import *
# from generate_output import drop_null_vals, covid_spread_animation, industry_visualization, str_to_datetime_index, predict_and_plot
# import pandas as pd
# import panel as pn
# import plotly as py
# import pickle
# import numpy as np

def test_func():
    ######################################################################
    # check drop_null_vals
    df = pd.DataFrame(np.random.randn(100,100))
    df[df > 0.9] = pd.np.nan
    dropped = drop_null_vals(df)
    cols_to_check = dropped.columns
    a = dropped[cols_to_check].isnull().apply(lambda x: all(x), axis=1)
    b = dropped[cols_to_check].isnull().apply(lambda x: all(x), axis=0)
    assert isinstance(dropped, pd.DataFrame)
    assert not(a.any() or b.any())

    ######################################################################
    # check covid_spread_animation
    us_confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    us_time_series_confirmed = drop_null_vals(pd.read_csv(us_confirmed_url,error_bad_lines=False))
    us_time_series_confirmed = us_time_series_confirmed.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region', 'Long_': 'Long'})
    assert isinstance(covid_spread_animation(us_time_series_confirmed,"north america",world=False),py.graph_objs._figure.Figure)

    ######################################################################
    # check industry_visualization
    with open('data/stocks_data.p', 'rb') as handle:
        dict_industries_pickle = pickle.load(handle)
    result_industry = industry_visualization(dict_industries_pickle)
    assert isinstance(result_industry,pn.layout.Column)

    ######################################################################
    # check str_to_datetime_index
    china_industry_data = pd.read_csv("data/china_industry_mean.csv")
    str_to_datetime_index(china_industry_data)
    assert isinstance(china_industry_data, pd.DataFrame)
    assert isinstance(china_industry_data.index, pd.core.indexes.datetimes.DatetimeIndex)

    ######################################################################
    # check predict_and_plot
    industries = list(china_industry_data.columns[2:])
    assert predict_and_plot(industries) is None
    ######################################################################