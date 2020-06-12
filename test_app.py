from app import *
import plotly as py

def test_fun():
    assert isinstance(drop_null_vals(pd.read_csv('data/time_series_covid_19_confirmed.csv')),pd.DataFrame)
    with open('data/stocks_data.p', 'rb') as handle:
        dict_industries_pickle = pickle.load(handle)
    assert isinstance(pick_top_six_stocks(dict_industries_pickle),dict)
    assert isinstance(plot_stock_timeseries("Airlines"),py.graph_objs._figure.Figure)
    assert isinstance(plot_pval_animation("Airlines"),py.graph_objs._figure.Figure)
    assert isinstance(predict_and_plot("Airlines"),py.graph_objs._figure.Figure)
