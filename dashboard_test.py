from dashboard import *

# testing
def test_final():
    '''final test'''
    testdata = pd.read_csv('data/time_series_covid_19_confirmed.csv')
    assert isinstance(covid_spread_animation(testdata),py.graph_objs._figure.Figure)
    with open('data/save.p', 'rb') as handle:
        dict_industries_pickle = pickle.load(handle)
    result_industry = industry_visualization(dict_industries_pickle)
    assert isinstance(result_industry,pn.layout.Column)
    industries = list(industry_pval_dict.keys())
    assert isinstance(calc_pval(industries,industry_pval_dict),pn.layout.WidgetBox)
    industries = list(china_industry_data.columns[2:])
    assert isinstance(predict_and_plot(industries),pn.layout.Column)

