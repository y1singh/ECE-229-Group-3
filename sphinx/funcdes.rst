Overview of functions
========================

.. py:function:: covid_spread_animation

   The function generates an html output file to store the increase in confirmed cases in United States on Data axis animation. 

   *Input:*

   .. py:data:: Data

   DataFrame containing the number of confirmed covid cases across US.

   .. py:data:: Mode

   String that tells which scatter geo map to plot on.

   .. py:data:: World

   Boolean value to confirm if it is world map or not.

   *Output:*

   Plotly scatter geo plot with increase of confirmed cases visualization. 


.. py:function:: industry_visualization

   The function generates a panel servable app to plot holoview time series analysis of stock market prices within different industries. 

   *Input:*

   .. py:data:: dict_industries

   Dictionary of dataframes that contains stock market prices with each key value as the industry type.

   *Output:*

   Panel app which updates the plot based on menu selection. 

.. py:function:: calc_pval

   The function generates a plotly scatter animation plot with bubble size representing the p value to compare the correlation of stock prices with confirmed cases over time. 

   *Input:*

   .. py:data:: Industries

   List of unique keys of the the industries dictionary for drop down selection. 

   .. py:data:: Industy_pval_dict

   Panel app which updates the plotly plot based on menu selection and provides p value animation. 

   *Output:*

   Panel app which updates the plot based on menu selection. 

.. py:function:: predict_and_plot

   The function trains a spline regression model based on the China Industries data and predicts the stock prices for US industries. 

   *Input:*

   .. py:data:: Industries

   List of china industries to train the model on. 

   *Output:*

   Panel app which updates the matplotlib plot based on menu selection and provides stock prices prediction plot. 