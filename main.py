import pandas as pd
import plotly.express as px
#import seaborn as sns
import streamlit as st
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#import matplotlib.pyplot as plt
import plotly.graph_objects as go





header = st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modelTraining=st.beta_container()

with header:
    st.title('Welcome to Rahul Tyagi Covid Data Analyser!')


with dataset:
    st.header('Los Angeles Covid Symptom DataSet')
    covid_data=pd.read_csv('COVID-19_Daily_Counts_of_Cases__Hospitalizations__and_Deaths.csv')

    st.write(covid_data.head())
    df = pd.DataFrame( covid_data)
   # df['month'] = pd.DatetimeIndex(df['DATE_OF_INTEREST']).year

    #lines = pd.DataFrame(covid_data['CASE_COUNT'].value_counts()).head(50)
    #st.line_chart(lines)

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('DEATH_COUNT'),
        y=alt.Y('HOSPITALIZED_COUNT'),
       # color=alt.Color("name:N")
    ).properties(title="Case vs Date Chart")
    st.altair_chart(chart, use_container_width=True)

   # fig = px.line(df, x='DATE_OF_INTEREST', y='CASE_COUNT')
    #fig = go.Figure()

   # fig.update_layout(
   #    margin=dict(l=20, r=20, t=20, b=20),
   #     paper_bgcolor="LightSteelBlue",
  #  )

   # fig.show()



with features:
    st.header('Below is the list of features that i Created ')

with modelTraining:
    st.header('Time to train')
    st.text('This is the time take to train the model')

    sel_col,disp_col=st.beta_columns(2)
    max_depth=sel_col.slider('What should be the max depth of the model',min_value=20,max_value=100,value=20,step=10)

    n_estimators=sel_col.selectbox('how many tress should be there',options=[100,200,300],index=0)

    sel_col.text('Here is the list of all feautres in my data :')
    sel_col.write(df.columns)

    input_feature=sel_col.text_input('which feature should be used ?','DEATH_COUNT')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else :
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
    X=df[[input_feature]]
    y=df[['DEATH_COUNT']]
    regr.fit(X,y)
    prediction=regr.predict(y)

    disp_col.subheader('Mean abs error is :')
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader('Mean squared error of the model is :')
    disp_col.write(mean_squared_error(y,prediction))
    disp_col.subheader('Mean R squared of the model is :')
    disp_col.write(r2_score(y,prediction))










