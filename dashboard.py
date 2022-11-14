from datetime import date

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from dash import dash_table
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

#creating dataframe
CIPLA_df = pd.read_csv('D:\Study\Project\Stocks-Visualizing-and-Analysis\Cleaned Datasets\Cleaned Cipla.csv')
HINDUNILVR_df = pd.read_csv('D:\Study\Project\Stocks-Visualizing-and-Analysis\Cleaned Datasets\Cleaned HUL.csv')
INFY_df = pd.read_csv('D:\Study\Project\Stocks-Visualizing-and-Analysis\Cleaned Datasets\Cleaned Infosys.csv')
RELIANCE_df = pd.read_csv('D:\Study\Project\Stocks-Visualizing-and-Analysis\Cleaned Datasets\Cleaned Reliance.csv')
df = pd.read_csv('D:\Study\Project\Stocks-Visualizing-and-Analysis\Cleaned Datasets\CombinedALL.csv')

#app creation
app=dash.Dash()

#creating web layout 
app.layout=html.Div(
    #creating tabs
    dcc.Tabs(
        id='dashboard-tabs', value='price-tab', children=[
            #crating tab1
            dcc.Tab(label='stock analysis', value='stock analysis',children=[
               
               #row1(name, code)
                html.Div(children=[   
                                   html.Div(html.H1('Stock Visulisation & Analysis')),   
                    dcc.Dropdown( df['Name'].unique(),
                        'select company',
         
                        id='Name',
                        
                        style={'width':'80%','margin':'auto','display':'inline-block', 'align':'center'}),



                ]),
                #row 2(selector)
                html.Div(children=[
                    dcc.RadioItems(id='selector',options=[
                        {'label': 'high', 'value': 'high'},
                        {'label': 'low', 'value': 'low'},
                        {'label': 'open', 'value': 'open'},
                        {'label': 'close', 'value': 'close'},
                        {'label': 'volume', 'value': 'volume'},
                        {'label': 'DaytoDay_Analysis', 'value': 'DaytoDay_Analysis'},
                        {'label': 'Trend_Analysis(SMA)', 'value': 'Trend_Analysis(SMA)'},
                        {'label': 'candle', 'value': 'candle'},],
                        value='candle',

                    style={'width':'60%','margin':'auto','display':'inline-block'}),
                ]

                ),
                html.Div(children=[
                    #stats of selected data works with Name
                    html.Div(children=[
                        dash_table.DataTable(id='stats'),
                    ],
                    
                    style={'width':'30%','border':'2px','margin':'auto','display':'inline-block'}
                    ),
                    html.Div(id='graph_div',children=[
                        dcc.Graph(id='indicator_graph'),
                    ]    
                    ),
                    
                ]), 

            ]),
            dcc.Tab(label='Combined Analysis', value='comparitive analysis',children=[
                html.H1('Stock Visulisation & Analysis'),
                dcc.Dropdown(['Trading_Analysis', 'Risk_Analysis', 'Open-price_Trend_Analysis','Close-price_Trend_Analysis','percent_change_in_price'], 'Trading_Analysis', id='comparitive_selector_dropdown'),
                html.Div(id='output_graph',children=[
                    dcc.Graph(id='comparission_graph'),
                ])

            ]),
        ]
    ),

)


@app.callback(
    Output('indicator_graph', 'figure'),
    Input('Name', 'value'),
    Input('selector', 'value'),

    #Input('my-date-picker-range', 'start_date'),
    #Input('my-date-picker-range', 'end_date')

)
def update_graph(Name,selector):
    dff=df[df['Name']== Name]


    #dff=dff.loc[start_date:end_date]
    print(dff)
    print(selector)
    #fig = px.scatter(x=dff[dff['Date'] ],
    #                 y=dff[dff['High'] ],
    #                 hover_name=dff[dff['Name']['high']])
    if selector=="candle":
        fig={
            'data':[
            go.Candlestick(
                x=dff.Date,
                open=dff.Open,
                high=dff.High,
                low=dff.Low,
                close=dff.Close
            
                )
            ]
        
        }
        return fig
    elif selector =="high":
        fig=px.line(x="Date",y="High",data_frame=dff,title=f"{Name} high price")

        return fig
    
    elif selector =="low":
        fig=px.line(x="Date",y="Low",data_frame=dff,title=f"{Name} low price")

        return fig

    elif selector =="open":
        fig=px.line(x="Date",y="Open",data_frame=dff,title=f"{Name} open price")
        

        return fig
    
    elif selector =="close":
        fig=px.line(x="Date",y="Close",data_frame=dff,title=f"{Name} Close price")
        

        return fig
    elif selector =="volume":
        fig=px.line(x="Date",y="Volume",data_frame=dff,title=f"{Name} volume price")
        

        return fig
    
    elif selector =="DaytoDay_Analysis":
        dff['Day_Perc_Change'] = dff['Adj Close'].pct_change()*100
        dff.dropna(axis = 0, inplace = True)
        fig=px.line(x="Date",y="Day_Perc_Change",data_frame=dff,title=f"{Name} Day to Day Analysis price")
        

        return fig
    elif selector =="Trend_Analysis(SMA)":
        ma_day = [10,20,50]

        df1 =pd.DataFrame()
        df1['Date']=dff['Date']
        for ma in ma_day:
            column_name = "MA for %s days" %(str(ma))
            df1[column_name]=pd.DataFrame.rolling(dff['Adj Close'],ma).mean()


        fig=px.line(x='Date',y=['MA for 10 days','MA for 20 days','MA for 50 days'],data_frame=df1,title=f"{Name} Trend Analysis (SMA) price")
        

        return fig 

    

@app.callback(
    Output('stats', 'figure'),
    Input('Name', 'value'),
    

 )
def stats(Name):
    dff=df[df['Name']== Name]
    stat=dff.describe()
    print(stat)
    #data=df.to_dict('stat')
    #print(data)
    # columns=[{"name": i, "id": i} for i in stat.columns]
    
    # return data, columns
    return {(stat.to_dict(dict='stat1'))}#, [{"name": i, "id": i} for i in stat.columns])}

@app.callback(
    Output('comparission_graph', 'figure'),
    Input('comparitive_selector_dropdown', 'value'),

)
def combinedAnalysis(csd):
    dff=df
    print(dff)
    print(csd)
    if csd=="Trading_Analysis":
        dff['MarketCap']=dff['Open']*dff['Volume']
        print(dff)
        fig=px.line(dff,x='Date',
                             y='Volume',title="Trading anlaysis",color="Name")

        return fig

    elif csd=="Risk_Analysis":
        dff['Daily Return'] = dff.apply(
            lambda x: diff(x['Adj Close'], x['Open']), axis=1)
        i=np.where(dff["Name"]=="INFOSYS")
        print(i)
        c=np.where(dff["Name"]=="CIPLA")
        print(c)
        r=np.where(dff["Name"]=="RELIANCE")
        print(r)
        h=np.where(dff["Name"]=="HINDUNILVR")
        print(h)
        imean=dff['Daily Return'].iloc[0:989].mean()
        print(imean)
        isd=dff['Daily Return'].iloc[0:989].std(axis=0)
        print(isd)
        cmean=dff['Daily Return'].iloc[2967:3956].mean(axis=0)
        print(cmean)
        csd=dff['Daily Return'].iloc[2967:3956].std(axis=0)
        print(csd)
        rmean=dff['Daily Return'].iloc[1978:2967].mean(axis=0)
        print(rmean)
        rsd=dff['Daily Return'].iloc[1987:2967].std(axis=0)
        print(rsd)
        hmean=dff['Daily Return'].iloc[989:1978].mean(axis=0)
        print(hmean)
        hsd=dff['Daily Return'].iloc[989:2967].std(axis=0)
        print(hsd)
        data={'Name':['RELIANCE','HINDUNILVR','CIPLA','INFOSYS'],
            'Expected Return':[rmean,hmean,cmean,imean],  #mean of the daily return of each company
            'Risk':[rsd,hsd,csd,isd],      #standard deviation of the each company
            }
        Correlation_data=pd.DataFrame(data)
        Correlation_data
        fig= px.scatter(Correlation_data, x="Expected Return", y="Risk", color="Name")
        fig.update_traces(marker=dict(size=15,line=dict(width=2,color='Black')))
        #fig.show()
        return fig

    elif csd=="Open-price_Trend_Analysis":

        CIPLA_df['SMA20 OPEN'] = CIPLA_df['Open'].rolling(20).mean()
        HINDUNILVR_df['SMA20 OPEN'] = HINDUNILVR_df['Open'].rolling(20).mean()
        RELIANCE_df['SMA20 OPEN'] = RELIANCE_df['Open'].rolling(20).mean()
        INFY_df['SMA20 OPEN'] = INFY_df['Open'].rolling(20).mean()
        
        open_comp=pd.concat([CIPLA_df['Date'],CIPLA_df['SMA20 OPEN'],HINDUNILVR_df['SMA20 OPEN'],RELIANCE_df['SMA20 OPEN'],INFY_df['SMA20 OPEN']],axis=1)
        open_comp.columns=['DATE','CIPLA','INFY','RELIANCE','HINDUNILVR']
        open_comp=pd.DataFrame(open_comp)
        open_comp.dropna()
        fig=px.line(x='DATE',y=['CIPLA','INFY','RELIANCE','HINDUNILVR'],data_frame=open_comp,title='Open Trend Analysis (SMA)')
        
        return fig

    elif csd=="Close-price_Trend_Analysis":
        CIPLA_df['SMA20 CLOSE'] = CIPLA_df['Close'].rolling(20).mean()
        HINDUNILVR_df['SMA20 CLOSE'] = HINDUNILVR_df['Close'].rolling(20).mean()
        RELIANCE_df['SMA20 CLOSE'] = RELIANCE_df['Close'].rolling(20).mean()
        INFY_df['SMA20 CLOSE'] = INFY_df['Close'].rolling(20).mean()
        
        close_comp=pd.concat([CIPLA_df['Date'],CIPLA_df['SMA20 CLOSE'],HINDUNILVR_df['SMA20 CLOSE'],RELIANCE_df['SMA20 CLOSE'],INFY_df['SMA20 CLOSE']],axis=1)
        close_comp.columns=['DATE','CIPLA','INFY','RELIANCE','HINDUNILVR']
        close_comp=pd.DataFrame(close_comp)
        close_comp.dropna()

        fig=px.line(x='DATE',y=['CIPLA','INFY','RELIANCE','HINDUNILVR'],data_frame=close_comp)
        
        return fig

    elif csd=="percent_change_in_price":
        CIPLA_df['percentage_change'] = (CIPLA_df['Close']/CIPLA_df['Close'].shift(1)) -1
        INFY_df['percentage_change'] = (INFY_df['Close']/INFY_df['Close'].shift(1))-1
        RELIANCE_df['percentage_change'] = (RELIANCE_df['Close']/RELIANCE_df['Close'].shift(1)) - 1
        HINDUNILVR_df['percentage_change'] = (HINDUNILVR_df['Close']/HINDUNILVR_df['Close'].shift(1)) -1

        fig = make_subplots(rows=2, cols=2,subplot_titles=("CIPLA", "INFOSYS", "RELIANCE", "HINDUNILVR"))

        fig.add_trace(
            go.Histogram(x=CIPLA_df['percentage_change']),
            row=1, col=1,
            
        )

        fig.add_trace(
            go.Histogram(x=INFY_df['percentage_change']),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=RELIANCE_df['percentage_change']),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=HINDUNILVR_df['percentage_change']),
            row=2, col=2
        )

        fig.update_layout(title_text="Price change")
        
        fig1=fig
        return fig1


#Calculating Daily Return
def diff(a, b):
    return b - a

#(x-> percentage change)
def trend(x):
    if x > -0.5 and x <= 0.5:
        return 'Slight or No change'
    elif x > 0.5 and x <= 1:
        return 'Slight Positive'
    elif x > -1 and x <= -0.5:
        return 'Slight Negative'
    elif x > 1 and x <= 3:
        return 'Positive'
    elif x > -3 and x <= -1:
        return 'Negative'
    elif x > 3 and x <= 7:
        return 'Among top gainers'
    elif x > -7 and x <= -3:
        return 'Among top losers'
    elif x > 7:
        return 'Bull run'
    elif x <= -7:
        return 'Bear drop'





if __name__ == '__main__':
    app.run_server(port=8000)