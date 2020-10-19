import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash


app = dash.Dash('__name__')
server = app.server


data = pd.read_csv('pg4_data.csv')
label = data['company'].values
labels = set(label)
app.layout = html.Div(children=[
        html.Div(className='row', children=[
                html.Div(className='four columns div-user-controls', children=[
                    html.H5('Stocks for companies listed in the NSE'),
                    html.P('Select a company down below'),
                    html.Label('COMPANY'),
                    dcc.Dropdown(id='my_dropdown', options=[

                        {'label': i, 'value': i} for i in labels],
                                 placeholder='Select company',
                                 multi=False,
                                 clearable=False,
                                 searchable=True,
                                 value=label[0]
                        )
                    ]
                ),
                html.Div(className='eight columns div-for-charts bg-grey', children=[
                    html.Div(dcc.Graph(id='graph', config={'displayModeBar': False}),)
                ])
            ]
        )
    ]
)
# callback for the graph


@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
# the plot function
def update_figure(plots):
    data1 = data
    dat = data1[data1['company'] == plots]
    dates = pd.to_datetime(dat.loc[:, 'date']).dt.date
    trace1 = go.Scatter(
        x=dates,
        y=dat['price'],
        mode='lines+markers',
        line=dict(color='#483D8B', dash='dot'),
        marker=dict(color='#A10197'),
        text=dat.company
    )

    fig = go.Figure(data=trace1)
    fig.update_layout(title=dict(text=plots, font_color='black', font_size=30, x=0.5),
                      xaxis=dict(title='Date', color='purple', showgrid=False, showspikes=True,
                                 spikethickness=1, spikedash='solid', spikemode='across',
                                 spikecolor='black'),
                      yaxis=dict(title='Price', color='green', showgrid=True, gridwidth=0.1,
                                 gridcolor='#BFCA90'),
                      hovermode='x',
                      autosize=True,
                      paper_bgcolor='#98A26A',
                      plot_bgcolor='#98A26A',
                      template='plotly_dark',
                      )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=700)
