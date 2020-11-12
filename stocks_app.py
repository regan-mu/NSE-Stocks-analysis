import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_daq as daq
import pandas as pd
import dash
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping


app = dash.Dash('__name__')
server = app.server


def generate_table(dataframe, max_rows=3):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style={'width': '100%', 'margin': 25, 'textAlign': 'center'})


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
                        ),
                    html.Div(id='stocks_table', className='row table-orders'),
                    html.P('Prediction for next day'),
                    daq.LEDDisplay(id='my-daq-leddisplay', value=0,
                                   color='#239B56', backgroundColor="#D6DBDF", size=30)
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
        line=dict(color='#483D8B', width=0.5),
        marker=dict(color='#A10197', size=4),
        text=dat.company
    )

    fig = go.Figure(data=trace1)
    fig.update_layout(title=dict(text=plots, font_color='black', font_size=30, x=0.5),
                      xaxis=dict(title='Date', color='purple', showgrid=False, showspikes=True,
                                 spikethickness=1, spikedash='solid', spikemode='toaxis+across+marker',
                                 spikecolor='black', spikesnap='cursor'),
                      yaxis=dict(title='Price', color='green', showgrid=True, gridwidth=0.1,
                                 gridcolor='#7AE2E6'),
                      hovermode='x',
                      autosize=True,
                      paper_bgcolor='#C2DBDC',
                      plot_bgcolor='#C2DBDC',
                      template='plotly_dark',
                      )

    return fig


@app.callback(
    Output(component_id='stocks_table', component_property='children'),
    Input(component_id='my_dropdown', component_property='value')
)
# Update table
def update_table(table):
    data1 = data
    dat = data1[data1['company'] == table]
    data2 = dat.loc[:, ['ticker', 'price', 'date']]
    table_data = generate_table(data2.iloc[-2:], 3)
    return table_data

@app.callback(
    Output(component_id='my-daq-leddisplay', component_property='value'),
    Input(component_id='my_dropdown', component_property='value')
)
# Model
def rnn(company):
    df = pd.read_csv('pg4_data.csv', parse_dates=True, index_col='date')
    df = df[df.company == company]
    df.drop(['ticker', 'company'], inplace=True, axis=1)
    df['price'] = df.price.apply(lambda x: x.replace(',', ''))
    df['price'] = pd.to_numeric(df.price, errors='coerce')
    train_data = df[:-7]
    test_data = df[-7:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    generator = TimeseriesGenerator(train_scaled, train_scaled, length=3, batch_size=1)
    model = Sequential()
    model.add(SimpleRNN(132, input_shape=(3, 1)))
    model.add(Dense(64))
    model.add(Dense(1))
    early_stops = EarlyStopping(monitor='val_loss', patience=2)
    validation = TimeseriesGenerator(test_scaled, test_scaled, length=3, batch_size=1)
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=20, validation_data=validation, callbacks=[early_stops])

    test_prediction = []
    first_eval_batch = test_scaled[-3:]
    current_batch = first_eval_batch.reshape(1, 3, 1)

    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_predictions = scaler.inverse_transform(test_prediction)
    return round(true_predictions[0][0], 2)

if __name__ == '__main__':
    app.run_server(debug=True, port=700)
