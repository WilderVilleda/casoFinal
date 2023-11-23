#Gráfica: Dashboard

import numpy as np
import plotly.express as px
import yfinance as yf
import datetime
import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pyfolio as pf
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


# Descargar datos
stocks = ["COF", "WMT", "BIMBOA.MX", "MCD", "CL", "PG", "TM"]
fecha_final = datetime.date.today()
fecha_inicial = fecha_final - datetime.timedelta(days=365 * 3)
data = yf.download(stocks, start=fecha_inicial, end=fecha_final)["Adj Close"]


#Portafolio_promedio_simple
returns = data.pct_change()
ret_acumulado = (1+returns).cumprod()
pesos = np.array([1/6, 1/6,1/6,1/6,0,1/6,1/6])
returns["Portafolio_promedio_simple"] = returns.dot(pesos)
ret_acumulado_2 = (1+returns).cumprod()

#Portafolio_max_Sharpe
pesos1 = np.array([0, 0,0.4713360636886999,0.5286639363113002,0,0,0])
returns2 = data.pct_change()
ret_acumulado_3 = (1+returns2).cumprod()
returns2["Portafolio_maximo_sharpe"]=returns2.dot(pesos1)
ret_acumulado_4 = (1+returns2).cumprod()

#Portafolio_min_Volatilidad
pesos2 = np.array([0.0090749879390336, 0.1894368675809966,0.1211972087037164,0.2512103107260549,0.2418873547154737,0.0854810739198495,
                0.1017121964148752])
returns3 = data.pct_change()
ret_acumulado_5 = (1+returns3).cumprod()
returns3["Portafolio_minima_varianza"]=returns3.dot(pesos2)
ret_acumulado_6 = (1+returns3).cumprod()

#Nuevo dataframe con los retornos acumulados

columna_portafolio_2 = ret_acumulado_6["Portafolio_minima_varianza"]
columna_portafolio_1 = ret_acumulado_4["Portafolio_maximo_sharpe"]
columna_portafolio = ret_acumulado_2["Portafolio_promedio_simple"]

conjunto_datos = pd.concat([columna_portafolio_2, columna_portafolio_1, columna_portafolio], axis=1)

conjunto_datos.reset_index(inplace=True)

# Crear aplicación Dash
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server

app.title="Dashboard"

# Definir el diseño de la aplicación
app.layout = html.Div([
    html.H1("Dashboard para acciones de empresas dedicadas al consumo masivo y retornos históricos de los portafolios"),

    dcc.Dropdown(
        id='selector-acciones',
        options=[{'label': accion, 'value': accion} for accion in stocks],
        multi=True,
        value=['PG', "CL"],
        style={'width': '50%'}
    ),

    dcc.Dropdown(
        id='indicador',
        options=[
            {'label': 'Precio', 'value': 'precio'},
            {'label': 'Rendimiento Acumulado', 'value': 'rendimiento'}
        ],
        value='precio',
        style={'width': '50%'}
    ),

    dcc.RangeSlider(
        id='periodo',
        marks={i: (fecha_inicial + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 365 * 3, 200)},
        min=0,
        max=365 * 3,
        step=30,
        value=[0, 365 * 3]
    ),

    dcc.Graph(id='grafico-lineas'),
    dcc.Graph(id='graph-ms'),
    dcc.Graph(id='graph-mv'),
])

# Definir la lógica de las callbacks
@app.callback(
    [Output('grafico-lineas', 'figure'),
     Output('graph-ms', 'figure'),
     Output('graph-mv', 'figure')],
    [Input('selector-acciones', 'value'),
     Input('indicador', 'value'),
     Input('periodo', 'value')]
)
def actualizar_grafico(selected_accion, selector, fechas):
    if selector == "precio":
        df_filtered = data[selected_accion].iloc[fechas[0]:fechas[1]]
    elif selector == "rendimiento":
        df_filtered = (1 + data[selected_accion].pct_change()).cumprod().iloc[fechas[0]:fechas[1]]
    else:
        df_filtered = pd.DataFrame()  # Maneja otras opciones aquí

    # Gráfico de líneas
    fig = px.line(df_filtered, x=df_filtered.index, y=df_filtered.columns, labels={'value': selector},
                  title=f'{selector} de {",  ".join(selected_accion)}')

    # Gráfico histograma para Portafolio máximo Sharpe (graph-ms)
    graph_ms = px.histogram(returns2, x="Portafolio_maximo_sharpe", 
                            title="Distribución del retorno histórico para el portafolio que maximiza el Sharpe", 
                            labels={"Portafolio_maximo_sharpe": "Rendimiento Diario"})

    # Gráfico histograma para Portafolio mínima varianza (graph-mv)
    graph_mv = px.histogram(returns3, x="Portafolio_minima_varianza", 
                            title="Distribución del retorno histórico para el portafolio que minimiza el riesgo",
                            labels={"Portafolio_minima_varianza": "Rendimiento Diario"})

    return fig, graph_ms, graph_mv

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=30000)
