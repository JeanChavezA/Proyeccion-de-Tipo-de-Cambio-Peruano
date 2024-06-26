# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:06:55 2024

@author: RIPLEY MIRAFLORES
"""

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cargar datos desde un archivo CSV
df = pd.read_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/tipo_cambio.csv')

# Verificar los nombres de las columnas
print("Columnas originales:")
print(df.columns)

# Eliminar espacios en blanco de los nombres de las columnas
df.columns = df.columns.str.strip()

# Verificar los nombres de las columnas nuevamente
print("Columnas después de eliminar espacios en blanco:")
print(df.columns)

# Asegurarse de que la columna 'Tipo_Cambio' exista
if 'Tipo_Cambio' not in df.columns:
    print("Error: La columna 'Tipo_Cambio' no se encuentra en el DataFrame.")
else:
    # Convertir la columna de fecha al tipo datetime especificando el formato correcto
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')

    # Verificar si hay fechas no válidas
    print(df[df['Fecha'].isna()])

    # Eliminar filas con fechas no válidas (opcional)
    df = df.dropna(subset=['Fecha'])

    # Establecer la columna de fecha como el índice del DataFrame
    df.set_index('Fecha', inplace=True)

    # Asegurarse de que los datos estén ordenados por fecha
    df.sort_index(inplace=True)

    # Rellenar huecos con el último valor conocido
    df = df.asfreq('D', method='ffill')

    # Verificar las primeras y últimas filas después del procesamiento
    print(df.head())
    print(df.tail())

    # Visualizar la serie temporal
    plt.figure(figsize=(10, 6))
    plt.plot(df['Tipo_Cambio'], label='Tipo de Cambio')
    plt.title('Tipo de Cambio Diario')
    plt.xlabel('Fecha')
    plt.ylabel('Tipo de Cambio')
    plt.legend()
    plt.show()

    # Función para realizar la prueba de Dickey-Fuller
    def adf_test(series):
        result = adfuller(series)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        return result[1]  # Devuelve el p-valor

    # Asegurarse de que la serie sea estacionaria
    p_value = adf_test(df['Tipo_Cambio'])
    print('p-value: %f' % p_value)

    # Si la serie no es estacionaria (p-value > 0.05), se recomienda diferenciación
    if p_value > 0.05:
        df['Tipo_Cambio_Dif'] = df['Tipo_Cambio'].diff().dropna()
        df = df.dropna()
        p_value = adf_test(df['Tipo_Cambio_Dif'])
        print('p-value after differencing: %f' % p_value)

    # Determinar los parámetros (p, d, q) para ARIMA
    p = d = q = range(0, 4)  # Ampliar el rango de búsqueda de parámetros
    pdq = list(itertools.product(p, d, q))

    # Buscar el mejor modelo
    best_aic = float("inf")
    best_param = None

    warnings.filterwarnings("ignore")

    for param in pdq:
        try:
            model = SARIMAX(df['Tipo_Cambio'], order=param, enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
        except Exception as e:
            print(f"Error con parámetros {param}: {e}")
            continue

    if best_param:
        print(f'Mejor parámetro encontrado: {best_param}')
        
        # Ajustar el modelo con los mejores parámetros
        best_model = SARIMAX(df['Tipo_Cambio'], order=best_param, enforce_stationarity=False, enforce_invertibility=False)
        best_results = best_model.fit()
        
        # Realizar la proyección para los próximos 365 días
        forecast = best_results.get_forecast(steps=365)
        forecast_index = pd.date_range(start=df.index[-1], periods=365, freq='D')
        forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
        
        # Guardar la proyección en un archivo CSV
        forecast_series.to_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/proyeccion_tipo_cambio_sarimax.csv', header=['Tipo_Cambio_Proyectado'])

        # Visualizar la serie temporal y la proyección
        plt.figure(figsize=(10, 6))
        plt.plot(df['Tipo_Cambio'], label='Tipo de Cambio')
        plt.plot(forecast_series, label='Proyección', color='red')
        plt.title('Tipo de Cambio Diario y Proyección a 365 Días')
        plt.xlabel('Fecha')
        plt.ylabel('Tipo de Cambio')
        plt.legend()
        plt.show()

    else:
        print("No se encontraron parámetros adecuados.")
