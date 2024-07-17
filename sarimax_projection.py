import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Cargar datos desde un archivo CSV
df = pd.read_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/tipo_cambio.csv')

# Procesar los datos como antes
df.columns = df.columns.str.strip()
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['Fecha'])
df.set_index('Fecha', inplace=True)
df.sort_index(inplace=True)
df = df.asfreq('D', method='ffill')

# Función para realizar la prueba de Dickey-Fuller
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result[1]  # Devuelve el p-valor

# Prueba de Dickey-Fuller
p_value = adf_test(df['Tipo_Cambio'])
if p_value > 0.05:
    df['Tipo_Cambio_Dif'] = df['Tipo_Cambio'].diff().dropna()
    df = df.dropna()
    p_value = adf_test(df['Tipo_Cambio_Dif'])

# Usar auto_arima para encontrar los mejores parámetros
stepwise_model = auto_arima(df['Tipo_Cambio'], start_p=1, start_q=1,
                            max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)

print(stepwise_model.summary())

# Ajustar el modelo con los mejores parámetros
best_model = SARIMAX(df['Tipo_Cambio'], order=stepwise_model.order, 
                     enforce_stationarity=False, enforce_invertibility=False)
best_results = best_model.fit()

# Realizar la proyección para los próximos 365 días
forecast = best_results.get_forecast(steps=365)
forecast_index = pd.date_range(start=df.index[-1], periods=365, freq='D')
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# Guardar la proyección en un archivo CSV
forecast_series.to_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/proyeccion_tipo_cambio_auto_arima.csv', header=['Tipo_Cambio_Proyectado'])

# Visualizar la serie temporal y la proyección
plt.figure(figsize=(10, 6))
plt.plot(df['Tipo_Cambio'], label='Tipo de Cambio')
plt.plot(forecast_series, label='Proyección', color='red')
plt.title('Tipo de Cambio Diario y Proyección a 365 Días')
plt.xlabel('Fecha')
plt.ylabel('Tipo de Cambio')
plt.legend()
plt.show()


