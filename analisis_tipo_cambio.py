import pandas as pd


# Cargar datos desde un archivo CSV
# Asegúrate de que el archivo 'tipo_cambio.csv' esté en la misma carpeta que este script
df = pd.read_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/tipo_cambio.csv')


# Visualizar los primeros datos
print(df.head())
    
# Verificar el tipo de datos de cada columna
print(df.dtypes)

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

# Verificar las primeras y últimas filas después del procesamiento
print(df.head())
print(df.tail())

import matplotlib.pyplot as plt

# Visualizar la serie temporal
plt.figure(figsize=(10, 6))
plt.plot(df, label='Tipo de Cambio')
plt.title('Tipo de Cambio Diario')
plt.xlabel('Fecha')
plt.ylabel('Tipo de Cambio')
plt.legend()
plt.show()

import itertools
import warnings
from statsmodels.tsa.stattools import adfuller

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
    df['Tipo de Cambio Dif'] = df['Tipo_Cambio'].diff().dropna()
    df = df.dropna()
    p_value = adf_test(df['Tipo de Cambio Dif'])
    print('p-value after differencing: %f' % p_value)

# Determinar los parámetros (p, d, q) para ARIMA
p = d = q = range(0, 3)
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
    except:
        continue

print(f'Mejor parámetro encontrado: {best_param}')