from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde un archivo CSV
df = pd.read_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/tipo_cambio.csv')

# Eliminar espacios en blanco de los nombres de las columnas
df.columns = df.columns.str.strip()

# Convertir la columna de fecha al tipo datetime especificando el formato correcto
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')

# Verificar si hay fechas no válidas
print(df[df['Fecha'].isna()])

# Eliminar filas con fechas no válidas
df = df.dropna(subset=['Fecha'])

# Establecer la columna de fecha como el índice del DataFrame
df.set_index('Fecha', inplace=True)

# Rellenar huecos con el último valor conocido
df = df.asfreq('D', method='ffill')

# Preparar los datos para Prophet
df.reset_index(inplace=True)
df.rename(columns={'Fecha': 'ds', 'Tipo_Cambio': 'y'}, inplace=True)

# Ajustar el modelo Prophet
model = Prophet()
model.fit(df)

# Realizar la proyección para los próximos 365 días
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Guardar la proyección en un archivo CSV
forecast[['ds', 'yhat']].to_csv('C:/Users/RIPLEY MIRAFLORES/ProyectoTipodeCambio/proyeccion_tipo_cambio_prophet.csv', index=False)

# Visual
