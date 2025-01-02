#Lab1 visualización de datos
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

print('Data read into a pandas dataframe!')

df_can.head(5)
df_can.tail(5)

df_can.info(verbose=False) # obtener información váliosa
df_can.columns
df_can.index

print(type(df_can.columns))
print(type(df_can.index))

df_can.columns.tolist()
df_can.index.tolist()

print(type(df_can.columns.tolist()))
print(type(df_can.index.tolist()))

df_can.shape # para ver las dimensiónes del dataframe

# Limpiar el dataset para remover algunas columnas innecesarias

# en pandas axis=0 representa filas (por defecto) y axis=1 representa columnas.
df_can.drop(['Area','REG', 'Dev', 'Type', 'Coverage'], axis=1, inplace=True)
df_can.head(2)

#Renombrar las columas con rename().
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns

#Agregar la clumna 'Total'
df_can['Total']=df_can.sum(axis=1)

#Checar cuantos objetos null tenemos en el dataset
df_can.isnull().sum()

#Finalmente podemos ver un pequeño resumen de cada columna en el dataframe 
# con describe()
df_can.describe()

'''Intermedio de Pandas

Seleccionar columnas 

Hay 2 métodos:

    1. Fácil y rápido pero solo funciona sí la columna 

no tiene espacios o caracteres especiales.

df.column_name   retorna series

    2. Más robusto y puede filtrar multiples columnas

df['column']   retorna series

df[['column 1', 'column 2']]   retorna dataframe'''


# Ejemplo filtrar en el listado de Countries
df_can.Country 

#Ejemplo filtrar la lista de países y los datos de los años 1980 - 1985

df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]]  # retorna un dataframe


'''Seleccionar Filas
Hay 2 maneras
1. df.loc[label]  filtra por las etiquetas de el índice del index/columna

2. df.iloc[index] filtra por las posiciones del index/columna
'''

# cambiamos los índices para más comodidad de busqueda con set_index()

df_can.set_index('Country', inplace=True)

df_can.head(3)

#remover el nombre del índice (opcional)
df_can.index.name = None


'''EJEMPLO
Veamos el número de immigrantes desde Japón para los siguientes escenarios
1. La fila completa de datos(todas las columnas)
2. Para el año 2013
3. Para los años 1980 a 1985
'''

#1 todas las columnas
df_can.loc['Japan']
df_can[df_can.index == 'Japan']

# método  alternativo

df_can.iloc[87]
df_can[df_can.index == 'Japan']

#2 Para el año 2013
df_can.loc['Japan', 2013]

# método alternativo
# el año 2013 es la última columna, con un indice posicional de 36

df_can.iloc[87, 36]

#3 para los años 1980 - 1985

df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984]]

# método alternativo
df_can.iloc[87, [3, 4, 5, 6, 7, 8]]

'''Cambiar de tipo las columnas de int a strings 
para no causar confusión'''

df_can.columns = list(map(str, df_can.columns))

# dado a que cambiamos los años a string, declararemos la 
# variable que nos permita llamar a todo el rango de años

#útil para plotearlo después
years = list(map(str, range(1980, 2014)))
years

'''Filtrar en base a los criterios

Para filtrar el dataframe basado en una condición, solo pasamos 
la condición como un vector booleano

Por ejemplo: filtremos el dataframe para mostrar los datos de 
los países Asiaticos'''

#1 crear la condición bolleana
condition = df_can['Continent'] == 'Asia'
print(condition)

#2 pasar la condición al dataframe
df_can[condition]

'''Podemos pasar diferentes criterios en la misma línea
filtramos AreaName = Asia y RegName = Southern Asia'''

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]

#Nota: pandas requiere & y | para 'and' y 'or'

# revisemos los cambios en el dataframe 
print('data dimensions: ', df_can.shape)
print(df_can.columns)
df_can.head(2)

'''Matplotlib: Librería de Visualización estandar 

Aplicar un estilo a Matplotlib:'''

print(plt.style.available)
mpl.style.use(['ggplot']) #Opcional

'''Line Plots'''

#Extraer las series de datos para Haití

haiti = df_can.loc['Haiti', years]
haiti.head()

#Dibujar una linea con .plot() en el dataframe de Haiti
haiti.plot()

'''Por defecto pandas pone el eje x con el índice (años),
y el eje y con la columna de valores (population).

Como sea los años no se muestran por el type que es string.
Por eso hay que cambiar ese type a int

También etiquetaremos los ejes x y y usando plt.title(), 
plt.ylabel() y plt.xlabel()
'''

haiti.index = haiti.index.map(int)
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.show()

'''Podemos notar el número de immigrantes de haití alcanzó
su máximo desde 2010 mientras que Canada aumento sus esfuerzos
para aceptar refugiados de Haití.

Anotemos el pico en el gráfico de la gráfica con el método 
plt.text()'''

haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

#anotemos el terremoto de 2010.
#sintáxis: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake')

plt.show()

'''plot() tiene bastantes opciones de gráficas como:

* bar para gráficas de barras verticales
* barh para gráficas de barras horizontales
* hist para histogramas 
* box para gráficas de cajas
* kde para gráficas de densidad
* area para gráficas de área
* pie para gráficas de pastel
* scatter para gráficas de distribución
* hexbin para gráficas hexbin

Autor: Alex Aklson 
IBM
'''