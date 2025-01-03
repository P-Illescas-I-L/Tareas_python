#2do Lab visualización de datos
import numpy as np
import pandas as pd

#descargar el dataset
df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

#print('Data downloaded and read into a dataframe!')

#Visualizar los primeros datos
#df_can.head()

#visualizar las dimensiónes del dataset
#print(df_can.shape)

#1. Limpieza de datos
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

#print(df_can.head())

#2. Renombrar alguna columnas para que haga más sentido
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

#df_can.head()

#3. Verificar el tipo de cada columna (que sea string)
all(isinstance(column,str) for column in df_can.columns)

#Cambiar a tipo string
df_can.columns = list(map(str, df_can.columns))

#verificar el tipo de dato otra vez
all(isinstance(column,str) for column in df_can.columns)

#4. Poner el nombre del país como índice

df_can.set_index('Country', inplace=True)

#print(df_can.head)

#5. Añadir la columna Total
df_can['Total'] = df_can.sum(axis=1)

#print(df_can.head)

print('data dimensions: ', df_can.shape)

#Crear una lista de años del 1980 - 2013
#útil para gráficar los datos
years = list(map(str, range(1980, 2014)))

print(years)

#Visualización de datos

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')

#Area Plots

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

#obtener el top 5
df_top5 = df_can.head()

#transponer el df
df_top5 = df_top5[years].transpose()

print(df_top5.head())

#para producir un gráfico sin stack, poner el
# párametro stacked al valor False

df_top5.index= df_top5.index.map(int)
df_top5.plot(kind='area',
            stacked=False,
            figsize=(20,10)) #pasar la tupla (x,y)

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

#Cambiar la transparencia con el valor alpha

df_top5.plot(kind='area',
            alpha = 0.25, # 0-1, el valor por defecto es 0.5 
            stacked=False,
            figsize=(20,10)) #pasar la tupla (x,y)

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

'''2 maneras de hacer plots

1. Mediante el Scripting Layer (método procedural),
usando matplotlib.pyplot as plt

df_top5.plot(kind='area',
            alpha = 0.25, # 0-1, el valor por defecto es 0.5 
            stacked=False,
            figsize=(20,10)) #pasar la tupla (x,y)

2. Usar el Artist Layer (método orientado a objetos),
usar la instancia Axes de Matplotlib (preferido)

esta es una forma más transparente y flexible de usar para
plots más avanzados.

ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

ax.set_title('Immigration Trend of 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')
'''

'''EJERCICIO Usar el método 1 de ploteo (scripting layer)
para crear el area plot de los 5 paises que tienen menor 
inmigración a Canada de 1980 a 2013. Usa transparencia con un
valor de 0.45
### type your answer here
df_last5 = df_can.tail(5)

df_last5 = df_last5[years].transpose()
df_last5.head()

df_last5.index = df_last5.index.map(int)


df_top5.plot(kind='area', 
            alpha=0.45,
            figsize=(20,10))

plt.title('Immigration Trend of 5 with least Immigration')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')


plt.show()'''

'''EJERCICIO 2 usa la artist layer para crear un area plot
sin apilar de los 5 paises que contribuyen menos a la inmigración
a Canada del 1980 al 2013. con un valor de transparencia de 0.55


ax = df_last5.plot(kind='area', alpha=0.55, stacked=False, figsize=(20, 10))

ax.set_title('Immigration least 5 countries to Canada')
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Years')'''


# Histogramas

'''Pregunta: Cual es la distribución de frecuencias del número
de populación de nuevos inmigrantes de varios paises a canada 
en 2013'''

#Examinar los datos del dataset en el año 2013
df_can['2013'].head()

#histograma de numpy (retorna 2 valores)
count, bin_edges = np.histogram(df_can['2013'])

print(count) #conteo de frecuencia
print(bin_edges) #rango de casillas, 10 por defecto


#podemos gráficar esta función fecilmente con .plot()

df_can.plot(kind='hist', figsize=(8,5))

plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()

# Para igualar el tamaño de las casillas podemos usar xticks

count, bin_edges = np.histogram(df_can['2013'])

df_can['2013'].plot(kind='hist', figsize=(8,5), xticks=bin_edges)

plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()

#Pregunta: Cual es la distribución de inmigración para 
#Dinamarca, Noruega y Suecia para los años 1980 - 2013?

#ver rápidamente el dataset
df_can.loc[['Denmark', 'Norway', 'Sweden'], years]

# generar el histograma
df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist()

#Esto no parece correcto
#Se puede solucionar haciendo una transposición del dataset
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_t.head()

#generar un histograma
df_t.plot(kind='hist', figsize=(10,6))

plt.title('Histogram of Immigration from Denmark, Norway, Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

'''Podemos mejorar la estetica e impacto de la gráfica anterior
* aumentar el tamaño de las casillas 15 con el parametro bins
* poner la transparencia a 60% con el parametro alpha
* etiquetar el eje x con el parametro xlabel
* Cambiar el color de la gráfica con el parametro color'''

#Obtener los valores x-tick
count, bin_edges = np.histogram(df_t, 15)

#histograma sin apilar
df_t.plot(kind='hist',
        figsize=(10,6),
        bins=15,
        alpha=0.6,
        xticks=bin_edges,
        color=['coral','darkslateblue','mediumseagreen'])

plt.title('Histogram of Immigration from Denmark, Norway, Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

'''si no queremos que se apilen las gráficas entre ellas,
podemos usar el parametro stacked.  También podemos ajustar
las etiquetas de eje x y y con xlim en la tupla (xmin,xmax) '''

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10 #el primer valor de casilla es 31, añadir buffer de 10 para estetica 
xmax = bin_edges[-1] - 10 #el primer valor de casilla es 308, añadir buffer de 10 para estetica 

#histograma apilado
df_t.plot(kind='hist',
        figsize=(10,6),
        bins=15,
        xticks=bin_edges,
        color=['coral','darkslateblue','mediumseagreen'],
        stacked=True,
        xlim=(xmin,xmax)
        )

plt.title('Histogram of Immigration from Denmark, Norway, Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()


'''Gráficos de Barras (Dataframe)
Se crean con .plot(kind='bar')
* kind=bar crea el gráfico de forma vertical
* kind=barh crea el gráfico de forma horizontal

Estudiemoslo analizando el efecto de la crisis financiera de Islandia

Actividad: comparar el número de inmigrantes originarios de Islandia
a Canada desde el año 1980 al 2013
'''

#paso 1: dibujar los datos
df_iceland = df_can.loc['Iceland', years]
df_iceland.head()

#paso 2: dibujar los datos
df_iceland.plot(kind='bar',figsize=(10,6))

plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.title('Icelandic immigrants to Canada from 1980 - 2013')

plt.show()

'''podemos anotar esto en el gráfico usando el método annotate
pasaremos los siguientes parametros:
* s: str, el texto de la anotación 
* xy: Tupla especificando el punto (x, y) a anotar
* xytext: Tupla especificando el punto (x, y) para poner el texto
* xycoords: El sistema de coordenadas que es dado en que xy
es dado en 'data' usa el sistema de coordenadas del objeto que es
anotado
* arrowprops: Toma un diccionario de propiedades para dibujar 
la flecha 
    * arrowstyle: Especifica el estilo de la flecha, '->'
es la fecha estandard
    * connectionstyle: Especifica el tipo de conexión 
    * c3: es una línea recta
    * color: especifica el color de la flecha
    * lw: especifica el ancho de la línea
'''

df_iceland.plot(kind='bar', figsize=(10,6), rot=90)
# rot=90 rota los xticks en 90 grados

plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.title('Icelandic immigrants to Canada from 1980 - 2013')

# Anotar flecha
plt.annotate('',#s, str. Se deja en blanco para no mostrar texto
            xy=(32,70),# apunta la cabeza de la flecha al punto dado (year 2013, pop 70)
            xytext=(28,20),# apunta la base de la flecha al punto dado (year 2008, pop 20)
            xycoords='data',# Usara el sistema de coordenadas del objeto que es anotado
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

#Anotar Texto
plt.annotate('2008 - 2011 Financial Crisis',#Texto a mostrar
            xy=(28,30),# Empieza el texto en el punto (year 2008, pop 30)
            rotation=72.5,# Echo por prueba y error para encajar con la flecha 
            va='bottom',# alinear el texto verticalmente de 'bottom'
            ha='left'# alinear el texto horizontalmente de 'left'
            )


plt.show()

'''EJERCICIO: Usa la capa de sripting con el dataset df_can,
crea un plot horizontal bar mostrando el número total de 
inmigrantes a canada desde el top 15 de paises, para el periodo
de 1980 al 2013: Etiqueta cada país con el conteo total de 
inmigrantes'''

### type your answer here
df_can.sort_values(by='Total', ascending=True, inplace=True)

df_top15 = df_can['Total'].tail(15)
df_top15

#Paso 2: Plot data
df_top15.plot(kind='barh', figsize=(12,12), color='steelblue')

plt.xlabel('Number of immigrants')
plt.title('Top 15 Countries Contributing to the Immigration to Canada between 1980 - 2013')

for index, value in enumerate(df_top15):
    label = format(int(value), ',')

plt.annotate(label, xy=(value - 47000, index - 10.0), color='white')

plt.show()