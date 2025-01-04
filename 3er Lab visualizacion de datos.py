#3er Lab visualizacion de datos
import numpy as np
import pandas as pd

#Descargar el dataset de inmigración a Canada
df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name= 'Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2
)

#print('Data downloaded and read into a dataframe')

#df_can.head()
#print(df_can.shape)

# Limpieza de datos

#Remover columnas innecesarias
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

#Renombramos las columnas para que tengan sentido
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

#para más consistencia, hagamos todas las columnas de tipo string
df_can.columns = list(map(str, df_can.columns))

#ponemos el nombre del país como índice- útil para usar el método .loc
df_can.set_index('Country', inplace=True)

#asegurar que las columnas de años son de tipo numérico
years = list(map(str, range(1980, 2014)))
df_can[years] = df_can[years].apply(pd.to_numeric, errors='coerce')

#Añadir la columna de total
df_can['Total'] = df_can.loc[:,'1980':'2013'].sum(axis=1)

# Verificamos la dimensión de los datos
print('Data Dimension: ', df_can.shape)

print(df_can.head())

#Visualización de datos Matplotlib

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')

'''Gráficos de Pastel
Paso 1: juntar los datos.

usaremos el método groupby para resumir los datos de inmigración
por Continent.

El proceso general de groupby requiere:
1. Split: Dividir los datos en grupos basados en algunos criterios
2. Apply: Aplicar una función a cada grupo independientemeente:
.sum() .count() .mean() .std() .aggregate() . apply() .etc...
3. Combine: Combinar los resultados en una estructura de datos.

#Agrupar por continentes y aplicar sum()
df_continents = df_can.groupby('Continent', axis=0).sum()

# nota que la salida del método groupby es un objeto 'groupby'
#no podemos usarlo más hasta aplicar una función
print(type(df_can.groupby('Continent', axis=0)))

print(df_continents.head())'''

'''Paso 2: Dibujar los datos.
Se hará con la palabra clave kind='Pie'. junto con sus
parámetros adicionales

*autopct: es un string o función usado para etiquetar los ejes
con su valor numerico. La etiqueta será puesta dentro del eje. 
Si es un string la etiqueta será fmt%pct.
*startangle: Rota el inicio del gráfico de Pastel en angulos
al sentido contrario del reloj del eje x
*shadow: Dibuja una sombra debajo del pastel

df_continents['Total'].plot(kind='pie',
                            figsize=(5,6),
                            autopct='%1.1f%%', #añade los percentajes
                            startangle=90, #empieza en el angulo 90° (Africa)
                            shadow=True, #añade sombreado
                            )

plt.title('Immigration to Canada by Continent [1980 - 2013]')
plt.axis('equal') #pone el gráfico de pastel para que se vea como un circulo

plt.show()'''

'''La vista anterior no es muy clara, los números y texto se translapan
en algunas instancias. Mejoremos esto con:

* Quita las etiquetas de texto en legend y 
la añade como una leyenda separada usando plt.legend() 
* Empuja los porcentajes para poner afuera del gráfico de pastel
pasandolo en el parámetro pctdistance
* Pasa en un conjunto de colores para los continentes con el 
parámetro colors
* Explota el gráfico de pastel para enfatizar los 3 continentes
más bajos pasandolos por el parámetro explode

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] #ratio de cada continente con su valor de offset cada wedge

df_continents['Total'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None, #Quita las etiquetas del gráfico
                            pctdistance=1.12, #el ratio entre el centro de cada pedazo y el inicio del texto
                            #generado por autopct
                            colors=colors_list, # añadir los colores personalizados
                            explode=explode_list) # 'explota' los 3 continentes más bajos

#escala el título a 12% para igualar a pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12)
plt.axis('equal')

# añade una leyenda
plt.legend(labels=df_continents.index, loc='upper left')

plt.show()'''


'''Diagrama de Cajas
Dibujemos el diagrama de cajas para los inmigrantes Japoneses
entre 1980 - 2013.

Paso 1. Obtener el subset del dataset. Aunque estamos extrayendo 
los datos para un solo país, lo obtendremos como un dataset. Esto
nos ayudará al llamar el método dataframe.describe() para ver 
percentiles.

#Obtener el dataframe
df_japan=df_can.loc[['Japan'], years].transpose()
df_japan.head()

# Paso 2: Dibujarlo con kind='box'
df_japan.plot(kind='box', figsize=(8, 6))

plt.title('Box Plot of Japanese Immigrant')
plt.ylabel('Number of Immigrants')

plt.show()

#Podemos ver los números llamando el método describe()
print(df_japan.describe())'''

'''EJERCICIO: Compara la distribución del número de 
nuevos inmigrantes desde India y China para el periodo de
1980 - 2013

df_CI = df_can.loc[['India','China'], years].transpose()
print(df_CI.head())

print(df_CI.describe())

df_CI.plot(kind='box', figsize=(10, 7))

plt.title('Boc Plot of Immigrants from China and India (1980 - 2013)')
plt.ylabel('Number of Immigrants')

plt.show()

#Para crear diagramas de caja horizontales, puedes poner el
#parámetro vert en la función plot y asignarlo como False

#Diagrama de cajas horizontal

df_CI.plot(kind='box', figsize=(10,7), color='blue', vert=False)

plt.title('Box Plot of Immigrants from China and India (1980 - 2013)')
plt.ylabel('Number of Immigrants')

plt.show()'''

'''Subplots. Para visualizar multiples plots juntos, podemos
crear una figure y  dividirlo en subplots, cada uno conteniendo 
un plot. Cuando usamos subplots, usualmente trabajamos con la 
capa de artista en vez de la capa de scripting

El sintaxís típico es:
fig = plt.figure() --> Crear la figura
ax = fig.add_subplot(nrows, ncols, plot_number) --> crear subplots

Donde:
* nrows y ncols sin usados para dividir nocionalmente la 
figura en sub-ejes(nrows * ncols)
* plot_number: se usa para identificar el subplot particular
que se crea dentro de esta cuadricula notacional.
plot_number empieza en 1, incrementa a lo largo de las primeras
filas y tiene un máximo de nrows * ncols

Podemos especificar que subplot para poner cada plot pasandolo
en el parametro ax en el método plot()

fig = plt.figure() 

ax0 = fig.add_subplot(1, 2, 1) # añade subplot 1 (1 fila, 2 columnas, primer plot)
ax1 = fig.add_subplot(1, 2, 2) # añade subplot 2 (1 fila, 2 columnas, segundo plot)

#Subplot 1: Box plot
df_CI.plot(kind='box', color='blue', vert=False, figsize=(20,6), ax=ax0) # añadir al subplot 1
ax1.set_title('Box Plots of immigrants from China and India (1980 - 2013)')
ax1.set_xlabel('Number of Immigrants')
ax1.set_ylabel('Countries')

#Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(20,6), ax=ax1) # añadir al subplot 2
ax1.set_title('Line Plots of immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()'''

'''Tip relacionados con la convención subplot
En el caso cuando nrows, ncols y plot_number son menores de 10, 
existe una convención en la cuál un número de 3 digitos puede
ser puesto, donde los cientos representan nrows, los 
dieces representan ncols y las unidades representan
plot_number

    subplot(211) == sublpot(2, 1, 1)

produce subejes en una figura que representa el top plot
(el primero), en una cuadrilla notacional de 2 filas por
1 columna

EJERCICIO: Crea un diagrama de cajas para visualizar la 
distribución del top de 15 paises, agrupados por las decadas
1980´s, 1990´s y 2000´s 

Paso 1: Obtener el top 15

df_top15 = df_can.sort_values(['Total'], ascending=False, axis=0).head(15)
df_top15'''

'''Paso 2: Crear un nuevo dataframe que contenga el agregado
para cada decada.
1. Crear una lista de años de las decadas 80s, 90s, 00s.
2. Cortar el df original para crear una serie para cada decada
y sumarla a lo largo de los años
3. Unir las 3 series en un nuevo dataframe

#crear una lista de todos los años
years_80s = list(map(str, range(1980, 1990)))
years_90s = list(map(str, range(1990, 2000)))
years_00s = list(map(str, range(2000, 2010)))

#cortar el df original para crear una serie para cada decada
df_80s = df_top15.loc[:, years_80s].sum(axis=1)
df_90s = df_top15.loc[:, years_90s].sum(axis=1)
df_00s = df_top15.loc[:, years_00s].sum(axis=1)

#unir las 3 series en un nuevo df
new_df = pd.DataFrame({'1980s':df_80s, '1990s':df_90s, '2000s':df_00s})
print(new_df.head())
print(new_df.describe())

#Paso 3: Dibujar el box plot
new_df.plot(kind='box', figsize=(10,6))
plt.title('Immigration from top 15 countries for decades 80s, 90s, 2000s')

plt.show()

#Chequemos cuantos datos caen en el outlier
new_df=new_df.reset_index()
new_df[new_df['2000s']> 209611.5]'''

'''Scatter Plots 
Usando un scatter plot, visualizemos la tendencia de la 
inmigración total a Canada para los años 1980 - 2013

Paso 1: Obtener el dataset, dado a que queremos la relación 
entre años y total population, convertiremos years al tipo int


#Usamos el método sum() para obtener el total de la población
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
#Cambiar los años a tipo int
df_tot.index = map(int, df_tot.index)
#reiniciar el índice para ponerlo de regreso en una columna
df_tot.reset_index(inplace=True)
#renombrar las columnas
df_tot.columns=['year', 'total']
#Ver el dataframe final
print(df_tot.head())'''

'''Paso 2: Dibujar los datos
En matplotlib podemos crear un diagrama de dispersión 
pasando kind='scatter' como argumenta de plot también 
necesitamos pasar las palabras claves x y y para especificar
las columnas que van en el eje x y y

df_tot.plot(kind='scatter', x='year', y='total', figsize=(10,6), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

plt.show()'''

'''Mira como el diagrama de dispersión no encaja.
Intentemos un plot lineal para hacer una mejor gráfica

Paso 1: Obtener la ecuación de linea. Utilizemos el método 
polyfit() de Numpy pasandole:
* x: coordenadas x de los datos
* y: coordenadas y de los datos
* deg: Grado de ajuste polinomial. 1 =lineal, 2 = cuadratico
y así

x = df_tot['year']
y = df_tot['total']
fit=np.polyfit(x,y, deg=1)

fit

#Paso 2: Dibujar la regresión lineal o diagrama de dispersión
df_tot.plot(kind='scatter', x='year', y='total',figsize=(10,6), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

#Dibujar la línea que ajuste mejor 

plt.plot(x, fit[0] * + fit[1], color='red')
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.show()'''

#imprimir la linea que encaje mejor
#'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1])

'''EJERCICIO: Crear un diagrama de dispersión del total de inmigración
de Dinamarca, Noruega y Suecia a Canada desde 1980 al 2013

Paso 1: Obtener los datos
* Crear el dataframe que consiste en los números asociados
con los paises
* Sumar los numeros de inmigración para hacer un total
* Reiniciar el índice a su lugar
* Renombrar las columnas a year y total
* Mostrar el dataframe resultante


df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))

df_total.reset_index(inplace=True)

df_total.columns = ['year','total']

df_total['year'] = df_total['year'].astype(int)

df_total.head()

#Paso 2: Generar el diagrama de dispersión con total vs year

df_total.plot(kind='scatter', x='year', y='total', figsize=(10,6), color='darkblue')
plt.title('Immigration from Denmark, Norway and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

plt.show()'''

'''Bubble Plots: Un diagrama de burbuja es una variación de un
diagrama de dispersión que muestra 3 dimensiones de datos (x, y, z)
Los puntos son remplazados con burbujas, y el tamaño de las burbujas 
son determinados por la variable z, también conocida como el peso.
En matplotlib podemos pasar el parametro s o plot en un arreglo o 
un escalar, que contenga el peso de cada punto.

Analicemos el efecto de la gran depresión de Argentina.

Analisemos el efecto de la crisis y comparemos la Inmigración 
de Argentina con la de Brasil con un bubble plot para los años 
de 1980 - 2013 

Paso 1: Obtener los datos para Brasil y Agrentina y convertir
years a int e incluirlo en el dataframe'''

#df transpuesto
df_can_t = df_can[years].transpose()

#pasar los años (el índice)a tipo int 
df_can_t.index = map(int, df_can_t.index)

#etiquetar el índice
df_can_t.index.name = 'Year'

#reiniciar el índice para poner el Year como una columna
df_can_t.reset_index(inplace=True)

#Ver los cambios
(df_can_t.head())

'''Paso 2: Crear los pesos normalizados.
Hay muchos métodos de normalización en estadística, cada uno
con su propio uso, usaremos feature scaling para poner todos los
valores dentro del rango [0,1]. La formula general es:

Z' = X-Xmin/Xmax-Xmin

Donde X es el valor original, X' es el valor 
normalizado correspondiente. La formula pone el valor máximo 
en el dataset a 1 y pone el valor mínimo a 0. El resto de los 
puntos son escalados entre 0-1'''

#datos normalizados de Brasil
norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())

#datos normalizados de Argentina
norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())

'''Paso 3: Dibujar los datos
* Para dibujar los dos diagramas de dispersión en un solo diagrama,
podemos incluir los ejes dentro de otro pasandolos por el parametro
ax

* También pasaremos los pesos usan el parametro s.
Dado que los pesos normalizados son entre 0 y 1, 
estos no serán visibles en el diagrama. 
Para esto haremos lo siguiente:
 * multiplicar los pesos por 2000 para escalarlos en la gráfica

 * añadir 10 para compensar para el valor mínimo (que tiene un
peso de 0 y después escalamos a x2000) '''

#Brasil
ax0 = df_can_t.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )
#Argentina
ax1 = df_can_t.plot(kind='scatter', 
                    x='Year', 
                    y='Argentina',
                    alpha=0.5,
                    color='blue',
                    s=norm_argentina * 2000 + 10,
                    ax=ax0)

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')
