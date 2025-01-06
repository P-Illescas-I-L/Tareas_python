#4to Lab de Visualizacion
import numpy as np
import pandas as pd
from PIL import Image
#Exploración de datos con pandas y Matplotlib

#Descargar el dataset
df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',    
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

#print('Data downloaded and read into a dataframe!')

#df_can.head()
#df_can.shape

#Limpieza de datos 

#remover las columnas innescesarias
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)

#Renombrar las columnas para que haga más sentido
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

#para consistencia hacer las etiquetas de columnas de tipo string
df_can.columns=list(map(str, df_can.columns))

#poner el nombre del país como índice
df_can.set_index('Country', inplace=True)

#Asegurar que las columnas sean de tipo númerico
years = list(map(str, range(1980, 2014)))
df_can[years] = df_can[years].apply(pd.to_numeric, errors='coerce')
#añadir la columna de total
df_can['Total'] = df_can.loc[:, '1980':'2013'].sum(axis=1)

#Verificar la dimensión de datos
print('data dimensions: ', df_can.shape)

#Visualización
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #para diagramas de waffles

mpl.style.use('ggplot')

'''Diagrama de Waffles
Veamos el caso de estudio acerca de Dinamarca, Noruega Y Suiza

Creemos el dataframe para esos países'''
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]

print(df_dsn)

'''Desafortunadamente, a diferencia de R los diagramas de waffles no se incluyen 
en librerias de Python

Paso 1: Determinar la proporción de cada categoría con respecto
al total

Computar la proporción de cada categoría con respecto al total'''

total_values = df_dsn['Total'].sum()
category_proportions =df_dsn['Total'] / total_values

#mostrar las proporciones
pd.DataFrame({"Category Proportion": category_proportions})

#Paso 2: Definir el tamaño general del diagrama de waffle
width = 40
height = 10

total_num_tiles = width * height

print(f'Total number of tiles is {total_num_tiles}.')

#Paso 3: Usar la proporción de cada categoría para determinar su respectivo 
#número de tiles
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)

pd.DataFrame({"Number of tiles": tiles_per_category})

'''Basado en las proporciones calculadas. Dinamarc ocupará
129 pedazos del diagrama de waffle, Noruega ocupará 77 y 
Suezia ocupará 194

Paso 4: Crear la matriz de resemnblanza del diagrama de waffle
y poblarla'''

#inicializar el diagrama como una matriz vacía
waffle_chart = np.zeros((height, width), dtype = np.uint)

#definir los índices con un loop a través del diagrama
category_index = 0
tile_index = 0

#Poblar el diagrama
for col in range(width):
    for row in range(height):
        tile_index += 1

        #si el número de casillas poblada para la categoría actual es igual a su correspondiente casilla
        if tile_index > sum(tiles_per_category[0:category_index]):
            #proceder a la siguiente categoría
            category_index += 1

        #poner el valor de clase a integer, que incremente con class
        waffle_chart[row,col] = category_index

print('Waffle chart populated!')

#veamos como se va viendo la matriz
waffle_chart

'''Como se esperaba, la matriz consiste de tres categorías
y el número total de cada instancias de categoría iguala el
número de casillas alojadas a cada categoría

Paso 5: Mapear el diagrama de waffle en una visualización

instanciar un nuevo objeto new figure'''
fig = plt.figure()

#usar matshow para mostrar el diagrama de waffle
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.show()

#Paso 6: Mejorar el diagrama
#instanciar un nuevo objeto new figure
fig = plt.figure()

#usar matshow para mostrar el diagrama de waffle
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

#obtener los ejes
ax = plt.gca()

#poner ticks menores
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (width), 1), minor=True)

#añadir las lineas de la cuadrilla basado en ticks menores
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
plt.show()

#Paso 7: Crear una leyenda y agregarla al diagrama
#instanciar un nuevo objeto new figure
fig = plt.figure()

#usar matshow para mostrar el diagrama de waffle
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

#obtener los ejes
ax = plt.gca()

#poner ticks menores
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (width), 1), minor=True)

#añadir las lineas de la cuadrilla basado en ticks menores
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

#Computar la suma acumulativa de categorías individuales para 
#igualar los esquemas de color entre diagrama y leyenda
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

#Crear la leyenda
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

#Añadir la leyenda al chart
plt.legend(handles=legend_handles,
        loc='lower center',
        ncol=len(df_dsn.index.values),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
        )
plt.show()

'''En resumen aquí están los pasos para hacer el diagrama
1. categories: categorías o clases unicas en un dataframe 
2. values: Valores correspondientes a categorías o clases
3. height: Altura definida del diagrama de waffle
4. width: Anchura definida del diagrama de waffle
5. colormap: clase Colormap
6. value_sign: En orden para hacer nuestra función más generalizable,
agregaremos este parametro para apuntar a los signos que pueden ser
asociados con un valor así como % $, y así. value_sign tiene un valor
por defecto de un string vacío '''

def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):
    #computar las proporciones de cada categoria con respecto al total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    #computar el numero total de casillas
    total_num_tiles = width * height # número total de casillas
    print('Total number of tiles is', total_num_tiles)

    #computar el número de casillas para cada categoría
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    #mostrar el número de casillas por categoría
    for i, tiles in enumerate(tiles_per_category):
        print(df_dsn.index.values[i] + ': ' + str(tiles))
    
    #inicializa el diagrama de waffles como una matriz vacía
    waffle_chart = np.zeros((height, width))

    #definir los índices a loopear a través del diagrama de waffles
    category_index = 0
    tile_index = 0

    #poblar el diagrama
    for col in range(width):
        for row in range(height):
            tile_index += 1

            #Sí el número de casillas ocupadas para la categoría actual
            #es igual a su correspondiente casilla 
            if tile_index > sum(tiles_per_category[0:category_index]):
                # proceder a la siguiente categoría
                category_index += 1
            
            #poner el valor de clase a integer, que incremente con clase
            waffle_chart[row, col] = category_index

    #instanciar un objeto new figure   
    fig = plt.figure()

    #usar matshow para mostrar el diagrama
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    #obtener el eje
    ax = plt.gca()

    #poner los ticks menores
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    #añadir las lineas de la cuadrilla
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    #computar la suma acumulativa de categorias individuales para iguakar los esquemas de color entre chart y leyenda
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    #Crear la leyenda
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str =  category + ' (' + value_sign + str(values[i]) + ')'

        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

        #añadir la leyenda al diagrama
        plt.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=len(categories),
            bbox_to_anchor=(0., -0.2, 0.95, .1)
        )
        plt.show()

'''Ahora para crear el diagrama de waffle, todo lo que tenemos que hacer
es llamar a la función create_waffle_chart.'''

width = 40
height = 40

categories = df_dsn.index.values #categorias
values = df_dsn['Total'] #valores correspondientes de las categorias

colormap = plt.cm.coolwarm #clase color map

#llamemos a la función para crear el diagrama de waffle

create_waffle_chart(categories, values, height, width, colormap)

'''Existe un paquete de python para generar los diagramas de waffle llamado pyWaffle
pero parece un repositorio que sigue desarrollandose'''

#Word Clouds
#importarlo junto a su set de stopwords
from wordcloud import WordCloud, STOPWORDS

#Lo analizaremos con una novela de Lewis Carroll
import urllib

#abrir el archivo y leerlo en una variable
alice_novel = urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt').read().decode("utf-8")

#usemos la función set para remover palabras redundantes
stopwords = set(STOPWORDS)

#Crear un objeto word cloud y generarlo
#instanciar un objeto word cloud
alice_wc = WordCloud(
    background_color = 'white',
    max_words=2000,
    stopwords=stopwords
)

#generar el word cloud
alice_wc.generate(alice_novel)

#Ahora que esta creado el word cloud, visualizemoslo
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#Reajustemos la nube para que podamos ver mejor las palabras menos frecuentes
fig = plt.figure(figsize=(14,18))

#mostrar la nube
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#vemos que said es una palabra no muy informativa. Así que hay
#que añadirla a los stopwords y re-generar la nube
stopwords.add('said') #añadir la palabra al stopword

#re-generar el word cloud
alice_wc.generate(alice_novel)

#mostrar la nube
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

'''Otra cosa que se puede hacer con el paquete word_cloud
es superponer las palabras en una máscara de cualquier forma.
Usemos una mascara de Alicia y su conejo con una imagen .png 
brindada por el curso'''

alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))

#Veamos como se ve la mascara
fig = plt.figure(figsize=(14,18))

plt.imshow(alice_wc, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()

#Formar el word cloud de acuerdo a la máscara es muy directo usand0
#el paquete word_cloud. 

#instanciar el objeto
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

#generar la nube
alice_wc.generate(alice_novel)

#mostrar la nube
fig = plt.figure(figsize=(14,18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis=('off')
plt.show()

'''Desafortunadamente el set de inmigración no tiene datos en texto, 
aun así hagamoslo

recordemos como se ven los datos'''
df_can.head()

#y cual era el total de inmigración desde 1980 a 2013?
total_immigration=df_can['Total'].sum()
total_immigration

#Usaremos los paises con una sola palabra como nombres, dupliquemos
#cada nombre de país basada en cuanto contribuye al total de inmigración

max_words = 90
word_string = ''
for country in df_can.index.values:
    #Checar si el nombre del país es de una sola palabra
    if country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

#mostrar el texto generado
word_string

#no hay necesidad de tener stopwords por lo que no hay necesidad de
#pasarlos cuando se crea un word cloud

wordcloud = WordCloud(background_color='white').generate(word_string)

print('Word cloud created!')

#mostrar la nube
plt.figure(figsize=(14,18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis=('off')
plt.show()

#Diagramas de regresión
import seaborn as sns

#Creemos un nuevo dataframe que almacene un total de inmigrantes 
#en canada desde el 1980 al 2013.

#podemos usar el método sum() para obtener el total de población
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

#Cambiar los años a tipo float
df_tot.index=map(float, df_tot.index)

#reiniciar el indice para ponerlo como una columna en el df fd_tot
df_tot.reset_index(inplace=True)

#renombrar las columnas
df_tot.columns=['year', 'total']

#ver el dataframe final
df_tot.head()

#con seaborn, generar un diagrama de regresión es tan simple como llamar 
#a la función regplot
sns.regplot(x='year', y='total', data=df_tot)

#puedes cambiar el color de la siguiente manera
sns.regplot(x='year', y='total', data=df_tot, color='green')
plt.show()

#puedes personalizar la forma del marcador de la siguiente manera
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()

#Hagamos la gráfica mas agradable a la vista con:
plt.figure(figsize=(15,10))
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()

#incrementaremos el tamaño de los marcadores para que encajen con el tamañp de la figura
plt.figure(figsize=(15,10))
ax=sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s':200})

ax.set(xlabel='year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

#finalmente incrementamos el tamaño de fuente de las etiquetas tickmark,
#el titulo y las etiquetas x y y
plt.figure(figsize=(15,10))

sns.set_theme(font_scale=1.5)

ax=sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s':200})
ax.set(xlabel='year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

#Puedes cambiar el estilo del background facilmente 

plt.figure(figsize=(15,10))

sns.set_theme(font_scale=1.5)
sns.set_style('ticks') # cambiar el background a blanco
#sns.set_style('whitegrid') # cambiar el background a blanco con lineas de cuadrilla

ax=sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s':200})
ax.set(xlabel='year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()








