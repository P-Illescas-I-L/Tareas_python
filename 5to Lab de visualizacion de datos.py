#5to Lab de visualizacion de datos
import numpy as np
import pandas as pd
import folium
import webbrowser

'''#Definir el mapa
world_map = folium.Map()

#mostrar el mapa
world_map

Usar folium es muy diferente en servicios como Jupyter
en vez de VS Code, en VS Code necesitas hacer el mapa como 
un archivo html, mientras que en Jupyter se hace como 
en el código de arriba'''

#Definir el mapa centrado alrededor de canada

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

#mostrar el mapa
map_path = "world_map.html"
world_map.save(map_path)
#Abrir el archivo en el navegador
webbrowser.open(map_path)

#Mapa de Mexico
mexico_latitude = 23.6345
mexico_longitude = -102.5528 

mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4)

#mostrar el mapa
mexico_path = "mexico_map.html"
world_map.save(mexico_path)
#Abrir el archivo en el navegador
webbrowser.open(mexico_path)



#Mapas Stamen Toner
world_map_1 = folium.Map(location=[56.130, -106.35], 
                        zoom_start=4, 
                        tiles="Stamen Toner",
                        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
)

#mostrar el mapa
map_path_1 = "world_map_1.html"
world_map.save(map_path_1)
#Abrir el archivo en el navegador
webbrowser.open(map_path_1)

#Mapas de Stamen Terrain
world_map_2 = folium.Map(location=[56.130, -106.35], 
                        zoom_start=4, 
                        tiles='https://{s}.tile.stamen.com/terrain/{z}/{x}/{y}.png',  # URL para Stamen Terrain
    attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
                        )

#mostrar el mapa
map_path_2 = "world_map_2.html"
world_map.save(map_path_2)
#Abrir el archivo en el navegador
webbrowser.open(map_path_2)

#Mapas con marcadores
#Descargar el dataset y leerlo dentro de un dataframe de pandas
df_incidents = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv')

print('Dataset descargado')

print(df_incidents.head())
print(df_incidents.shape)

#Obtener los primeros 100 crimenesen el df
limit=100
df_incidents = df_incidents.iloc[0:limit, :]

print(df_incidents.shape)

#valores de san francisco de longitud y latitud
latitude = 37.77
longitude = -122.42

#crear el mapa y mostrarlo
sanfran_map = folium.Map(location=[latitude,longitude], zoom_start=12)

#mostrar el mapa
sanfran_map

'''Ahora agreguemos las locaciones de crimenes en el mapa.
Esto se hace en Folium mediante crear el grupo de características 
"feature group" con sus propias caracteerísticas y estilos
para después agregar el sanfran_map'''

#Instanciar un feature group para los incidentes en el df
incidents = folium.map.FeatureGroup()

#Hacer loop a través de los 100 crimenes y añadirlos al
#incident feature group
for lat, lng in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

#Añadir los incidentes al mapa
sanfran_map.add_child(incidents)

'''También puedes añadir algunos textos pop-up que será mostrado
cuando hagas hover (poner el cursor encima) del marcador.
Hagamos esto a continuación:'''


#Instanciar un feature group para los incidentes en el df
incidents = folium.map.FeatureGroup()

#Hacer loop a través de los 100 crimenes y añadirlos al
#incident feature group
for lat, lng in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

#añadir el texto pop-up para cada marcador del mapa
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)

#Añadir los incidentes al mapa
sanfran_map.add_child(incidents)

'''Como puedes ver el mapa esta algo congestionado con 
todos esos marcadores, hay 2 maneras dde solucionar esto.
Lo más simple es quitar estos marcadores de localización 
y solo agregar el texto a los marcadores circulares 
como se ve a continuación:'''

#Crear el mapa y mostrarlo
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

#Hacer loop a través de los 100 crimenes y añadirlos al
#mapa
for lat, lng, label in zip(df_incidents.Y,df_incidents.X,df_incidents.Category):
    folium.features.CircleMarker(
        [lat, lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(sanfran_map)

# mostrar el mapa
sanfran_map

'''La otra solución es agrupar los marcadores en diferentes
clusters. Cada cluster es representado por el número de crimenes
en cada vecindario. '''

from folium import plugins 

#Empecemos desde una copia limpia del mapa de SF
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

#instanciar un objeto cluster de marca para los incidentes en el dataframe
incidents = plugins.MarkerCluster().add_to(sanfran_map)

#Hacer loop a través del dataframe y cada punto de dato al mark cluster
for lat, lng, label in zip(df_incidents.Y,df_incidents.X,df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
        ).add_to(incidents)

# mostrar el mapa
sanfran_map

'''Mapas Choropleth

creemos nuestro mapa Choropleth describiendo la inmigración
de varios paises a Canada'''

df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

print('Data downloaded and read into a dataframe!')

df_can.head
print(df_can.shape)

#Limpiar los datos
#remover columnas innecesarias
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

#renombremos las columnas
df_can.rename(columns={'OdName':'Country','AreaName':'Continent','RegName':'Region'}, inplace=True)

#cambiar el tipo a string
df_can.columns = list(map(str, df_can.columns))

#Añadir la columna 'total'
df_can['Total'] = df_can.sum(axis=1)

#Años que ocuparemos en este Lab
years = list(map(str, range(1980, 2014)))
print('data dimensions: ', df_can.shape)

#verificar los datos
df_can.head()

'''Para crear el mapa choropleth necesitamos el arcchivo 
geoJSON que define las áreas/límites del estado o país
en el que estamos interesados. 
Desarguemos los datos:'''

#archivo geojson
#! wget --quiet https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json
datosjson = pd.read_json('world_countries.json')

datosjson.head()

#Crear el mapa mundial, centrado en [0,0]
#world_geo = r'world_countries.json'
world_geo = datosjson

world_map = folium.Map(location=[0,0], zoom_start=2)

'''Para crear el mapa choropleth se usan los siguientes 
parametros
1. geo_data: Archivo GeoJSON
2. data: El dataframe que contiene los datos
3. columns: las columnas del df
4. key_on: es la llave/variable en el archivo
GeoJSON que contiene el nombre de la variable de interés'''

world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)

#mostrar el mapa
world_map

'''Mira como la leyenda muestra un límite negativo o 
treshold, Arreglemoslo definiendo nuestros propios tresholds
y empezando con 0 en vez de -6,918'''

world_geo=r'world_countries.json'

#crear el arreglo numpy de longitud 6 y con espaciado lineal
treshold_scale = np.linspace(df_can['Total'].min(),
                            df_can['Total'].max(),
                            6, dtype=int)
treshold_scale = treshold_scale.tolist() #cambiar el arreglo numpy a lista
treshold_scale[-1] = treshold_scale[-1] + 1 #asegurarse que el último valor de la lista es mayor al maximo de inmigración

#dejar que folium determine la escala
world_map = folium.Map(location=[0,0], zoom_start=2)
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map


