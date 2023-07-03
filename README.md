# PI1_camila_criado : SISTEMA DE RECOMENDACION DE PELICULAS
En este Repositorio se presenta el Primer Trabajo Individual De Camila Criado, Alumna de Part-Time Data Science en Henry

El trabajo consiste en el analisis de dos data sets sobre peliculas. Incluye una limpieza y proceso de trasnsformación de datos, un análisis exploratorio
y finalmente un modelo de Machine Learning para recomendar 5 peliculas a partir de 1 pasada como parámetro. Además los datos se disponibilizan en una API. 

A continuación se brinda una breve descripción de los archivos presentes:

NOTEBOOKS:

1. ETL: Notebook en donde se realiza la limpieza de datos y las transformaciones solicitadas. Ademas se incluyen algunas transformaciones adicionales que luego
        serán útiles para el resto de los puntos.

2. EDA: Notebook en donde se realiza el Analisis Exploratorio de los datos, llegandose a la conclusión que en este caso,convendrá un Modelo asociado al contenido
        de las películas y no a los usuarios (se explica mas adelante). Cabe aclarar que este tema podria profundizarse muchisimo más, aunque no es el fin de este trabajo.
        En el presente trabajo se presentó una idea de los alcances que puede tener una exploración de datos.

3. Funciones API: Notebooks donde se desarrollaron las funciones API. Esto se hizo por prevención por si la API no llegaba a resultar, asi se puede ver que todas las funciones
                 cumplen con lo solicitado. La estructura que se les dio (return en formato diccionario) tiene que ver con la manera en se suben las funciones a la API.


4. ML: Se desarrolla el modelo aquí. Se tuvo en cuenta los analisis del EDA. Se ha hecho una investigación sobre los diferentes sistemas posibles de recomendación existentes.
       Al respecto, hay sistemas de recomendación basados en usuarios y otros en contenido. En los primeros se tiene en cuenta la popularidad, los votos, las criticas, etc.
       En los segundos se tienen en cuenta las características de las películas, tales como el género, los actores, el director, el titulo, el idioma, etc.
       Según el EDA, se llegó a la conclusión que en este caso, por la poca disponibilidad de datos en la columna popularity y vote_count convenia hacer un sistema por contenido.


   main.py En este archivo se desarrolla el codigo para crear la API.

   DataSets:
   movies: archivo original
   credits: archivo original
   etl, eda, ml: los resultantes de las diferentes notebooks.

   Requirements txt. Aquí se presentan las librerias y dependencias a instalar para poder correr todas las notebooks y archivos.


   Camila Criado


   
