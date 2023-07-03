from fastapi import FastAPI
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#creamos la aplicacion 
#http://127.0.0.1:8000 
app = FastAPI(title='Proyecto Individual Camila Criado')

df = pd.read_csv('PI/Dataset/df_etl.csv')
df_ml = pd.read_csv('PI/Dataset/df_eda.csv')


@app.get('/')    #CREO LA RUTA PRINCIPAL
def read_root():
    return {'Hello': 'World'}

# CONSULTA 1 CANTIDAD DE PELICULAS POR MES
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes):

    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

    if mes.lower() in meses:
        mask = df.release_month == mes.lower().capitalize()
        cantidad_peliculas = df.loc[mask].shape[0]


     # Crear el diccionario de resultados
        resultado = {
        'Mes de estreno': mes.capitalize(),
        'Cantidad de películas estrenadas': cantidad_peliculas
        }

        return resultado
    else:
        resultado = {
        'ERROR': 'MES INCORRECTO',
        
        }
        return resultado
    
    # CONSULTA 2
    
@app.get('/cantidad_filmaciones_dia{dia}')    
def cantidad_filmaciones_dia(dia:str):
   
     '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia historicamente'''
     
     respuesta = 0
     dia = dia.lower().capitalize()
     if dia in df.release_day.unique():
        mask_cantidad_filmaciones_dia = df['release_day'] == dia
        respuesta = df[mask_cantidad_filmaciones_dia].shape[0]
        return {'Peliculas estrenadas un':dia, 'Cantidad':respuesta}
     else:
        return {'ERROR EN PALABRA INGRESADA ':dia, 'Dia Ingresado':'No es correcto'}
    
    
    # CONSULTA 3
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''
    
    respuesta2 = 0
    respuesta3 = 0
    mask_titulo = 0
    
    if df['title'].str.lower().str.contains(titulo.lower()).any():
        mask_titulo = df['title'].str.lower() == titulo.lower()
        respuesta2 = df.loc[mask_titulo, 'release_year'].values[0]
        respuesta3 = df.loc[mask_titulo, 'popularity'].values[0]
        return {'Pelicula':titulo.title(), 'Se estrenó el año':respuesta2, 'La misma tiene Score':respuesta3}
    else:
        return {'Pelicula':titulo.title(), 'ERROR':'NO SE ENCONTRO'}
    
    
#CONSULTA 4
    
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
    La misma variable deberá de contar con al menos 2000 valoraciones, 
    caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''
    
    mask_titulo = df['title'].str.lower() == titulo.lower()
    
    if df.loc[mask_titulo, 'vote_count'].values[0] < 2000:
        return {'ERROR':'No se cumplen los requisitos'}
    else:
        return {'titulo':titulo, 'anio':df.loc[mask_titulo, 'release_year'].values[0], 'voto_total':df.loc[mask_titulo, 'vote_count'].values[0], 'voto_promedio':df.loc[mask_titulo, 'vote_average'].values[0]}
    
 
 # CONSULTA 5   
@app.get('/get_actor/{nombre_actor}')
def get_actor(actor_name):
    global df  # Declarar el DataFrame como una variable global
    count = 0  # Variable para almacenar el contador de películas
    revenue_acumulado = 0  # Variable para almacenar la suma de revenue
    
    for actors, revenue in zip(df['Actors'], df['revenue']):
        if isinstance(actors, str):
            try:
                actors_list = ast.literal_eval(actors)
                if any(actor_name.lower() in actor.lower() for actor in actors_list):
                    count += 1
                    if pd.notnull(revenue):
                        revenue_acumulado += revenue
            except ValueError:
                pass
    
    revenue_promedio = revenue_acumulado / count if count > 0 else 0
    
    actor_data = {
        'Actor': actor_name,
        'Películas': count,
        'Revenue Total': revenue_acumulado,
        'Revenue Promedio': revenue_promedio
    }
    
    return actor_data

# CONSULTA 6

@app.get('/get_director/{nombre_director}')
def get_director(director_name):
    global df
    
    # Filtrar el dataframe por el director especificado
    director_movies = df[df['Director'] == director_name]
    
    if director_movies.empty:
        print(f"No se encontraron películas para el director {director_name}.")
        return None
    
    # Calcular el revenue acumulado
    total_revenue = director_movies['revenue'].sum()
    
    # Crear una lista para almacenar la información de las películas del director
    movies_info = []
    
    # Iterar sobre cada película del director
    for index, row in director_movies.iterrows():
        movie_title = row['title']
        release_date = row['release_date']
        revenue = row['revenue']
        budget = row['budget']
        profit = revenue - budget
        
        # Verificar si la fecha de lanzamiento es una cadena de texto
        if isinstance(release_date, str):
            release_date_str = release_date
        else:
            release_date_str = release_date.strftime('%Y-%m-%d') if not pd.isnull(release_date) else 'No especificado'
        
        movie_info = {
            'Película': movie_title,
            'Fecha de lanzamiento': release_date_str,
            'Revenue': revenue if revenue != 0 else 'No especificado',
            'Presupuesto': budget if budget != 0 else 'No especificado',
            'Ganancia': profit if revenue != 0 and budget != 0 else 'No especificado'
        }
        
        movies_info.append(movie_info)
    
    # Crear el diccionario de resultado
    director_data = {
        'Director': director_name,
        'Películas': movies_info,
        'Revenue acumulado': total_revenue if total_revenue != 0 else 'No especificado'
    }
    
    return director_data


# MODELO DE MACHINE LEARNING
@app.get('/recomendacion/{titulo}')
def content_based_recommender(movie_title):
    # Crear un objeto TfidfVectorizer para convertir el texto en vectores
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(' '), # Personalizamos el tokenizer para que no separe las palabras del género
        stop_words='english', # Agregamos stop words para eliminar palabras comunes que no aportan mucho significado
        use_idf=True, # Utilizamos la frecuencia inversa del documento (IDF) para dar más peso a las palabras poco frecuentes
        smooth_idf=True, # Evitamos dividir por cero al suavizar el IDF
        norm=None, # No normalizamos los vectores para poder ajustar manualmente la relevancia de 'genre_name'
        sublinear_tf=True, # Aplicamos sublineal TF para suavizar la frecuencia de términos en el texto
        min_df=0.001 # Filtro de términos que aparecen en menos del 0.1% de los documentos
    )

    # Obtener las columnas relevantes del DataFrame
    df_relevant = df_ml[['title', 'genre_name', 'Director', 'first_actor', 'original_language']]

    # Preprocesar las columnas de texto para su uso en el modelo
    df_relevant['genre_name'] = df_relevant['genre_name'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df_relevant['text'] = df_relevant['title'] + ' ' + df_relevant['genre_name'].apply(lambda x: ' '.join(x)) + ' ' + df_relevant['Director'] + ' ' + df_relevant['first_actor'] + ' ' + df_relevant['original_language']

    # Calcular la matriz de vectores TF-IDF
    tfidf_matrix = vectorizer.fit_transform(df_relevant['text'])

    # Calcular la similitud coseno entre las películas
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Obtener el índice de la película de entrada si existe
    if movie_title in df_relevant['title'].values:
        idx = df_relevant[df_relevant['title'] == movie_title].index[0]
    else:
        print("La película no se encuentra en el conjunto de datos.")
        return None

    # Obtener las puntuaciones de similitud coseno para todas las películas
    scores = list(enumerate(cosine_similarities[idx]))

    # Ordenar las películas por puntuación de similitud
    scores = sorted(scores, key=lambda x: (x[1], x[0]), reverse=True)

    # Obtener las 5 películas recomendadas
    top_recommendations = scores[1:6]  # Excluir la película de entrada

    # Crear un diccionario con los datos de las películas recomendadas
    recommendations = []
    for idx, score in top_recommendations:
        title = df_relevant.iloc[idx]['title']
        genre = df_relevant.iloc[idx]['genre_name']
        director = df_relevant.iloc[idx]['Director']
        actor = df_relevant.iloc[idx]['first_actor']
        language = df_relevant.iloc[idx]['original_language']
        recommendation = {
            'Title': title,
            'Genre': genre,
            'Director': director,
            'Actor': actor,
            'Language': language
        }
        recommendations.append(recommendation)

    # Obtener los datos de la película de entrada
    movie_data = df_relevant[df_relevant['title'] == movie_title].iloc[0]
    movie_info = {
        'Title': movie_data['title'],
        'Genre': movie_data['genre_name'],
        'Director': movie_data['Director'],
        'Actor': movie_data['first_actor'],
        'Language': movie_data['original_language']
    }

    # Imprimir los datos de la película de entrada
    print("Película:", movie_title)
    print("Título:", movie_info['Title'])
    print("Género:", movie_info['Genre'])
    print("Director:", movie_info['Director'])
    print("Actor:", movie_info['Actor'])
    print("Idioma:", movie_info['Language'])

    # Imprimir las películas recomendadas
    print("\nRecomendaciones para la película:", movie_title)
    for i, recommendation in enumerate(recommendations):
        print(f"{i+1}. {recommendation['Title']} - Género: {recommendation['Genre']}, Director: {recommendation['Director']}, Actor: {recommendation['Actor']}, Idioma: {recommendation['Language']}")

    return recommendations