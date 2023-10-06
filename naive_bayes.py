import pandas as pd
import numpy as np
import json
from scipy.stats import norm
import csv
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos_entrenamiento(nombre_archivo):
    df = pd.read_csv(nombre_archivo)
    return df

def cargar_datos_prueba(nombre_archivo):
    data = []
    with open(nombre_archivo, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Crear un nuevo diccionario sin la última columna
            new_row = {key: value for key, value in row.items() if key != 'Iris'}
            data.append(new_row)

    return data

def calcular_tabla_de_frecuencia_categorico(df):
    columnas_a_contar = df.select_dtypes(include=['object']).columns
    resultados = {}

    for i, columna in enumerate(columnas_a_contar):
        conteo = pd.crosstab(index=df[columna], columns=df['Iris'], colnames=['Iris'], rownames=[columna])

        if i < len(columnas_a_contar) - 1:
            conteo += 1

        resultados[columna] = conteo
    print('\nFRECUENCIAS - ATRIBUTOS CATEGORICOS:\n')
    for columna, conteo in resultados.items():
        print(f"Frecuencia para la columna '{columna}':")
        conteo_dict = conteo.to_dict()
        conteo_str = json.dumps(conteo_dict, separators=(', \n', ': ')).encode('utf-8').decode('unicode_escape')
        print(conteo_str)
        print('\n')

    return resultados

def calcular_verosimilitud_categorico(conteo_resultados, especies, total_clase):
    verosimilitud = {}

    for columna, conteo in conteo_resultados.items():
        verosimilitud_columna = {}

        for especie in especies:
            conteo_especie = conteo[[especie]]

            if columna == list(conteo_resultados.keys())[-1]:
                verosimilitud_columna[especie] = conteo_especie / total_clase
            else:
                total_apariciones = conteo_especie.sum().sum()
                verosimilitud_columna[especie] = conteo_especie / total_apariciones

        verosimilitud[columna] = verosimilitud_columna
    
    verosimilitud_c = {}

    print('VEROSIMILITUDES - ATRIBUTOS CATEGORICOS:\n')

    for columna, verosimilitud_columna in verosimilitud.items():
        print(f"Verosimilitud para la columna '{columna}':")
        
        # Crea un diccionario interno para esta columna
        columna_dict = {especie: conteo.to_dict() for especie, conteo in verosimilitud_columna.items()}
        
        # Agrega el diccionario de la columna al diccionario principal
        verosimilitud_c[columna] = columna_dict

        print(json.dumps(columna_dict, indent=4, ensure_ascii=False)) 
        print('\n')

    return verosimilitud_c

def contar_apariciones_ultima_columna(conteo_resultados):
    conteo_ultima_columna = conteo_resultados[list(conteo_resultados.keys())[-1]]
    total_apariciones = conteo_ultima_columna.sum().sum()
    return total_apariciones

def calcular_tabla_de_frecuencia_numerico(df):
    valores_por_clase = {}

    clases = df["Iris"].unique()
    for clase in clases:
        clase_df = df[df["Iris"] == clase]
        atributos_numericos = clase_df[["Sepal length", "Petal length"]].values.tolist()
        nombres_atributos = ["Sepal length", "Petal length"]
        valores_por_clase[clase] = {"atributos": nombres_atributos, "valores": atributos_numericos}

    return valores_por_clase

def calcular_verosimilitud_numerico(valores_por_clase):
    resultados = {}
    verosimilitud_n= {}

    for clase, valores_dict in valores_por_clase.items():
        nombres_atributos = valores_dict["atributos"]
        valores = valores_dict["valores"]
        arreglo_valores = np.array(valores)
        media = np.mean(arreglo_valores, axis=0)
        desviacion_estandar = np.std(arreglo_valores, axis=0)

        resultados[clase] = {
            "Atributos": nombres_atributos,
            "Media": media.tolist(),
            "Desviación Estándar": desviacion_estandar.tolist()
        }

    for clase, datos in resultados.items():
        
        media_num = {
            'Sepal length': datos['Media'][0],
            'Petal length': datos['Media'][1]
        }
        
        desviacion_num = {
            'Sepal length': datos['Desviación Estándar'][0],
            'Petal length': datos['Desviación Estándar'][1]
        }
        
        # Almacena las impresiones en el diccionario bajo la clave correspondiente (clase)
        verosimilitud_n[clase] = {
            "media": media_num,
            "desviacion_estandar": desviacion_num
        }
    
    print('VEROSIMILITUDES - ATRIBUTOS NUMERICOS:\n')
    print("\nVerosimilitud_n:")
    print(json.dumps(verosimilitud_n, indent=4))        

    return verosimilitud_n

def calcular_probabilidad_posterior(data, verosimilitud_categorico, verosimilitud_numerico):
    print('\nPROBABILIDADES POSTERIORES:')

    clases_seleccionadas = [] 
    
    for index, instancia in enumerate(data):
        print(f'\nInstancia {index+1}:')
        
        probabilidades_posteriores = {}
        
        for clase in verosimilitud_categorico['Iris']:
            prob_cat = 1.0
            prob_num = 1.0
            
            for atributo, valor in instancia.items():
                if atributo in verosimilitud_categorico and valor in verosimilitud_categorico[atributo][clase]:
                    prob_cat *= verosimilitud_categorico[atributo][clase][valor]
                
                if atributo in verosimilitud_numerico[clase]['media']:
                    media = verosimilitud_numerico[clase]['media'][atributo]
                    desviacion = verosimilitud_numerico[clase]['desviacion_estandar'][atributo]
                    prob_num *= norm.pdf(float(valor), loc=media, scale=desviacion)
            
            prob_previa = verosimilitud_categorico['Iris'][clase][clase][clase]  # Acceder al valor numérico
            prob_total = prob_cat * prob_num * prob_previa
            probabilidades_posteriores[clase] = prob_total
        
        # Normalizar las probabilidades posteriores
        suma_probabilidades = sum(probabilidades_posteriores.values())
        for clase in probabilidades_posteriores:
            probabilidades_posteriores[clase] /= suma_probabilidades
        
        # Encontrar la clase con la mayor probabilidad
        clase_seleccionada = max(probabilidades_posteriores, key=probabilidades_posteriores.get)
        clases_seleccionadas.append(clase_seleccionada)
        
        # Imprimir las probabilidades posteriores
        for clase, prob in probabilidades_posteriores.items():
            print(f'Probabilidad - "{clase}": {prob:.3f}')
    
    return clases_seleccionadas

def matriz_confusion(clases_seleccionadas, archivo_csv):
        # Cargar las clases reales desde el archivo prueba.csv
    clases_reales = []
    with open(archivo_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            clase_real = row[-1]  
            clases_reales.append(clase_real)

    # Comparar clases seleccionadas con clases reales
    aciertos = sum(1 for clase_seleccionada, clase_real 
                in zip(clases_seleccionadas, clases_reales) 
                if clase_seleccionada == clase_real)
    
    total_instancias = len(clases_reales)
    precision = aciertos / total_instancias

    print(f'\nClases seleccionadas: \n {clases_seleccionadas}')
    print(f'\nClases reales: \n {clases_reales}')
    print(f'\nPrecisión del modelo: {precision * 100:.2f}%')

    # Crear la matriz de confusión
    confusion = confusion_matrix(clases_reales, clases_seleccionadas)

    # Calcular el recall, precisión y exactitud
    recall_clases = recall_score(clases_reales, clases_seleccionadas, average=None, zero_division=0.0) * 100
    precision_clases = precision_score(clases_reales, clases_seleccionadas, average=None) * 100
    exactitud = accuracy_score(clases_reales, clases_seleccionadas) * 100

    # Redondear las métricas a números enteros
    recall_clases = [round(recall) for recall in recall_clases]
    precision_clases = [round(precision) for precision in precision_clases]
    exactitud = round(exactitud)

    # Crear un DataFrame para visualizar la matriz de confusión
    clases_unicas = sorted(list(set(clases_reales + clases_seleccionadas)))
    df_confusion = pd.DataFrame(confusion, index=clases_unicas, columns=clases_unicas)

    # Visualizar la matriz de confusión junto con el recall, precisión y exactitud en el mismo gráfico
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)  # Subplot para la matriz de confusión
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Clases Seleccionadas')
    plt.ylabel('Clases Reales')
    plt.title('Matriz de Confusión')

    plt.subplot(2, 2, 3)  # Subplot para recall
    sns.barplot(x=recall_clases, y=clases_unicas, color="skyblue")
    plt.xlabel("Recall")
    plt.ylabel("Clases")
    plt.title("Recall por Clase")

    plt.subplot(2, 2, 4)  # Subplot para precisión
    sns.barplot(x=precision_clases, y=clases_unicas, color="salmon")
    plt.xlabel("Precisión")
    plt.ylabel("Clases")
    plt.title("Precisión por Clase")

    # Muestra la exactitud en la parte superior del gráfico
    plt.subplot(2, 2, 2)
    sns.barplot(x=[exactitud], y=["Exactitud"], color="limegreen")
    plt.xlabel("Métrica")
    plt.title("Exactitud del Modelo")

    plt.tight_layout()
    plt.show()

    
