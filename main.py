import naive_bayes
import pandas as pd
import random

def main(archivo_train, archivo_test):

    train = naive_bayes.cargar_datos_entrenamiento(archivo_train)

    test = naive_bayes.cargar_datos_prueba(archivo_test)

    especies = ['iris-setosa', 'iris-versicolor', 'iris-virginica']
    resultados = naive_bayes.calcular_tabla_de_frecuencia_categorico(train)
    total_clase = naive_bayes.contar_apariciones_ultima_columna(resultados)
    verosimilitud_c = naive_bayes.calcular_verosimilitud_categorico(resultados, especies, total_clase)


    tabla = naive_bayes.calcular_tabla_de_frecuencia_numerico(train)
    verosimilitud_n = naive_bayes.calcular_verosimilitud_numerico(tabla)

    clases_seleccionadas = naive_bayes.calcular_probabilidad_posterior(test, verosimilitud_c, verosimilitud_n)

    naive_bayes.matriz_confusion(clases_seleccionadas,'prueba.csv')

if __name__ == "__main__":

    for pruebas in range(0, 5):
        
        print('\nEVALUACION - ', pruebas + 1)

        archivo_original = "datos.csv"
        df_original = pd.read_csv(archivo_original)

        total_filas = len(df_original)

        porcentaje_70 = int(0.7 * total_filas)

        indices_aleatorios = random.sample(range(total_filas), porcentaje_70)

        df_70 = df_original.iloc[indices_aleatorios]
        df_30 = df_original.drop(indices_aleatorios)

        df_70.to_csv('entrenamiento.csv', index=False)
        df_30.to_csv('prueba.csv', index=False)

        main('entrenamiento.csv', 'prueba.csv')

