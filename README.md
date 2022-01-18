# Aprendizaje automático

Repositorio de actividades del curso de aprendizaje automático. La descripción del curso y las actividades se pueden encontrar la siguiente  [curso](https://github.com/satuelisa/StatisticalLearning)

Los datos a usar del libro están en [dataset](https://hastie.su.domains/ElemStatLearn/datasets/)


## Tarea 1: Introducción
>Instructions: Identify one or more learning problems in your thesis work and identify goals and elements.

En el trabajo de investigación del alumno se tienen resultados preliminares de 320 instancias resueltas de la programación de los generadores (Unit Commitment) para el día siguiente resuelto con un modelo de programación entera mixta.

Como entrada se tienen:
- Datos de disponibilidad de las unidades (variable categórica).
- La demanda eléctrica por regiones (variable escalar).
- La región a la que pertenecen los generadores (variable categórica).
- Requerimientos de reservas por tipo de reserva (variable escalar).


En las salidas de cada instancia se tiene el resultado de:
- Los generadores que fueron seleccionados para ser prendidos o apagados en cada hora (variable categórica).
- La producción de los generadores  en cada hora (variable escalar).
- Líneas de transmisión que fueron violadas (variable categórica).
- Las pérdidas en el sistema (variable escalar).
- Tiempos de ejecución de cada instancia (variable escalar).

Preguntas de investigación:
1. ¿Es posible determinar que conjunto de lineas de transmisión serán violadas conociendo los generadores que serán prendidos o la demanda por regiones?
2. ¿Es posible predecir las pérdidas en el sistema conociendo la salida de potencia de cada generador?
3. ¿Cómo puedo agrupar los generadores de acuerdo a aquellos que seguramente serán elegidos y a aquellos que es mas o menos probable que sean elegidos y aquellos que probablemente no?
4. ¿Es posible predecir el tiempo de ejecución de la programación conociendo la combinación de las lineas de transmisión violadas?
5. ¿Puede hacerse un modelo que explique los precios de las reservas en el sistema en función de los requerimientos de reservas requeridas?

Otras preguntas indirectamente relacionadas pero que se tienen datos para estudiar:

1. ¿Es posible predecir la aportación de agua de un embalse de un mes teniendo en cuenta las aportaciones mensuales de los demás embalses en una cuenca?. 

## Tarea 2: Aprendizaje supervisado

>Instructions: First carry out Exercise 2.8 of the textbook with their ZIP-code data and then replicate the process the best you manage to some data from you own problem that was established in the first homework.

El código completo de esta tarea se encuentra en [Tarea2.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea2.ipynb) 
Aquí solo se presentaran los resultados y partes importantes del código:

Los datos utilizados en este cuaderno están disponibles aquí: [Datasets](https://drive.google.com/drive/folders/159GnBJQDxTY9oYqPBZzdNghyb4Gd9pDS?usp=sharing)

### **Regresión líneal**
A continuación usaremos un modelo de regresión líneal para resolver el problema de ZIP-code [2,3] del libro [liga](https://link.springer.com/book/10.1007/978-0-387-84858-7). Usando la librería sklearn obtenemos un modelo de predicción de los datos de entrenamiento. Posteriomente, calculamos los errores entre la predicción $y\_pred$ y los datos de entrenamiento "Y". Además, los errores de predicción son representados por un histograma.
```python
model = LinearRegression().fit(X, Y) #https://realpython.com/linear-regression-in-python/
y_pred = model.predict(X)
error = Y - y_pred
dfx = pd.DataFrame(error,Y)
dfx.to_csv('error.csv')
plt = dfx.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf91', zorder=2, rwidth=0.9)
err_regress = mean_absolute_error(Y,y_pred)
```
![image](https://user-images.githubusercontent.com/54382451/150030914-b242c594-95f1-4124-b4a1-12f2a5f19f11.png)

