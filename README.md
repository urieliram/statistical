# Aprendizaje automático

Repositorio de actividades del curso de aprendizaje automático. La descripción del curso y las actividades se pueden encontrar la siguiente  [liga](https://github.com/satuelisa/StatisticalLearning)

Los datos a usar del libro están en [liga](https://hastie.su.domains/ElemStatLearn/datasets/)


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

```python
s = "Aquí el código usado"
print s
```
Los datos utilizados pa

