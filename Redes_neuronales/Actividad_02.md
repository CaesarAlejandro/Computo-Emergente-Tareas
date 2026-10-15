### **3. Implementación del Modelo ADALINE (Regla Delta)**

#### **Descripción Computacional y Matemática**
El algoritmo implementa una red neuronal de tipo ADALINE (Widrow 1962), una estructura propuesta por Widrow y Hoff que funciona como un combinador lineal adaptativo. Este modelo pertenece a la clase de redes unidireccionales organizadas en capas (feed-forward) y con aprendizaje supervisado, las cuales son empleadas como clasificadores de patrones y estimadores de funciones. A diferencia del perceptrón simple, el aprendizaje en la red ADALINE utiliza la salida real continua calculada por la red, sin someterla previamente a una función umbral o escalón. La salida de la red se define matemáticamente como la suma ponderada de sus entradas:

$$y = \sum_{j=0}^{n} w_j x_j$$

Los pesos sinápticos se inicializan de manera aleatoria con valores entre 0 y 1. Durante las épocas de entrenamiento, el algoritmo calcula el error de forma iterativa tomando la diferencia directa entre el valor deseado y la salida real continua:

$$e = d - y$$

Para evaluar el rendimiento general del modelo, se calcula el Error Cuadrático Medio (MSE) sobre todo el conjunto de patrones:

$$MSE = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2} (d_i - y_i)^2$$

Con el objetivo de minimizar esta superficie de error, los pesos se actualizan utilizando la **Regla Delta**, basada en el método del descenso del gradiente. Cada peso se modifica de manera estrictamente proporcional al error, a la entrada correspondiente y a una tasa de aprendizaje constante ($\alpha$):

$$w_j(t+1) = w_j(t) + \alpha \cdot e \cdot x_j$$

#### **Datos de Trabajo y Definición de Patrones**
El conjunto de datos está diseñado para resolver un problema de descodificador de binario a decimal utilizando patrones de entrenamiento de dimensión 3. La matriz de entradas contiene el número máximo de combinaciones posibles para 3 bits ($2^3 = 8$). 

Cada vector de entrada está precedido por un valor de `1` fijo, que cumple la función de señal de entrada para el parámetro del umbral (bias). Las salidas deseadas contienen los valores escalares decimales del 0 al 7, que corresponden a la traducción exacta de cada respectivo número binario.

#### **Salida Teórica Esperada**
Teóricamente, existe una expresión matemática analítica capaz de realizar la descodificación de binario a decimal mediante una suma ponderada lineal. Dado que las salidas del problema son una función estrictamente lineal de las entradas, el modelo ADALINE es capaz de aproximar esta expresión de forma autónoma. Se espera que el algoritmo ajuste iterativamente sus pesos hasta que el error cuadrático total tienda a cero, encontrando una configuración de parámetros globalmente óptima.

#### **Resultados y Evolución del Error**
Al ejecutar el código, se graficó la curva de aprendizaje (MSE vs. Épocas), evidenciando que la medida del error cuadrático medio cae de manera exponencial durante las primeras iteraciones a medida que el descenso del gradiente ajusta los parámetros, estabilizándose cerca de cero mucho antes de alcanzar las 150 épocas máximas. 

Al finalizar, la red arroja pesos óptimos que convergen a los valores teóricos esperados: `[0.0, 4.0, 2.0, 1.0]`. En la prueba de eficiencia final, la predicción continua de la red acierta con una precisión absoluta:

*   **Entrada Binaria:** `[0, 0, 0]` $\rightarrow$ **Decimal Esperado:** 0 | **Predicción Red:** ~0.00
*   **Entrada Binaria:** `[0, 1, 0]` $\rightarrow$ **Decimal Esperado:** 2 | **Predicción Red:** ~2.00
*   **Entrada Binaria:** `[1, 0, 1]` $\rightarrow$ **Decimal Esperado:** 5 | **Predicción Red:** ~5.00
*   **Entrada Binaria:** `[1, 1, 1]` $\rightarrow$ **Decimal Esperado:** 7 | **Predicción Red:** ~7.00

#### **Conclusiones**
Los resultados demuestran la eficacia de la regla Delta para la optimización de pesos en problemas de naturaleza continua. El problema de conversión de base binaria a base decimal es, por definición analítica, una combinación puramente lineal donde cada posición de bit representa una potencia de 2 ($2^2=4$, $2^1=2$, $2^0=1$). 

El modelo ADALINE demostró ser la arquitectura idónea para este problema. Al analizar los pesos finales obtenidos (`w1=4.0`, `w2=2.0`, `w3=1.0`, con un `bias=0.0`), se concluye que la red neuronal no se limitó simplemente a memorizar o clasificar los patrones, sino que logró deducir de forma autónoma la función matemática subyacente (las potencias de la base 2) que rige el sistema de numeración binario.
