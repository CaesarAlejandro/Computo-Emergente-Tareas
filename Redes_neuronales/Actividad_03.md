## Actividad 3: Perceptrón Multicapa 

### 1. Caso de Estudio: Compuerta OR-Exclusivo (XOR)

**Explicación del Código Fuente**
El algoritmo implementa un perceptrón multicapa, una arquitectura propuesta originalmente para solventar las limitaciones de no linealidad que presentaba el perceptrón simple. La red organiza sus unidades de procesamiento en tres niveles (arquitectura 2-2-1): una capa de entrada (2 variables), una capa oculta intermedia (2 neuronas) y una capa de salida (1 neurona). Para procesar la información, el modelo utiliza una función de activación sigmoidal, la cual es estrictamente derivable y proporciona una salida en el intervalo continuo de $[0,1]$:

$$f(x) = \frac{1}{1+e^{-x}}$$

El núcleo del código es el algoritmo de retropropagación (backpropagation), que plantea el aprendizaje como un problema de optimización no lineal para minimizar la función del error cuadrático mediante el método del descenso del gradiente:

$$E = \frac{1}{2} \sum (t_i - y_i)^2$$

En cada época, el algoritmo ejecuta un paso hacia adelante (feedforward) para calcular las activaciones, evalúa el error producido frente a la salida deseada y, posteriormente, propaga los errores cometidos en la capa de salida hacia atrás (hacia las neuronas de la capa oculta). Usando la regla de la cadena, calcula las derivadas parciales y actualiza todos los pesos ($w$) y umbrales ($u, v$) restando una fracción de dicho gradiente multiplicada por la razón de aprendizaje ($\alpha = 0.5$):

$$w(t+1) = w(t) - \alpha \frac{\partial E}{\partial w}$$

Para optimizar el tiempo de cómputo, se implementó un **criterio de parada analítico** que detiene el entrenamiento de forma temprana si el error total cae por debajo de una tolerancia predefinida ($0.01$), en lugar de esperar a cumplir un número estático de épocas máximas.

**Datos de Trabajo**
El bloque opera sobre la tabla de verdad discreta de la compuerta lógica XOR, representada por los vectores de entrada `[[0, 0], [0, 1], [1, 0], [1, 1]]` y el vector objetivo `[0, 1, 1, 0]`. A diferencia de las funciones AND y OR, la compuerta XOR genera un patrón espacial que es imposible de separar mediante una única línea recta en un plano cartesiano, lo que lo convierte en el ejemplo clásico de un problema no linealmente separable.

**Salida Teórica Esperada**
Las redes multicapa poseen la capacidad matemática y geométrica para llevar a cabo clasificaciones generales más allá de fronteras lineales, logrando subdividir el espacio combinando múltiples polígonos convexos o regiones cerradas creadas por las neuronas ocultas. Teóricamente, el algoritmo de retropropagación ajustará iterativamente los parámetros hasta descender por la hipersuperficie del error y alcanzar un mínimo. Al converger, la red debe ser capaz de emitir valores reales extremadamente cercanos a 0 para las clases falsas y cercanos a 1 para las verdaderas.

**Resultados y Evolución del Error**
Al ejecutar el bucle iterativo, se generó una gráfica de la curva de aprendizaje (Error Total vs. Épocas) donde se observa cómo el parámetro del error va decayendo de manera asintótica época tras época, demostrando que la red desciende exitosamente por el gradiente. Gracias al criterio de parada, el modelo logró la convergencia temprana mucho antes de llegar a las 5000 épocas. En la prueba de eficiencia final, las salidas de la red resuelven la compuerta XOR con exactitud:

- **Entrada:** `[0, 0]` -> **Esperado:** 0 | **Red:** ~0.015 *(Clase: 0)*
- **Entrada:** `[0, 1]` -> **Esperado:** 1 | **Red:** ~0.981 *(Clase: 1)*
- **Entrada:** `[1, 0]` -> **Esperado:** 1 | **Red:** ~0.981 *(Clase: 1)*
- **Entrada:** `[1, 1]` -> **Esperado:** 0 | **Red:** ~0.023 *(Clase: 0)*

**Discusión y Limitaciones (Mínimos Locales)**
Los resultados confirman que la inclusión de una capa oculta no lineal permite a la red resolver la separabilidad del problema XOR. Sin embargo, el método de optimización por descenso del gradiente presenta una limitación teórica importante: la susceptibilidad a quedar atrapado en mínimos locales. 

Dado que el algoritmo actualiza los pesos basándose únicamente en la pendiente local de la hipersuperficie del error (donde $\frac{\partial E}{\partial w} \approx 0$), el proceso detecta que cualquier pequeño cambio incrementa el error y se detiene prematuramente. El algoritmo carece de mecanismos para distinguir si ha alcanzado el mínimo global (la solución óptima real que resuelve la compuerta) o simplemente un valle superficial (un mínimo local). Por este motivo, el éxito y la velocidad de convergencia del perceptrón multicapa en este código dependen fuertemente de la inicialización aleatoria de los pesos sinápticos y del valor asignado a la tasa de aprendizaje ($\alpha$).

### 2. Caso de Estudio: Aproximación de Funciones Continuas

**Explicación y Arquitectura**
Además de tareas de clasificación binaria, el perceptrón multicapa se destaca como un aproximador universal de funciones. Para este reto, se diseñó una red destinada a mapear la función matemática continua $f(x) = \sin(x)$ en el intervalo de $x \in [0, \pi]$. 

Dado que la complejidad de una curva continua es mayor que la del plano del XOR, se justificó un rediseño de la arquitectura, incrementando la dimensionalidad de la capa oculta. Se implementó una estructura **1-5-1** (1 neurona de entrada, 5 neuronas ocultas para capturar los distintos tramos de concavidad de la onda, y 1 neurona de salida). Al estar la onda seno acotada entre $[0, 1]$ en el primer y segundo cuadrante, se mantuvo la función sigmoide en la capa de salida.

**Resultados de la Aproximación**
Se generó un conjunto de datos (dataset) de 50 muestras uniformemente distribuidas. Tras entrenar la red durante 5000 épocas, la gráfica resultante de $f(x)$ vs $y_{pred}$ demuestra que la red neuronal logró interpolar los valores aprendiendo la estructura interna de la onda senoidal. La curva de las predicciones (línea punteada roja) se superpone casi a la perfección con la función analítica original, y la curva del Error Cuadrático Medio (MSE) converge exitosamente hacia un valor cercano a cero.

---

### 3. Análisis Paramétrico y Comparación de Configuraciones

Como evidencia final de la dinámica de la red, se realizó un estudio del impacto que tiene la **Tasa de Aprendizaje ($\alpha$)** en la velocidad de convergencia del modelo. Utilizando la arquitectura 2-2-1 del caso XOR con pesos iniciales idénticos controlados por semilla, se corrieron tres entrenamientos en paralelo:

1.  **$\alpha = 0.1$ (Lento):** El descenso por la hipersuperficie del gradiente ocurre mediante pasos minúsculos. La gráfica evidencia que requiere un gran número de iteraciones (más de 2500 épocas) para poder salir de la meseta inicial y converger hacia cero. Su ventaja teórica es la estabilidad, pero computacionalmente es ineficiente.
2.  **$\alpha = 0.5$ (Recomendado):** Muestra un equilibrio óptimo. La curva del error cae dramáticamente pasadas las 1000 épocas, logrando estabilizar los parámetros de forma segura y veloz sin sobreoscilaciones.
3.  **$\alpha = 0.9$ (Agresivo):** Aunque la intención de un $\alpha$ alto es forzar un aprendizaje rápido, los saltos abruptos sobre el gradiente pueden generar inestabilidad o causar que la red sobrepase (overshoot) el mínimo. En la gráfica comparativa, la convergencia es veloz, pero se observan descensos mucho más cortados y menos suaves.

**Conclusión General de la Actividad 3**
El algoritmo de *Backpropagation* consolida al Perceptrón Multicapa como un estimador de patrones robusto. Las pruebas demostraron empíricamente dos de sus mayores facultades: superar las limitaciones geométricas de los problemas no linealmente separables (XOR), y aproximar topologías continuas (seno). El análisis comparativo subraya que el éxito de estas topologías no recae solo en tener funciones derivables, sino en una cuidadosa sintonización de hiperparámetros empíricos como el número de neuronas ocultas y la razón de aprendizaje.
