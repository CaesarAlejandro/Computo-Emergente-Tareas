# Fundamentos de Computación Neuronal: Implementaciones en Python Puro

Este repositorio alberga un cuaderno de Google Colab enfocado en explorar las bases matemáticas y algorítmicas de la inteligencia artificial. A través de implementaciones estructuradas estrictamente en Python puro —sin la intervención de librerías de aprendizaje automático—, el objetivo es desmitificar cómo "aprenden" las máquinas reconstruyendo paso a paso los modelos clásicos de la computación neuronal.

## Introducción a las Redes Neuronales Artificiales

Las Redes Neuronales Artificiales (RNA) son modelos matemáticos no lineales y multiparamétricos capaces de inducir una correspondencia entre conjuntos de patrones de información. En esencia, estos sistemas intentan modelar la estructura y el funcionamiento lógico de los componentes del sistema nervioso animal.

En la biología, el cerebro procesa los estímulos a través de millones de células (neuronas) que reciben señales, las suman y producen una respuesta si superan un umbral específico. Los modelos artificiales emulan este comportamiento utilizando "elementos de proceso" o neuronas artificiales. En estos modelos, cada señal de entrada se multiplica por un valor variable llamado "peso sináptico", simulando la fuerza de la conexión entre neuronas. Posteriormente, las señales se suman y pasan por una función de activación que determina la salida final de la unidad.

Lo que convierte a las RNA en dispositivos de cómputo extraordinariamente poderosos es su capacidad de adaptación. A diferencia de los programas tradicionales que ejecutan instrucciones secuenciales, las redes neuronales no se programan de forma explícita para resolver una tarea, sino que adquieren el conocimiento mediante un proceso de aprendizaje. Este aprendizaje consiste en ajustar iterativamente los pesos de las interconexiones en función de la experiencia.

Dependiendo del entorno y del problema a resolver, este proceso puede adoptar distintos paradigmas:

* **Aprendizaje Supervisado:** La red se entrena utilizando pares de ejemplos que incluyen la entrada y la salida deseada (un "profesor" indica la respuesta correcta).


* **Aprendizaje No Supervisado:** A la red solo se le proporcionan los datos de entrada y esta debe descubrir por sí misma la estructura subyacente o las similitudes para organizar la información.



## Modelos Desarrollados

A continuación, se presenta un desglose de los algoritmos implementados en este proyecto, desde las primeras unidades lógicas estáticas hasta arquitecturas capaces de autoorganizarse o resolver problemas no linealmente separables:

### Modelos Implementados

1. **Modelo de McCulloch-Pitts (1943):** Unidad lógica binaria estática que demuestra la capacidad de cómputo universal mediante pesos y umbrales fijos.


2. **Perceptrón Simple (1958):** Primera red neuronal con aprendizaje supervisado, diseñada para resolver problemas linealmente separables ajustando pesos mediante una función de activación tipo escalón.


3. **ADALINE (1960):** Elemento combinador adaptativo que introduce la Regla Delta para minimizar el error cuadrático medio utilizando valores continuos antes de aplicar la función umbral.


4. **Perceptrón Multicapa:** Arquitectura avanzada que supera la limitación de la separabilidad lineal incorporando capas ocultas no lineales y empleando el algoritmo de retropropagación (descenso del gradiente) para el aprendizaje.


5. **Red de Hopfield:** Memoria autoasociativa de entrada binaria con aprendizaje supervisado, capaz de recuperar patrones a partir de estímulos ruidosos utilizando la regla de Hebb.


6. **Algoritmo de Clustering (K-medio):** Modelo de aprendizaje no supervisado para clasificación de patrones de entrada continua, agrupando datos por proximidad espacial.


7. **Clasificador de K-vecinos más cercano (KNN):** Sistema supervisado para datos continuos que clasifica nuevos patrones basándose en la votación mayoritaria de los datos memorizados más próximos en el espacio.


8. **Mapas de Kohonen (SOM):** Red no supervisada de entrada continua donde las neuronas compiten para activarse, logrando una autoorganización topológica que descubre estructuras ocultas en los datos.


9. **Red de Hamming:** Clasificador supervisado de entrada binaria que determina la similitud entre un patrón y las clases memorizadas midiendo la distancia lógica (bits de diferencia).


10. **Matriz Memoria Asociativa (Willshaw, 1969):** Modelo fundamental de computación neuronal que funciona como memoria heteroasociativa, vinculando un patrón de entrada específico con un patrón de salida diferente.


11. **Clasificador Carpenter / Grossberg (ART-1):** Modelo de aprendizaje no supervisado para entradas binarias que emplea un parámetro de vigilancia para crear nuevas categorías dinámicamente sin sobrescribir el conocimiento previo.


12. **Clasificador Gausiano:** Clasificador supervisado para entrada continua que modela las clases basándose en sus distribuciones estadísticas (medias y varianzas) y la probabilidad de la campana de Gauss.


13. **Clasificador Óptimo:** Modelo probabilístico supervisado para entrada binaria que toma decisiones evaluando las probabilidades a priori y condicionales de los datos de entrenamiento.

La organización de los algoritmos en este repositorio no es arbitraria; responde a una progresión histórica, pedagógica y estructural que traza la evolución de la resolución de problemas en inteligencia artificial.

* **Evolución Histórica y Complejidad Matemática (Modelos 1 al 4):** El cuaderno inicia estableciendo los fundamentos del procesamiento neuronal. Arranca con el modelo estático de McCulloch-Pitts, avanza hacia los primeros modelos iterativos unicapa limitados por la separabilidad geométrica (Perceptrón y ADALINE), y culmina con el Perceptrón Multicapa. Este último representa el salto matemático que permitió resolver fronteras de decisión complejas mediante el cálculo diferencial y la retropropagación de errores.


* **Paradigmas de Memoria, Distancia y Autoorganización (Modelos 5 al 9):** Una vez dominada la optimización de pesos por minimización del error, la estructura transita hacia el diagrama clásico de clasificación (supervisado vs. no supervisado, binario vs. continuo). Este bloque agrupa algoritmos que abordan el aprendizaje desde perspectivas geométricas o asociativas, abarcando reglas biológicas de refuerzo sináptico (Hopfield), medidas de distancia euclidiana (K-medio, KNN, Kohonen) y cálculo de diferencias lógicas (Hamming).


* **Modelos Probabilísticos y Topologías Especializadas (Modelos 10 al 13):** El segmento final cubre enfoques estadísticos y de categorización dinámica. Presenta desde matrices tempranas de asociación heterogénea (Willshaw) y redes capaces de equilibrar la retención de memoria con el aprendizaje de nueva información (Carpenter/Grossberg), hasta clasificadores puramente probabilísticos (Gausiano y Óptimo). Esta secuencia final demuestra cómo el campo se expandió integrando la estadística formal para estimar la pertenencia a una clase.

Antes de continuar, vale la pena mencionar de que en el desarrollo de los algoritmos se usaron dos módulos de la librería estándar de Python: Math y random. Estos módulos tiene herramientas para trabajar con funciones matemáticas y aleatorizar valores. No tiene integrada funciones que manejen directamente los algoritmos cómo si lo sería Scikitlearn. Dicho esto a continuación se detalla el algoritmo usado para cada modelo.

**(1) Modelo de McCulloch-Pitts (Compuerta Lógica AND)**

**Explicación del Código Fuente**
El algoritmo implementa una función `mcculloch_pitts_and` que simula la unidad computacional neuronal más elemental utilizando neuronas de tipo binario. Se establecen dos parámetros estructurales que se mantienen fijos: los pesos sinápticos (`w1 = 1`, `w2 = 1`) y el nivel del umbral de activación (`umbral = 2`). La operación principal calcula la entrada total a la neurona mediante la suma ponderada de las señales (`y_in = (x1 * w1) + (x2 * w2)`). Inmediatamente después, esta suma se evalúa a través de una función de activación del tipo escalón. Si la suma ponderada es mayor o igual al umbral, la neurona dispara y retorna 1; en caso contrario, retorna 0. Al mantener los pesos y umbrales inmutables, este algoritmo no ejecuta ningún ciclo de aprendizaje.

**Datos de Trabajo**
El bloque opera sobre una lista de patrones discretos de entrada puramente binarios: `[(0, 0), (0, 1), (1, 0), (1, 1)]`. Estas tuplas representan exhaustivamente las cuatro combinaciones de estados posibles para las dos variables independientes de una compuerta lógica.

**Salida Teórica Esperada**
De acuerdo con la tabla de verdad matemática para la función lógica AND, la neurona debe permanecer inactiva (salida 0) para las entradas `(0, 0)`, `(0, 1)` y `(1, 0)`. La salida teórica solo alcanza la activación (salida 1) en el escenario donde ambas entradas se disparan simultáneamente en `(1, 1)`.

**Resultados de Ejecución Reales**
Al iterar la función sobre la lista de patrones, el algoritmo procesa la aritmética interna y devuelve exactamente la tabla de verdad esperada para la función AND:

* **Entrada: (0, 0)** -> Salida de la neurona: **0** *(Suma ponderada de 0, no supera el umbral)*.
* **Entrada: (0, 1)** -> Salida de la neurona: **0** *(Suma ponderada de 1, no supera el umbral)*.
* **Entrada: (1, 0)** -> Salida de la neurona: **0** *(Suma ponderada de 1, no supera el umbral)*.
* **Entrada: (1, 1)** -> Salida de la neurona: **1** *(Suma ponderada de 2, iguala el umbral)*.

El resultado confirma el principio del modelo: las funciones lógicas se pueden describir mediante combinaciones de estas neuronas binarias básicas.

** (2) Perceptrón Simple (Algoritmo Perceptrónico)**

**Explicación del Código Fuente**
El algoritmo implementa una red unicapa con aprendizaje supervisado, orientada específicamente a la resolución de problemas linealmente separables. El ciclo inicia estableciendo los pesos sinápticos en cero y definiendo una razón de aprendizaje constante de $\alpha = 0.5$. Por cada patrón de entrenamiento, la red calcula la suma ponderada de las entradas y la evalúa mediante una función de activación bipolar antisimétrica, la cual produce un 1 para valores positivos, -1 para negativos, o 0 (punto de indeterminación) si el valor es exactamente cero. Si la salida calculada por la red ($y$) no coincide con la salida deseada ($d$), se activa un mecanismo de corrección que adapta las sinapsis sumando al peso actual una fracción proporcional al error y a la entrada: $W_{ji}(n+1) = W_{ji}(n) + \alpha \cdot d \cdot x_i(n)$. El bucle de entrenamiento se detiene automáticamente en el momento en que los pesos logran procesar la totalidad del conjunto de datos sin cometer errores en una misma época.

**Datos de Trabajo**
El conjunto de datos representa la tabla de verdad de la compuerta lógica AND, pero codificada en formato bipolar (-1 para falso, 1 para verdadero). Cada vector de entrada tiene la estructura `[x0, x1, x2]`, donde el primer elemento es un $1$ constante que cumple la función de entrada para la neurona de inclinación (o bias), desplazando la frontera de decisión en el plano. El vector `salidas_deseadas` establece que la única condición verdadera (`1`) ocurre cuando tanto $x_1$ como $x_2$ son positivos.

**Salida Teórica Esperada**
Dado que la compuerta AND es un problema en el cual los puntos de una clase pueden separarse geométricamente de la otra mediante una línea recta, la teoría establece que el perceptrón debe alcanzar la convergencia. El algoritmo encontrará matemáticamente al menos un hiperplano (representado por los pesos finales) capaz de clasificar correctamente todas las entradas sin caer en un bucle infinito.

**Resultados de Ejecución Reales**
Al ejecutar este código, el perceptrón detecta los errores iniciales y modifica los pesos progresivamente. La convergencia se alcanza de manera exitosa en pocas épocas, y la prueba de eficiencia final corrobora que la frontera lineal ajustada discrimina los valores a la perfección:

* Entrada: `[-1, -1]` -> Salida de la red: **-1**
* Entrada: `[-1, 1]` -> Salida de la red: **-1**
* Entrada: `[1, -1]` -> Salida de la red: **-1**
* Entrada: `[1, 1]` -> Salida de la red: **1**

** (3) ADALINE (Adaptive Linear Neuron)**

**Explicación del Código Fuente**
El algoritmo implementa una red neuronal de tipo ADALINE, una estructura propuesta por Widrow y Hoff que funciona como un combinador lineal adaptativo. A diferencia del perceptrón simple, el aprendizaje en la red ADALINE utiliza la salida real continua calculada por la red, sin pasarla previamente por una función umbral o escalón. Los pesos sinápticos se inicializan de manera aleatoria con valores entre 0 y 1. Durante las épocas de entrenamiento, el algoritmo calcula el error iterativamente tomando la diferencia directa entre el valor real producido en la salida y la salida esperada ($d - y$). Para minimizar el error cuadrático medio para todos los patrones de aprendizaje, los pesos se actualizan usando la regla Delta, basada en el método del descenso del gradiente. Cada peso se modifica proporcionalmente al error, a la entrada correspondiente y a la tasa de aprendizaje.

**Datos de Trabajo**
El conjunto de datos está diseñado para resolver un problema de descodificador de binario a decimal utilizando patrones de entrenamiento de dimensión 3. La matriz de `entradas` contiene el número máximo de combinaciones posibles para 3 bits ($2^3 = 8$). Cada vector de entrada está precedido por un valor de `1` fijo, que cumple la función de señal de entrada para el parámetro del umbral (bias). Las `salidas_deseadas` contienen los valores escalares decimales del 0 al 7, que corresponden a la traducción exacta de cada número binario.

**Salida Teórica Esperada**
Teóricamente, existe una expresión matemática analítica capaz de realizar la descodificación de binario a decimal mediante una suma ponderada lineal. Dado que las salidas del problema son una función estrictamente lineal de las entradas, el modelo ADALINE es capaz de aproximar esta expresión por sí solo. Se espera que el algoritmo ajuste iterativamente sus pesos hasta que el error cuadrático total tienda a cero, encontrando una configuración de parámetros óptima.

**Resultados de Ejecución Reales**
Al ejecutar el código, la medida del error cuadrático medio cae drásticamente durante las 150 épocas a medida que el descenso del gradiente ajusta la superficie del error. Al finalizar, la red arroja pesos óptimos muy cercanos a los valores teóricos de la base binaria (`[0.0, 4.0, 2.0, 1.0]`). En la prueba de eficiencia, la predicción continua (lineal) de la red acierta con altísima precisión:

* Entrada Binaria: `[0, 0, 0]` -> Decimal Esperado: **0** | Predicción Red: **~0.00**
* Entrada Binaria: `[0, 1, 0]` -> Decimal Esperado: **2** | Predicción Red: **~2.00**
* Entrada Binaria: `[1, 0, 1]` -> Decimal Esperado: **5** | Predicción Red: **~5.00**
* Entrada Binaria: `[1, 1, 1]` -> Decimal Esperado: **7** | Predicción Red: **~7.00**

** (4) Perceptrón Multicapa con Retropropagación (Backpropagation)**

**Explicación del Código Fuente**
El algoritmo implementa un perceptrón multicapa, una arquitectura propuesta originalmente en 1986 para solventar las limitaciones de no linealidad que presentaba el perceptrón simple. La red organiza sus unidades de procesamiento en tres niveles: una capa de entrada (2 variables), una capa oculta intermedia (2 neuronas) y una capa de salida (1 neurona). Para procesar la información, el modelo utiliza una función de activación sigmoidal $f(x) = \frac{1}{1+e^{-x}}$, la cual es estrictamente derivable y proporciona una salida en el intervalo continuo de $[0, +1]$.

El núcleo del código es el algoritmo de retropropagación (back-propagation), que plantea el aprendizaje como un problema de optimización no lineal para minimizar la función del error cuadrático medio mediante el método del descenso del gradiente. En cada época, el algoritmo ejecuta un paso hacia adelante (feedforward) para calcular las activaciones, evalúa el error producido frente a la salida deseada y, posteriormente, propaga los errores cometidos en la capa de salida hacia atrás (hacia las neuronas de la capa oculta). Usando la regla de la cadena, calcula las derivadas parciales y actualiza todos los pesos ($w$) y umbrales ($u, v$) restando una fracción de dicho gradiente multiplicada por la razón de aprendizaje ($\alpha = 0.5$).

**Datos de Trabajo**
El bloque opera sobre la tabla de verdad discreta de la compuerta lógica XOR (O exclusivo), representada por los vectores de entrada `[[0, 0], [0, 1], [1, 0], [1, 1]]` y el vector objetivo `[0, 1, 1, 0]`. A diferencia de las funciones AND y OR, la compuerta XOR genera un patrón espacial que es imposible de separar mediante una única línea recta en un plano cartesiano, lo que lo convierte en el ejemplo clásico de un problema no linealmente separable.

**Salida Teórica Esperada**
Las redes multicapa poseen la capacidad matemática y geométrica para llevar a cabo clasificaciones generales más allá de fronteras lineales, logrando subdividir el espacio combinando múltiples polígonos convexos o regiones cerradas. Teóricamente, el algoritmo de retropropagación ajustará iterativamente los parámetros de las neuronas ocultas hasta descender por la hipersuperficie del error y alcanzar un mínimo (ya sea local o global). Al converger, la red debe ser capaz de emitir valores reales extremadamente cercanos a 0 para las clases falsas y cercanos a 1 para las verdaderas.

**Resultados de Ejecución Reales**
Al ejecutar este bucle iterativo, se observa cómo el parámetro del error total va decayendo época tras época, demostrando que la red desciende exitosamente por el gradiente. Al término de las 5000 épocas, las funciones sigmoidales arrojan números continuos que, al ser redondeados, resuelven la compuerta XOR con exactitud:

* Entrada: `[0, 0]` -> Esperado: **0** | Red: **~0.015** *(Clase: 0)*
* Entrada: `[0, 1]` -> Esperado: **1** | Red: **~0.981** *(Clase: 1)*
* Entrada: `[1, 0]` -> Esperado: **1** | Red: **~0.981** *(Clase: 1)*
* Entrada: `[1, 1]` -> Esperado: **0** | Red: **~0.023** *(Clase: 0)*
