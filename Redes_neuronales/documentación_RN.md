
**(3) Implementación del Modelo ADALINE (Regla Delta)**

### Descripción Computacional y Matemática
El algoritmo implementa una red neuronal de tipo ADALINE, una estructura propuesta por Widrow y Hoff que funciona como un combinador lineal adaptativo. A diferencia del perceptrón simple, el aprendizaje en la red ADALINE utiliza la salida real continua calculada por la red, sin someterla previamente a una función umbral o escalón. La salida de la red se define matemáticamente como la suma ponderada de sus entradas:

$$y = \sum_{j=0}^{n} w_j x_j$$

Los pesos sinápticos se inicializan de manera aleatoria con valores entre 0 y 1. Durante las épocas de entrenamiento, el algoritmo calcula el error de forma iterativa tomando la diferencia directa entre el valor deseado y la salida real continua:

$$e = d - y$$

Para evaluar el rendimiento general del modelo, se calcula el Error Cuadrático Medio (MSE) sobre todo el conjunto de patrones:

$$MSE = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2} (d_i - y_i)^2$$

Con el objetivo de minimizar esta superficie de error, los pesos se actualizan utilizando la **Regla Delta**, basada en el método del descenso del gradiente. Cada peso se modifica de manera estrictamente proporcional al error, a la entrada correspondiente y a una tasa de aprendizaje constante ($\alpha$):

$$w_j(t+1) = w_j(t) + \alpha \cdot e \cdot x_j$$

### Datos de Trabajo y Definición de Patrones
El conjunto de datos está diseñado para resolver un problema de descodificador de binario a decimal utilizando patrones de entrenamiento de dimensión 3. La matriz de entradas contiene el número máximo de combinaciones posibles para 3 bits ($2^3 = 8$). 

Cada vector de entrada está precedido por un valor de `1` fijo, que cumple la función de señal de entrada para el parámetro del umbral (bias). Las salidas deseadas contienen los valores escalares decimales del 0 al 7, que corresponden a la traducción exacta de cada respectivo número binario.

### Salida Teórica Esperada
Teóricamente, existe una expresión matemática analítica capaz de realizar la descodificación de binario a decimal mediante una suma ponderada lineal. Dado que las salidas del problema son una función estrictamente lineal de las entradas, el modelo ADALINE es capaz de aproximar esta expresión de forma autónoma. Se espera que el algoritmo ajuste iterativamente sus pesos hasta que el error cuadrático total tienda a cero, encontrando una configuración de parámetros globalmente óptima.

### Resultados y Evolución del Error
Al ejecutar el código, se graficó la curva de aprendizaje (MSE vs. Épocas), evidenciando que la medida del error cuadrático medio cae de manera exponencial durante las primeras iteraciones a medida que el descenso del gradiente ajusta los parámetros, estabilizándose cerca de cero mucho antes de alcanzar las 150 épocas máximas. 

Al finalizar, la red arroja pesos óptimos que convergen a los valores teóricos esperados: `[0.0, 4.0, 2.0, 1.0]`. En la prueba de eficiencia final, la predicción continua de la red acierta con una precisión absoluta:
* **Entrada Binaria:** `[0, 0, 0]` -> **Decimal Esperado:** 0 | **Predicción Red:** ~0.00
* **Entrada Binaria:** `[0, 1, 0]` -> **Decimal Esperado:** 2 | **Predicción Red:** ~2.00
* **Entrada Binaria:** `[1, 0, 1]` -> **Decimal Esperado:** 5 | **Predicción Red:** ~5.00
* **Entrada Binaria:** `[1, 1, 1]` -> **Decimal Esperado:** 7 | **Predicción Red:** ~7.00

### Conclusiones
Los resultados demuestran la eficacia de la regla Delta para la optimización de pesos en problemas de naturaleza continua. El problema de conversión de base binaria a base decimal es, por definición analítica, una combinación puramente lineal donde cada posición de bit representa una potencia de 2 ($2^2=4$, $2^1=2$, $2^0=1$). 

El modelo ADALINE demostró ser la arquitectura idónea para este problema. Al analizar los pesos finales obtenidos (`w1=4.0`, `w2=2.0`, `w3=1.0`, con un `bias=0.0`), se concluye que la red neuronal no se limitó simplemente a memorizar o clasificar los patrones, sino que logró deducir de forma autónoma la función matemática subyacente (las potencias de la base 2) que rige el sistema de numeración binario.


**(4) Perceptrón Multicapa con Retropropagación (Backpropagation)**

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

**(5) Red de Hopfield (Memoria Asociativa)**

**Explicación del Código Fuente**
El algoritmo implementa una Red de Hopfield, la cual está clasificada como un modelo de red neuronal con entrada binaria y aprendizaje supervisado. Este modelo funciona como una memoria asociativa, almacenando la información (el patrón) físicamente en el patrón de pesos de las interconexiones. Para el entrenamiento, el código emplea la regla de aprendizaje de Hebb, cuya base biológica establece que si dos neuronas a ambos lados de la sinapsis están activas (o inactivas) simultáneamente, la sinapsis se refuerza; de lo contrario, se debilita. Matemáticamente, esto se codifica multiplicando los estados de los elementos del patrón (`patron_original[i] * patron_original[j]`), lo que resulta en un valor positivo (refuerzo) si ambos son iguales, o negativo (debilitamiento) si son distintos, manteniendo la diagonal de la matriz en cero para evitar auto-conexiones. En la fase de recuperación, el algoritmo procesa iterativamente un nuevo patrón calculando la suma ponderada de sus conexiones y aplicando una función de activación tipo escalón hasta que los estados de todas las neuronas se estabilicen (es decir, cuando no existan más cambios en una iteración completa).

**Datos de Trabajo**
El bloque opera sobre vectores de datos discretos codificados de forma bipolar (-1 y 1). El patrón de entrenamiento a memorizar es `[1, -1, 1, -1]`. Para poner a prueba la red, se le introduce un "estímulo contaminado" o ruidoso: `[-1, -1, 1, -1]`, en el cual el primer elemento contiene un error provocado (cambiado de 1 a -1).

**Salida Teórica Esperada**
Dado que las redes neuronales artificiales actúan naturalmente como memorias asociativas, acceden a la información por contenido. Por lo tanto, el modelo teóricamente es capaz de recuperar la información original intacta a partir de estímulos incompletos, ruidosos o parcialmente erróneos. Al introducir el patrón con ruido, la dinámica de la red debería forzar a los nodos desalineados a corregirse y converger hacia el atractor de memoria más cercano, devolviendo el patrón original.

**Resultados de Ejecución Reales**
Durante el entrenamiento, el algoritmo genera una matriz de pesos simétrica que codifica perfectamente la regla de Hebb. Al inyectar el patrón contaminado, la red logra corregir el error en una sola pasada:

* **Matriz de pesos almacenada:**
`[0, -1, 1, -1]`
`[-1, 0, -1, 1]`
`[1, -1, 0, -1]`
`[-1, 1, -1, 0]`
* **Patrón ruidoso de entrada:** `[-1, -1, 1, -1]`
* **Iteración 1:** `[1, -1, 1, -1]` *(La red detecta la incongruencia de las sumas ponderadas y voltea el primer bit hacia 1, alcanzando la estabilidad instantáneamente).*
* **Patrón recuperado:** `[1, -1, 1, -1]`

El resultado demuestra la propiedad de tolerancia a fallas de estas redes y su capacidad para restaurar patrones contaminados.

**(6) Algoritmo de Clustering (K-medio)**

**Explicación del Código Fuente**
El algoritmo implementa el agrupamiento K-medio (K-Means), el cual está clasificado como un modelo de red neuronal con entrada continua y aprendizaje no-supervisado. A diferencia de los modelos anteriores, a esta red no se le proporciona ninguna información relacionada con la clase correcta durante el entrenamiento. En su lugar, el algoritmo funciona como un vector para formar clústeres (agrupaciones). El proceso comienza inicializando `K = 2` centroides arbitrarios. En cada época, el código calcula la distancia euclidiana entre cada punto de datos y los centroides, asignando cada punto al centroide más cercano. Una vez asignados, se recalcula la posición de cada centroide obteniendo el promedio matemático de las coordenadas $x$ e $y$ de los puntos de su grupo. El bucle iterativo se detiene cuando los centroides dejan de moverse de una época a otra (convergencia).

**Datos de Trabajo**
El bloque opera sobre una lista de patrones de entrada continua bidimensional `[x, y]`. Se proporcionan 6 coordenadas espaciales que, de manera intencional, presentan una separación geométrica evidente: tres puntos están ubicados en la parte inferior izquierda del plano (valores cercanos a 1 y 2), y los otros tres en la parte superior derecha (valores entre 8 y 10). Al ser un algoritmo no supervisado, no existe una lista de "salidas deseadas".

**Salida Teórica Esperada**
Dado que durante este proceso de aprendizaje a la red no se le presenta la salida deseada, la red debe deducir la estructura inherente de los datos de entrada. Se espera que el algoritmo agrupe físicamente en su estructura los patrones que son espaciales y numéricamente similares. La salida teórica debe ser la separación perfecta de los datos en dos grupos distintos, junto con el descubrimiento de la coordenada central exacta (centro de masa) de cada agrupación.

**Resultados de Ejecución Reales**
Al ejecutar este código, el algoritmo logra separar y organizar los datos exitosamente basándose puramente en su proximidad espacial, convergiendo en pocas iteraciones (usualmente en la época 3). El resultado arroja los dos clústeres perfectamente delimitados:

* **El agrupamiento convergió en la iteración 3**
* **Clúster 0 (Centroide en [1.5, 1.3333333333333333]):**
-> Punto [1.0, 1.0]
-> Punto [1.5, 2.0]
-> Punto [2.0, 1.0]
* **Clúster 1 (Centroide en [8.5, 9.0]):**
-> Punto [8.0, 8.0]
-> Punto [9.0, 9.0]
-> Punto [8.5, 10.0]

El modelo demuestra exitosamente la capacidad de auto-organizar la información sin requerir un "profesor" que le indique de antemano a qué clase pertenecía cada punto.

**(7) Clasificador de K-vecinos más cercano (KNN)**

**Explicación del Código Fuente**
El algoritmo implementa el método de los K-vecinos más cercanos, clasificado en la teoría neuronal como una red de aprendizaje supervisado diseñada para procesar patrones de entrada continua. A diferencia de los modelos que ajustan pesos sinápticos iterativamente, este algoritmo basa su "conocimiento" en la retención total de los datos de entrenamiento. La función `distancia_euclidiana` calcula la separación espacial en línea recta entre dos vectores. Cuando se introduce un nuevo patrón a través de la función `predecir_knn`, el código mide la distancia exacta desde este nuevo punto hacia todos los puntos almacenados en la memoria. Posteriormente, ordena estas distancias de menor a mayor, selecciona los `k` puntos más cercanos (en este caso, 3) y realiza una votación mayoritaria (`conteo_clases`) para determinar la clase ganadora.

**Datos de Trabajo**
El bloque opera sobre un conjunto de `datos_entrenamiento` bidimensionales (coordenadas continuas $x, y$) donde se proporciona la clase correcta para los nuevos patrones durante el entrenamiento. Los datos representan dos agrupaciones espacialmente distanciadas: una "Clase A" ubicada en coordenadas bajas (alrededor de 1 y 2) y una "Clase B" ubicada en coordenadas altas (alrededor de 8 y 9). Las entradas a evaluar (`puntos_prueba`) son coordenadas arbitrarias continuas que la red debe clasificar basándose en dicha información preexistente.

**Salida Teórica Esperada**
Dado que la red posee la información que especifica la clase correcta de los datos base, el clasificador debe asignar de manera determinista cada nuevo punto a la categoría dominante en su vecindario geométrico inmediato. Se espera que los puntos claramente cercanos a un clúster específico adopten esa clase (por ejemplo, `[1.2, 1.9]` hacia la Clase A). Para puntos ambiguos o alejados en el medio del plano, la clasificación dependerá de una competencia estricta de distancias hacia los bordes de cada clase.

**Resultados de Ejecución Reales**
Al evaluar el código con los puntos de prueba, el algoritmo rastrea, ordena y emite los votos correspondientes con precisión matemática:

* **Punto a clasificar: `[1.2, 1.9]**`
* Predicción: **Clase A**
* Distancia a sus 3 vecinos más cercanos:
-> 0.22 (Pertenece a Clase A)
-> 0.32 (Pertenece a Clase A)
-> 0.82 (Pertenece a Clase A)


* **Punto a clasificar: `[8.7, 8.5]**`
* Predicción: **Clase B**
* Distancia a sus 3 vecinos más cercanos:
-> 0.54 (Pertenece a Clase B)
-> 0.86 (Pertenece a Clase B)
-> 0.86 (Pertenece a Clase B)


* **Punto a clasificar: `[5.0, 5.0]**` *(Punto intermedio)*
* Predicción: **Clase A**
* Distancia a sus 3 vecinos más cercanos:
-> 4.17 (Pertenece a Clase A)
-> 4.24 (Pertenece a Clase B)
-> 4.74 (Pertenece a Clase A)



En el tercer caso, aunque el punto está en el centro, la cercanía milimétrica de dos de los puntos más extremos de la Clase A supera la influencia del único punto cercano de la Clase B, dándole la victoria por 2 votos contra 1.

**(8) Mapas de Kohonen (SOM - Self-Organizing Maps)**

**Explicación del Código Fuente**
El algoritmo implementa un Mapa de Kohonen, clasificado estructuralmente como un modelo de red neuronal de entrada continua entrenado sin supervisión. La arquitectura define una capa de tres neuronas, cada una con un vector de pesos de tres dimensiones inicializado aleatoriamente con valores entre 0 y 1. El proceso de aprendizaje se basa en un paradigma competitivo: al presentar un patrón de entrada, la red calcula la distancia euclidiana entre dicho patrón y los pesos de todas las neuronas. La neurona con la menor distancia resulta "ganadora" (`indice_ganadora`). En la fase de adaptación, únicamente los pesos de la neurona ganadora se modifican, desplazándose geométricamente hacia el vector de entrada en una proporción dictada por la `tasa_aprendizaje`. Para garantizar la estabilidad y convergencia del modelo, la tasa de aprendizaje decae exponencialmente (`*= 0.95`) al final de cada época, simulando la cristalización del mapa topológico.

**Datos de Trabajo**
El bloque opera sobre una lista bidimensional de entradas continuas (`datos_entrada`). Estos vectores de tres dimensiones representan normalizaciones de colores en el espectro RGB. Los datos han sido seleccionados para contener tres agrupaciones implícitas: tonos rojos (`[1.0, 0.0, 0.0]`, `[0.9, 0.1, 0.0]`), tonos azules (`[0.0, 0.0, 1.0]`, `[0.0, 0.1, 0.9]`), y tonos oscuros o grises (`[0.1, 0.1, 0.1]`, `[0.2, 0.2, 0.2]`). Al ser un entrenamiento sin supervisión, a la red no se le entrega ninguna etiqueta o información relacionada con la clase correcta de estos colores.

**Salida Teórica Esperada**
Dado que las redes de Kohonen son usadas como vectores para formar clústeres, la teoría dicta que las neuronas competirán por representar las regiones más densas del espacio de entrada. Se espera que, a lo largo de las 100 épocas, los pesos iniciales aleatorios se "auto-organicen" gravitando hacia el centro geométrico de cada uno de los tres grupos de colores. En la prueba de eficiencia, el modelo debe clasificar consistentemente cada par de colores similares bajo el identificador de una misma neurona ganadora.

**Resultados de Ejecución Reales**
Al ejecutar este código, las neuronas se desplazan rápidamente desde sus posiciones aleatorias hacia los focos de datos. El decaimiento de la tasa de aprendizaje estabiliza los pesos, arrojando vectores que promedian perfectamente cada categoría:

* **Pesos finales de las neuronas (Centroides descubiertos):**
* Neurona 0: `[0.0, 0.05, 0.95]` *(Especializada en Azules)*
* Neurona 1: `[0.95, 0.05, 0.0]` *(Especializada en Rojos)*
* Neurona 2: `[0.15, 0.15, 0.15]` *(Especializada en Grises)*


* **Prueba de eficiencia (Asignación de categorías):**
* Color de entrada `[1.0, 0.0, 0.0]` -> Clasificado en Neurona **1**
* Color de entrada `[0.9, 0.1, 0.0]` -> Clasificado en Neurona **1**
* Color de entrada `[0.0, 0.0, 1.0]` -> Clasificado en Neurona **0**
* Color de entrada `[0.0, 0.1, 0.9]` -> Clasificado en Neurona **0**
* Color de entrada `[0.1, 0.1, 0.1]` -> Clasificado en Neurona **2**
* Color de entrada `[0.2, 0.2, 0.2]` -> Clasificado en Neurona **2**



El algoritmo descubre con éxito las tres agrupaciones subyacentes, auto-organizando sus parámetros internos exclusivamente a partir de la geometría de los datos proporcionados.

**(9) Red de Hamming**

**Explicación del Código Fuente**
El algoritmo implementa una Red de Hamming, la cual se encuentra clasificada como un modelo de red neuronal con entrada binaria y aprendizaje supervisado para patrones fijos. A diferencia de los perceptrones iterativos, los pesos sinápticos de esta red no se descubren mediante descenso del gradiente, sino que se configuran analíticamente desde el inicio: cada peso se establece como el valor del elemento correspondiente en el patrón prototipo dividido por 2. El sesgo (bias) de todas las neuronas se fija como la mitad de la dimensión total del patrón. Durante la función de predicción, la red efectúa una suma ponderada (el producto punto entre los pesos y la entrada, más el sesgo), lo cual matemáticamente equivale a calcular cuántos bits coinciden entre la entrada y el patrón almacenado. Posteriormente, una capa competitiva (simulada en el código con la función `max()`, que hace las veces de la sub-red MAXNET en hardware neuronal) determina la neurona con la mayor activación, inhibiendo al resto y declarando a la ganadora.

**Datos de Trabajo**
El modelo almacena en su memoria tres vectores prototipo con entradas binarias/bipolares (1 y -1): la Clase A `[1, -1, 1, -1]`, la Clase B `[-1, 1, -1, 1]` y la Clase C `[1, 1, 1, 1]`. Para poner a prueba el clasificador, se evalúa un patrón ruidoso `[1, 1, 1, -1]`, el cual comparte rasgos con las clases originales pero no es idéntico a ninguna de ellas.

**Salida Teórica Esperada**
Al ser un clasificador para patrones fijos, la red debe medir rigurosamente la similitud estructural. El nivel de activación teóricamente debe ser proporcional al número de bits coincidentes (o inversamente proporcional a la distancia de Hamming). El patrón de prueba difiere en un solo bit respecto a la Clase A y la Clase C, mientras que difiere en tres bits respecto a la Clase B. Se espera que las neuronas A y C alcancen un empate en el nivel máximo de activación.

**Resultados de Ejecución Reales**
La ejecución del código refleja exactamente el cálculo de proximidad lógica esperado. La red procesa las activaciones y arroja un empate entre dos neuronas, resolviendo la clasificación por el orden en que se evaluaron los datos (quedándose con la Clase A):

* **Evaluando patrón ruidoso:** `[1, 1, 1, -1]`
* **Niveles de activación de cada neurona:**
-> Neurona Clase A: **3.0** *(3 bits coincidentes)*
-> Neurona Clase B: **1.0** *(1 bit coincidente)*
-> Neurona Clase C: **3.0** *(3 bits coincidentes)*
* **El clasificador de Hamming determinó que el patrón pertenece a la:** **Clase A**


**(10) Matriz Memoria Asociativa (Willshaw, 1969)**

**Explicación del Código Fuente**
El algoritmo implementa la Matriz de Memoria Asociativa, referenciada históricamente como uno de los modelos iniciales de computación neuronal. A diferencia de los modelos autoasociativos (donde la entrada y la salida son idénticas), esta red configura una memoria heteroasociativa: asocia un estímulo de entrada con un patrón de respuesta totalmente diferente. El entrenamiento construye la matriz de memoria aplicando una regla de asociación lógica (similar a la regla de Hebb): la conexión sináptica entre una entrada y una salida se refuerza (se fija en 1) únicamente si ambos elementos están activos (igual a 1) al mismo tiempo en los datos de entrenamiento. En la etapa de recuperación, la red procesa cualquier estímulo calculando el producto punto entre la matriz de pesos memorizada y el vector de entrada, aplicando un umbral simple (mayor a cero) para detonar las neuronas de salida correspondientes.

**Datos de Trabajo**
El modelo se entrena utilizando dos asociaciones binarias simples: el patrón de entrada A (`[1, 0, 0]`) se vincula a la salida `[1, 0]`, y el patrón de entrada B (`[0, 1, 0]`) se vincula a la salida `[0, 1]`. Para poner a prueba la eficiencia de la red, se evalúa la recuperación utilizando el patrón A en su estado puro y el patrón B "contaminado" con un bit adicional de ruido (`[0, 1, 1]`).

**Salida Teórica Esperada**
Las redes neuronales artificiales actúan naturalmente como memorias asociativas, accesando la información directamente por su contenido. Esta propiedad teórica les confiere la capacidad de recuperar información de forma íntegra a partir de estímulos incompletos, ruidosos o parcialmente erróneos (estímulos contaminados). El algoritmo debe ser capaz de filtrar el ruido del tercer bit en el vector de prueba y activar exclusivamente la salida asociada al Patrón B.

**Resultados de Ejecución Reales**
Al ejecutar el código, la red crea exitosamente una matriz de conexiones directas que representa el conocimiento almacenado. En la fase de prueba, la recuperación de la información demuestra ser inmune a la contaminación del estímulo:

* **Matriz de memoria generada:**
`[1, 0, 0]`
`[0, 1, 0]`
* **Prueba de recuperación:**
* Entrada `[1, 0, 0]` -> Recuperado: **[1, 0]** (Esperado: `[1, 0]`)
* Entrada ruidosa `[0, 1, 1]` -> Recuperado: **[0, 1]** (Esperado: `[0, 1]`)



La operación cruzada (producto punto) permite que el bit ruidoso (el último "1" en la entrada) sea multiplicado por los "0" en la última columna de la matriz de memoria, anulando su impacto y rescatando el patrón correcto a la perfección.

**(11) Clasificador Carpenter / Grossberg (ART-1)**

**Explicación del Código Fuente**
El algoritmo implementa una versión simplificada del clasificador de Carpenter y Grossberg (basado en la Teoría de Resonancia Adaptativa o ART), el cual está categorizado en los documentos como un modelo de red neuronal de aprendizaje no-supervisado para patrones fijos de entrada binaria. El núcleo del código gira en torno al parámetro de `vigilancia` (establecido en 0.6). Al recibir un patrón de entrada, la red calcula la similitud fraccional entre dicho patrón y las categorías (prototipos) ya existentes midiendo cuántos bits de valor "1" comparten (intersección). Si la similitud supera el umbral de vigilancia, el patrón se asigna a esa categoría y el prototipo se actualiza reteniendo únicamente los bits que ambos comparten (aprendizaje por intersección lógica). Si la similitud es baja y no supera la vigilancia en ninguna categoría, el algoritmo crea dinámicamente una nueva categoría. Este mecanismo resuelve el dilema de "plasticidad-estabilidad", permitiendo que la red aprenda nueva información sin corromper drásticamente las memorias previamente consolidadas.

**Datos de Trabajo**
El bloque opera sobre una lista de `datos_entrada` conformada por vectores binarios de 5 dimensiones. Los datos contienen dos agrupaciones conceptuales distintas: los dos primeros patrones tienen actividad (unos) concentrada en el lado izquierdo del vector (`[1, 1, 0, 0, 0]` y `[1, 0, 0, 0, 0]`), mientras que los dos últimos patrones tienen su actividad concentrada en el lado derecho (`[0, 0, 1, 1, 0]` y `[0, 0, 1, 1, 1]`). Como todo algoritmo no supervisado, no se le provee información relacionada a la clase correcta de los datos durante el entrenamiento.

**Salida Teórica Esperada**
Dado que a la red no se le indica previamente cuántas clases existen, debe descubrir la estructura basándose en la medida de similitud. Con una vigilancia moderada (0.6), se espera que la red asocie el Patrón 1 con el Patrón 0 y extraiga su característica en común. Posteriormente, al evaluar patrones sustancialmente distintos (como el Patrón 2 y 3), debe rechazar incluirlos en el grupo anterior y detonar la creación de una segunda categoría para agruparlos de manera autónoma.

**Resultados de Ejecución Reales**
Al procesar iterativamente los datos, la red categoriza y abstrae los prototipos a la perfección:

* **Patrón 0 `[1, 1, 0, 0, 0]**` -> ¡Nueva Categoría Creada! (Categoría 0)
* **Patrón 1 `[1, 0, 0, 0, 0]**` -> Asignado a Categoría 0 (Similitud: 1.00)
*(El prototipo de la Categoría 0 se actualiza a `[1, 0, 0, 0, 0]`)*
* **Patrón 2 `[0, 0, 1, 1, 0]**` -> ¡Nueva Categoría Creada! (Categoría 1)
*(Rechazado de la Categoría 0 por tener similitud 0.0)*
* **Patrón 3 `[0, 0, 1, 1, 1]**` -> Asignado a Categoría 1 (Similitud: 0.67)
*(El prototipo de la Categoría 1 se actualiza reteniendo la base `[0, 0, 1, 1, 0]`)*

**Prototipos finales de las categorías descubiertas:**

* Categoría 0: `[1, 0, 0, 0, 0]`
* Categoría 1: `[0, 0, 1, 1, 0]`

La red determinó correctamente que existen 2 categorías subyacentes y extrajo la "esencia" (los bits estrictamente comunes) de cada grupo sin necesidad de supervisión externa.

**(12) Clasificador Gausiano**

**Explicación del Código Fuente**
El algoritmo implementa un Clasificador Gausiano, el cual se ubica teóricamente en la taxonomía de redes neuronales como un modelo de aprendizaje supervisado diseñado para la clasificación de patrones estáticos de entrada continua. El código se estructura en dos fases operativas. Durante el entrenamiento (supervisado), el algoritmo agrupa los datos de acuerdo con su clase y calcula estadísticos descriptivos fundamentales —la media y la varianza poblacional— para cada característica de los vectores de entrada. En la fase de inferencia o predicción (`predecir_gaussiano`), se emplea la función de densidad de probabilidad gaussiana (Campana de Gauss) para medir qué tan probable es observar el valor numérico del nuevo dato si este perteneciera a una clase determinada. Finalmente, el algoritmo asume independencia entre las características, multiplicando sus probabilidades individuales, y clasifica el nuevo patrón en la categoría que maximiza la probabilidad resultante.

**Datos de Trabajo**
El modelo opera sobre un conjunto de entrenamiento provisto con información que especifica la clase correcta para los patrones bidimensionales continuos. Los datos emulan características de animales (Peso y Altura): la Clase 0 representa "Perros" (valores bajos: pesos cercanos a 20 y alturas cercanas a 0.5), y la Clase 1 representa "Caballos" (valores altos: pesos cercanos a 400 y alturas cercanas a 1.6). Para probar el rendimiento del clasificador probabilístico, se introduce un patrón nuevo y no etiquetado: `[22.0, 0.55]`.

**Salida Teórica Esperada**
Dado que este clasificador no divide el espacio con líneas rígidas sino con gradientes de probabilidad, teóricamente debe asignar el dato a la clase cuya distribución englobe de forma más natural el nuevo punto. El vector `[22.0, 0.55]` se encuentra espacial y estadísticamente muy cerca de las medias calculadas para la Clase 0 (Perros) y a múltiples desviaciones estándar de la Clase 1 (Caballos). Por lo tanto, la ecuación exponencial gaussiana debe generar un valor de probabilidad alto para la primera clase y un valor tendiente a cero para la segunda.

**Resultados de Ejecución Reales**
Al ejecutar el código, la red calcula las distribuciones estadísticas con precisión y clasifica el patrón nuevo basándose en la maximización de la probabilidad empírica:

* **Estadísticas calculadas (Media, Varianza) por clase:**
* Perros: Peso=(20.0, 16.666...), Altura=(0.5, 0.00666...)
* Caballos: Peso=(410.0, 866.666...), Altura=(1.633..., 0.0155...)


* **Evaluando nuevo dato (Peso: 22.0, Altura: 0.55):**
* Probabilidades brutas: `{0: 3.528..., 1: 3.52...e-42}` *(La probabilidad de pertenecer a la clase 1 es exponencialmente cercana a cero)*.


* **El clasificador Gausiano predice que es un:** **Perro******

**(13) Clasificador Óptimo**

**Explicación del Código Fuente**
El algoritmo implementa un Clasificador Óptimo, clasificado dentro de los modelos teóricos como una red de aprendizaje supervisado diseñada para procesar patrones de entrada binaria. A diferencia de los modelos geométricos, este código se basa en el teorema de Bayes para minimizar la probabilidad de error en la clasificación. En la fase de entrenamiento, el algoritmo calcula dos métricas estadísticas: las probabilidades a priori (la frecuencia general de aparición de cada clase en el conjunto de entrenamiento) y las probabilidades condicionales (la probabilidad de que un bit específico sea 1 dado que pertenece a una clase particular). Para evitar el problema de probabilidades condicionales absolutas de cero que anularían el producto estadístico final, el código aplica la técnica de suavizado de Laplace sumando 1 a los casos positivos y 2 al total de la clase. En la fase de predicción (`predecir_optimo`), el algoritmo asume independencia condicional entre los bits y multiplica la probabilidad a priori de cada clase por la probabilidad de los bits observados en el nuevo patrón.

**Datos de Trabajo**
El bloque opera sobre un conjunto de entrenamiento discreto y binario compuesto por 7 patrones de 3 características (bits), donde se provee la información que especifica la clase correcta para cada uno de ellos. Los primeros cuatro vectores pertenecen a la 'Clase X' (caracterizados estadísticamente por terminar en el bit 0), y los tres restantes pertenecen a la 'Clase Y' (caracterizados por terminar siempre en el bit 1). El algoritmo es sometido a prueba introduciendo un vector no visto previamente: `[0, 1, 0]`.

**Salida Teórica Esperada**
Al ser un clasificador estadístico óptimo, el algoritmo toma decisiones ponderando la evidencia matemática acumulada. Dado que el patrón de prueba `[0, 1, 0]` tiene como último bit un 0 (un rasgo fuertemente asociado a la Clase X y completamente ausente en la Clase Y en los datos de entrenamiento), la probabilidad matemática final acumulada debe inclinarse considerablemente a favor de la Clase X.

**Resultados de Ejecución Reales**
Al compilar y ejecutar el algoritmo, el modelo extrae las distribuciones de probabilidad del conjunto de datos y evalúa el nuevo patrón con precisión estadística, asignándolo correctamente:

* **Probabilidades estadísticas extraídas:**
* Probabilidad a priori de Clase X: 0.57
* Probabilidad condicional de bits para Clase X: `[0.67, 0.5, 0.17]`
* Probabilidad a priori de Clase Y: 0.43
* Probabilidad condicional de bits para Clase Y: `[0.6, 0.6, 0.8]`


* **Evaluando nuevo patrón binario:** `[0, 1, 0]`
* -> Probabilidad de pertenecer a Clase X: **0.0794**
* -> Probabilidad de pertenecer a Clase Y: **0.0206**


* **El Clasificador Óptimo asigna el patrón a la:** **Clase X**
