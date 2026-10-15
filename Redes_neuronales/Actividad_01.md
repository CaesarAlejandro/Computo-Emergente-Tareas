### **1. Modelo de McCulloch-Pitts (Compuerta Lógica AND)**

El algoritmo implementa una función `mcculloch_pitts_and` que simula la unidad computacional neuronal más elemental utilizando neuronas de tipo binario. En este modelo inicial, la función básica del elemento de proceso es sumar entradas y producir una determinada salida si dicha suma supera un umbral establecido. Para este caso, se establecen dos parámetros estructurales que se mantienen fijos: los pesos sinápticos ($w_1 = 1$, $w_2 = 1$) y el nivel del umbral de activación ($\theta = 2$). La operación principal calcula la entrada total a la neurona mediante la suma ponderada de las señales:

$$y_{in} = \sum_{i=1}^{2} x_i w_i = (x_1 \cdot w_1) + (x_2 \cdot w_2)$$

Inmediatamente después, esta suma se evalúa a través de una función de activación del tipo escalón:

$$y = \begin{cases} 1 & \text{si } y_{in} \geq \theta \\ 0 & \text{si } y_{in} < \theta \end{cases}$$

Si la suma ponderada es mayor o igual al umbral, la neurona dispara y retorna 1; en caso contrario, retorna 0. Al mantener los pesos y umbrales inmutables, este algoritmo no ejecuta ningún ciclo de aprendizaje.

El bloque opera sobre una lista de patrones discretos de entrada puramente binarios: `[(0, 0), (0, 1), (1, 0), (1, 1)]`. Estas tuplas representan exhaustivamente las cuatro combinaciones de estados posibles para las dos variables independientes de una compuerta lógica.

De acuerdo con la tabla de verdad matemática para la función lógica AND, la neurona debe permanecer inactiva (salida 0) para las entradas `(0, 0)`, `(0, 1)` y `(1, 0)`. La salida teórica solo alcanza la activación (salida 1) en el escenario donde ambas entradas se disparan simultáneamente en `(1, 1)`.

Al iterar la función sobre la lista de patrones, el algoritmo procesa la aritmética interna y devuelve exactamente la tabla de verdad esperada para la función AND:

*   **Entrada:** `(0, 0)` $\rightarrow$ **Salida de la neurona:** `0` (Suma ponderada de 0, no supera el umbral $\theta = 2$).
*   **Entrada:** `(0, 1)` $\rightarrow$ **Salida de la neurona:** `0` (Suma ponderada de 1, no supera el umbral $\theta = 2$).
*   **Entrada:** `(1, 0)` $\rightarrow$ **Salida de la neurona:** `0` (Suma ponderada de 1, no supera el umbral $\theta = 2$).
*   **Entrada:** `(1, 1)` $\rightarrow$ **Salida de la neurona:** `1` (Suma ponderada de 2, iguala el umbral $\theta = 2$).

#### **Análisis de Separabilidad Lineal (AND)**
La capacidad del modelo para resolver la compuerta AND radica en que este es un problema linealmente separable[cite: 6]. Geométricamente, la ecuación de la suma ponderada igualada al umbral ($w_1 x_1 + w_2 x_2 = \theta$) define una recta en un plano bidimensional. Al sustituir los valores ($1 \cdot x_1 + 1 \cdot x_2 = 2$, o bien $x_2 = -x_1 + 2$), obtenemos una frontera de decisión lineal que aísla perfectamente el caso de activación verdadera $(1, 1)$ del resto de las combinaciones nulas.

---

### **1.5 Modelo de McCulloch-Pitts (Compuerta Lógica OR)**

La adaptación del modelo para la función OR requiere una modificación mínima pero fundamental en la función `mcculloch_pitts_or`: el umbral de activación se reduce ($\theta = 1$), manteniendo los pesos sinápticos idénticos al modelo anterior ($w_1 = 1$, $w_2 = 1$). Al igual que la configuración AND, este modelo es una simplificación teórica de una neurona que evalúa estímulos simples. Las ecuaciones matemáticas subyacentes de suma ponderada y activación escalón permanecen exactamente iguales. Al reducir el umbral, basta con que una sola de las entradas esté activa para que la suma ponderada iguale o supere la barrera de activación.

La tabla de verdad matemática para la función lógica OR dicta que la neurona debe activarse (salida 1) siempre que al menos una de las entradas sea 1. La salida teórica solo permanece inactiva (salida 0) en el escenario donde ambas entradas son nulas `(0, 0)`.

La iteración sobre los mismos patrones de entrada confirma el comportamiento de la compuerta OR:

*   **Entrada:** `(0, 0)` $\rightarrow$ **Salida de la neurona:** `0` (Suma ponderada de 0, no supera el umbral $\theta = 1$).
*   **Entrada:** `(0, 1)` $\rightarrow$ **Salida de la neurona:** `1` (Suma ponderada de 1, iguala el umbral $\theta = 1$).
*   **Entrada:** `(1, 0)` $\rightarrow$ **Salida de la neurona:** `1` (Suma ponderada de 1, iguala el umbral $\theta = 1$).
*   **Entrada:** `(1, 1)` $\rightarrow$ **Salida de la neurona:** `1` (Suma ponderada de 2, supera el umbral $\theta = 1$).

#### **Análisis de Separabilidad Lineal (OR)**
Al igual que la compuerta AND, la función OR es un problema linealmente separable. Al aplicar la ecuación del umbral ($1 \cdot x_1 + 1 \cdot x_2 = 1$, que se traduce en $x_2 = -x_1 + 1$), la frontera de decisión se desplaza hacia el origen. Esta nueva recta corta los ejes en $1$, aislando eficazmente el único punto de inactividad $(0, 0)$ de los tres puntos de activación $(0, 1), (1, 0)$ y $(1, 1)$.

Las implementaciones conjuntas demuestran la viabilidad teórica y práctica de simular operaciones lógicas básicas mediante aritmética ponderada. Se validó con éxito que el simple ajuste manual del umbral de activación (de $\theta = 2$ a $\theta = 1$) es suficiente para alterar la función lógica de la neurona sin cambiar sus pesos sinápticos.

Sin embargo, el modelo de McCulloch-Pitts presenta dos limitaciones críticas para arquitecturas computacionales modernas:

*   **Ausencia de Aprendizaje:** Carece de un algoritmo o regla de aprendizaje de adaptación de pesos; los parámetros ($w$ y $\theta$) no se ajustan ante el error, debiendo ser deducidos analíticamente e ingresados por el programador.
*   **Restricción de Linealidad:** Como todo modelo perceptivo de capa única, está limitado a problemas estrictamente linealmente separables. Es estructuralmente incapaz de trazar fronteras de decisión para problemas no lineales (como la compuerta XOR) a menos que se integren arquitecturas multicapa.

---

### **2. Perceptrón Simple (Algoritmo Perceptrónico)**

#### **Explicación del Código Fuente**
El algoritmo implementa una red unicapa con aprendizaje supervisado, orientada específicamente a la resolución de problemas linealmente separables actuando como un clasificador de patrones fijos. En este modelo, provisto con información que especifica la clase correcta, el ciclo inicia estableciendo los pesos sinápticos en cero ($w_j = 0$) y definiendo una razón de aprendizaje constante de $\alpha = 0.5$. Por cada patrón de entrenamiento, la red calcula la suma ponderada de las entradas:

$$y_{in} = \sum_{j=0}^{2} w_j x_j$$

Inmediatamente, se evalúa mediante una función de activación bipolar antisimétrica, la cual produce un 1 para valores positivos, -1 para negativos, o 0 (punto de indeterminación) si el valor es exactamente cero:

$$y = \begin{cases} 1 & \text{si } y_{in} > 0 \\ 0 & \text{si } y_{in} = 0 \\ -1 & \text{si } y_{in} < 0 \end{cases}$$

Si la salida calculada por la red ($y$) no coincide con la salida deseada ($d$), se activa un mecanismo de corrección que adapta las sinapsis sumando al peso actual una fracción proporcional al error y a la entrada:

$$w_j(n+1) = w_j(n) + \alpha \cdot d \cdot x_j(n)$$

El bucle de entrenamiento se detiene automáticamente en el momento en que los pesos logran procesar la totalidad del conjunto de datos sin cometer errores en una misma época.

#### **Datos de Trabajo**
El conjunto de datos representa la tabla de verdad de la compuerta lógica AND, pero codificada en formato bipolar (-1 para falso, 1 para verdadero). Cada vector de entrada tiene la estructura `[x0, x1, x2]`, donde el primer elemento ($x_0$) es un $1$ constante que cumple la función de entrada para la neurona de inclinación (o bias), desplazando la frontera de decisión en el plano. El vector `salidas_deseadas` establece que la única condición verdadera (1) ocurre cuando tanto $x_1$ como $x_2$ son positivos.

#### **Salida Teórica Esperada**
Dado que la compuerta AND es un problema en el cual los puntos de una clase pueden separarse geométricamente de la otra mediante una línea recta, la teoría establece que el perceptrón debe alcanzar la convergencia. El algoritmo encontrará matemáticamente al menos un hiperplano (representado por los pesos finales) capaz de clasificar correctamente todas las entradas sin caer en un bucle infinito.

#### **Resultados de Ejecución Reales**
Al ejecutar este código, el perceptrón detecta los errores iniciales y modifica los pesos progresivamente. La convergencia se alcanza de manera exitosa en pocas épocas, y la prueba de eficiencia final corrobora que la frontera lineal ajustada discrimina los valores a la perfección:

*   **Entrada:** `[-1, -1]` $\rightarrow$ **Salida de la red:** `-1`
*   **Entrada:** `[-1, 1]` $\rightarrow$ **Salida de la red:** `-1`
*   **Entrada:** `[1, -1]` $\rightarrow$ **Salida de la red:** `-1`
*   **Entrada:** `[1, 1]` $\rightarrow$ **Salida de la red:** `1`

#### **Análisis de Separabilidad Lineal**
El éxito en la convergencia del perceptrón demuestra empíricamente la separabilidad lineal del problema AND bipolar. La inclusión del término de sesgo (bias o $x_0$) resulta fundamental en este aspecto; al permitir que la frontera de decisión ($w_1 x_1 + w_2 x_2 + bias = 0$) no esté obligada a cruzar por el origen de coordenadas $(0,0)$, el algoritmo goza de los grados de libertad geométricos necesarios para trazar una recta óptima que aísla el único caso positivo $(1, 1)$ del resto de los patrones negativos.

#### **Análisis de Resultados y Limitaciones**
A diferencia de McCulloch-Pitts, el perceptrón supera la necesidad de establecer umbrales manuales gracias a su regla de aprendizaje, logrando adaptar sus pesos de forma autónoma basándose en la retroalimentación de sus propios errores. Sin embargo, comparte la misma limitación arquitectónica estructural: al ser una red de capa única, su capacidad de clasificación se circunscribe exclusivamente a problemas linealmente separables. Ante conjuntos de datos no lineales, como la función lógica XOR, el algoritmo iteraría indefinidamente sin lograr que el error llegue a cero, evidenciando la necesidad de implementar Perceptrones Multicapa (MLP) y algoritmos avanzados como Backpropagation para tareas de mayor complejidad geométrica.
