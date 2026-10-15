# Fundamentos de Computación Neuronal: Implementaciones en Python Puro

Esta carpeta del repositorio alberga un cuaderno de Google Colab enfocado en explorar las bases matemáticas y algorítmicas de la inteligencia artificial. A través de implementaciones estructuradas estrictamente en Python puro —sin la intervención de librerías de aprendizaje automático cómo podría ser scikit learn—, el objetivo es construir los algoritmos presentados sin la necesidad de usar funciones externas con dichos algoritmos integrados. Cabe destacar que para el desarrollo de los mismos se usaron tres librerías en total, dos de ellas pertenecientes a la librería estándar de Python (random y math) y una librería externa con la cual se haces los gráficos de los resultados (matplotlib). 

## Introducción a las Redes Neuronales Artificiales

Las Redes Neuronales Artificiales (RNA) son modelos matemáticos multiparamétricos no-lineales capaces de inducir una correspondencia entre conjuntos de patrones de información. En esencia, estos sistemas intentan modelar la estructura y el funcionamiento de algunos componentes del sistema nervioso animal.

En la biología, el encéfalo o cerebro está constituido por millones de neuronas que reciben señales, las suman y producen una determinada salida si la suma es mayor que un umbral determinado. Los modelos artificiales emulan este comportamiento utilizando una simplificación teórica de las neuronas a la que se llama "elemento de proceso". En estos modelos, cada señal de entrada se convierte al pasar por la conexión multiplicándose por un valor llamado "peso o fuerza de la conexión" (peso sináptico). Posteriormente, las señales pasan por un sumador que produce una suma ponderada y, a continuación, por una función de activación que determina la salida final de la unidad.

Lo que convierte a las RNA en dispositivos de cómputo poderosos es su capacidad de adaptación. En vez de ejecutar un programa de instrucciones secuencialmente como en un computador de von Neumann, las redes neuronales no se programan, sino que aprenden o se entrenan en la tarea a computar. El conocimiento se almacena y el aprendizaje se plasma mediante la modificación de la fuerza o peso sináptico de las distintas uniones entre neuronas.

Dependiendo de la forma en que el entorno influye en el proceso, se puede adoptar distintos paradigmas de aprendizaje:

*   **Aprendizaje Supervisado:** La red se entrena asociando un vector de entrada con un vector de salida deseada, por lo que son provistas con la información que especifica la clase correcta para los patrones durante el entrenamiento.
*   **Aprendizaje No Supervisado:** En su entrenamiento no se presentan las salidas objetivo que se desean asociar a cada patrón de entrada. La red debe descubrir por sí misma rasgos comunes, regularidades, correlaciones o categorías en los datos de entrada para autoorganizarse en función de las relaciones de similitud existentes.

## Modelos Desarrollados

A continuación, se presenta un desglose de los algoritmos implementados en este proyecto, desde las primeras unidades lógicas estáticas hasta arquitecturas capaces de autoorganizarse o resolver problemas no linealmente separables:

### Modelos Implementados

1. **Modelo de McCulloch-Pitts (1943):** Unidad lógica binaria estática que demuestra la capacidad de cómputo universal mediante pesos y umbrales fijos.

2. **Perceptrón Simple:** Red neuronal unidireccional con aprendizaje supervisado diseñada para procesar información de entrada continua. Es utilizada tradicionalmente para resolver problemas linealmente separables.

3. **ADALINE:** Red unidireccional organizada en capas y con aprendizaje supervisado (conocida como adalina). Elemento combinador adaptativo que minimiza el error cuadrático medio utilizando valores continuos antes de aplicar la función umbral.

4. **Perceptrón Multicapa:** Arquitectura unidireccional organizada en capas que incluye capas ocultas y emplea aprendizaje supervisado. Supera limitaciones lineales al utilizar el método de optimización de la función coste denominado descenso por el gradiente y el popular algoritmo de aprendizaje back-propagation (retropropagación).

5. **Red de Hopfield:** Modelo empleado como memoria asociativa y en la eliminación de ruido en imágenes. Procesa valores discretos o de entrada binaria. Almacena patrones en forma de estados estables utilizando la regla de Hebb, considerada un aprendizaje no supervisado de tipo hebbiano, si bien también se le clasifica en esquemas de redes entrenadas con supervisión.

6. **Algoritmo de Clustering (K-medio):** Modelo de red neuronal de aprendizaje no supervisado diseñado para la clasificación de patrones de entrada continua, agrupando datos por proximidad.

7. **Clasificador de K-vecinos más cercano (KNN):** Modelo de red neuronal de aprendizaje supervisado para el procesamiento de información de entrada continua. Clasifica nuevos patrones basándose en la votación mayoritaria de los datos memorizados más próximos en el espacio.

8. **Mapas de Kohonen (SOM):** Modelo neuronal no supervisado para patrones de entrada continua. Las neuronas se organizan y compiten para activarse (neurona vencedora), logrando una autoorganización que refleja un orden topológico para descubrir estructuras ocultas o clústeres en los datos sensoriales.

9. **Red de Hamming:** Clasificador neuronal con aprendizaje supervisado destinado al procesamiento de patrones de entrada binaria. Determina la similitud entre un patrón y las clases memorizadas.

10. **Matriz Memoria Asociativa (Willshaw, 1969):** Modelo fundamental de computación neuronal que funciona como memoria heteroasociativa, vinculando un patrón de entrada específico con un patrón de salida diferente.

11. **Clasificador Carpenter / Grossberg (ART-1):** Modelo de red neuronal de aprendizaje no supervisado utilizado para la clasificación de patrones de entrada binaria. Emplea un parámetro de vigilancia para crear nuevas categorías dinámicamente sin sobrescribir el conocimiento previo.

12. **Clasificador Gausiano:** Modelo de red neuronal de aprendizaje supervisado utilizado para el procesamiento de información de entrada continua. Modela las clases basándose en sus distribuciones estadísticas y la probabilidad de la campana de Gauss.

13. **Clasificador Óptimo:** Modelo de red neuronal de aprendizaje supervisado para el procesamiento de información de entrada binaria. Toma decisiones evaluando probabilidades a priori y condicionales.


El cuaderno traza una progresión histórica y pedagógica de las redes neuronales que inicia con los fundamentos lógicos y los primeros modelos iterativos unicapa limitados por la separabilidad lineal (McCulloch-Pitts, Perceptrón y ADALINE), para luego dar el salto matemático hacia fronteras complejas con el Perceptrón Multicapa y laretropropagación de errores. A partir de allí, la estructura transita hacia algoritmos basados en la memoria, la distancia geométrica y la autoorganización (Hopfield, K-medio, KNN, Kohonen y Hamming), los cuales abordan el aprendizaje desde perspectivas asociativas y biológicas. Finalmente, el documento culmina expandiendo el campo hacia modelos estadísticos y topologías dinámicas especializadas (Willshaw, Carpenter/Grossberg, Gausiano y Óptimo), demostrando cómo la computación neuronal integró el cálculo de probabilidades formales y la categorización avanzada para equilibrar la memoria y estimar la pertenencia a distintas clases.

Toda la lógica de los algoritmos la van a encontrar en el cuaderno de Jyputer dentro de esta carpeta. En los siguientes archivos .md se van a encontrar una descripción detallada de cada actividad así cómo los resultados obtenidos. 
