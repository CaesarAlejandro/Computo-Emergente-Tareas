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
