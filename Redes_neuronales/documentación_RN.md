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
