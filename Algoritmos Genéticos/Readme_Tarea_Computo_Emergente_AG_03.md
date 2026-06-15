# Documentación del Código — Tarea 03: Algoritmo Genético Aplicado al Problema del Viajante de Comercio (TSP)

> Archivo original: `Tarea_Cómputo_Emergente_AG_03.ipynb`

---

## Visión General

Este notebook implementa un **Algoritmo Genético (AG)** para resolver el **Problema del Viajante de Comercio (TSP)** con 15 ciudades. A diferencia de la tarea anterior (AGS con representación binaria para F(x)=x²), este problema requiere una **codificación por permutación**, lo que implica operadores genéticos especializados (Order Crossover y mutación por intercambio) para garantizar que cada solución sea válida — es decir, que cada ciudad aparezca exactamente una vez en la ruta.

El notebook explora la configuración del AG, su ejecución, la visualización de rutas, y un análisis de sensibilidad variando las probabilidades de cruce (Pc) y mutación (Pm).

### Características del problema

| Característica | Valor |
|---|---|
| Problema | TSP — minimizar distancia total del recorrido |
| Codificación | Permutación de enteros (0 a N-1) |
| Número de ciudades | 15 (12 en círculo + 3 interiores) |
| Función de costo | Suma de distancias euclidianas entre ciudades consecutivas |
| Función de aptitud | 1 / distancia_total (inversión para maximizar) |
| Selección | Rueda de ruleta (RWS) |
| Cruce | Order Crossover (OX), Pc=0.9 |
| Mutación | Swap (intercambio de dos ciudades), Pm=0.1 |
| Elitismo | 2 individuos conservados |
| Población | 100 |
| Generaciones | 500 |

---

## Celda 0 — Descripción del Problema (Markdown)

**Tipo:** Markdown

**Contenido:** Presenta el título de la tarea y una descripción completa del Problema del Viajante de Comercio (TSP). Explica que el objetivo es encontrar el recorrido de longitud mínima que visite todas las ciudades exactamente una vez y regrese al punto de partida. Detalla la codificación usada (permutación de enteros del 1 al N), la función de costo (suma de distancias entre ciudades consecutivas), el objetivo de minimización, y menciona que se graficará la mejor solución encontrada a lo largo de las generaciones. Esta celda establece el marco conceptual para todo el notebook.

---

## Celda 1 — Título de Sección 1 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado de sección "1. Importaciones y configuración". Sirve como separador visual para indicar el inicio del bloque de configuración del entorno de trabajo.

---

## Celda 2 — Importaciones y Configuración de Matplotlib

**Tipo:** Código

```python
import numpy as np
import matplotlib.pyplot as plt
import pygad
import math
import random
import copy
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
```

**Descripción:** Importa todas las bibliotecas necesarias y configura los parámetros globales de visualización:

- **`numpy` (`np`)**: Biblioteca fundamental para cálculos numéricos y manipulación de arreglos. Se usa para calcular distancias euclidianas, generar posiciones de ciudades, y manipular la matriz de distancias.
- **`matplotlib.pyplot` (`plt`)**: Biblioteca de graficación para visualizar las ciudades, las rutas, y la evolución de la distancia. Se configura con resolución de 120 DPI, fuente DejaVu Sans, y manejo correcto del signo menos.
- **`pygad`**: Librería de algoritmos genéticos que proporciona la clase `pygad.GA`. En este notebook se usa con operadores personalizados (cruce OX y mutación swap) adaptados a la representación por permutación.
- **`math`**: Funciones matemáticas estándar, usado para cálculos de techo (`ceil`) al organizar los subplots de visualización.
- **`random`**: Módulo de generación de números aleatorios de Python, utilizado extensivamente en los operadores genéticos personalizados (selección de puntos de corte en OX, selección de índices para swap, y mezcla de individuos iniciales).
- **`copy`**: Módulo de copia profunda, importado como utilidad potencial para manipular soluciones sin efectos secundarios.
- **`warnings.filterwarnings("ignore")`**: Suprime advertencias para mantener la salida limpia.
- **`%matplotlib inline`**: Directiva de Jupyter para mostrar las gráficas directamente en el notebook.

---

## Celda 3 — Título de Sección 2 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "2. Definición de las ciudades". Explica que se definen 15 ciudades en un plano 2D: 12 dispuestas alrededor de un círculo (configuración donde la solución óptima es el perímetro) y 3 ciudades interiores que hacen el problema más interesante al introducir atajos potenciales.

---

## Celda 4 — Definición de las 15 Ciudades

**Tipo:** Código

```python
NUM_CITIES = 15
CITY_RADIUS = 10

np.random.seed(42)
cities = []

for i in range(12):
    angle = 2 * np.pi * i / 12
    x = CITY_RADIUS * np.cos(angle)
    y = CITY_RADIUS * np.sin(angle)
    cities.append((x, y))

cities.append((3.0, 2.0))
cities.append((-4.0, 3.0))
cities.append((1.0, -5.0))

cities = np.array(cities)
city_labels = list(range(1, NUM_CITIES + 1))
```

**Descripción:** Define las posiciones de las 15 ciudades en un plano 2D:

1. **Constantes**: `NUM_CITIES = 15` (total de ciudades) y `CITY_RADIUS = 10` (radio del círculo sobre el que se distribuyen las primeras 12).
2. **Ciudades circulares (12)**: Se generan 12 ciudades uniformemente espaciadas sobre una circunferencia de radio 10. Para cada ciudad `i`, el ángulo es `2π × i / 12`, y las coordenadas son `(R × cos(θ), R × sin(θ))`. Con semilla 42, estas posiciones son deterministas y reproducibles.
3. **Ciudades interiores (3)**: Se añaden manualmente tres ciudades dentro del círculo en posiciones estratégicas: (3.0, 2.0), (-4.0, 3.0) y (1.0, -5.0). Estas ciudades rompen la simetría circular y crean la posibilidad de rutas más cortas que atraviesan el interior del círculo, haciendo el problema más desafiante.
4. **Conversión a arreglo NumPy**: La lista de tuplas se convierte a un arreglo NumPy de forma (15, 2) para facilitar cálculos vectorizados de distancia.
5. **Etiquetas**: Se crean etiquetas numéricas del 1 al 15 para la visualización.

Se imprimen las posiciones de todas las ciudades para verificación.

---

## Celda 5 — Visualización de las Ciudades

**Tipo:** Código

**Descripción:** Genera un gráfico de dispersión que muestra la ubicación de las 15 ciudades en el plano 2D. Cada ciudad se representa con un punto rojo con borde negro, y se anota su número junto a ella. Se configura el aspecto igual (`set_aspect('equal')`) para que las distancias euclidianas se perciban correctamente sin distorsión. La cuadrícula con transparencia 0.3 facilita la lectura de las coordenadas. Esta visualización permite verificar la distribución de las ciudades y entender la geometría del problema antes de ejecutar el AG.

---

## Celda 6 — Título de Sección 3 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "3. Matriz de distancias". Indica que se calculará la distancia euclidiana entre cada par de ciudades.

---

## Celda 7 — Cálculo de la Matriz de Distancias

**Tipo:** Código

```python
def compute_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

dist_matrix = compute_distance_matrix(cities)
```

**Descripción:** Define y ejecuta la función que calcula la **matriz de distancias** entre todos los pares de ciudades:

- **`compute_distance_matrix(cities)`**: Crea una matriz cuadrada de tamaño n×n (15×15 en este caso). Para cada par de ciudades (i, j), calcula la **distancia euclidiana** usando `np.linalg.norm(cities[i] - cities[j])`, que equivale a √((x₂-x₁)² + (y₂-y₁)²). La diagonal principal contiene ceros (distancia de una ciudad a sí misma), y la matriz es simétrica (distancia de i a j = distancia de j a i).
- **`dist_matrix`**: Variable global que almacena la matriz calculada. Esta matriz se usa intensivamente en la función de aptitud y en la visualización de rutas, permitiendo consultas de distancia en tiempo O(1) sin recalcular.

Se imprime la matriz redondeada a 2 decimales para inspección visual.

---

## Celda 8 — Título de Sección 4 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "4. Función de aptitud (fitness)". Explica que la función de costo es la distancia total del recorrido, y que como PyGAD maximiza la aptitud, se usa `fitness = 1 / distancia_total` para convertir el problema de minimización en uno de maximización.

---

## Celda 9 — Funciones de Aptitud y Costo

**Tipo:** Código

### `route_distance(route, dist_matrix)`

```python
def route_distance(route, dist_matrix):
    total = 0.0
    n = len(route)
    for i in range(n):
        city_from = int(route[i])
        city_to = int(route[(i + 1) % n])
        total += dist_matrix[city_from][city_to]
    return total
```

Función de **costo** que calcula la distancia total de un recorrido. Recorre la ruta secuencialmente, sumando la distancia entre cada par de ciudades consecutivas. El operador módulo `(i + 1) % n` garantiza que la última ciudad se conecte de vuelta con la primera, cerrando el ciclo (el viajante regresa al punto de partida). Los índices se convierten a `int` para asegurar compatibilidad con la matriz de distancias.

### `fitness_tsp(ga_instance, solution, solution_idx)`

```python
def fitness_tsp(ga_instance, solution, solution_idx):
    distance = route_distance(solution, dist_matrix)
    if distance == 0:
        return float('inf')
    return 1.0 / distance
```

Función de **aptitud** compatible con PyGAD. Calcula la distancia total de la ruta y retorna su inversa (`1 / distancia`). Esta transformación es necesaria porque PyGAD maximiza la aptitud por defecto, mientras que el TSP requiere minimizar la distancia. Una distancia menor produce una aptitud mayor, lo que guía al AG hacia rutas más cortas. Se incluye una verificación de distancia cero (que no debería ocurrir con ciudades distintas) para evitar división por cero.

---

## Celda 10 — Título de Sección 5 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "5. Operadores genéticos para permutaciones". Explica la necesidad crítica de operadores especializados: los operadores estándar de cruce y mutación (como los usados en la tarea anterior para representación binaria) generarían soluciones inválidas al producir duplicados o ciudades faltantes en la ruta. Define los dos operadores que se usarán:
- **Cruce**: Order Crossover (OX) — preserva el orden relacional de las ciudades.
- **Mutación**: Inversión de subrutas (swap de dos ciudades) — preserva la permutación.

---

## Celda 11 — Operadores Genéticos para Permutaciones

**Tipo:** Código

### `order_crossover(parents, offspring_size, ga_instance)`

```python
def order_crossover(parents, offspring_size, ga_instance):
```

Implementa el operador de **Order Crossover (OX)**, un cruce diseñado específicamente para representaciones por permutación que garantiza que el descendiente sea una permutación válida. El procedimiento para generar cada descendiente es:

1. **Selección de padres**: Se toman dos padres alternados del arreglo `parents` (padre 1 = `parents[k % num_parents]`, padre 2 = `parents[(k+1) % num_parents]`).
2. **Puntos de corte**: Se seleccionan dos puntos de corte aleatorios (`start`, `end`) que definen un segmento del cromosoma.
3. **Copia del segmento**: El segmento entre los puntos de corte se copia directamente del padre 1 al hijo, manteniendo las posiciones exactas.
4. **Completar con el padre 2**: Las ciudades restantes (que no están en el segmento copiado) se toman del padre 2 **en el orden en que aparecen**, y se colocan en las posiciones vacías del hijo comenzando después del segundo punto de corte. Este mecanismo preserva el orden relacional de las ciudades del padre 2, lo que transmite información sobre la secuencia de visitas.

La firma de la función cumple con la convención de PyGAD para operadores de cruce personalizados: recibe los padres, el tamaño de la descendencia y la instancia del GA, y retorna un arreglo NumPy con los descendientes.

### `swap_mutation(offspring, ga_instance)`

```python
def swap_mutation(offspring, ga_instance):
```

Implementa la **mutación por intercambio (swap)**, que es segura para permutaciones porque simplemente intercambia la posición de dos ciudades seleccionadas aleatoriamente. El procedimiento es:

1. Para cada cromosoma en la descendencia, se genera un número aleatorio uniforme.
2. Si este número es menor que la probabilidad de mutación `pm`, se seleccionan dos índices aleatorios distintos y se intercambian las ciudades en esas posiciones.
3. Si el número aleatorio es mayor o igual a `pm`, el cromosoma permanece sin cambios.

A diferencia de la mutación bit-flip usada en la tarea anterior (que invertía genes individuales), la mutación swap opera sobre pares de genes y preserva automáticamente la propiedad de permutación — no puede crear duplicados ni omitir ciudades.

---

## Celda 12 — Título de Sección 6 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "6. Callback para registrar la evolución". Describe los observables que se registran en cada generación: la distancia de la mejor ruta encontrada y la distancia promedio de la población. También menciona que se almacenan las mejores rutas periódicamente para su visualización posterior.

---

## Celda 13 — Callback de Evolución

**Tipo:** Código

```python
best_distance_history = []
avg_distance_history = []
best_routes_log = {}
SAVE_ROUTE_EVERY = 10

def on_generation_tsp(ga_instance):
```

**Descripción:** Define variables globales y el callback que se ejecuta al finalizar cada generación del AG:

### Variables globales

- **`best_distance_history`**: Lista que almacena la distancia de la mejor ruta en cada generación. Equivalente al observable "a" (mejor individuo) de la tarea anterior, pero ahora en términos de distancia (menor es mejor).
- **`avg_distance_history`**: Lista que almacena la distancia promedio de toda la población en cada generación. Equivalente al observable "b".
- **`best_routes_log`**: Diccionario que mapea números de generación a las mejores rutas encontradas en esas generaciones. Permite visualizar cómo evoluciona la ruta geométricamente.
- **`SAVE_ROUTE_EVERY = 10`**: Frecuencia con la que se guarda una copia de la mejor ruta para visualización posterior (cada 10 generaciones).

### `on_generation_tsp(ga_instance)`

Callback que realiza las siguientes operaciones en cada generación:

1. **Obtiene la mejor solución** actual usando `ga_instance.best_solution()`, que retorna (cromosoma, aptitud, índice).
2. **Calcula la distancia** de la mejor ruta usando `route_distance()`, y la agrega a `best_distance_history`.
3. **Calcula la distancia promedio** de toda la población recorriendo cada individuo, calculando su distancia, y promediando con `np.mean()`. Se agrega a `avg_distance_history`.
4. **Guarda la ruta periódicamente**: Si la generación actual es múltiplo de `SAVE_ROUTE_EVERY` (o es la primera), se almacena una copia de la mejor ruta en `best_routes_log` usando como clave el número de generación.

Se usa `global` para modificar las listas y diccionarios globales desde dentro de la función.

---

## Celda 14 — Título de Sección 7 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "7. Configuración y ejecución del Algoritmo Genético". Presenta una tabla con los parámetros del AG para el TSP: población=100, generaciones=500, padres para cruce=50, selección RWS, cruce OX con Pc=0.9, mutación swap con Pm=0.1, y elitismo de 2 individuos.

---

## Celda 15 — Configuración del AG para el TSP

**Tipo:** Código

```python
POP_SIZE = 100
NUM_GENERATIONS = 500
NUM_PARENTS_MATING = 50
PC = 0.9
PM = 0.1
KEEP_ELITISM = 2
```

**Descripción:** Define los hiperparámetros del AG y crea la instancia de PyGAD. Este es un bloque extenso que realiza varias operaciones críticas:

### 1. Parámetros del AG

| Parámetro | Valor | Justificación |
|---|---|---|
| `POP_SIZE` | 100 | Población mayor que en la tarea anterior (20→100) debido a que el espacio de búsqueda es mucho más grande (14!/2 rutas posibles) |
| `NUM_GENERATIONS` | 500 | Más generaciones para permitir convergencia en un espacio combinatorio |
| `NUM_PARENTS_MATING` | 50 | La mitad de la población seleccionada como padres |
| `PC` | 0.9 | Probabilidad de cruce alta para favorecer la recombinación de segmentos de rutas |
| `PM` | 0.1 | Probabilidad de mutación moderada, mayor que en la tarea anterior (0.002→0.1) porque swap muta por cromosoma, no por gen |
| `KEEP_ELITISM` | 2 | Se conservan los 2 mejores individuos entre generaciones |

### 2. Población inicial

```python
initial_population = []
for _ in range(POP_SIZE):
    individual = list(range(NUM_CITIES))
    random.shuffle(individual)
    initial_population.append(individual)
```

Se genera la población inicial creando 100 permutaciones aleatorias de los enteros [0, 1, ..., 14]. Cada individuo es una permutación válida generada con `random.shuffle()`. A diferencia de la tarea anterior donde se usaba `gene_space=[0, 1]` para representación binaria, aquí se proporciona una `initial_population` explícita a PyGAD porque los genes no son independientes entre sí (deben formar una permutación).

### 3. Creación de la instancia de PyGAD

Se configuran los parámetros del AG, incluyendo los operadores personalizados (`crossover_type=order_crossover`, `mutation_type=swap_mutation`), la función de aptitud TSP, y el callback de evolución. Se establece `allow_duplicate_genes=True` (necesario para la representación por permutación) y `gene_type=int` para que los genes sean enteros.

---

## Celda 16 — Ejecución del AG

**Tipo:** Código

```python
ga_tsp.run()
```

**Descripción:** Ejecuta el ciclo evolutivo completo del algoritmo genético. Durante 500 generaciones, el AG realiza: selección de padres por rueda de ruleta, cruce Order Crossover con probabilidad 0.9, mutación swap con probabilidad 0.1, conservación de élite (2 individuos), y evaluación de aptitud. El callback `on_generation_tsp` se ejecuta al final de cada generación, registrando los observables. La ejecución puede tardar varios segundos dependiendo del hardware.

---

## Celda 17 — Título de Sección 8 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "8. Mejor solución encontrada". Indica que se mostrará la mejor ruta, su distancia total, y se comparará con las distancias promedio inicial y final.

---

## Celda 18 — Resultados de la Mejor Solución

**Tipo:** Código

**Descripción:** Extrae y presenta los resultados del AG finalizado:

1. **Mejor solución**: Se obtiene con `ga_tsp.best_solution()`, que retorna el cromosoma (ruta), la aptitud y el índice.
2. **Ruta**: Se convierte a enteros y se muestra en dos formatos — índices base 0 (interno) y números de ciudad base 1 (legible).
3. **Distancia total**: Se calcula con `route_distance()`.
4. **Aptitud**: Se muestra el valor de 1/distancia.
5. **Comparación**: Se muestra la distancia promedio inicial (generación 1), la distancia promedio final (generación 500), y el porcentaje de mejora respecto al promedio inicial.

Esta celda permite evaluar cuantitativamente el desempeño del AG y verificar que la distancia de la mejor ruta es significativamente menor que la del promedio poblacional inicial.

---

## Celda 19 — Título de Sección 9 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "9. Evolución de la distancia a lo largo de las generaciones". Indica que se graficarán dos observables: (a) la distancia de la mejor ruta y (b) la distancia promedio de la población.

---

## Celda 20 — Gráfica de Evolución del AG

**Tipo:** Código

**Descripción:** Genera una gráfica de líneas que muestra la evolución del AG a lo largo de las 500 generaciones:

- **Línea azul** (`'b-'`): Distancia de la mejor ruta encontrada en cada generación (observable "a"). Se espera una curva descendente que se estabiliza, indicando que el AG encuentra rutas progresivamente más cortas.
- **Línea roja** (`'r-'`, con transparencia 0.7): Distancia promedio de toda la población en cada generación (observable "b"). Muestra cómo la población en general converge hacia soluciones de menor distancia.

Se configuran etiquetas de ejes, título con los parámetros del AG, leyenda y cuadrícula. La gráfica permite identificar la velocidad de convergencia y si el AG sigue mejorando o se ha estancado.

---

## Celda 21 — Título de Sección 10 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "10. Visualización de la mejor ruta encontrada".

---

## Celda 22 — Función de Visualización de Rutas

**Tipo:** Código

```python
def plot_route(cities, route, title="Ruta del viajante", ax=None):
```

**Descripción:** Define la función reutilizable `plot_route()` que grafica una ruta del TSP en un plano 2D. Sus parámetros son:

- **`cities`**: Arreglo de posiciones (x, y) de las ciudades.
- **`route`**: Permutación que define el orden de visita.
- **`title`**: Título del gráfico (por defecto "Ruta del viajante").
- **`ax`**: Eje de Matplotlib opcional, que permite reutilizar la función en subgráficas.

El procedimiento de graficación es:

1. **Cierre del ciclo**: Se añade la primera ciudad al final de la ruta (`route_list = list(route) + [route[0]]`) para dibujar la línea de regreso al origen.
2. **Línea de ruta**: Se traza una línea azul continua (`'o-'`) que conecta las ciudades en el orden de la ruta, con marcadores rojos y bordes negros.
3. **Etiquetas de ciudades**: Se anota el número de cada ciudad con texto azul oscuro en negrita, desplazado ligeramente para no superponerse con los marcadores.
4. **Marcador de inicio**: Se marca la ciudad de inicio con un triángulo verde (`'g^'`) de mayor tamaño para identificar fácilmente el punto de partida.
5. **Información**: Se muestra la distancia total de la ruta en el título, se configura aspecto igual, cuadrícula y leyenda.

La función retorna el objeto `ax` para permitir encadenamiento o personalización adicional.

---

## Celda 23 — Gráfica de la Mejor Ruta

**Tipo:** Código

**Descripción:** Invoca `plot_route()` para graficar la mejor ruta encontrada por el AG después de 500 generaciones. Se usa el título "Mejor ruta encontrada (Gen 500)". Esta visualización es el resultado principal del notebook: muestra geométricamente el recorrido óptimo (o cercano a óptimo) que el AG ha descubierto, permitiendo verificar visualmente si la ruta tiene sentido (sin cruces excesivos, trayectoria suave).

---

## Celda 24 — Título de Sección 11 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "11. Evolución visual de la mejor ruta en el tiempo". Explica que se mostrarán las mejores rutas encontradas en diferentes generaciones, permitiendo observar cómo el algoritmo mejora progresivamente la solución.

---

## Celda 25 — Visualización Multi-Generación de Rutas

**Tipo:** Código

**Descripción:** Genera una cuadrícula de subgráficas que muestra la evolución visual de la mejor ruta a lo largo de las generaciones. El proceso es:

1. **Selección de generaciones**: Se obtienen las generaciones registradas en `best_routes_log` (guardadas cada 10 generaciones). Si hay más de 6, se seleccionan 6 equiespaciadas usando `np.linspace()`. Se asegura que la generación final (500) esté incluida.
2. **Layout de subgráficas**: Se calcula el número de filas y columnas (3 columnas, filas según necesidad) y se crea la figura con `plt.subplots()`.
3. **Graficación**: Para cada generación seleccionada, se llama a `plot_route()` pasando el eje correspondiente, mostrando la ruta y su distancia en ese punto de la evolución.
4. **Subgráficas vacías**: Si el número de generaciones seleccionadas no llena la cuadrícula, las subgráficas sobrantes se ocultan.

Esta visualización es especialmente reveladora: permite ver cómo el AG elimina progresivamente los cruces de ruta, acorta los segmentos, y converge hacia una ruta más eficiente generación tras generación.

---

## Celda 26 — Título de Sección 12 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "12. Comparación de la ruta inicial vs. la ruta final". Indica que se comparará la primera ruta aleatoria con la mejor ruta encontrada por el AG.

---

## Celda 27 — Comparación Visual: Ruta Inicial vs. Final

**Tipo:** Código

**Descripción:** Genera una comparación lado a lado de dos rutas:

1. **Ruta inicial (izquierda)**: Toma el primer individuo de la población inicial (`initial_population[0]`), que es una permutación aleatoria generada al inicio. Calcula y muestra su distancia.
2. **Mejor ruta (derecha)**: Muestra la mejor ruta encontrada por el AG después de 500 generaciones.

Ambas rutas se grafican usando `plot_route()` en subgráficas adyacentes. Al final se imprime un resumen numérico con la mejora porcentual:

```
Mejora respecto a la ruta inicial: XX.XX%
  Distancia inicial: XXX.XX
  Distancia final:   XXX.XX
  Reducción:         XXX.XX
```

Esta comparación cuantifica el beneficio del AG: muestra cuánta reducción de distancia se obtuvo respecto a una solución aleatoria, demostrando la efectividad del proceso evolutivo.

---

## Celda 28 — Título de Sección 13 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado "13. Análisis de sensibilidad: Efecto de Pm y Pc". Introduce la sección de experimentación paramétrica para evaluar cómo cambian los resultados al variar los hiperparámetros del AG.

---

## Celda 29 — Función de Experimento TSP

**Tipo:** Código

```python
def run_tsp_experiment(pop_size=100, num_generations=300, pc=0.9, pm=0.1,
                      keep_elitism=2, seed=42):
```

**Descripción:** Define la función `run_tsp_experiment()` que encapsula toda la lógica de configuración y ejecución del AG para el TSP, permitiendo realizar experimentos paramétricos de forma sistemática. Sus parámetros son:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `pop_size` | 100 | Tamaño de la población |
| `num_generations` | 300 | Número de generaciones (reducido a 300 vs. 500 para acelerar los experimentos) |
| `pc` | 0.9 | Probabilidad de cruce |
| `pm` | 0.1 | Probabilidad de mutación |
| `keep_elitism` | 2 | Número de élites conservados |
| `seed` | 42 | Semilla aleatoria |

Dentro de la función:

1. **Reinicia las variables globales** (`best_distance_history`, `avg_distance_history`, `best_routes_log`) a sus estados vacíos para que cada experimento registre datos limpios.
2. **Fija semillas aleatorias** tanto de `random` como de `numpy` para reproducibilidad.
3. **Genera la población inicial** con las semillas fijadas, creando permutaciones aleatorias.
4. **Configura y ejecuta** una instancia de PyGAD con los parámetros especificados.
5. **Retorna un diccionario** con: la mejor distancia encontrada, la mejor ruta, el historial de mejor distancia por generación, y el historial de distancia promedio por generación.

Esta función es análoga a `run_aga()` de la tarea anterior, pero adaptada al TSP con representación por permutación.

---

## Celda 30 — Título de Sección 13.1 (Markdown)

**Tipo:** Markdown

**Contenido:** Subtítulo "13.1 Efecto de la Probabilidad de Cruce (Pc)".

---

## Celda 31 — Experimento: Variación de Pc en el TSP

**Tipo:** Código

```python
pc_values_tsp = [0.9, 0.7, 0.5]
results_pc = {}

for pc in pc_values_tsp:
    result = run_tsp_experiment(pc=pc, pm=0.1, seed=42)
    results_pc[pc] = result
```

**Descripción:** Ejecuta tres experimentos del AG para el TSP, variando la probabilidad de cruce (0.9, 0.7 y 0.5) mientras se mantiene Pm=0.1 fijo. Para cada valor de Pc, se ejecuta el AG completo (300 generaciones, población 100) con semilla 42 para reproducibilidad. Los resultados se almacenan en el diccionario `results_pc`, donde cada clave es el valor de Pc y el valor contiene la mejor distancia, la mejor ruta, y los historiales de evolución. Se imprime la mejor distancia alcanzada en cada ejecución para comparación rápida.

---

## Celda 32 — Gráficas Comparativas: Efecto de Pc en el TSP

**Tipo:** Código

**Descripción:** Genera dos subgráficas lado a lado que comparan el efecto de Pc:

- **Subgráfica izquierda** (observable "a"): Mejor distancia por generación para cada valor de Pc. Se espera que Pc=0.9 produzca la convergencia más rápida hacia distancias menores, mientras que Pc=0.5 converja más lentamente al reducirse la recombinación de segmentos de ruta.
- **Subgráfica derecha** (observable "b"): Distancia promedio de la población por generación para cada valor de Pc. Muestra cómo la población en general se beneficia de una mayor tasa de cruce.

Cada curva está etiquetada con su valor de Pc. El título general indica "Efecto de la Probabilidad de Cruce (Pc) en el TSP".

---

## Celda 33 — Título de Sección 13.2 (Markdown)

**Tipo:** Markdown

**Contenido:** Subtítulo "13.2 Efecto de la Probabilidad de Mutación (Pm)".

---

## Celda 34 — Experimento: Variación de Pm en el TSP

**Tipo:** Código

```python
pm_values_tsp = [0.05, 0.1, 0.3]
results_pm = {}

for pm in pm_values_tsp:
    result = run_tsp_experiment(pc=0.9, pm=pm, seed=42)
    results_pm[pm] = result
```

**Descripción:** Ejecuta tres experimentos del AG para el TSP, variando la probabilidad de mutación (0.05, 0.1 y 0.3) mientras se mantiene Pc=0.9 fijo. Los valores de Pm son diferentes a los de la tarea anterior (0.002, 0.01, 0.1) porque la mutación swap opera por cromosoma completo (no por gen individual), por lo que las probabilidades efectivas son distintas. Se usa semilla 42 para reproducibilidad. Los resultados se almacenan en `results_pm` y se imprime la mejor distancia de cada ejecución.

---

## Celda 35 — Gráficas Comparativas: Efecto de Pm en el TSP

**Tipo:** Código

**Descripción:** Genera dos subgráficas lado a lado que comparan el efecto de Pm:

- **Subgráfica izquierda** (observable "a"): Mejor distancia por generación para cada valor de Pm. Se espera que Pm=0.05 y Pm=0.1 produzcan buenos resultados (con Pm=0.1 explorando más), mientras que Pm=0.3 introduzca demasiada aleatoriedad, degradando la convergencia.
- **Subgráfica derecha** (observable "b"): Distancia promedio de la población por generación para cada valor de Pm. Con Pm alto (0.3), la población mantendrá mayor diversidad pero no convergerá tan eficientemente.

Cada curva está etiquetada con su valor de Pm. El título general indica "Efecto de la Probabilidad de Mutación (Pm) en el TSP".

---

## Celda 36 — Resumen y Conclusiones (Markdown)

**Tipo:** Markdown

**Contenido:** Presenta las conclusiones del notebook, organizadas en tres secciones:

### Resultados principales

El AG logró encontrar una ruta considerablemente mejor que una selección aleatoria, demostrando la capacidad de los algoritmos evolutivos para problemas de optimización combinatoria con 15 ciudades.

### Observaciones clave

1. **Representación por permutación**: Fue esencial usar operadores especializados (OX, swap) para garantizar soluciones válidas.
2. **Convergencia**: El AG converge rápidamente en las primeras generaciones, con mejoras incrementales posteriores.
3. **Efecto de Pc**: Una probabilidad de cruce alta (0.9) favorece la exploración del espacio de búsqueda al combinar segmentos de rutas diferentes.
4. **Efecto de Pm**: Una mutación moderada (0.1) ayuda a escapar de óptimos locales, mientras que valores muy altos degradan el rendimiento.
5. **Naturaleza NP-completa**: Para 15 ciudades, el espacio de búsqueda contiene 14!/2 ≈ 4.36 × 10¹⁰ posibles rutas. El AG explora eficientemente una fracción minúscula de este espacio y encuentra soluciones de buena calidad.

### Limitaciones

- El AG no garantiza encontrar la solución óptima global.
- Para más ciudades, se requiere ajustar parámetros y posiblemente incorporar técnicas híbridas (meméticos, enfriamiento simulado).

---

## Resumen de la Estructura del Notebook

| Sección | Celdas | Descripción |
|---|---|---|
| Descripción del problema | 0 | Título y marco conceptual del TSP |
| Importaciones | 1-2 | Librerías y configuración de matplotlib |
| Ciudades | 3-5 | Definición de 15 ciudades y visualización |
| Matriz de distancias | 6-7 | Cálculo de distancias euclidianas entre pares |
| Función de aptitud | 8-9 | Costo (distancia total) y aptitud (1/distancia) |
| Operadores genéticos | 10-11 | Order Crossover (OX) y mutación swap |
| Callback de evolución | 12-13 | Registro de observables por generación |
| Configuración y ejecución | 14-16 | Parámetros del AG, población inicial y ejecución |
| Mejor solución | 17-18 | Extracción y presentación de resultados |
| Evolución de la distancia | 19-20 | Gráfica de evolución (mejor + promedio) |
| Visualización de rutas | 21-23 | Función plot_route() y gráfica de la mejor ruta |
| Evolución visual | 24-25 | Cuadrícula de rutas en diferentes generaciones |
| Comparación inicial vs. final | 26-27 | Ruta aleatoria vs. mejor ruta encontrada |
| Análisis de sensibilidad | 28-35 | Experimentos variando Pc y Pm con gráficas |
| Conclusiones | 36 | Resumen de resultados, observaciones y limitaciones |

---

## Comparación con la Tarea 02 (AGS Binario)

| Aspecto | Tarea 02 (AGS) | Tarea 03 (TSP) |
|---|---|---|
| Problema | Maximizar F(x)=x² | Minimizar distancia total |
| Codificación | Binaria (6 o 13 bits) | Permutación de enteros |
| Cruce | Un punto (estándar) | Order Crossover (OX) |
| Mutación | Bit-flip por gen | Swap por cromosoma |
| Aptitud | Directa (x²) | Inversa (1/distancia) |
| Población | 20-50 | 100 |
| Generaciones | 100 | 500 |
| Elitismo | 0 | 2 |
| Pm | 0.002 (por gen) | 0.1 (por cromosoma) |

---

## Dependencias

| Paquete | Uso |
|---|---|
| `pygad` | Motor del algoritmo genético |
| `numpy` | Cálculos numéricos, arreglos, distancias |
| `matplotlib` | Visualización de ciudades, rutas y evolución |
| `math` | Cálculos de ceiling para subgráficas |
| `random` | Operadores genéticos personalizados (OX, swap) |
| `copy` | Utilidad de copia para manipulación de soluciones |
