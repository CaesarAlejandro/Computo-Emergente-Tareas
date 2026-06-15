# Documentación del Código — Tarea 02: Algoritmo Genético Simple (AGS)

> Archivo original: `Tarea_Cómputo_Emergente_AG_02.ipynb`

---

## Visión General

Este notebook implementa un **Algoritmo Genético Simple (AGS)** con representación binaria para optimizar la función **F(x) = x²** en dos dominios:

1. **x entero** en el intervalo [0, 63], usando 6 bits.
2. **x real** en el intervalo [0, 63] con precisión de 2 cifras decimales, usando 13 bits.

El notebook explora además el efecto de variar la probabilidad de cruce (Pc) y la probabilidad de mutación (Pm) sobre la convergencia del algoritmo.

### Especificaciones del AGS

| Característica | Valor |
|---|---|
| Representación | Binaria |
| Población | Constante n |
| Inicialización | Aleatoria |
| Renormalización | Ninguna |
| Selección | Rueda de ruleta (RWS) |
| Operadores | Cruce un punto + Mutación |
| Sustitución | Total (no se conservan padres) |
| Generaciones | 100 |

---

## Celda 0 — Encabezado del Notebook (Markdown)

**Tipo:** Markdown

**Contenido:** Presenta el título de la tarea ("Tarea 02 - Algoritmo Genético Simple (AGS)") y una tabla con las especificaciones del AGS que se implementará a lo largo del notebook. Esta tabla resume los hiperparámetros y decisiones de diseño del algoritmo: representación binaria de cromosomas, tamaño de población constante, inicialización aleatoria, selección por rueda de ruleta, cruce de un punto, mutación bit-flip, sustitución total de la población en cada generación, y 100 generaciones de ejecución.

---

## Celda 1 — Título de Sección 0 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado de sección "0. Importaciones y configuración general". Sirve como separador visual para indicar el inicio del bloque de configuración del entorno de trabajo.

---

## Celda 2 — Instalación de PyGAD

**Tipo:** Código

```python
!pip install pygad
```

**Descripción:** Ejecuta el comando de instalación del paquete `pygad` desde PyPI. PyGAD es una biblioteca de Python que proporciona una implementación flexible de algoritmos genéticos. En este notebook se utiliza como motor principal para la evolución de la población, configurando sus parámetros de forma personalizada (selección, cruce, mutación, etc.). La instalación se realiza al inicio para asegurar que todas las celdas posteriores tengan acceso a la librería.

---

## Celda 3 — Importaciones y Configuración de Matplotlib

**Tipo:** Código

```python
import numpy as np
import matplotlib.pyplot as plt
import pygad
import math
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
```

**Descripción:** Importa todas las bibliotecas necesarias para el notebook y configura los parámetros globales de visualización:

- **`numpy` (`np`)**: Biblioteca fundamental para cálculos numéricos y manipulación de arreglos. Se usa extensivamente para generar valores aleatorios, calcular promedios, y manipular cromosomas como arreglos binarios.
- **`matplotlib.pyplot` (`plt`)**: Biblioteca de graficación que permite visualizar la evolución de la aptitud a lo largo de las generaciones. Se configuran los parámetros globales: fuente sans-serif DejaVu Sans, manejo correcto del signo menos en ejes, y resolución de figura de 120 DPI para gráficas nítidas.
- **`pygad`**: Librería de algoritmos genéticos que proporciona la clase `pygad.GA`, la cual encapsula el ciclo evolutivo completo (selección, cruce, mutación, evaluación). Se configura con parámetros personalizados en las celdas posteriores.
- **`math`**: Módulo de funciones matemáticas estándar, utilizado para calcular logaritmos en base 2 y techos (`ceil`) al determinar el número de bits necesarios para la representación binaria.
- **`warnings.filterwarnings("ignore")`**: Suprime todas las advertencias del intérprete para mantener la salida limpia durante las múltiples ejecuciones del AG.

---

## Celda 4 — Título de Sección 1 (Markdown)

**Tipo:** Markdown

**Contenido:** Encabezado de la sección principal "1. Implementación del AGS para F(x) = x², x entero [0, 63]". Indica los parámetros base (n ≈ 20, Pm ≈ 0.002, Pc ≈ 0.8, 100 generaciones) y especifica que la representación binaria utiliza **6 bits** porque 2⁶ = 64 > 63, lo cual permite codificar todos los valores enteros del 0 al 63 inclusive.

---

## Celda 5 — Definición de Funciones del AGS (Entero)

**Tipo:** Código

Esta celda es el núcleo funcional del notebook. Define todas las funciones auxiliares y la función principal `run_aga()` para ejecutar el algoritmo genético.

### `NUM_GENES = 6`

Constante global que indica el número de bits (genes) del cromosoma para la representación entera. Con 6 bits se pueden representar valores de 0 a 63 (2⁶ = 64 valores).

### `binary_to_int(chromosome)`

```python
def binary_to_int(chromosome):
    x = 0
    for i, bit in enumerate(chromosome):
        x += bit * (2 ** (NUM_GENES - 1 - i))
    return int(x)
```

Función de **decodificación** que convierte un cromosoma binario (arreglo de 0s y 1s) en su valor entero equivalente. Recorre cada bit del cromosoma y suma su contribución posicional: el bit más significativo (posición 0) tiene peso 2^(L-1), y el menos significativo (posición L-1) tiene peso 2⁰. Por ejemplo, el cromosoma `[1, 1, 1, 1, 1, 1]` se decodifica como 63.

### `fitness_func_integer(ga_instance, solution, solution_idx)`

```python
def fitness_func_integer(ga_instance, solution, solution_idx):
    x = binary_to_int(solution)
    return x ** 2
```

**Función de aptitud (fitness)** para el problema entero. Decodifica el cromosoma a su valor entero `x` y retorna `x²`. Esta función define el objetivo del AG: maximizar F(x) = x² en el intervalo [0, 63], cuyo máximo teórico es F(63) = 3969. La firma de la función cumple con la convención requerida por PyGAD, que pasa la instancia del GA, la solución (cromosoma) y su índice como argumentos.

### `custom_mutation(offspring, ga_instance)`

```python
def custom_mutation(offspring, ga_instance):
    pm = ga_instance.mutation_probability
    if pm is None:
        pm = 0.002
    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < pm:
                offspring[chromosome_idx, gene_idx] = 1 - offspring[chromosome_idx, gene_idx]
    return offspring
```

Operador de **mutación personalizado** que implementa la mutación bit-flip. Para cada gen de cada cromosoma en la descendencia, se genera un número aleatorio uniforme en [0, 1). Si este número es menor que la probabilidad de mutación `pm`, el bit se invierte (0 → 1 o 1 → 0). Este operador introduce diversidad genética en la población, permitiendo al AG explorar nuevas regiones del espacio de búsqueda. Se define como función personalizada en lugar de usar la mutación nativa de PyGAD para tener control exacto sobre la probabilidad de mutación por gen.

### `on_generation(ga_instance)`

```python
def on_generation(ga_instance):
    best_fitness = ga_instance.best_solution()[1]
    ga_instance.best_fitness_history.append(best_fitness)

    fitness = ga_instance.last_generation_fitness
    avg_fitness = np.mean(fitness)
    ga_instance.avg_fitness_history.append(avg_fitness)
```

**Callback** que se ejecuta al finalizar cada generación. Su función es registrar dos observables clave:

1. **`best_fitness_history`**: La aptitud del mejor individuo de la generación actual (observable "a"). Se obtiene llamando a `ga_instance.best_solution()`, que retorna una tupla (cromosoma, aptitud).
2. **`avg_fitness_history`**: El promedio de las aptitudes de todos los individuos de la población en la generación actual (observable "b"). Se calcula con `np.mean()` sobre el arreglo de aptitudes de la última generación.

Estos historiales se almacenan como atributos dinámicos del objeto `ga_instance` y permiten graficar la evolución del AG después de la ejecución.

### `run_aga(...)`

```python
def run_aga(pop_size=20, num_generations=100, pc=0.8, pm=0.002, num_genes=6,
            fitness_func=None, on_gen_callback=None, seed=None):
```

Función principal que **configura y ejecuta** el Algoritmo Genético Simple. Sus parámetros son:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `pop_size` | 20 | Tamaño de la población |
| `num_generations` | 100 | Número de generaciones |
| `pc` | 0.8 | Probabilidad de cruce |
| `pm` | 0.002 | Probabilidad de mutación por gen |
| `num_genes` | 6 | Número de bits del cromosoma |
| `fitness_func` | `fitness_func_integer` | Función de aptitud |
| `on_gen_callback` | `on_generation` | Callback por generación |
| `seed` | None | Semilla aleatoria para reproducibilidad |

Dentro de la función se realizan las siguientes operaciones:

1. **Asignación de funciones por defecto**: Si no se especifica una función de aptitud o callback, se usan las definidas anteriormente.
2. **Fijación de semilla**: Si se proporciona `seed`, se fija la semilla de NumPy para resultados reproducibles.
3. **Cálculo de padres**: `num_parents_mating = pop_size // 2`, es decir, la mitad de la población es seleccionada como padres en cada generación.
4. **Creación de la instancia de PyGAD**: Se configuran todos los parámetros del AG:
   - `gene_type=int` y `gene_space=[0, 1]`: Los genes son enteros binarios (0 o 1).
   - `init_range_low=0, init_range_high=2`: Rango para la inicialización aleatoria (genera 0s y 1s).
   - `parent_selection_type="rws"`: Selección por rueda de ruleta (Roulette Wheel Selection), donde la probabilidad de selección de un individuo es proporcional a su aptitud.
   - `crossover_type="single_point"`: Cruce de un solo punto, donde se elige un punto de corte aleatorio y se intercambian los segmentos de los padres.
   - `crossover_probability=pc`: Probabilidad de que se aplique el operador de cruce a un par de padres.
   - `mutation_type=custom_mutation`: Se usa la mutación bit-flip personalizada definida arriba.
   - `keep_parents=0, keep_elitism=0`: No se conserva ningún padre ni élite entre generaciones (sustitución total).
   - `allow_duplicate_genes=True`: Permite genes duplicados (necesario para representación binaria).
5. **Inicialización de historiales**: Se crean listas vacías `best_fitness_history` y `avg_fitness_history` como atributos de la instancia.
6. **Ejecución**: Se llama a `ga_instance.run()` para iniciar el ciclo evolutivo.
7. **Retorno**: Devuelve una tupla con la instancia del GA, el historial del mejor individuo y el historial del promedio.

Al finalizar la celda se imprime un mensaje de confirmación y el rango representable con 6 bits.

---

## Celda 6 — Título de Sección 1.1 (Markdown)

**Tipo:** Markdown

**Contenido:** Introduce la sección "1.1 Medir la ejecución del AGS", indicando que se graficarán dos observables: (a) la adaptación del mejor individuo y (b) el promedio de las adaptaciones de la población. Reafirma los parámetros: n=20, Pc=0.8, Pm=0.002, 100 generaciones.

---

## Celda 7 — Ejecución del AGS (Parámetros Base)

**Tipo:** Código

```python
ga_base, best_hist_base, avg_hist_base = run_aga(
    pop_size=20, num_generations=100, pc=0.8, pm=0.002, seed=42
)
```

**Descripción:** Ejecuta el AGS con los parámetros base del experimento y semilla 42 (para reproducibilidad). Almacena tres resultados:

- `ga_base`: La instancia completa del GA ejecutado.
- `best_hist_base`: Historial de aptitud del mejor individuo por generación.
- `avg_hist_base`: Historial de aptitud promedio de la población por generación.

Luego extrae y muestra la mejor solución encontrada: el cromosoma óptimo, su valor decodificado `x`, el valor de la función F(x) = x², y lo compara con el máximo teórico F(63) = 3969.

---

## Celda 8 — Gráfica de Evolución del AGS (Parte 1.1)

**Tipo:** Código

**Descripción:** Genera una gráfica de líneas con Matplotlib que muestra la evolución del AG a lo largo de las generaciones:

- **Línea azul** (`'b-'`): Aptitud del mejor individuo por generación (observable "a"). Muestra cómo el mejor cromosoma encontrado mejora con el tiempo hasta converger al óptimo.
- **Línea roja** (`'r-'`): Aptitud promedio de toda la población por generación (observable "b"). Muestra cómo el rendimiento promedio de la población se aproxima gradualmente al mejor, indicando convergencia y homogeneización genética.

Se configuran etiquetas de ejes, título, leyenda y cuadrícula. Se usa `plt.tight_layout()` para evitar superposiciones.

---

## Celda 9 — Título de Sección 1.2 (Markdown)

**Tipo:** Markdown

**Contenido:** Plantea la pregunta de investigación para la sección "1.2 Efecto de variar Pc (Pm=0.002 fijo)": ¿qué cambios experimentan los observables (a) y (b) si se mantiene Pm=0.002 fijo y se varía Pc a 0.6 y 0.4?

---

## Celda 10 — Experimento: Variación de Pc

**Tipo:** Código

```python
pc_values = [0.8, 0.6, 0.4]
results_1_2 = {}

for pc in pc_values:
    ga, best_h, avg_h = run_aga(pop_size=20, num_generations=100, pc=pc, pm=0.002, seed=42)
    results_1_2[pc] = {'best': best_h, 'avg': avg_h, 'ga': ga}
```

**Descripción:** Ejecuta el AGS tres veces, una por cada valor de probabilidad de cruce (0.8, 0.6 y 0.4), manteniendo todos los demás parámetros constantes (Pm=0.002, n=20, 100 generaciones, semilla 42). Los resultados se almacenan en el diccionario `results_1_2`, donde cada clave es el valor de Pc y el valor es otro diccionario con los historiales de mejor aptitud, aptitud promedio y la instancia del GA. Para cada ejecución se imprime el mejor x encontrado, su aptitud, la mejor aptitud final y el promedio final.

---

## Celda 11 — Gráficas Comparativas: Efecto de Pc

**Tipo:** Código

**Descripción:** Genera dos subgráficas lado a lado (`1, 2` layout):

- **Subgráfica izquierda** (observable "a"): Aptitud del mejor individuo vs. generación para cada valor de Pc. Permite visualizar cómo la velocidad de convergencia del mejor individuo varía con la probabilidad de cruce.
- **Subgráfica derecha** (observable "b"): Aptitud promedio de la población vs. generación para cada valor de Pc. Muestra el impacto de Pc sobre la convergencia poblacional.

Cada curva está etiquetada con su valor de Pc correspondiente. El título general indica "Efecto de la Probabilidad de Cruce (Pc)".

---

## Celda 12 — Análisis de Convergencia por Pc

**Tipo:** Código

**Descripción:** Realiza un análisis numérico de los resultados del experimento de variación de Pc. Para cada valor de Pc, imprime:

- La mejor aptitud final alcanzada.
- El promedio final de la población.
- La generación en la que el mejor individuo alcanzó por primera vez el máximo teórico (3969 = 63²). Esto se determina recorriendo el historial de mejor aptitud y buscando la primera generación donde el valor es ≥ 3969. Si no se alcanza, se indica "No convergió".

Este análisis permite cuantificar objetivamente el impacto de Pc sobre la velocidad de convergencia y la capacidad del AG para encontrar el óptimo global.

---

## Celda 13 — Respuesta 1.2: Análisis del Efecto de Pc (Markdown)

**Tipo:** Markdown

**Contenido:** Presenta la respuesta analítica al experimento de variación de Pc:

- **Pc=0.8 (alto)**: El AG explota eficientemente las buenas soluciones mediante recombinación. Convergencia rápida tanto para el mejor individuo como para el promedio.
- **Pc=0.6**: Se reduce la tasa de recombinación, lo que implica que más individuos se reproducen sin cruce. Esto ralentiza la convergencia porque se generan menos combinaciones genéticas nuevas.
- **Pc=0.4 (bajo)**: La mayoría de los descendientes son copias directas de los padres, limitando severamente la exploración del espacio de búsqueda. El promedio crece muy lentamente.

**Conclusión**: Reducir Pc frena la convergencia del promedio (observable b) y puede retrasar la aparición del mejor individuo (observable a), porque el cruce es el operador principal que combina material genético favorable.

---

## Celda 14 — Título de Sección 1.3 (Markdown)

**Tipo:** Markdown

**Contenido:** Plantea la pregunta de investigación para la sección "1.3 Efecto de variar Pm (Pc=0.8 fijo)": ¿qué cambios experimentan los observables (a) y (b) si se mantiene Pc=0.8 fijo y se varía Pm a 0.01 y 0.1?

---

## Celda 15 — Experimento: Variación de Pm

**Tipo:** Código

```python
pm_values = [0.002, 0.01, 0.1]
results_1_3 = {}

for pm in pm_values:
    ga, best_h, avg_h = run_aga(pop_size=20, num_generations=100, pc=0.8, pm=pm, seed=42)
    results_1_3[pm] = {'best': best_h, 'avg': avg_h, 'ga': ga}
```

**Descripción:** Ejecuta el AGS tres veces, una por cada valor de probabilidad de mutación (0.002, 0.01 y 0.1), manteniendo Pc=0.8 fijo y los demás parámetros constantes (n=20, 100 generaciones, semilla 42). Los resultados se almacenan en el diccionario `results_1_3` con la misma estructura que la sección anterior. Para cada ejecución se imprime un resumen de resultados.

---

## Celda 16 — Gráficas Comparativas: Efecto de Pm

**Tipo:** Código

**Descripción:** Genera dos subgráficas lado a lado, análogas a la celda 11 pero para la variación de Pm:

- **Subgráfica izquierda** (observable "a"): Aptitud del mejor individuo vs. generación para cada valor de Pm.
- **Subgráfica derecha** (observable "b"): Aptitud promedio de la población vs. generación para cada valor de Pm.

Se espera observar que valores altos de Pm producen oscilaciones erráticas en ambas curvas, ya que la mutación excesiva destruye las buenas soluciones acumuladas por selección y cruce.

---

## Celda 17 — Análisis de Resultados por Pm

**Tipo:** Código

**Descripción:** Imprime un análisis numérico simplificado de los resultados del experimento de Pm. Para cada valor de probabilidad de mutación, muestra la mejor aptitud final y el promedio final. A diferencia de la sección de Pc, no se calcula la generación de convergencia porque con Pm=0.1 el AG probablemente no convergerá al óptimo.

---

## Celda 18 — Respuesta 1.3: Análisis del Efecto de Pm (Markdown)

**Tipo:** Markdown

**Contenido:** Presenta la respuesta analítica al experimento de variación de Pm:

- **Pm=0.002 (bajo)**: La mutación introduce muy poca diversidad. El AG converge rápidamente hacia el óptimo gracias al cruce, pero la baja diversidad puede causar convergencia prematura a un subóptimo.
- **Pm=0.01**: Mayor diversidad genética que beneficia la exploración en generaciones iniciales. Sin embargo, puede causar oscilaciones en las generaciones finales: el mejor individuo puede alcanzar el óptimo pero luego ser "destruido" por mutaciones.
- **Pm=0.1 (alto)**: Convierte el AG en esencialmente una búsqueda aleatoria. La diversidad es tan alta que la presión selectiva no puede concentrar a la población alrededor del óptimo. El mejor individuo oscila erráticamente y el promedio no converge.

**Conclusión**: Aumentar Pm más allá del valor óptimo destruye la convergencia. El observable "a" se vuelve inestable y el observable "b" no logra converger.

---

## Celda 19 — Título de Sección 1.4 (Markdown)

**Tipo:** Markdown

**Contenido:** Introduce la sección "1.4 F(x) = x², x real [0, 63] con precisión de 2 cifras decimales" con tamaño de población 50. Plantea tres preguntas:

1. ¿Qué cambio debo realizar en el código del AGS?
2. ¿En cuántas partes debo dividir el intervalo [0, 63]?
3. ¿Cuál es el largo del cromosoma o número de bits?

---

## Celda 20 — Cálculo de Bits para Representación Real

**Tipo:** Código

```python
precision = 0.01
a, b = 0.0, 63.0
num_intervals = (b - a) / precision
num_values = num_intervals + 1
num_bits_real = math.ceil(math.log2(num_values))
```

**Descripción:** Calcula los parámetros necesarios para representar un valor real en [0, 63] con precisión de 0.01 (2 cifras decimales) usando codificación binaria:

1. **Número de subintervalos**: (b - a) / precision = 63 / 0.01 = 6300. El intervalo continuo se discretiza en 6300 subintervalos iguales.
2. **Número de valores posibles**: 6300 + 1 = 6301 (incluyendo ambos extremos).
3. **Bits necesarios**: ⌈log₂(6301)⌉ = ⌈12.6214⌉ = **13 bits**. Con 13 bits se pueden representar 2¹³ = 8192 valores, más que los 6301 necesarios.
4. **Resolución real**: (b - a) / (2¹³ - 1) ≈ 0.0077, que es menor que la precisión requerida de 0.01, confirmando que 13 bits son suficientes.

Se imprime un detalle completo de todos los cálculos intermedios.

---

## Celda 21 — Definición de Funciones para Representación Real

**Tipo:** Código

Define las funciones análogas a las de la celda 5 pero adaptadas para la representación de valores reales con 13 bits.

### `NUM_GENES_REAL = num_bits_real`

Constante que almacena el número de bits necesarios (13) para la representación real.

### `binary_to_real(chromosome)`

```python
def binary_to_real(chromosome):
    int_val = 0
    for i, bit in enumerate(chromosome):
        int_val += bit * (2 ** (NUM_GENES_REAL - 1 - i))
    x_real = a + int_val * (b - a) / (2 ** NUM_GENES_REAL - 1)
    return round(x_real, 2)
```

Función de decodificación que convierte un cromosoma binario de 13 bits a un valor real en [0, 63]. El proceso es:

1. Decodificar el cromosoma binario a un valor entero (como antes).
2. Mapear el valor entero al intervalo real usando la fórmula de normalización: `x_real = a + int_val × (b - a) / (2^L - 1)`. Esta fórmula distribuye uniformemente los 8192 valores enteros en el rango [0, 63].
3. Redondear a 2 cifras decimales para cumplir con la precisión requerida.

### `fitness_func_real(ga_instance, solution, solution_idx)`

Función de aptitud para el problema real. Decodifica el cromosoma a su valor real `x` y retorna `x²`. El máximo teórico sigue siendo F(63) = 3969.

### `custom_mutation_real(offspring, ga_instance)`

Mutación bit-flip idéntica a `custom_mutation` pero definida como función separada para mayor claridad conceptual. La lógica es la misma: cada bit tiene probabilidad `pm` de invertirse.

### `on_generation_real(ga_instance)`

Callback idéntico a `on_generation` pero para la instancia del AG con representación real. Registra los historiales de mejor aptitud y aptitud promedio por generación.

---

## Celda 22 — Ejecución del AGS con Representación Real

**Tipo:** Código

```python
ga_real = pygad.GA(
    num_generations=100,
    num_parents_mating=25,
    sol_per_pop=50,
    num_genes=NUM_GENES_REAL,
    ...
)
```

**Descripción:** Configura y ejecuta directamente una instancia de PyGAD para el problema con representación real, sin usar la función `run_aga()`. Los cambios clave respecto a la configuración entera son:

- **`sol_per_pop=50`**: Población aumentada a 50 (en lugar de 20) para manejar el espacio de búsqueda más grande.
- **`num_parents_mating=25`**: La mitad de la población (50/2).
- **`num_genes=13`**: Cromosoma de 13 bits en lugar de 6.
- **`fitness_func=fitness_func_real`**: Usa la función de aptitud para valores reales.
- **`mutation_type=custom_mutation_real`**: Usa la mutación definida para este contexto.

Se inicializan los historiales como atributos dinámicos de la instancia, se ejecuta el AG, y se imprimen los resultados: cromosoma óptimo, valor entero decodificado, valor real x, F(x), y comparación con el máximo teórico.

---

## Celda 23 — Gráfica de Evolución para Representación Real

**Tipo:** Código

**Descripción:** Genera una gráfica de líneas similar a la de la celda 8, pero para el AG con representación real de 13 bits. Muestra:

- **Línea azul**: Evolución de la aptitud del mejor individuo.
- **Línea roja**: Evolución de la aptitud promedio de la población.

El título incluye la especificación "13 bits" para distinguirla de la versión entera. Esta gráfica permite verificar que el AG con representación real también converge correctamente al óptimo.

---

## Celda 24 — Respuesta 1.4: Análisis de la Representación Real (Markdown)

**Tipo:** Markdown

**Contenido:** Responde las tres preguntas planteadas en la sección 1.4:

### 1. ¿Qué cambio debo realizar en el código del AGS implementado previamente?

Se debe cambiar la **función de decodificación** del cromosoma. En lugar de interpretar directamente los bits como un entero, ahora se debe:

1. Decodificar el cromosoma binario a un valor entero (como antes).
2. Mapear ese valor entero al intervalo real [0, 63] usando la fórmula: **x_real = a + valor_entero × (b - a) / (2^L - 1)** donde a=0, b=63 y L=13.
3. Redondear el resultado a 2 cifras decimales.

Además, se debe aumentar el número de bits del cromosoma (de 6 a 13) y ajustar el tamaño de la población a 50.

### 2. ¿En cuántas partes debo dividir el intervalo [0, 63]?

- Número de subintervalos = (b - a) / precisión = 63 / 0.01 = **6300**
- Número de valores posibles = **6301**

### 3. ¿Cuál es el largo del cromosoma o número de bits?

- L = ⌈log₂(6301)⌉ = ⌈12.6214⌉ = **13 bits**
- Con 13 bits se representan 8192 valores, más que los 6301 necesarios.
- La resolución real obtenida es Δx ≈ 0.0077 < 0.01, cumpliendo la precisión requerida.

---

## Resumen de la Estructura del Notebook

| Sección | Celdas | Descripción |
|---|---|---|
| Encabezado y especificaciones | 0-1 | Título y tabla de parámetros del AGS |
| Importaciones | 2-3 | Instalación de PyGAD e importación de librerías |
| Implementación AGS (entero) | 4-5 | Funciones de decodificación, aptitud, mutación, callback y ejecución |
| Medición del AGS | 6-8 | Ejecución base y gráfica de evolución |
| Efecto de Pc | 9-13 | Experimento variando Pc (0.8, 0.6, 0.4) con gráficas y análisis |
| Efecto de Pm | 14-18 | Experimento variando Pm (0.002, 0.01, 0.1) con gráficas y análisis |
| Representación real | 19-24 | Cálculo de bits, funciones adaptadas, ejecución y respuestas |

---

## Dependencias

| Paquete | Versión | Uso |
|---|---|---|
| `pygad` | 3.7.0 | Motor del algoritmo genético |
| `numpy` | ~2.0 | Cálculos numéricos y arreglos |
| `matplotlib` | — | Visualización de gráficas |
| `math` | estándar | Logaritmos y funciones de redondeo |
