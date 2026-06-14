# Documentación Técnica — Tarea 02

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Especificaciones del AGS](#2-especificaciones-del-ags)
3. [Dependencias y Configuración](#3-dependencias-y-configuración)
4. [Arquitectura del Código](#4-arquitectura-del-código)
5. [Módulo: Funciones de Decodificación](#5-módulo-funciones-de-decodificación)
6. [Módulo: Función de Aptitud — Dominio Entero](#6-módulo-función-de-aptitud--dominio-entero)
7. [Módulo: Operador de Mutación Personalizado](#7-módulo-operador-de-mutación-personalizado)
8. [Módulo: Callback de Generación](#8-módulo-callback-de-generación)
9. [Módulo: Función de Ejecución del AGS](#9-módulo-función-de-ejecución-del-ags)
10. [Experimento 1.1 — Ejecución Base](#10-experimento-11--ejecución-base)
11. [Experimento 1.2 — Efecto de Variar Pc](#11-experimento-12--efecto-de-variar-pc)
12. [Experimento 1.3 — Efecto de Variar Pm](#12-experimento-13--efecto-de-variar-pm)
13. [Experimento 1.4 — Extensión a Dominio Real](#13-experimento-14--extensión-a-dominio-real)
14. [Glosario](#14-glosario)

---

## 1. Visión General

Este proyecto implementa un **Algoritmo Genético Simple (AGS)** para optimizar la función **F(x) = x²** en dos variantes de dominio:

- **Dominio entero:** x ∈ [0, 63], representación binaria de 6 bits.
- **Dominio real:** x ∈ [0.00, 63.00] con precisión de 2 cifras decimales, representación binaria de 13 bits.

La implementación se realiza sobre la biblioteca **PyGAD**, personalizando los operadores de mutación y los callbacks de generación para registrar históricos de aptitud. Se llevan a cabo cuatro experimentos que analizan el efecto de los parámetros del AGS sobre la convergencia: la ejecución base con parámetros estándar, la variación de la probabilidad de cruce (Pc), la variación de la probabilidad de mutación (Pm) y la extensión al dominio real con mayor resolución cromosómica.

---

## 2. Especificaciones del AGS

| Característica       | Valor                              |
|----------------------|------------------------------------|
| Representación       | Binaria                            |
| Población            | Constante n (20 o 50 según caso)   |
| Inicialización       | Aleatoria uniforme                 |
| Renormalización      | Ninguna                            |
| Selección            | Rueda de ruleta (RWS)              |
| Operadores           | Cruce un punto + Mutación bit-flip |
| Sustitución          | Total (keep_parents=0, keep_elitism=0) |
| Generaciones         | 100                                |
| Probabilidad de cruce (Pc) | 0.8 (base), 0.6, 0.4          |
| Probabilidad de mutación (Pm) | 0.002 (base), 0.01, 0.1    |

### Parámetros por experimento

| Experimento | n  | Pc   | Pm    | Bits | Dominio           |
|-------------|----|------|-------|------|-------------------|
| 1.1         | 20 | 0.8  | 0.002 | 6    | Entero [0, 63]    |
| 1.2         | 20 | variable | 0.002 | 6  | Entero [0, 63]    |
| 1.3         | 20 | 0.8  | variable | 6  | Entero [0, 63]    |
| 1.4         | 50 | 0.8  | 0.002 | 13   | Real [0, 63]      |

---

## 3. Dependencias y Configuración

### Bibliotecas requeridas

| Biblioteca     | Versión  | Propósito                                         |
|----------------|----------|---------------------------------------------------|
| `numpy`        | 2.0.2    | Operaciones numéricas y arreglos                  |
| `matplotlib`   | —        | Visualización de gráficas de evolución            |
| `pygad`        | 3.7.0    | Framework de algoritmos genéticos                 |
| `math`         | stdlib   | Funciones matemáticas (log2, ceil)                |
| `warnings`     | stdlib   | Supresión de advertencias                          |

### Instalación

```python
!pip install pygad
```

### Configuración de Matplotlib

```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
```

Se configura `DejaVu Sans` como fuente principal, se desactiva el manejo especial del signo menos Unicode, y se establece la resolución de figuras a 120 DPI para obtener gráficas nítidas en el notebook.

---

## 4. Arquitectura del Código

El código se organiza en módulos funcionales interconectados que fluyen de la siguiente manera:

```
┌─────────────────────────────────────────────────────┐
│                  run_aga()                          │
│  Función orquestadora del AGS                      │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │fitness_func │  │custom_mutation│  │on_generation│ │
│  │ (aptitud)   │  │ (mutación)    │  │ (callback)  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬─────┘ │
│         │                │                  │        │
│         ▼                ▼                  ▼        │
│  ┌──────────────────────────────────────────────┐   │
│  │            pygad.GA (motor)                  │   │
│  │  - RWS selection                            │   │
│  │  - Single-point crossover                   │   │
│  │  - Bit-flip mutation (custom)               │   │
│  │  - Full substitution                        │   │
│  └──────────────────────────────────────────────┘   │
│         │                                            │
│         ▼                                            │
│  Resultados: ga_instance, best_hist, avg_hist       │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  binary_to_int() /   │
│  binary_to_real()    │
│  (decodificación)    │
└──────────────────────┘
```

### Flujo de datos

1. **Inicialización:** `run_aga()` configura la instancia de `pygad.GA` con los parámetros proporcionados y crea las listas `best_fitness_history` y `avg_fitness_history` como atributos dinámicos del objeto GA.
2. **Evolución:** En cada generación, PyGAD evalúa la aptitud de toda la población mediante `fitness_func`, selecciona padres con RWS, aplica cruce de un punto con probabilidad Pc, y ejecuta la mutación personalizada con probabilidad Pm.
3. **Registro:** El callback `on_generation` captura la mejor aptitud y la aptitud promedio de la generación actual, almacenándolas en los históricos.
4. **Decodificación:** Tras la ejecución, las funciones `binary_to_int()` o `binary_to_real()` traducen el cromosoma binario óptimo al valor fenotípico correspondiente.

---

## 5. Módulo: Funciones de Decodificación

### `binary_to_int(chromosome)`

Convierte un cromosoma binario (arreglo de 0s y 1s) a su valor entero correspondiente utilizando el esquema de codificación binaria posicional estándar.

```python
NUM_GENES = 6

def binary_to_int(chromosome):
    x = 0
    for i, bit in enumerate(chromosome):
        x += bit * (2 ** (NUM_GENES - 1 - i))
    return int(x)
```

**Parámetros:**
- `chromosome` (array-like de int): Arreglo de longitud `NUM_GENES` (6) con valores 0 o 1.

**Retorna:**
- `int`: El valor entero decodificado en el rango [0, 63].

**Detalle de funcionamiento:** Recorre cada gen del cromosoma desde la posición más significativa (índice 0) hasta la menos significativa. Cada bit se multiplica por 2 elevado a la potencia correspondiente a su posición, y los resultados se acumulan. Por ejemplo, el cromosoma `[1, 1, 1, 1, 1, 1]` produce 32 + 16 + 8 + 4 + 2 + 1 = 63.

### `binary_to_real(chromosome)`

Convierte un cromosoma binario a su valor real en el intervalo [a, b] con resolución determinada por el número de bits.

```python
NUM_GENES_REAL = 13  # Calculado dinámicamente

def binary_to_real(chromosome):
    int_val = 0
    for i, bit in enumerate(chromosome):
        int_val += bit * (2 ** (NUM_GENES_REAL - 1 - i))
    x_real = a + int_val * (b - a) / (2 ** NUM_GENES_REAL - 1)
    return round(x_real, 2)
```

**Parámetros:**
- `chromosome` (array-like de int): Arreglo de longitud `NUM_GENES_REAL` (13) con valores 0 o 1.

**Retorna:**
- `float`: El valor real decodificado en el intervalo [0, 63] con 2 cifras decimales.

**Detalle de funcionamiento:** Primero decodifica el cromosoma a un valor entero, igual que `binary_to_int`. Luego aplica la fórmula de mapeo lineal `x_real = a + int_val * (b - a) / (2^L - 1)` para proyectar el valor entero al intervalo continuo [0, 63]. Finalmente, redondea a 2 decimales. Con 13 bits, la resolución real es Δx ≈ 0.0077, que es inferior a 0.01, garantizando la precisión de 2 cifras decimales.

---

## 6. Módulo: Función de Aptitud — Dominio Entero

### `fitness_func_integer(ga_instance, solution, solution_idx)`

```python
def fitness_func_integer(ga_instance, solution, solution_idx):
    x = binary_to_int(solution)
    return x ** 2
```

**Parámetros:**
- `ga_instance` (pygad.GA): Instancia actual del algoritmo genético (requerido por la firma de PyGAD).
- `solution` (ndarray): Cromosoma binario a evaluar.
- `solution_idx` (int): Índice de la solución en la población (requerido por la firma de PyGAD).

**Retorna:**
- `int`: Valor de aptitud F(x) = x², donde x es la decodificación entera del cromosoma.

**Nota de diseño:** La firma de tres parámetros es impuesta por PyGAD. La función maximiza x², por lo que el óptimo global se alcanza en x = 63 con F(63) = 3969. Esta función es monótonamente creciente, lo que facilita la convergencia del AG pero la hace un buen caso de prueba para observar el efecto de los parámetros del algoritmo.

### `fitness_func_real(ga_instance, solution, solution_idx)`

```python
def fitness_func_real(ga_instance, solution, solution_idx):
    x = binary_to_real(solution)
    return x ** 2
```

Función análoga a `fitness_func_integer` pero utiliza `binary_to_real()` para la decodificación. Retorna un valor float que representa x² en el dominio real.

---

## 7. Módulo: Operador de Mutación Personalizado

### `custom_mutation(offspring, ga_instance)` / `custom_mutation_real(offspring, ga_instance)`

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

**Parámetros:**
- `offspring` (ndarray de shape `(n_offspring, n_genes)`): Matriz de descendientes producida por el cruce.
- `ga_instance` (pygad.GA): Instancia del GA de donde se extrae `mutation_probability`.

**Retorna:**
- `ndarray`: La matriz de descendientes con mutaciones aplicadas.

**Detalle de funcionamiento:** Implementa una **mutación bit-flip** independiente por gen. Para cada gen de cada cromosoma descendiente, se genera un número aleatorio uniforme en [0, 1). Si este número es menor que la probabilidad de mutación Pm, el bit se invierte (0→1 o 1→0). Esta es la estrategia de mutación canónica para representaciones binarias y garantiza que cada gen tenga exactamente una probabilidad Pm de cambiar, independientemente de los demás.

**Consideración de diseño:** Se implementa como operador personalizado en lugar del operador nativo de PyGAD para tener control explícito sobre la semilla y el comportamiento de la probabilidad de mutación. Se incluyen dos versiones (`custom_mutation` y `custom_mutation_real`) que son funcionalmente idénticas pero se definen por separado para mantener la consistencia con sus respectivos módulos de decodificación.

---

## 8. Módulo: Callback de Generación

### `on_generation(ga_instance)` / `on_generation_real(ga_instance)`

```python
def on_generation(ga_instance):
    best_fitness = ga_instance.best_solution()[1]
    ga_instance.best_fitness_history.append(best_fitness)

    fitness = ga_instance.last_generation_fitness
    avg_fitness = np.mean(fitness)
    ga_instance.avg_fitness_history.append(avg_fitness)
```

**Parámetros:**
- `ga_instance` (pygad.GA): Instancia del GA al final de la generación actual.

**Funcionalidad:**
1. Obtiene la aptitud del mejor individuo de la generación actual mediante `ga_instance.best_solution()[1]` y la agrega a la lista `best_fitness_history`.
2. Calcula la aptitud promedio de toda la población actual usando `np.mean(ga_instance.last_generation_fitness)` y la agrega a `avg_fitness_history`.

**Nota de diseño:** Las listas `best_fitness_history` y `avg_fitness_history` se inician como atributos vacíos del objeto `ga_instance` antes de ejecutar `ga_instance.run()`. Este patrón de agregar atributos dinámicos a la instancia de PyGAD es seguro dentro del ciclo de vida de una ejecución, pero requiere que las listas se inicialicen antes de llamar a `run()`. Al final de la ejecución, contienen exactamente 100 entradas (una por generación).

---

## 9. Módulo: Función de Ejecución del AGS

### `run_aga(pop_size=20, num_generations=100, pc=0.8, pm=0.002, num_genes=6, fitness_func=None, on_gen_callback=None, seed=None)`

```python
def run_aga(pop_size=20, num_generations=100, pc=0.8, pm=0.002, num_genes=6,
            fitness_func=None, on_gen_callback=None, seed=None):

    if fitness_func is None:
        fitness_func = fitness_func_integer
    if on_gen_callback is None:
        on_gen_callback = on_generation

    if seed is not None:
        np.random.seed(seed)

    num_parents_mating = pop_size // 2

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=pop_size,
        num_genes=num_genes,
        fitness_func=fitness_func,
        gene_type=int,
        gene_space=[0, 1],
        init_range_low=0,
        init_range_high=2,
        parent_selection_type="rws",
        crossover_type="single_point",
        crossover_probability=pc,
        mutation_type=custom_mutation,
        mutation_probability=pm,
        keep_parents=0,
        keep_elitism=0,
        on_generation=on_gen_callback,
        allow_duplicate_genes=True,
        suppress_warnings=True,
        random_seed=seed,
    )

    ga_instance.best_fitness_history = []
    ga_instance.avg_fitness_history = []

    ga_instance.run()

    return (ga_instance,
            ga_instance.best_fitness_history,
            ga_instance.avg_fitness_history)
```

**Parámetros:**

| Parámetro           | Tipo        | Default | Descripción                                        |
|---------------------|-------------|---------|----------------------------------------------------|
| `pop_size`          | int         | 20      | Tamaño de la población                             |
| `num_generations`   | int         | 100     | Número de generaciones a ejecutar                  |
| `pc`                | float       | 0.8     | Probabilidad de cruce                              |
| `pm`                | float       | 0.002   | Probabilidad de mutación por gen                   |
| `num_genes`         | int         | 6       | Longitud del cromosoma (número de bits)            |
| `fitness_func`      | callable    | None    | Función de aptitud (default: `fitness_func_integer`)|
| `on_gen_callback`   | callable    | None    | Callback por generación (default: `on_generation`) |
| `seed`              | int/None    | None    | Semilla para reproducibilidad                      |

**Retorna:**
- `tuple`: `(ga_instance, best_fitness_history, avg_fitness_history)` donde `ga_instance` es el objeto PyGAD completo y los dos últimos son listas con los históricos de aptitud.

**Detalle de configuración de PyGAD:**

| Parámetro PyGAD          | Valor                | Justificación                                          |
|--------------------------|----------------------|--------------------------------------------------------|
| `num_parents_mating`     | `pop_size // 2`      | La mitad de la población se selecciona como padres     |
| `gene_type`              | `int`                | Los genes son valores enteros (0 o 1)                  |
| `gene_space`             | `[0, 1]`             | Cada gen solo puede tomar valor 0 o 1                  |
| `init_range_low/high`    | 0, 2                 | Rango para inicialización aleatoria (enteros en [0,1]) |
| `parent_selection_type`  | `"rws"`              | Selección por rueda de ruleta (Roulette Wheel Selection)|
| `crossover_type`         | `"single_point"`     | Cruce de un punto                                      |
| `keep_parents`           | 0                    | No se conservan padres de la generación anterior       |
| `keep_elitism`           | 0                    | No se aplica elitismo (sustitución total)              |
| `allow_duplicate_genes`  | True                 | Permite genes duplicados (necesario para binario)      |
| `suppress_warnings`      | True                 | Suprime advertencias de PyGAD                          |

**Consideraciones sobre la sustitución total:** Con `keep_parents=0` y `keep_elitism=0`, toda la población es reemplazada en cada generación por los descendientes generados. Esto significa que no hay preservación garantizada del mejor individuo entre generaciones, lo que puede causar que el mejor individuo se pierda si la mutación o el cruce lo degradan. Esta decisión es deliberada y se alinea con la definición canónica de un AGS con sustitución total.

---

## 10. Experimento 1.1 — Ejecución Base

### Objetivo

Ejecutar el AGS con los parámetros estándar y visualizar la evolución de la aptitud del mejor individuo y del promedio de la población a lo largo de 100 generaciones.

### Parámetros

| Parámetro | Valor |
|-----------|-------|
| n         | 20    |
| Pc        | 0.8   |
| Pm        | 0.002 |
| Bits      | 6     |
| Generaciones | 100 |
| Semilla   | 42    |

### Ejecución

```python
ga_base, best_hist_base, avg_hist_base = run_aga(
    pop_size=20, num_generations=100, pc=0.8, pm=0.002, seed=42
)
```

### Visualización

Se genera una gráfica con dos curvas:
- **Curva azul:** Aptitud del mejor individuo por generación (observable a).
- **Curva roja:** Aptitud promedio de la población por generación (observable b).

### Resultados esperados

El AGS debe converger al óptimo global x = 63 (F(x) = 3969). La curva del mejor individuo crece de forma monotónica (o cuasi-monotónica) hasta alcanzar el máximo, mientras que la curva del promedio muestra un crecimiento gradual que refleja la homogeneización progresiva de la población alrededor del óptimo. Con los parámetros base, la convergencia suele producirse en las primeras 20-40 generaciones, tras las cuales ambas curvas se estabilizan.

---

## 11. Experimento 1.2 — Efecto de Variar Pc

### Objetivo

Analizar cómo la probabilidad de cruce (Pc) afecta la velocidad de convergencia y la calidad de la solución cuando la probabilidad de mutación se mantiene fija en Pm = 0.002.

### Parámetros evaluados

| Configuración | Pc  | Pm    | n  | Bits | Generaciones |
|---------------|-----|-------|----|------|--------------|
| Alta          | 0.8 | 0.002 | 20 | 6    | 100          |
| Media         | 0.6 | 0.002 | 20 | 6    | 100          |
| Baja          | 0.4 | 0.002 | 20 | 6    | 100          |

### Metodología

Se ejecuta `run_aga()` para cada valor de Pc con semilla fija (seed=42), y se comparan las curvas de aptitud del mejor individuo y del promedio poblacional en gráficas lado a lado. Se analiza además la generación de convergencia, definida como la primera generación en la que la mejor aptitud alcanza o supera 3969 (63²).

### Resultados y análisis

**Pc = 0.8 (valor alto):** El AG explota eficientemente las buenas soluciones mediante recombinación. La convergencia es rápida tanto para el mejor individuo como para el promedio de la población. El cruce permite combinar bloques génicos favorables de diferentes individuos, acelerando la acumulación de bits "1" en la población.

**Pc = 0.6:** Se reduce la tasa de recombinación, lo que implica que una fracción mayor de la población se reproduce sin cruce (el 40% de los pares seleccionados no se cruzan). Esto ralentiza la convergencia, ya que se generan menos combinaciones nuevas de genes favorables. El mejor individuo puede alcanzar el óptimo, pero el promedio de la población converge más lentamente porque la diversidad genética favorable se propaga con menor velocidad.

**Pc = 0.4:** La baja probabilidad de cruce significa que la mayoría de los descendientes son copias directas de los padres (sin recombinación). Esto limita significativamente la exploración del espacio de búsqueda. El mejor individuo puede tardar mucho más en aparecer (o no aparecer), y el promedio de la población crece muy lentamente, ya que la diversidad genética solo proviene de la mutación (que con Pm=0.002 es muy baja).

**Conclusión:** Reducir Pc frena la convergencia del promedio (observable b) y puede retrasar la aparición del mejor individuo (observable a). El cruce es el operador principal que combina material genético favorable de diferentes individuos, por lo que su reducción debilita la capacidad de explotación del AG.

---

## 12. Experimento 1.3 — Efecto de Variar Pm

### Objetivo

Analizar cómo la probabilidad de mutación (Pm) afecta la estabilidad de la convergencia y la diversidad poblacional cuando la probabilidad de cruce se mantiene fija en Pc = 0.8.

### Parámetros evaluados

| Configuración | Pc  | Pm    | n  | Bits | Generaciones |
|---------------|-----|-------|----|------|--------------|
| Baja          | 0.8 | 0.002 | 20 | 6    | 100          |
| Media         | 0.8 | 0.01  | 20 | 6    | 100          |
| Alta          | 0.8 | 0.1   | 20 | 6    | 100          |

### Metodología

Análoga al Experimento 1.2: ejecución con semilla fija y comparación de curvas de aptitud en gráficas lado a lado.

### Resultados y análisis

**Pm = 0.002 (valor bajo):** La mutación introduce muy poca diversidad. El AG converge rápidamente hacia el óptimo gracias al cruce, y una vez que la población se homogeneiza, el mejor individuo se mantiene estable. Sin embargo, la baja diversidad puede causar que el AG se quede atrapado en un subóptimo si la población converge prematuramente antes de descubrir el cromosoma óptimo. En este caso particular (función monótona), el riesgo es mínimo.

**Pm = 0.01:** La mayor tasa de mutación introduce más diversidad genética. Esto puede ser beneficioso en las generaciones iniciales (mayor exploración), ayudando a escapar de subóptimos. Sin embargo, en las generaciones finales puede causar oscilaciones: el mejor individuo puede alcanzar el óptimo pero luego ser "destruido" por mutaciones, y el promedio de la población muestra más fluctuaciones respecto al caso base.

**Pm = 0.1 (valor alto):** La alta tasa de mutación convierte el AG esencialmente en una búsqueda aleatoria. La diversidad es tan alta que la presión selectiva no puede concentrar a la población alrededor del óptimo. El mejor individuo (observable a) oscila erráticamente, mientras que el promedio (observable b) se mantiene bajo y con grandes fluctuaciones, sin convergencia. Con Pm=0.1, cada gen tiene un 10% de probabilidad de invertirse, lo que significa que en un cromosoma de 6 bits se espera que ~0.6 bits cambien por descendiente, destruyendo sistemáticamente las buenas combinaciones genéticas.

**Conclusión:** Aumentar Pm más allá del valor óptimo destruye la convergencia. El observable a (mejor individuo) se vuelve inestable, y el observable b (promedio) no logra converger, mostrando que una mutación excesiva impide la explotación de buenas soluciones. La mutación debe ser lo suficientemente baja como para no perturbar las soluciones prometedoras, pero lo suficientemente alta como para mantener diversidad y escapar de subóptimos.

---

## 13. Experimento 1.4 — Extensión a Dominio Real

### Objetivo

Extender el AGS para optimizar F(x) = x² en el dominio real x ∈ [0, 63] con precisión de 2 cifras decimales, utilizando una población de tamaño 50.

### Preguntas de diseño

#### 1. ¿Qué cambio debo realizar en el código del AGS implementado previamente?

Se debe cambiar la **función de decodificación** del cromosoma binario. En lugar de interpretar directamente los bits como un número entero, ahora se debe:

1. Decodificar el cromosoma binario a un valor entero (como antes).
2. Mapear ese valor entero al intervalo real [0, 63] usando la fórmula de mapeo lineal.
3. Redondear el resultado a 2 cifras decimales.

Además, se debe aumentar el número de bits del cromosoma (L) para representar la mayor cantidad de valores discretos necesarios, y ajustar el tamaño de la población a 50 como indica el enunciado.

#### 2. ¿En cuántas partes debo dividir el intervalo [0, 63]?

Con precisión de 0.01 (2 cifras decimales):

- Número de subintervalos = (b - a) / precisión = 63 / 0.01 = **6300**
- Número de valores posibles = 6300 + 1 = **6301**

#### 3. ¿Cuál es el largo del cromosoma o número de bits?

```
L = ceil(log2(6301)) = ceil(12.6214) = 13 bits
```

Con 13 bits se pueden representar 2¹³ = 8192 valores, lo cual es suficiente para los 6301 valores requeridos. La resolución real obtenida es:

```
Δx = (b - a) / (2^L - 1) = 63 / 8191 ≈ 0.007693
```

Como Δx ≈ 0.0077 < 0.01, se cumple la precisión de 2 decimales.

### Implementación

El experimento 1.4 no utiliza la función `run_aga()`, sino que configura la instancia de PyGAD directamente para usar las funciones de decodificación y callback específicas del dominio real:

```python
ga_real = pygad.GA(
    num_generations=100,
    num_parents_mating=25,        # 50 // 2
    sol_per_pop=50,
    num_genes=NUM_GENES_REAL,     # 13
    fitness_func=fitness_func_real,
    gene_type=int,
    gene_space=[0, 1],
    init_range_low=0,
    init_range_high=2,
    parent_selection_type="rws",
    crossover_type="single_point",
    crossover_probability=0.8,
    mutation_type=custom_mutation_real,
    mutation_probability=0.002,
    keep_parents=0,
    keep_elitism=0,
    on_generation=on_generation_real,
    allow_duplicate_genes=True,
    suppress_warnings=True,
    random_seed=42,
)
```

### Visualización

Se genera una gráfica con las mismas dos curvas (mejor individuo y promedio), ahora con subtítulo que indica "13 bits" para reflejar la codificación extendida.

### Resultados esperados

El AGS debe converger a un valor cercano a x = 63.00, con F(x) cercano a 3969.00. Debido al mayor tamaño del espacio de búsqueda (8192 valores posibles vs. 64 en el caso entero) y al mayor tamaño de cromosoma, la convergencia puede requerir más generaciones o mostrar un comportamiento ligeramente diferente, pero con Pc=0.8 y Pm=0.002 el algoritmo típicamente alcanza el óptimo antes de la generación 100.

---

## 14. Glosario

| Término                  | Definición                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------|
| **AGS**                  | Algoritmo Genético Simple. Metaheurística poblacional inspirada en la evolución biológica.              |
| **Cromosoma**            | Representación codificada de una solución candidata (en este caso, un arreglo binario).                 |
| **Gen**                  | Unidad mínima del cromosoma (en este caso, un bit: 0 o 1).                                             |
| **Fenotipo**             | La expresión observable del cromosoma (el valor x decodificado).                                        |
| **Genotipo**             | La codificación interna de la solución (el cromosoma binario).                                          |
| **Aptitud (Fitness)**    | Valor numérico que mide la calidad de una solución. En este caso, F(x) = x².                            |
| **Pc**                   | Probabilidad de cruce. Fracción de pares de padres que se recombinan.                                   |
| **Pm**                   | Probabilidad de mutación. Probabilidad de que cada gen individual se invierta.                          |
| **RWS**                  | Roulette Wheel Selection (Selección por Rueda de Ruleta). Método de selección proporcional a la aptitud. |
| **Cruce un punto**       | Operador de recombinación que elige un punto de corte aleatorio e intercambia segmentos entre padres.   |
| **Mutación bit-flip**    | Operador que invierte un bit con probabilidad Pm.                                                       |
| **Sustitución total**    | Estrategia donde toda la población es reemplazada por los descendientes en cada generación.             |
| **Convergencia**         | Estado donde la población se homogeneiza alrededor de una solución, típicamente el óptimo.              |
| **Presión selectiva**    | Fuerza con la que el AG favorece a los individuos más aptos. Una presión alta acelera la convergencia.  |
| **Exploración**          | Capacidad del AG de buscar en regiones diversas del espacio de soluciones.                              |
| **Explotación**          | Capacidad del AG de refinar y concentrarse en las mejores soluciones encontradas.                       |
| **Mapeo lineal**         | Fórmula para proyectar un valor entero decodificado al intervalo real [a, b]: x = a + int × (b-a)/(2^L-1). |
| **Resolución (Δx)**      | Diferencia mínima entre dos valores adyacentes representables.                                          |
