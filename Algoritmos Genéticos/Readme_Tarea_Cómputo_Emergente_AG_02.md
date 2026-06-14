# Tarea 01 — Algoritmo Genético Simple (AGS)

## Información general

| Campo | Detalle |
|-------|---------|
| **Curso** | Introducción a la Computación Emergente |
| **Institución** | Universidad Central de Venezuela — Escuela de Física |
| **Profesor** | Esteban Álvarez |
| **Archivo** | `tarea01_AG.ipynb` |
| **Lenguaje** | Python 3.12 |
| **Librería principal** | PyGAD 3.7 |

---

## Objetivo

Implementar un Algoritmo Genético Simple (AGS) para encontrar el máximo de la función **F(x) = x²**, analizando el efecto de variar los parámetros de probabilidad de cruce (Pc) y mutación (Pm).

---

## Requisitos

```bash
pip install numpy matplotlib pygad
```

| Paquete | Versión usada | Propósito |
|---------|---------------|-----------|
| `numpy` | ≥ 1.24 | Operaciones numéricas y arreglos |
| `matplotlib` | ≥ 3.7 | Generación de gráficas |
| `pygad` | 3.7 | Framework de algoritmos genéticos |

---

## Estructura del notebook

El notebook está dividido en las siguientes secciones, correspondientes a cada inciso de la tarea:

### Sección 0 — Importaciones y configuración general

Celda de código única que importa las librerías, configura los parámetros de matplotlib (`%matplotlib inline`, `axes.unicode_minus = False`) y define la constante `DOWNLOAD_DIR`.

### Sección 1 — Implementación del AGS para F(x) = x², x entero [0, 63]

Define todas las funciones base del algoritmo genético:

| Función | Descripción |
|---------|-------------|
| `binary_to_int(chromosome)` | Decodifica un cromosoma binario de 6 bits a un entero en [0, 63]. Cada bit `i` contribuye con `bit × 2^(5-i)`. |
| `fitness_func_integer(ga_instance, solution, solution_idx)` | Función de aptitud: retorna `x²` donde `x = binary_to_int(solution)`. |
| `custom_mutation(offspring, ga_instance)` | Mutación por **flip de bits**: recorre cada gen del cromosoma y con probabilidad Pm lo invierte (0→1 o 1→0). Es necesaria porque la mutación por defecto de PyGAD no preserva la representación binaria. |
| `on_generation(ga_instance)` | Callback que registra en cada generación: (1) la aptitud del mejor individuo y (2) el promedio de aptitudes de la población. Almacena los valores en listas adjuntas al objeto `ga_instance`. |
| `run_aga(pop_size, num_generations, pc, pm, num_genes, fitness_func, on_gen_callback, seed)` | Función orquestadora. Crea una instancia de `pygad.GA` con los parámetros dados, inicializa los historiales, ejecuta el AG y retorna la instancia junto con los historiales de mejor aptitud y aptitud promedio. |

**Parámetros por defecto del AGS:**

| Parámetro | Valor | Configuración PyGAD |
|-----------|-------|---------------------|
| Tamaño de población | 20 | `sol_per_pop=20` |
| Número de genes | 6 | `num_genes=6` |
| Tipo de gen | Entero binario | `gene_type=int, gene_space=[0,1]` |
| Selección | Rueda de ruleta | `parent_selection_type="rws"` |
| Cruce | Un punto | `crossover_type="single_point"` |
| Prob. cruce | 0.8 | `crossover_probability=0.8` |
| Mutación | Flip de bits (custom) | `mutation_type=custom_mutation` |
| Prob. mutación | 0.002 | `mutation_probability=0.002` |
| Sustitución | Total | `keep_parents=0, keep_elitism=0` |
| Generaciones | 100 | `num_generations=100` |

### Sección 1.1 — Medir la ejecución del AGS

Ejecuta el AGS con los parámetros base (n=20, Pc=0.8, Pm=0.002) y produce dos outputs:

1. **Impresión en consola**: Cromosoma óptimo, valor de x, F(x), y máximo teórico (F(63)=3969).
2. **Gráfica inline**: Curvas de (a) aptitud del mejor individuo y (b) promedio de la población vs. generación.

### Sección 1.2 — Efecto de variar Pc (Pm=0.002 fijo)

Ejecuta el AGS tres veces con Pc ∈ {0.8, 0.6, 0.4}, manteniendo Pm=0.002. Incluye:

- **Celda de ejecución**: Corre los 3 experimentos con semilla 42 para reproducibilidad.
- **Celda de gráfica**: Dos subgráficas lado a lado: (a) mejor individuo y (b) promedio, cada una con las 3 curvas de Pc.
- **Celda de análisis**: Calcula aptitud final, promedio final y generación de convergencia al máximo (3969) para cada Pc.
- **Celda Markdown con respuesta**: Explicación cualitativa de que reducir Pc frena la convergencia porque el cruce es el operador principal de recombinación de material genético favorable.

### Sección 1.3 — Efecto de variar Pm (Pc=0.8 fijo)

Ejecuta el AGS tres veces con Pm ∈ {0.002, 0.01, 0.1}, manteniendo Pc=0.8. Incluye:

- **Celda de ejecución**: Corre los 3 experimentos.
- **Celda de gráfica**: Dos subgráficas comparativas: (a) mejor individuo y (b) promedio.
- **Celda de análisis**: Resume aptitud y promedio final por cada Pm.
- **Celda Markdown con respuesta**: Explicación de que Pm bajo puede causar estancamiento, Pm moderado ayuda a escapar subóptimos, y Pm alto (0.1) destruye la convergencia (búsqueda aleatoria).

### Sección 1.4 — F(x) = x², x real [0, 63] con precisión de 2 decimales

Aborda la extensión del AGS a valores reales. Incluye:

1. **Cálculo de bits necesarios**:
   - Subintervalos = (63 − 0) / 0.01 = **6300**
   - Valores posibles = **6301**
   - Bits = ⌈log₂(6301)⌉ = **13** (2¹³ = 8192 > 6301)
   - Resolución real = 63/8191 ≈ 0.0077 < 0.01 ✓

2. **Funciones nuevas para representación real**:

| Función | Descripción |
|---------|-------------|
| `binary_to_real(chromosome)` | Decodifica 13 bits a entero, luego mapea linealmente: `x = 0 + entero × 63/8191`, redondeando a 2 decimales. |
| `fitness_func_real(ga_instance, solution, solution_idx)` | Retorna `x²` donde `x = binary_to_real(solution)`. |
| `custom_mutation_real(offspring, ga_instance)` | Flip de bits idéntico a `custom_mutation` pero aplicado a cromosomas de 13 bits. |
| `on_generation_real(ga_instance)` | Callback idéntico a `on_generation` para la versión real. |

3. **Ejecución del AG con n=50** y las funciones de aptitud real.
4. **Gráfica** de evolución (mejor individuo + promedio).
5. **Respuesta Markdown** con las 3 preguntas: cambio de código, número de subintervalos, largo del cromosoma.

---

## Diseño de la representación binaria

### x entero [0, 63]

```
Cromosoma:  [b5  b4  b3  b2  b1  b0]
Valor:      b5×32 + b4×16 + b3×8 + b2×4 + b1×2 + b0×1
Ejemplo:    [1   1   0   1   1   1] → 32+16+0+4+2+1 = 55
```

### x real [0, 63] con 2 decimales

```
Cromosoma:  [b12  b11  ...  b1  b0]     (13 bits)
Entero:     Σ bi × 2^(12-i)            ∈ [0, 8191]
Real:       entero × 63 / 8191          ∈ [0.00, 63.00]
Resolución: 63/8191 ≈ 0.0077           (< 0.01 requerido)
```

---

## Notas de implementación

1. **Mutación personalizada**: PyGAD no tiene un operador de mutación nativo para flip de bits en representación binaria. La función `custom_mutation` itera sobre cada gen y con probabilidad Pm lo invierte. Esto es equivalente a la mutación bit-flip estándar de la literatura de AGs.

2. **Sustitución total**: Se logra con `keep_parents=0` y `keep_elitism=0`, lo que significa que ninguna solución de la generación anterior sobrevive a la siguiente. La nueva población se forma completamente con los descendientes generados.

3. **Selección por ruleta (RWS)**: PyGAD implementa la selección proporcional al fitness mediante `parent_selection_type="rws"`. La probabilidad de selección de un individuo es proporcional a su aptitud relativa.

4. **Semilla aleatoria**: Se usa `seed=42` en todos los experimentos para garantizar reproducibilidad. Esto permite comparar los efectos de variar un solo parámetro a la vez.

5. **Historiales de aptitud**: PyGAD no expone directamente un historial por generación del mejor y promedio. Se implementa mediante el callback `on_generation`, que captura `ga_instance.best_solution()[1]` y `np.mean(ga_instance.last_generation_fitness)` al final de cada generación.

---

## Salidas esperadas

| Sección | Output |
|---------|--------|
| 1.1 | Gráfica de convergencia (mejor + promedio) para Pc=0.8, Pm=0.002 |
| 1.2 | Gráfica comparativa de 3 valores de Pc + tabla de convergencia |
| 1.3 | Gráfica comparativa de 3 valores de Pm + tabla de resultados |
| 1.4 | Cálculos de bits + gráfica de convergencia para x real |
