# Documentación Técnica — Tarea_Cómputo_Emergente_AG_01

## Información general

| Campo | Detalle |
|-------|---------|
| **Archivo** | `Tarea_Cómputo_Emergente_AG_01.ipynb` |
| **Lenguaje** | Python 3 |
| **Librería principal** | PyGAD |
| **Tipo de notebook** | Jupyter Notebook (formato 4.0) |
| **Total de celdas** | 11 (4 Markdown + 7 Code) |

---

## Objetivo

Implementar un Algoritmo Genético (AG) para minimizar la función bidimensional:

```
f(x, y) = x² + 2y² − 0.3·cos(3πx) − 0.4·cos(4πy) + 0.7
```

con x, y ∈ [−5, 5], usando representación binaria. Se comparan tres variantes del AG (Simple, con renormalización lineal, con renormalización por ventana) y se realiza un barrido de parámetros Pc y Pm para encontrar la combinación óptima.

---

## Requisitos

```bash
pip install pygad numpy matplotlib
```

| Paquete | Propósito |
|---------|-----------|
| `pygad` | Framework de algoritmos genéticos |
| `numpy` | Operaciones numéricas, promedios, arreglos |
| `matplotlib` | Gráficas de evolución y mapa de calor |

---

## Estructura del notebook

### Celda 0–2 — Librerías e importaciones

- **Celda 1**: Instala PyGAD vía `pip install pygad`.
- **Celda 2**: Importa `pygad`, `numpy` y `matplotlib.pyplot`.

### Celda 3–4 — Funciones del programa

Define todas las funciones centrales del AG:

---

#### `f(x, y)`

```python
def f(x, y):
    return x**2 + 2*y**2 - 0.3*np.cos(3*np.pi*x) - 0.4*np.cos(4*np.pi*y) + 0.7
```

Función objetivo a minimizar. Es una superficie con un mínimo global cerca del origen, modulada por términos cosenoidales que introducen múltiples mínimos locales.

**Propiedades:**
- Mínimo global: aproximadamente f(0, 0) = −0.3·cos(0) − 0.4·cos(0) + 0.7 = −0.3 − 0.4 + 0.7 = **0.0**
- Los términos cuadráticos (x² + 2y²) dominan lejos del origen.
- Los términos cosenoidales generan oscilaciones locales con periodo 2/3 en x y 1/2 en y.

---

#### `binario_a_float(array, bits=16, valor_maximo=5.0, valor_minimo=-5.0)`

```python
def binario_a_float(array, bits=16, valor_maximo=5.0, valor_minimo=-5.0):
    escala = (valor_maximo - valor_minimo) / (2**bits - 1)
    string_binario = "".join(str(int(b)) for b in array)
    decimal = int(string_binario, 2)
    return valor_minimo + (decimal * escala)
```

Decodifica un segmento binario del cromosoma a un valor real en [−5, 5].

**Algoritmo:**

1. Calcula la resolución: `escala = 10 / (2¹⁶ − 1) ≈ 0.0001526`
2. Convierte el arreglo de bits a un string binario.
3. Parsea el string a entero decimal con `int(string, 2)`.
4. Mapea linealmente: `valor = −5 + decimal × escala`

**Ejemplo:**

```
bits = [0, 0, ..., 0] (16 ceros) → decimal = 0 → x = −5.0
bits = [1, 1, ..., 1] (16 unos)  → decimal = 65535 → x = 5.0
bits = [1, 0, ..., 0]            → decimal = 32768 → x ≈ 0.0008
```

**Resolución**: Δx = 10/65535 ≈ 1.526 × 10⁻⁴, suficiente para explorar el espacio con gran precisión.

---

#### `funcion_de_adaptacion(instancia_ga, solucion, indice_solucion)`

```python
def funcion_de_adaptacion(instancia_ga, solucion, indice_solucion):
    binario_x = solucion[:16]
    binario_y = solucion[16:]
    x = binario_a_float(binario_x)
    y = binario_a_float(binario_y)
    funcion = f(x, y)
    return -funcion
```

Función de aptitud (fitness) para PyGAD.

**Diseño:**

1. Divide el cromosoma de 32 bits en dos mitades: primeros 16 bits → x, últimos 16 bits → y.
2. Decodifica cada mitad a un valor real en [−5, 5].
3. Evalúa f(x, y).
4. Retorna `−f(x, y)` para convertir la minimización en maximización (PyGAD maximiza).

**Estructura del cromosoma:**

```
|<------ 16 bits ------>|<------ 16 bits ------>|
|      x (binario)      |      y (binario)      |
|  b15 b14 ... b1 b0    |  b15 b14 ... b1 b0    |
```

---

#### `ejecucion_simulacion(numero_de_simulaciones=30, tipo_de_escalado=None, conversion_de_ventana=None, **kwargs)`

```python
def ejecucion_simulacion(numero_de_simulaciones=30, tipo_de_escalado=None,
                         conversion_de_ventana=None, **kwargs)
```

Función orquestadora que ejecuta múltiples simulaciones independientes del AG y promedia los resultados.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `numero_de_simulaciones` | int | 30 | Cantidad de ejecuciones independientes a promediar |
| `tipo_de_escalado` | str/None | None | Tipo de renormalización: `None`, `"lineal"` o `"ventana"` |
| `conversion_de_ventana` | float/None | None | Parámetro para la renormalización por ventana (no implementado completamente) |
| `**kwargs` | — | — | Parámetros adicionales pasados a `pygad.GA` (ej: `crossover_probability`, `mutation_probability`) |

**Algoritmo:**

1. Itera `numero_de_simulaciones` veces:
   - Define un callback `llamada_generaciones` que captura el promedio de fitness de cada generación.
   - Crea una instancia de `pygad.GA` con la configuración fija y los kwargs variables.
   - Ejecuta el AG (`carga_genetica.run()`).
   - Almacena el historial de mejores soluciones y los promedios por generación.
2. Calcula el promedio entre simulaciones para cada generación:
   - `media_mejores`: promedio de los mejores fitness por generación (arreglo de longitud 101: gen 0 a gen 100).
   - `media_promedios`: promedio de los promedios poblacionales por generación (arreglo de longitud 100: gen 1 a gen 100).

**Configuración fija del AG:**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `num_generations` | 100 | Generaciones por ejecución |
| `sol_per_pop` | 100 | Tamaño de población |
| `num_parents_mating` | 20 | Padres seleccionados para reproducción |
| `fitness_func` | `funcion_de_adaptacion` | Aptitud = −f(x, y) |
| `gene_type` | int | Genes enteros (0 o 1) |
| `gene_space` | [0, 1] | Representación binaria |
| `num_genes` | 32 | 16 bits para x + 16 bits para y |
| `mutation_by_replacement` | True | La mutación reemplaza el gen en vez de sumar |
| `random_mutation_min_val` | 0 | Valor mínimo de mutación |
| `random_mutation_max_val` | 2 | Valor máximo (0 o 1 efectivamente) |

**Nota sobre la mutación:** La combinación `mutation_by_replacement=True` con `random_mutation_min_val=0` y `random_mutation_max_val=2` genera valores aleatorios enteros en {0, 1} que reemplazan al gen original. Esto es equivalente a un flip de bits con cierta probabilidad, pero implementado a través de los parámetros nativos de PyGAD en lugar de una función de mutación personalizada.

**Retorna:**

| Valor | Tipo | Descripción |
|-------|------|-------------|
| `media_mejores` | `ndarray` | Promedio del mejor fitness por generación (101 puntos: gen 0–100) |
| `media_promedios` | `ndarray` | Promedio del fitness promedio por generación (100 puntos: gen 1–100) |

---

### Celda 5–6 — Simulación AG Simple

Ejecuta 30 simulaciones del AG con parámetros base:

| Parámetro | Valor |
|-----------|-------|
| `crossover_probability` | 0.7 |
| `mutation_probability` | 0.005 |

Genera una gráfica con dos curvas:
- **Mejor individuo** (línea sólida azul oscuro): Evolución del mejor fitness promediado.
- **Promedio de la población** (línea discontinua azul): Evolución del fitness promedio promediado.

**Interpretación:** La curva del mejor individuo debería converger hacia 0 (máximo de la aptitud `−f(x,y)` cuando `f(x,y) → 0`), mientras que el promedio de la población converge más lentamente.

---

### Celda 7–8 — Simulación AG Modificado (Renormalización)

Implementa y compara tres variantes del AG mediante la función `f_con_escalado`:

#### Mecanismo de renormalización

La variable global `MODO_ESCALADO` controla cómo se modifica la función objetivo antes de calcular la aptitud:

```python
MODO_ESCALADO = None  # Sin renormalización (AG Simple)
MODO_ESCALADO = "lineal"  # Renormalización lineal
MODO_ESCALADO = "ventana"  # Renormalización por ventana
```

| Modo | Transformación | Efecto |
|------|---------------|--------|
| `None` (Simple) | `f(x, y)` sin cambios | Aptitud = `−f(x, y)` directamente |
| `"lineal"` | `f(x, y) × 0.5` | Escala la función al 50%, reduciendo las diferencias de aptitud entre individuos. Esto **disminuye la presión selectiva**, permitiendo mayor diversidad y exploración. |
| `"ventana"` | `f(x, y) − 10.0` | Desplaza la función restando 10. Como la aptitud es `−f(x,y)`, restar 10 a f equivale a sumar 10 a la aptitud de todos los individuos. Esto **eleva el piso de aptitud**, reduciendo la ventaja de los individuos más aptos y promoviendo diversidad. |

**Función `f_con_escalado(x, y)`:**

```python
def f_con_escalado(x, y):
    valor_crudo = f_original(x, y)
    if MODO_ESCALADO is None:
        return valor_crudo
    if MODO_ESCALADO == "ventana":
        return valor_crudo - 10.0
    if MODO_ESCALADO == "lineal":
        return valor_crudo * 0.5
    return valor_crudo
```

Esta función se asigna temporalmente a la variable global `f` para que `funcion_de_adaptacion` la use indirectamente. Después de las simulaciones, se restaura `f = f_original`.

**Ejecución:**

Se ejecutan 3 conjuntos de 30 simulaciones cada uno (Simple, Lineal, Ventana) con los mismos Pc=0.7 y Pm=0.005, generando una gráfica comparativa con 6 curvas (3 mejores + 3 promedios).

**Interpretación esperada:**

- **Renormalización lineal**: Al reducir las diferencias de aptitud, la selección por ruleta se vuelve más equitativa. Esto puede evitar convergencia prematura pero también ralentiza la convergencia.
- **Renormalización por ventana**: Al elevar el piso de aptitud, todos los individuos tienen una probabilidad de selección más similar. El efecto es análogo a la lineal pero con un desplazamiento constante.

---

### Celda 9–10 — Barrido de parámetros cruzados (Pc × Pm)

Realiza un barrido sistemático de combinaciones de parámetros para encontrar la configuración óptima.

**Diseño experimental:**

| Parámetro | Valores explorados |
|-----------|--------------------|
| Probabilidad de cruce (Pc) | [0.6, 0.75, 0.9] |
| Probabilidad de mutación (Pm) | [0.001, 0.005, 0.01] |
| Simulaciones por combinación | 10 (reducido vs. 30 de las secciones anteriores) |

**Total de ejecuciones:** 3 × 3 × 10 = 90 simulaciones del AG.

**Métrica de rendimiento:** El mejor fitness de la última generación, promediado sobre las 10 simulaciones. Se almacena en una matriz 3×3 (`matriz_rendimiento`).

**Visualización:** Mapa de calor (heatmap) con:
- Eje X: Probabilidad de mutación (Pm)
- Eje Y: Probabilidad de cruce (Pc)
- Color: Fitness final promedio (mayor es mejor, es decir, más cercano a 0)
- Anotaciones numéricas en cada celda con el valor exacto del fitness

**Interpretación esperada:**

- Pc alto (0.9) + Pm moderado (0.005–0.01): Mejor rendimiento por buena combinación de exploración (cruce) y diversidad (mutación).
- Pc bajo (0.6) + Pm bajo (0.001): Posible convergencia prematura por falta de exploración y diversidad.
- Pc alto + Pm alto: Puede introducir demasiada aleatoriedad, degradando la convergencia.

---

## Diseño de la representación binaria

### Cromosoma de 32 bits

```
|<--- 16 bits: x --->|<--- 16 bits: y --->|
| b15 b14 ... b1 b0  | b15 b14 ... b1 b0  |
```

### Mapeo a valores reales

```
bits_x → entero_x → x = −5 + entero_x × (10/65535)
bits_y → entero_y → y = −5 + entero_y × (10/65535)
```

### Resolución

- 16 bits → 65,536 valores posibles por variable
- Rango [−5, 5] → Resolución = 10/65535 ≈ 1.526 × 10⁻⁴
- Espacio de búsqueda: 65,536² ≈ 4.3 × 10⁹ puntos

---

## Notas de implementación

### 1. Mutación nativa vs. personalizada

A diferencia de la Tarea 01 donde se implementó una función `custom_mutation` explícita para flip de bits, este notebook utiliza los parámetros nativos de PyGAD:

```python
mutation_by_replacement=True,
random_mutation_min_val=0,
random_mutation_max_val=2
```

Con `mutation_by_replacement=True`, PyGAD reemplaza el gen original con un valor aleatorio uniforme en [0, 2) truncado a entero, es decir, {0, 1}. Esto equivale a asignar un bit aleatorio, no a un flip. La diferencia sutil es:

| Método | Comportamiento |
|--------|----------------|
| Flip (Tarea 01) | Si bit=0 → 1; si bit=1 → 0. Siempre cambia. |
| Reemplazo (este notebook) | Asigna 0 o 1 aleatoriamente. 50% de probabilidad de cambiar. |

La probabilidad efectiva de cambio por gen es `Pm × 0.5`, mientras que con flip es `Pm × 1.0`.

### 2. Renormalización vía modificación de la función objetivo

La renormalización se implementa reemplazando temporalmente la función global `f` con `f_con_escalado`. Esto funciona porque `funcion_de_adaptacion` llama a `f(x, y)` en tiempo de ejecución, y Python resuelve el nombre `f` dinámicamente.

**Flujo de ejecución:**

```
f = f_original           # Guardar referencia
f = f_con_escalado       # Reemplazar globalmente
MODO_ESCALADO = "lineal" # Configurar modo
# ... ejecutar simulación ...
f = f_original           # Restaurar
```

**Limitación:** Este enfoque usa una variable global mutable (`MODO_ESCALADO`) que debe ser configurada antes de cada llamada a `ejecucion_simulacion`. No es thread-safe y puede causar errores si se ejecutan simulaciones en paralelo.

### 3. Promediado estadístico

La función `ejecucion_simulacion` promedia los resultados de múltiples simulaciones independientes, lo que:

- Suaviza la variabilidad estocástica del AG.
- Permite comparaciones más robustas entre configuraciones.
- Reduce el efecto de runs atípicamente buenos o malos.

El número de simulaciones se reduce de 30 a 10 en el barrido de parámetros para mantener un tiempo de ejecución razonable (90 simulaciones totales vs. 270 si se usaran 30).

### 4. Callback `llamada_generaciones`

El callback se redefine dentro del bucle de simulaciones para capturar el contexto de cada iteración. Sin embargo, la variable `tipo_de_escalado` se recibe como parámetro pero la lógica de escalado por ventana (`conversion_de_ventana`) está incompleta (solo tiene un `pass`). Esto sugiere que la renormalización por ventana fue simplificada a un desplazamiento constante (−10.0) en `f_con_escalado`.

---

## Resumen de secciones y outputs

| Celda | Sección | Output |
|-------|---------|--------|
| 0–2 | Librerías | Instalación e importación |
| 3–4 | Funciones | Definición de funciones (sin output visual) |
| 5–6 | AG Simple | Gráfica de evolución (30 simulaciones, Pc=0.7, Pm=0.005) |
| 7–8 | AG Modificado | Gráfica comparativa Simple vs. Lineal vs. Ventana (30 sim c/u) |
| 9–10 | Barrido Pc×Pm | Mapa de calor 3×3 con fitness final (10 sim por celda) |
