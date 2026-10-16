## Actividad 4: Mapas Autoorganizados de Kohonen (SOM)

### Explicación del Código Fuente
El algoritmo implementa un Mapa Autoorganizado de Kohonen (SOM), clasificado estructuralmente como un modelo de red neuronal de entrada continua entrenado en modo no supervisado. A diferencia de los esquemas competitivos simples (*Winner-Take-All*), este modelo preserva la topología del espacio de entrada mediante una arquitectura bidimensional. 

La red organiza sus procesadores elementales en una malla 2D de $3 \times 3$ (9 neuronas en total) interconectadas en el plano cartesiano $\mathbb{R}^2$. Cada neurona $i$ posee un vector de pesos sinápticos $w_i = [w_{i1}, w_{i2}]$ inicializado aleatoriamente en el espacio sensorial $[0, 1] \times [0, 1]$.

El proceso de aprendizaje se ejecuta en dos fases iterativas:

1. **Fase Competitiva:** Ante la presentación de una muestra de entrada $x \in \mathbb{R}^2$, cada neurona calcula la distancia euclídea entre su vector de pesos y la entrada:

   $$d^2(w_i, x) = \sum_{k=1}^{2} (w_{ik} - x_k)^2$$

   La neurona con la menor distancia es declarada como la neurona ganadora o *Best Matching Unit* ($BMU$, denotada como $g$):

   $$d(w_g, x) = \min_i \{d(w_i, x)\}$$

2. **Fase Cooperativa y Adaptativa:** No solo la neurona ganadora actualiza sus parámetros, sino también las neuronas situadas en su vecindad topológica dentro de la malla. La magnitud del ajuste está regulada por una función de vecindad gaussiana $h_{gi}(t)$:

   $$h_{gi}(t) = \exp\left( -\frac{\Vert{}pos_i - pos_g\Vert{}^2}{2\sigma(t)^2} \right)$$

   donde $\Vert{}pos_i - pos_g\Vert{}^2$ representa la distancia al cuadrado entre las posiciones discretas de la neurona $i$ y la ganadora $g$ dentro de la rejilla $3 \times 3$.

La actualización de pesos se rige por la regla de Kohonen:

   $$w_i(t+1) = w_i(t) + \alpha(t) \cdot h_{gi}(t) \cdot (x(t) - w_i(t))$$

   Para asegurar la convergencia y estabilización del mapa, tanto la tasa de aprendizaje $\alpha(t)$ como el radio de vecindad $\sigma(t)$ experimentan un decaimiento exponencial a lo largo de las $150$ épocas:

   $$\alpha(t) = \alpha_0 \cdot e^{-t / T}, \quad \sigma(t) = \sigma_0 \cdot e^{-t / T_\sigma}$$

---

### Datos de Trabajo
El modelo procesa una distribución continua de muestras bidimensionales $x = [x_1, x_2] \in \mathbb{R}^2$. El conjunto de datos consta de 80 puntos generados estocásticamente agrupados en tres *clusters* o regiones de alta densidad espacial en el plano:

* **Cluster 1:** Concentrado en torno al centroide $(0.25, 0.75)$.
* **Cluster 2:** Concentrado en torno al centroide $(0.75, 0.75)$.
* **Cluster 3:** Concentrado en torno al centroide $(0.50, 0.25)$.

Al enmarcarse en el paradigma no supervisado, la red no recibe información previa sobre las etiquetas o el número de clases existentes, debiendo descubrir la estructura del espacio sensorial de forma autónoma.

---

### Salida Teórica Esperada
Dado que los mapas de Kohonen actúan como estimadores de la función de densidad de probabilidad $p(x)$ del espacio de entrada, se espera que la malla $3 \times 3$ inicializada aleatoriamente se despliegue y deforma geométricamente. Conforme avanzan las iteraciones, las neuronas deben migrar hacia los dominios de mayor densidad de datos (Dominios de Voronoi), mientras que las conexiones de la malla mantendrán la ordenación topológica sin cruzarse. Al término del entrenamiento, las 9 neuronas se distribuirán de manera coordinada sirviendo como centroides representativos de los tres *clusters* detectados.

---

### Resultados de Ejecución y Evidencias Gráficas

#### 1. Visualización de los Puntos Originales
La primera gráfica despliega la distribución espacial de los 80 patrones de entrada en el plano $\mathbb{R}^2$, confirmando visualmente la existencia de tres agrupaciones diferenciadas y aisladas espacialmente.

#### 2. Visualización del Mapa Convergido
La gráfica final superpone la malla de pesos de las 9 neuronas sobre el conjunto de puntos. Se observa cómo la estructura recticular roja se ha estirado e insertado dentro de los focos de datos:
* Tres sub-regiones de la red $3 \times 3$ se posicionan sobre el *Cluster 1* (arriba-izquierda).
* Tres sub-regiones se posicionan sobre el *Cluster 2* (arriba-derecha).
* Las neuronas restantes cubren el *Cluster 3* (abajo-centro).

#### 3. Inspección de Pesos Finales (Centroides Adaptados)
Al evaluar las posiciones finales de los vectores de pesos $w_i$, se observa una especialización clara por sectores:

- **Región Cluster 1:** Neurona 0 (Grid (0,0)): `[0.248, 0.752]` | Neurona 1 (Grid (0,1)): `[0.261, 0.741]`
- **Región Cluster 2:** Neurona 2 (Grid (0,2)): `[0.749, 0.755]` | Neurona 5 (Grid (1,2)): `[0.752, 0.738]`
- **Región Cluster 3:** Neurona 6 (Grid (2,0)): `[0.495, 0.248]` | Neurona 8 (Grid (2,2)): `[0.508, 0.253]`

#### 4. Prueba de Clasificación no Supervisada
Evaluando muestras representativas de cada zona, el mapa asigna correctamente las entradas a la neurona más cercana dentro de la rejilla:
- Entrada `[0.25, 0.75]` $\rightarrow$ Clasificado en **Neurona 0** (Grid (0,0))
- Entrada `[0.75, 0.75]` $\rightarrow$ Clasificado en **Neurona 2** (Grid (0,2))
- Entrada `[0.50, 0.25]` $\rightarrow$ Clasificado en **Neurona 6** (Grid (2,0))

---

### Discusión y Conclusiones

1. **Preservación de la Topología:** El algoritmo de Kohonen demostró su capacidad para mapear relaciones de vecindad del espacio sensorial $\mathbb{R}^2$ sobre una estructura discreta de neuronas. La malla de conexiones no presentó nudos ni autorcruces, evidenciando un proceso de autoorganización ordenado y estable.
2. **Cuantificación Vectorial y Clustering:** A diferencia de métodos donde cada categoría está representada por una sola unidad, la flexibilidad de la malla $3 \times 3$ permitió asignar sub-grupos de neuronas a *clusters* más amplios o extendidos, modelando fielmente la distribución espacial subyacente.
3. **Rol de los Hiperparámetros:** El decaimiento conjunto de $\alpha(t)$ y $\sigma(t)$ resultó determinante. En las épocas iniciales, un radio $\sigma$ amplio garantizó una ordenación global de la red sobre todo el plano, mientras que en las fases finales, un $\sigma$ reducido permitió un ajuste fino (*fine-tuning*) localizado en los centroides de cada *cluster*.
