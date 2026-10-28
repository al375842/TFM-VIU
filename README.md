# Sistema de apoyo a la decisión financiera basado en aprendizaje automático

Lucas Niño Ruiz

Código del Trabajo Fin de Máster del Máster Universitario en Big Data y Ciencia de Datos (VIU), dirigido por el Dr. Arturo Peralta Martín-Palomino.

El sistema predice si un activo va a subir en la sesión siguiente y comprueba, mediante backtesting, si esas predicciones sirven para invertir. La pregunta del trabajo es si, usando solo precio y volumen, un modelo gana a una referencia que no usa ninguna información (el clasificador naïve, que siempre predice la clase más frecuente) y a comprar y mantener (Buy & Hold). El diseño completo y la discusión de los resultados están en la memoria.

## Contenido

```text
src/tfm/          paquete de Python con todo el pipeline
notebooks/        un notebook por módulo, que llama al paquete
tests/            58 pruebas automáticas (pytest)
docs/ejecucion/   registro de la ejecución final: entorno, huellas de los datos y logs
data/, results/   datos y resultados, que no se versionan porque se regeneran
```

El paquete tiene cinco módulos, uno por etapa, y cada uno responde a uno de los objetivos específicos de la memoria:

| Módulo | Fichero | Qué hace |
|---|---|---|
| M1 | `data.py` | descarga los datos con yfinance y los guarda en `data/raw/` |
| M2 | `features.py` | construye las variables predictoras y la variable objetivo |
| M3 | `models.py`, `validation.py` | define los modelos y la validación temporal |
| M4 | `backtest.py` | simula la estrategia frente a Buy & Hold |
| M5 | `evaluation.py` | genera las tablas y figuras de resultados |

Además, `lstm.py` y `walkforward.py` son las dos extensiones, `metrics.py` reúne las métricas y los contrastes estadísticos, `config.py` guarda la configuración y `pipeline.py` lo encadena todo desde la línea de comandos.

## Instalación

Hace falta Python 3.10 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -e ".[dev]"         # con ".[dev,lstm]" se instala también TensorFlow
pytest
```

La LSTM no necesita GPU: tiene 19.265 parámetros y entrena en unos 35 segundos por activo en CPU.

## Uso

```bash
python -m tfm.pipeline                          # descarga los datos y ejecuta el experimento
python -m tfm.pipeline --offline --lstm         # reutiliza los datos guardados e incluye la LSTM
python -m tfm.pipeline --tickers AAPL "^GSPC"   # solo algunos activos
python -m tfm.pipeline --offline --walk-forward --wf-backtest --lstm   # validación walk-forward
```

En Windows, los tickers con `^` necesitan comillas. Las tablas se guardan en `results/tables/` (en CSV y como fragmento LaTeX) y las figuras en `results/figures/`. Las figuras de resumen se pueden regenerar sin volver a entrenar:

```bash
python -c "from tfm import evaluation; evaluation.build_summary_figures()"
```

El experimento principal con la LSTM tarda unos 12 minutos en un procesador de 12 núcleos, y el walk-forward completo, algo menos de hora y media. Los notebooks del 01 al 06 recorren los mismos pasos módulo a módulo; el 01 descarga los datos que usan los demás.

## Diseño del experimento

Se usan catorce activos: cuatro índices, un ETF global, siete acciones y dos criptomonedas, con precios diarios ajustados desde 2000 (o desde que cotiza cada activo) hasta 2025. Cada activo mantiene su propio calendario, y los días sin dato se eliminan, porque rellenarlos sería inventar precios.

Con esos datos se calculan diez variables, todas con información hasta el cierre de cada día: rentabilidad logarítmica, medias móviles, volatilidad, variación del volumen, RSI y MACD. Lo que se predice es si el precio de cierre de la sesión siguiente supera al de hoy.

Se comparan seis familias de modelos: el naïve, la regresión logística, random forest, XGBoost, una SVM y, como extensión, una red LSTM. Se entrena con los datos anteriores a 2025 y se prueba en 2025. La última fila del entrenamiento se descarta, ya que su etiqueta usa un precio que ya es de 2025, y lo mismo se hace en la validación cruzada (`TimeSeriesSplit`). Los hiperparámetros se eligen con balanced accuracy y no con F1, porque el F1 le da un 0,70 a un modelo que siempre dice «sube».

En el backtesting, si al cierre el modelo predice subida, se compra en la apertura del día siguiente, y si no, el dinero queda en efectivo; nunca se apuesta a la baja. Una estrategia aporta valor si gana a Buy & Hold en rentabilidad acumulada y en ratio de Sharpe a la vez. La simulación no tiene comisiones, así que los resultados positivos son un máximo.

Como un buen resultado en 2025 podría deberse a ese año en concreto, el experimento se repite año a año, de 2016 a 2025, con una validación walk-forward de 129 pliegues. Las criptomonedas tienen menos historia, por lo que entre activos se comparan proporciones y no recuentos.

## Resultados

| Hipótesis | Resultado |
|---|---|
| H1. Los modelos aciertan más que el naïve | No recibe respaldo suficiente. En 2025, 20 de los 70 pares activo-modelo lo superan, pero por muy poco; solo uno tiene un p-valor inferior a 0,05 (random forest en Ethereum) y con la corrección de Bonferroni no queda ninguno. |
| H2. Los modelos no lineales mejoran a la regresión logística | No se confirma. Solo una de las 42 comparaciones con el contraste de McNemar es significativa, y a favor del modelo lineal. |
| H3. Acertar más no implica ganar más | Recibe apoyo descriptivo. 30 de los 70 pares ganan a Buy & Hold en rentabilidad y Sharpe, pero cuatro superan al naïve sin aportar valor y catorce aportan valor sin superarlo. |
| H4. La LSTM no compensa su coste | Es compatible con los resultados. Es la familia con la accuracy media más baja y aporta valor en cuatro activos. |

En el walk-forward, los modelos aciertan de media un 0,507, menos que el 0,531 del naïve, y solo un 29 % de las evaluaciones lo supera, con mucha diferencia entre años (un 63 % en 2022, que fue bajista, y un 6 % en 2024, muy alcista). En dinero, juntando los diez años, solo 6 de los 70 pares ganan a Buy & Hold, cuando en 2025 eran 30. El propio naïve lo consigue en 7 de los 14 activos, porque el simulador no replica exactamente comprar y mantener, razón por la cual una ventaja pequeña frente a Buy & Hold no basta para hablar de capacidad predictiva.

En resumen, usando solo precio y volumen los modelos no muestran una ventaja predictiva generalizable, cosa que encaja con la forma débil de la Hipótesis del Mercado Eficiente, aunque no la demuestra. Además, el modelo que más acierta no es necesariamente el que más gana, así que una estrategia hay que evaluarla también en dinero.

## Reproducibilidad

Todos los modelos con componente aleatoria usan la semilla 42, y los datos descargados se guardan en `data/raw/`, de modo que con `--offline` el experimento no depende de la red. El proveedor recalcula los precios ajustados con cada dividendo, así que una descarga nueva puede diferir un poco. Por eso [`docs/ejecucion/`](docs/ejecucion/) guarda las huellas SHA-256 de los datos utilizados, las versiones exactas de los paquetes y los logs de la ejecución final.

El experimento se ejecutó muchas veces durante su depuración, y cada corrección quedó cubierta por una prueba. Entre otras cosas, las pruebas comprueban que ninguna variable use información del futuro y que ninguna etiqueta del entrenamiento lea el bloque de prueba.

## Limitaciones y líneas futuras

La limitación que más pesa es que la simulación no tiene costes, así que los resultados positivos son un máximo. Además, son catorce activos y solo se usan precio y volumen. Como líneas futuras quedan validar el sistema con datos posteriores a 2025, añadir costes y controles de exposición, probar otras variables, como noticias o datos macroeconómicos, y predecir la volatilidad.
