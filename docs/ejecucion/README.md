# Registro de la ejecución definitiva

Material de reproducibilidad al que remite el Anexo A de la memoria. Los datos
descargados y los resultados no se versionan: se regeneran ejecutando el
*pipeline* (ver el [README principal](../../README.md)).

| Fichero | Contenido |
|---|---|
| `requirements-ejecucion.txt` | Versiones exactas de los paquetes del entorno en el que se ejecutaron los experimentos definitivos (`pip freeze`). |
| `sha256_datos.json` | Huella SHA-256 de cada uno de los 14 ficheros de `data/raw/`. Identifica la copia exacta de los datos usada: una descarga nueva puede diferir, porque el proveedor recalcula los precios ajustados con cada dividendo. |
| `principal.log` | Salida completa del experimento principal (`python -m tfm.pipeline --offline --lstm`). |
| `walkforward.log` | Salida completa del análisis walk-forward (`--offline --walk-forward --wf-backtest --lstm`). |
| `ejecuciones.json` | Orden ejecutada y duración de cada experimento: 743,9 s el principal y 5.216,6 s el walk-forward. |
| `resultados_primera_ejecucion.zip` | Tablas de resultados de la primera ejecución (agosto de 2026), conservadas solo como registro. Las cifras de la memoria proceden de la ejecución definitiva. |

Para comprobar que los datos locales coinciden con los de la memoria:

```bash
python -c "import hashlib, json; h = json.load(open('docs/ejecucion/sha256_datos.json')); print(all(hashlib.sha256(open(p, 'rb').read()).hexdigest() == d for p, d in h.items()))"
```
