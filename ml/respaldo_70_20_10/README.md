# Respaldo del artifact 70/20/10 (esquema anterior)

Copia íntegra del artifact que servía el sistema **antes** del cambio al esquema
de partición 80/20. Se conserva para poder reproducir y citar los resultados
publicados con el esquema anterior sin volver a entrenarlo.

## Qué contiene

| fichero | qué es |
|---|---|
| `modelo_meta.json` | metadatos, encoders, catálogos y métricas del artifact 70/20/10 |
| `modelo_xgboost_flete.pkl` | modelo puntual |
| `modelo_xgboost_flete_q_lo.pkl` | modelo de cuantil 2.5% |
| `modelo_xgboost_flete_q_hi.pkl` | modelo de cuantil 97.5% |
| `data_pipeline_70_20_10.py.bak` | el `data_pipeline.py` tal y como estaba antes del cambio |

## Cómo era ese esquema

Cortes por **cuantil posicional** sobre las fechas (0.70 y 0.90), redondeados al
límite del día siguiente:

```
TRAIN | 56,419 filas | 2021-04-02 → 2024-10-10
VAL   | 17,208 filas | 2024-10-11 → 2025-08-15
TEST  |  8,613 filas | 2025-08-16 → 2025-12-31
```

Métricas que publicaba: MAPE de test **22.08 %**, escenario congelado 26.64 %,
cobertura del IC95 en test 99.56 %.

## Por qué NO son comparables con las del esquema 80/20

El TEST de este esquema eran **cinco meses** del tramo final de 2025, un periodo
excepcionalmente calmo de mercado. El holdout del esquema 80/20 son los **doce
meses** de 2025, con hasta once meses de horizonte para el modelo congelado. Que
el MAPE pase de 22.08 % a ~27 % **no significa que el modelo haya empeorado**:
es el mismo tipo de modelo evaluado sobre un problema más largo y más difícil.
Comparar las dos cifras directamente es el error que este README existe para
evitar.

## Cómo reproducirlo

El código del esquema sigue vivo y es seleccionable — no hay que restaurar nada
para volver a construir sus particiones:

```python
from ml import data_pipeline as dp
p = dp.construir("resultado_combinado.csv", esquema="70_20_10")
```

Aviso: reconstruir las *particiones* no reconstruye el *artifact*. El modelo se
entrena hoy con una receta distinta (sin early stopping, número de árboles por
origen rodante interno — ver `ml/seleccion_iteraciones.py`), así que entrenar
con `esquema="70_20_10"` da un modelo parecido pero no idéntico a estos `.pkl`.
Los ficheros de esta carpeta son la única copia exacta del artifact publicado.

## Cómo volver a servirlo (si alguna vez hiciera falta)

Copiar los cuatro ficheros a `ml/` y recargar el artifact:

```bash
cp ml/respaldo_70_20_10/modelo_*.pkl ml/respaldo_70_20_10/modelo_meta.json ml/
curl -X POST http://localhost:8000/api/maintenance/model/reload -H "Authorization: Bearer <token admin>"
```

Ese meta **no** contiene la clave `walk_forward_2025`, así que el panel de
mantenimiento mostrará el apartado de origen rodante vacío. Es correcto: ese
artifact es anterior al protocolo.
