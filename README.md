# Métrica de privacidad para *prompts* de LLMs

Sistema que estima, de forma interpretable, el riesgo de reidentificación de un *prompt* dirigido a un modelo de lenguaje (LLM) y sugiere reformulaciones para reducirlo. Se basa en una adaptación del modelo clásico de **k-anonimato** a texto libre: dado un *prompt*, se estima con cuántos individuos de un universo de comparación sería compatible la información que revela. El resultado es un número *k* con una lectura directa: *"soy uno entre k"*.

Código asociado al Trabajo de Fin de Grado **"Diseño de un GPT para la evaluación y protección de la privacidad en prompts de usuario"**.

- **Autor:** Miguel Pérez de Ciriza Lacunza
- **Director:** Javier Parra Arnau (Departamento de Ingeniería Telemática)
- **Titulación:** Grado en Ciencia e Ingeniería de Datos — UPC (FIB · ETSETB · FME)

> **Aviso:** este es un prototipo académico. El valor de *k* se computa sobre un universo de comparación **sintético** (`personas_sinteticas.csv`), por lo que sirve para comparar el grado de identificabilidad relativo entre *prompts* y validar las propiedades de la métrica, **no** como una estimación calibrada sobre una población real ni como garantía jurídica de anonimato.

---

## ¿Qué hace?

El sistema recibe un *prompt* en lenguaje natural y produce:

1. Los **atributos personales** (cuasi-identificadores) detectados en el texto.
2. El valor de **k** sobre el universo de comparación y una **clasificación cualitativa** del riesgo (crítico / alto / medio / bajo).
3. Un conjunto de **recomendaciones de generalización** y el *prompt* reescrito, si *k* queda por debajo del objetivo.

Se organiza en tres módulos funcionales más un orquestador:

- un **extractor** basado en LLM que convierte el texto en atributos estructurados,
- un **motor de k** que cuenta candidatos compatibles sobre el universo sintético,
- un **recomendador** voraz que generaliza atributos minimizando la pérdida de utilidad.

---

## Estructura del repositorio

| Fichero | Descripción |
|---|---|
| `extractor.py` | Extractor de cuasi-identificadores y relevancia vía OpenAI API (*structured outputs*). |
| `metrica_k.py` | Configuración del dominio, normalización, cálculo de *k* y sesión acumulativa multi-prompt. |
| `generar_dataset.py` | Generador reproducible del universo sintético `personas_sinteticas.csv`. |
| `modificador_prompt.py` | Loss Metric, coste de utilidad, búsqueda voraz del recomendador, reescritura y clasificación de riesgo. |
| `pipeline.py` | Orquestador y CLI: análisis puntual de un *prompt* y modo interactivo multi-prompt. |
| `evaluar.py` | Experimentos H1, H3, H4, H5 y H6 sobre el corpus sintético. |
| `evaluar_estabilidad.py` | Experimento H2 (estabilidad ante reformulaciones) con el extractor real. |
| `evaluar_tria.py` | Comprobación de coherencia: relación entre *k* y un ataque de reidentificación por *embeddings*. |
| `personas_sinteticas.csv` | Universo de comparación sintético (se regenera con `generar_dataset.py`). |
| `resultados_eval/` | Salidas de los experimentos (JSON y figuras). |

---

## Requisitos e instalación

Requiere **Python 3.10+**.

```bash
# Dependencias del sistema principal
pip install openai python-dotenv pandas
pip install matplotlib            # opcional, solo para las gráficas

# Dependencias adicionales del experimento de coherencia (evaluar_tria.py)
pip install sentence-transformers torch numpy
```

O, si prefieres, crea un `requirements.txt` con:

```
openai
python-dotenv
pandas
matplotlib
sentence-transformers
torch
numpy
```

### Clave de API

El extractor llama a la API de OpenAI. Define tu clave en un fichero `.env` en la raíz del repositorio:

```
OPENAI_API_KEY=sk-...
```

> El fichero `.env` está excluido y **nunca** debe subirse al repositorio.

---

## Uso

### 1. Generar el universo sintético

```bash
python generar_dataset.py
```

Produce `personas_sinteticas.csv` (1500 individuos ficticios, semilla fija `seed=42`, totalmente reproducible).

### 2. Analizar un *prompt*

```bash
python pipeline.py "Soy investigadora en IA en una universidad pública de Valencia, nací en 1981. ¿Qué congresos me recomiendas?"
```

Sin clave de API, usando una extracción simulada:

```bash
python pipeline.py --mock
```

### 3. Modo interactivo (sesión multi-prompt)

```bash
python pipeline.py -i
python pipeline.py -i --mock     # interactivo sin API
```

La sesión acumula la información revelada a lo largo de varios *prompts* y muestra el *k* acumulado en cada turno. Comandos: `/salir`, `/reset`, `/estado`, `/ayuda`.

### 4. Reproducir los experimentos de la memoria

```bash
python evaluar.py                # H1, H3, H4, H5, H6 (corpus sintético)
python evaluar_estabilidad.py    # H2 (requiere API real)
python evaluar_tria.py           # comprobación de coherencia (embeddings)
```

Los resultados se guardan en `resultados_eval/`.

---

## Reproducibilidad

- Generación del dataset con semilla fija (`seed=42`).
- Extractor invocado con `temperature=0` y *structured outputs* en modo `strict`.
- Universo de comparación sintético versionado en el repositorio.

Estas tres condiciones hacen que la ejecución sobre un mismo *prompt* sea, dentro de los márgenes propios de la API, reproducible.

---

## Correspondencia con la memoria

| Componente | Secciones de la memoria |
|---|---|
| Extractor | §4.2 (diseño) y Anexo B (*system prompt* completo) |
| Universo de comparación / cálculo de *k* | §4.3 y §4.4 |
| Recomendador (Loss Metric, búsqueda voraz, Algoritmo 1) | §4.5 |
| Arquitectura y flujo end-to-end | §4.1 y §4.6 |
| Evaluación (H1–H6) | §5.4 |
| Comprobación de coherencia con TRIA | §5.4.7 |


