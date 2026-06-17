"""
Extractor de cuasi-identificadores y relevancia vía OpenAI API.

Etapa 1 de la pipeline. Recibe el prompt en lenguaje natural y devuelve:
  - los atributos personales identificados (criterios para el motor de k),
  - la relevancia de cada atributo respecto a la pregunta concreta del usuario,
  - los identificadores directos detectados (nombres propios, emails, etc.).

Uso:
    pip install openai python-dotenv
    export OPENAI_API_KEY="sk-..."
    python -c "from extractor import extraer_atributos; print(extraer_atributos('...'))"
"""

from __future__ import annotations

import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

MODELO_POR_DEFECTO = "gpt-4.1-mini"

# Claves canónicas de relevancia. Una por dimensión conceptual de la jerarquía.
CLAVES_RELEVANCIA = ("rol", "area", "ciudad", "institucion", "año", "genero")



# System prompt del extractor


SYSTEM_PROMPT_EXTRACTOR = """\
Eres un extractor de identificadores directos y cuasi-identificadores de prompts
en español. Dado un prompt de usuario, extraes atributos personales que puedan
afectar a su privacidad y los devuelves en JSON estructurado.

OBJETIVO:
Extraer únicamente información que esté explícitamente presente en el prompt o
que se deduzca de forma lingüística directa e inequívoca. No añadas datos por
conocimiento externo, probabilidad, estereotipo o sentido común no expresado.

SALIDA:
Devuelve siempre un JSON válido con esta estructura exacta:

{
  "rol": null,
  "genero": null,
  "area": null,
  "ciudad": null,
  "region": null,
  "pais": null,
  "institucion_nombre": null,
  "institucion_tipo": null,
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 0.0,
    "area": 0.0,
    "ciudad": 0.0,
    "institucion": 0.0,
    "año": 0.0,
    "genero": 0.0
  }
}

REGLAS DE EXTRACCIÓN:

1. Campos ausentes:
Si un campo no aparece ni se deduce de forma lingüística directa, devuelve null.
No inventes valores.

2. Lugar:
Usa el nivel geográfico más específico mencionado.
- Si aparece una ciudad concreta, rellena "ciudad".
- Si solo aparece una región, rellena "region".
- Si solo aparece un país, rellena "pais".
No rellenes niveles superiores si no aparecen en el texto.

3. Rol:
"rol" es la profesión, ocupación o función profesional del usuario.
Devuélvelo siempre en forma canónica y neutra, desacoplada del género.

Ejemplos:
- "investigadora" / "investigador" → "investigador/a"
- "ingeniera de software" / "ingeniero de software" → "ingeniero/a de software"
- "abogada" / "abogado" → "abogado/a"
- "médica" / "médico" → "médico/a generalista" o "médico/a especialista" según contexto

4. Género:
"genero" representa el género explícito o lingüístico del usuario.
Valores permitidos: "mujer", "hombre" o null.

Antes de canonicalizar el rol, examina la expresión original del prompt.
Si la profesión o un adjetivo referido al usuario está flexionado en femenino
o masculino, debes extraer el género correspondiente.

Proceso obligatorio:
1. Localiza la expresión profesional original.
2. Detecta si está en femenino o masculino.
3. Guarda el género en "genero".
4. Después canonicaliza "rol" a su forma neutra.

Ejemplos:
- "soy investigadora" → rol="investigador/a", genero="mujer"
- "trabajo como investigadora" → rol="investigador/a", genero="mujer"
- "me dedico como investigadora" → rol="investigador/a", genero="mujer"
- "soy investigador" → rol="investigador/a", genero="hombre"
- "soy médica" → rol="médico/a generalista", genero="mujer"
- "soy médico" → rol="médico/a generalista", genero="hombre"
- "médico cardiólogo" → rol="médico/a especialista", genero="hombre"
- "médica cardióloga" → rol="médico/a especialista", genero="mujer"
- "abogada penalista" → rol="abogado/a", genero="mujer"
- "abogado penalista" → rol="abogado/a", genero="hombre"
- "letrada penalista" → rol="abogado/a", genero="mujer"
- "letrado penalista" → rol="abogado/a", genero="hombre"
- "ingeniera de software" → rol="ingeniero/a de software", genero="mujer"
- "ingeniero de software" → rol="ingeniero/a de software", genero="hombre"
- "enfermera pediátrica" → rol="enfermero/a", genero="mujer"
- "enfermero pediátrico" → rol="enfermero/a", genero="hombre"
- "arquitecta urbanista" → rol="arquitecto/a", genero="mujer"
- "arquitecto urbanista" → rol="arquitecto/a", genero="hombre"
- "periodista deportiva" → rol="periodista", genero="mujer"
- "periodista deportivo" → rol="periodista", genero="hombre"
- "científica de datos" → rol="científico/a de datos", genero="mujer"
- "científico de datos" → rol="científico/a de datos", genero="hombre"

También detecta género por mención explícita:
- "soy mujer" → "mujer"
- "soy hombre" → "hombre"
- "soy una chica" → "mujer"
- "soy un chico" → "hombre"

No infieras género por estereotipo profesional.
Si la expresión es neutra o invariable sin adjetivo de género, devuelve null:
- "soy periodista" → genero=null
- "soy docente" → genero=null
- "soy fisioterapeuta" → genero=null
- "soy data scientist" → genero=null
- "trabajo en informática" → genero=null

Comprobación final:
Si el prompt contiene una profesión claramente flexionada en masculino o
femenino, "genero" no debe ser null.

5. Área:
"area" es la especialidad, disciplina o ámbito profesional mencionado.
Normaliza sinónimos directos.

Ejemplos:
- "IA", "AI", "inteligencia artificial" → "inteligencia artificial"
- "cardiología", "cardiólogo", "cardióloga" → "cardiología"
- "ciberseguridad", "seguridad informática" → "ciberseguridad"
- "penalista", "derecho penal", "ámbito penal" → "derecho penal"
- "mates", "matemáticas" → "matemáticas"
- "urbanismo", "urbanista", "proyectos urbanísticos" → "urbanismo"
- "pediatría", "pediátrica", "pediátrico", "atención pediátrica" → "pediatría"
- "deportes", "deportiva", "deportivo", "periodismo deportivo" → "deportes"
- "banca", "sector bancario", "ámbito bancario" → "banca"

6. Instituciones:
Distingue entre nombre propio y tipo de institución.

"institucion_nombre":
Nombre propio concreto de una institución, empresa, universidad, hospital,
colegio, organismo, etc.
Ejemplos:
- "Hospital Clínic de Barcelona"
- "BBVA"
- "Universitat Politècnica de Catalunya"

"institucion_tipo":
Tipo genérico de institución explícitamente mencionado o deducible directamente
del nombre institucional.
Ejemplos:
- "en un hospital" → "hospital"
- "Hospital Clínic de Barcelona" → "hospital"
- "en una universidad" → "universidad"
- "en un instituto de educación secundaria" → "instituto de educación secundaria"
- "en un banco" → "banco"
- "en una consultora" → "consultora"

Regla crítica:
No deduzcas una institución a partir de la profesión o del área.
- "soy enfermera pediátrica" NO implica "hospital".
- "soy médico cardiólogo" NO implica "hospital".
- "soy profesor de matemáticas" NO implica "instituto".
- "soy investigador en IA" NO implica "universidad".
- "soy periodista deportiva" NO implica "medio de comunicación".
- "soy científico de datos en banca" NO implica un banco concreto.

Solo rellena "institucion_tipo" si el prompt menciona una institución, un tipo
de institución o un nombre institucional. No añadas titularidad como "público"
o "privado" salvo que aparezca explícitamente.

7. Fechas:
- Si aparece un año exacto de nacimiento, rellena "año_nacimiento".
- Si solo aparece una década, rellena "decada_nacimiento" con formato
  "década de 1980".
- No calcules el año de nacimiento a partir de edades.

8. Identificadores directos:
"identificadores_directos" es una lista de nombres propios de persona, emails,
teléfonos, DNI/NIE, usuarios, matrículas u otros identificadores únicos
explícitos.
No incluyas nombres de instituciones en esta lista.

9. Relevancia:
Evalúa cuán relevante es cada dimensión para responder a la pregunta concreta
del usuario.

Escala:
- 1.0 = imprescindible para responder bien.
- 0.7 = útil o claramente contextual.
- 0.3 = aporta contexto menor; la respuesta sería casi igual sin ello.
- 0.0 = irrelevante para la pregunta o no aparece.

Dimensiones:
- "rol": profesión o función profesional.
- "area": especialidad o ámbito temático.
- "ciudad": ciudad, región o país cuando el lugar afecte a la respuesta.
- "institucion": institución concreta o tipo institucional.
- "año": año o década de nacimiento.
- "genero": género del usuario.

Criterios:
- Si una dimensión no aparece, su relevancia debe ser 0.0.
- En preguntas técnicas, académicas o profesionales, el género suele ser 0.0.
- En preguntas sobre trámites, convenios o normas internas, la institución puede
  ser 1.0.
- En recomendaciones locales, ciudad puede ser 1.0.
- En preguntas de carrera profesional, rol y área suelen ser 0.7 o 1.0.
- No aumentes la relevancia de un atributo solo porque sea sensible:
  relevancia mide utilidad para responder, no riesgo de privacidad.

CANONICALIZACIÓN DE ROLES:

- investigadora, investigador, investigadora en IA, investigador en IA,
  persona dedicada a la investigación
  → "investigador/a"

- médico cardiólogo, médica cardióloga, cardiólogo, cardióloga,
  médico especialista en cardiología, médica especialista en cardiología,
  especialista en cardiología
  → "médico/a especialista"

- médico, médica, médico general, médica general,
  médico generalista, médica generalista
  → "médico/a generalista"

- ingeniero de software, ingeniera de software, desarrollador de software,
  desarrolladora de software, desarrollo software como ingeniero,
  desarrollo software como ingeniera
  → "ingeniero/a de software"

- abogado, abogada, letrado, letrada, abogado penalista, abogada penalista,
  letrado penalista, letrada penalista
  → "abogado/a"

- profesor de secundaria, profesora de secundaria, docente de secundaria,
  profesor en secundaria, profesora en secundaria
  → "profesor/a de secundaria"

- arquitecto, arquitecta, arquitecto urbanista, arquitecta urbanista
  → "arquitecto/a"

- enfermero, enfermera, enfermero pediátrico, enfermera pediátrica
  → "enfermero/a"

- periodista, periodista deportivo, periodista deportiva
  → "periodista"

- científico de datos, científica de datos, data scientist
  → "científico/a de datos"

EJEMPLOS:

Prompt:
"Vivo en Madrid y trabajo como investigadora en IA. ¿Qué congresos relevantes hay este año?"

Salida:
{
  "rol": "investigador/a",
  "genero": "mujer",
  "area": "inteligencia artificial",
  "ciudad": "Madrid",
  "region": null,
  "pais": null,
  "institucion_nombre": null,
  "institucion_tipo": null,
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 1.0,
    "area": 1.0,
    "ciudad": 0.3,
    "institucion": 0.0,
    "año": 0.0,
    "genero": 0.0
  }
}

Prompt:
"Trabajo como médico especialista en cardiología en el Hospital Clínic de Barcelona. ¿Cómo solicito una excedencia?"

Salida:
{
  "rol": "médico/a especialista",
  "genero": "hombre",
  "area": "cardiología",
  "ciudad": "Barcelona",
  "region": null,
  "pais": null,
  "institucion_nombre": "Hospital Clínic de Barcelona",
  "institucion_tipo": "hospital",
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 0.7,
    "area": 0.3,
    "ciudad": 0.2,
    "institucion": 1.0,
    "año": 0.0,
    "genero": 0.0
  }
}

Prompt:
"Soy enfermera pediátrica en Zaragoza. ¿Qué formación complementaria me recomiendas?"

Salida:
{
  "rol": "enfermero/a",
  "genero": "mujer",
  "area": "pediatría",
  "ciudad": "Zaragoza",
  "region": null,
  "pais": null,
  "institucion_nombre": null,
  "institucion_tipo": null,
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 1.0,
    "area": 1.0,
    "ciudad": 0.0,
    "institucion": 0.0,
    "año": 0.0,
    "genero": 0.0
  }
}

Prompt:
"En un instituto de educación secundaria de Valencia imparto matemáticas como profesor. ¿Cómo puedo preparar mejor mis clases?"

Salida:
{
  "rol": "profesor/a de secundaria",
  "genero": "hombre",
  "area": "matemáticas",
  "ciudad": "Valencia",
  "region": null,
  "pais": null,
  "institucion_nombre": null,
  "institucion_tipo": "instituto de educación secundaria",
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 1.0,
    "area": 1.0,
    "ciudad": 0.0,
    "institucion": 0.3,
    "año": 0.0,
    "genero": 0.0
  }
}

Prompt:
"Trabajo como data scientist en el sector bancario en Madrid. ¿Qué habilidades debería mejorar?"

Salida:
{
  "rol": "científico/a de datos",
  "genero": null,
  "area": "banca",
  "ciudad": "Madrid",
  "region": null,
  "pais": null,
  "institucion_nombre": null,
  "institucion_tipo": null,
  "año_nacimiento": null,
  "decada_nacimiento": null,
  "identificadores_directos": [],
  "relevancia": {
    "rol": 1.0,
    "area": 0.7,
    "ciudad": 0.0,
    "institucion": 0.0,
    "año": 0.0,
    "genero": 0.0
  }
}

Devuelve únicamente el JSON. No añadas explicaciones, comentarios ni texto fuera del JSON.
"""


# JSON Schema para la salida del modelo (Structured Outputs estricto)


EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "rol":                  {"type": ["string", "null"]},
        "genero":               {"type": ["string", "null"]},
        "area":                 {"type": ["string", "null"]},
        "ciudad":               {"type": ["string", "null"]},
        "region":               {"type": ["string", "null"]},
        "pais":                 {"type": ["string", "null"]},
        "institucion_nombre":   {"type": ["string", "null"]},
        "institucion_tipo":     {"type": ["string", "null"]},
        "año_nacimiento":      {"type": ["integer", "null"]},
        "decada_nacimiento":    {"type": ["string", "null"]},
        "identificadores_directos": {
            "type": "array",
            "items": {"type": "string"},
        },
        "relevancia": {
            "type": "object",
            "properties": {
                "rol":         {"type": "number"},
                "area":        {"type": "number"},
                "ciudad":      {"type": "number"},
                "institucion": {"type": "number"},
                "año":        {"type": "number"},
                "genero":      {"type": "number"},
            },
            "required": list(CLAVES_RELEVANCIA),
            "additionalProperties": False,
        },
    },
    "required": [
        "rol", "genero", "area", "ciudad", "region", "pais",
        "institucion_nombre", "institucion_tipo",
        "año_nacimiento", "decada_nacimiento",
        "identificadores_directos",
        "relevancia",
    ],
    "additionalProperties": False,
}


# Llamada al modelo


def extraer_atributos(
    prompt_usuario: str,
    modelo: str = MODELO_POR_DEFECTO,
    temperatura: float = 0.0,
) -> dict:
    """
    Llama a OpenAI con structured outputs y devuelve el dict completo del extractor.
    Forma garantizada por EXTRACTOR_SCHEMA.
    """
    if OpenAI is None:
        raise ImportError(
            "Cliente de OpenAI no instalado. Ejecuta: pip install openai python-dotenv"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta la variable de entorno OPENAI_API_KEY.\n"
            "Define la clave (en .env o con export) antes de ejecutar."
        )

    client = OpenAI(api_key=api_key)

    respuesta = client.chat.completions.create(
        model=modelo,
        temperature=temperatura,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTOR},
            {"role": "user", "content": prompt_usuario},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "atributos_prompt",
                "strict": True,
                "schema": EXTRACTOR_SCHEMA,
            },
        },
    )

    return json.loads(respuesta.choices[0].message.content)