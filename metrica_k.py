"""
Cálculo de k-anonymity sobre prompts.

Este módulo es puro: solo configuración, normalización y matching contra el
dataset. No conoce nada del LLM ni del modificador de prompts. Lo importan
tanto el modificador como el orquestador (pipeline.py).

Componentes:
  - HIERARCHIES, ALIAS_A_CRITERIO, SINONIMOS: configuración del dominio.
  - norm, aplicar_sinonimos, normalizar_criterios: pre-procesado.
  - calcular_k_estricto: k clásico (matching exacto en QIDs presentes).
  - SesionPrivacidad: cálculo acumulativo entre prompts.
"""

from __future__ import annotations


import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# =====================================================================
# Configuración del dominio
# =====================================================================

DATASET_PATH = Path("personas_sinteticas.csv")

# Jerarquía de generalización: cada clave conceptual lista las columnas del
# dataset de más específico (índice 0) a más general. "*" es supresión total.
HIERARCHIES: dict[str, list[str]] = {
    "rol":         ["profesion_rol", "profesion_rol_general", "*"],
    "area":        ["area_especializacion", "area_general", "*"],
    "ciudad":      ["ciudad", "region", "pais", "*"],
    "institucion": ["institucion_nombre", "institucion_tipo", "institucion_general", "*"],
    "año":        ["año_nacimiento", "decada_nacimiento", "*"],
    "genero":      ["genero", "*"],
}

# Mapeo de los nombres que saca el GPT a (clave_canonica, nivel).
ALIAS_A_CRITERIO: dict[str, tuple[str, int]] = {
    "rol": ("rol", 0),
    "area": ("area", 0),
    "ciudad": ("ciudad", 0),
    "institucion": ("institucion", 0),
    "año": ("año", 0),
    "genero": ("genero", 0),
    "profesion_rol":          ("rol", 0),
    "profesion_rol_general":  ("rol", 1),
    "area_especializacion":   ("area", 0),
    "area_general":           ("area", 1),
    "region":                 ("ciudad", 1),
    "pais":                   ("ciudad", 2),
    "institucion_nombre":     ("institucion", 0),
    "institucion_tipo":       ("institucion", 1),
    "institucion_general":    ("institucion", 2),
    "año_nacimiento":        ("año", 0),
    "decada_nacimiento":      ("año", 1),
}

# Sinónimos para canonicalizar valores extraídos del prompt al formato del
# dataset. IMPORTANTE: las CLAVES van normalizadas (minúsculas, sin diacríticos),
# porque la búsqueda se hace sobre norm(valor). Los VALORES pueden llevar acento;
# se normalizan en el matching contra el dataset.
SINONIMOS: dict[str, dict[str, str]] = {
    "rol": {
        # Investigación
        "investigador": "investigador/a",
        "investigadora": "investigador/a",
        "investigador en ia": "investigador/a",
        "investigadora en ia": "investigador/a",
        "persona dedicada a la investigacion": "investigador/a",
        # Medicina generalista
        "medico": "médico/a generalista",
        "medica": "médico/a generalista",
        "medico general": "médico/a generalista",
        "medica general": "médico/a generalista",
        "medico generalista": "médico/a generalista",
        "medica generalista": "médico/a generalista",
        "medico de familia": "médico/a generalista",
        "medica de familia": "médico/a generalista",
        # Medicina especialista
        "medico especialista": "médico/a especialista",
        "medica especialista": "médico/a especialista",
        "cardiologo": "médico/a especialista",
        "cardiologa": "médico/a especialista",
        "medico cardiologo": "médico/a especialista",
        "medica cardiologa": "médico/a especialista",
        "especialista en cardiologia": "médico/a especialista",
        # Ingeniería de software
        "ingeniero de software": "ingeniero/a de software",
        "ingeniera de software": "ingeniero/a de software",
        "desarrollador de software": "ingeniero/a de software",
        "desarrolladora de software": "ingeniero/a de software",
        "desarrollador": "ingeniero/a de software",
        "desarrolladora": "ingeniero/a de software",
        "programador": "ingeniero/a de software",
        "programadora": "ingeniero/a de software",
        # Derecho
        "abogado": "abogado/a",
        "abogada": "abogado/a",
        "letrado": "abogado/a",
        "letrada": "abogado/a",
        "abogado penalista": "abogado/a",
        "abogada penalista": "abogado/a",
        # Docencia
        "profesor de secundaria": "profesor/a de secundaria",
        "profesora de secundaria": "profesor/a de secundaria",
        "docente de secundaria": "profesor/a de secundaria",
        "profesor en secundaria": "profesor/a de secundaria",
        "profesora en secundaria": "profesor/a de secundaria",
        # Arquitectura
        "arquitecto": "arquitecto/a",
        "arquitecta": "arquitecto/a",
        "arquitecto urbanista": "arquitecto/a",
        "arquitecta urbanista": "arquitecto/a",
        # Enfermería
        "enfermero": "enfermero/a",
        "enfermera": "enfermero/a",
        "enfermero pediatrico": "enfermero/a",
        "enfermera pediatrica": "enfermero/a",
        # Ciencia de datos
        "cientifico de datos": "científico/a de datos",
        "cientifica de datos": "científico/a de datos",
        "data scientist": "científico/a de datos",
        "analista de datos": "científico/a de datos",
        # Periodismo (invariable, pero cubrimos especialidades)
        "periodista deportivo": "periodista",
        "periodista deportiva": "periodista",
    },
    "area": {
        # IA / ML
        "ia": "inteligencia artificial",
        "ai": "inteligencia artificial",
        "machine learning": "aprendizaje automático",
        "ml": "aprendizaje automático",
        "aprendizaje maquina": "aprendizaje automático",
        "deep learning": "aprendizaje automático",
        "nlp": "procesamiento de lenguaje natural",
        "pln": "procesamiento de lenguaje natural",
        "procesamiento del lenguaje natural": "procesamiento de lenguaje natural",
        "computer vision": "visión por computador",
        "vision artificial": "visión por computador",
        "vision por computador": "visión por computador",
        # Seguridad
        "ciberseguridad": "ciberseguridad",
        "seguridad informatica": "ciberseguridad",
        "seguridad de la informacion": "ciberseguridad",
        "infosec": "ciberseguridad",
        # Derecho
        "penalista": "derecho penal",
        "ambito penal": "derecho penal",
        "penal": "derecho penal",
        # Salud
        "cardiologia": "cardiología",
        "cardiologo": "cardiología",
        "cardiologa": "cardiología",
        "pediatrica": "pediatría",
        "pediatrico": "pediatría",
        "atencion pediatrica": "pediatría",
        # Otros dominios
        "mates": "matemáticas",
        "urbanista": "urbanismo",
        "proyectos urbanisticos": "urbanismo",
        "deportiva": "deportes",
        "deportivo": "deportes",
        "periodismo deportivo": "deportes",
        "sector bancario": "banca",
        "ambito bancario": "banca",
        "finanzas bancarias": "banca",
    },
    "ciudad": {
        "bcn": "barcelona",
        "vlc": "valencia",
        "mad": "madrid",
        "vitoria": "vitoria-gasteiz",
        "gasteiz": "vitoria-gasteiz",
        "donosti": "donostia-san sebastian",
        "donostia": "donostia-san sebastian",
        "san sebastian": "donostia-san sebastian",
        "la coruna": "a coruna",
        "coruna": "a coruna",
    },
    "genero": {
        "femenino": "mujer",
        "masculino": "hombre",
        "f": "mujer",
        "m": "hombre",
        "fem": "mujer",
        "masc": "hombre",
        "chica": "mujer",
        "chico": "hombre",
        "varon": "hombre",
    },
}


# =====================================================================
# Normalización
# =====================================================================

def norm(x) -> str:
    """
    Normalización para matching exacto: strip, lower y eliminación de
    diacríticos. Tras esto, "València", "Valencia" y "VALENCIA " son iguales.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return s


def aplicar_sinonimos(clave: str, valor) -> str:
    """Mapea un valor a su forma canónica según el diccionario de sinónimos."""
    v = norm(valor)
    return SINONIMOS.get(clave, {}).get(v, v)


def normalizar_criterios(criterios_raw: dict) -> dict[str, dict]:
    """
    Convierte la salida del extractor (pares atributo → valor en lenguaje natural)
    al formato canónico { clave: {"valor": str, "nivel": int} }.
    Atributos no contemplados o vacíos se descartan.
    """
    perfil: dict[str, dict] = {}
    for k, v in criterios_raw.items():
        if v is None or norm(v) == "":
            continue
        if k not in ALIAS_A_CRITERIO:
            continue
        clave, nivel = ALIAS_A_CRITERIO[k]
        valor_canon = aplicar_sinonimos(clave, v)
        # Si ya hay valor para esta clave en el mismo prompt, conservar el más específico.
        if clave in perfil and perfil[clave]["nivel"] <= nivel:
            continue
        perfil[clave] = {"valor": valor_canon, "nivel": nivel}
    return perfil


# =====================================================================
# Cálculo de k
# =====================================================================

def calcular_k_estricto(df: pd.DataFrame, perfil: dict[str, dict]):
    """
    k-anonymity clásico: matching exacto contra el dataset.

    Se ignora silenciosamente cualquier criterio cuya columna no exista en el
    dataset, cuyo nivel sea supresión ("*") o cuyo valor esté vacío.
    Devuelve (k, candidatos_df, descripcion_atributos_usados).
    """
    mascara = pd.Series([True] * len(df), index=df.index)
    usados: list[str] = []

    for clave, info in perfil.items():
        if clave not in HIERARCHIES:
            continue
        niveles = HIERARCHIES[clave]
        nivel = info["nivel"]
        if nivel >= len(niveles):
            continue
        columna = niveles[nivel]
        if columna == "*" or columna not in df.columns:
            continue
        valor = info["valor"]
        if not valor:
            continue
        mascara &= df[columna].apply(norm).eq(norm(valor))
        usados.append(f"{clave}@{columna}={valor}")

    candidatos = df[mascara]
    return len(candidatos), candidatos, usados


# =====================================================================
# Sesión acumulativa
# =====================================================================

@dataclass
class SesionPrivacidad:
    """
    Mantiene el perfil del usuario a lo largo de varios prompts. Cada nuevo
    prompt puede añadir atributos nuevos, refinar (más específico), confirmar
    (mismo nivel y valor) o contradecir (mismo nivel, valor distinto: conflicto).

    El k acumulado es monótonamente decreciente o igual respecto al prompt anterior.
    """
    df: pd.DataFrame
    perfil: dict[str, dict] = field(default_factory=dict)
    historico: list[dict] = field(default_factory=list)
    conflictos: list[dict] = field(default_factory=list)

    def añadir_prompt(self, criterios_raw: dict) -> tuple[int, pd.DataFrame]:
        nuevos = normalizar_criterios(criterios_raw)

        for clave, info_nueva in nuevos.items():
            if clave not in self.perfil:
                self.perfil[clave] = info_nueva
                continue
            info_actual = self.perfil[clave]
            if info_nueva["nivel"] < info_actual["nivel"]:
                self.perfil[clave] = info_nueva
            elif info_nueva["nivel"] == info_actual["nivel"]:
                if info_nueva["valor"] != info_actual["valor"]:
                    self.conflictos.append({
                        "prompt_idx": len(self.historico) + 1,
                        "clave": clave,
                        "valor_previo": info_actual["valor"],
                        "valor_nuevo": info_nueva["valor"],
                    })

        k, candidatos, usados = calcular_k_estricto(self.df, self.perfil)
        self.historico.append({
            "n_prompt": len(self.historico) + 1,
            "atributos_acumulados": dict(self.perfil),
            "atributos_usados": usados,
            "k": k,
        })
        return k, candidatos

    def resumen(self) -> str:
        lineas = ["=== Histórico de sesión ==="]
        for h in self.historico:
            lineas.append(
                f"  Prompt #{h['n_prompt']}: "
                f"k = {h['k']:>4}   "
                f"({len(h['atributos_usados'])} atributos efectivos)"
            )
        if self.conflictos:
            lineas.append("\nConflictos detectados:")
            for c in self.conflictos:
                lineas.append(
                    f"  - {c['clave']}: '{c['valor_previo']}' vs '{c['valor_nuevo']}' "
                    f"(prompt #{c['prompt_idx']})"
                )
        return "\n".join(lineas)
