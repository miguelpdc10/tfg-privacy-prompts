"""
evaluar_estabilidad.py — Test de la hipótesis H2 (estabilidad ante reformulaciones).

Para cada grupo de prompts, todos los miembros expresan el mismo contenido
informativo con variaciones triviales (orden de cláusulas, sinónimos del
diccionario, palabras de relleno). El sistema debería producir el MISMO
perfil normalizado para todas las variantes del grupo.

Métrica primaria: igualdad de los perfiles normalizados (clave, valor, nivel)
extraídos por el sistema. Esto evita que coincidencias accidentales en el
valor de \widehat{k} oculten inestabilidad real del extractor (caso clásico:
en un grupo donde \widehat{k}=0 para todas las variantes, distintas
extracciones podrían dar el mismo k=0 por falta de candidatos en el dataset,
sin que eso implique que el extractor sea estable).

Métrica secundaria (informativa): variación del valor de \widehat{k} sobre el
dataset actual.

Uso:
    python evaluar_estabilidad.py            # con API real
"""

from __future__ import annotations

import sys
import json
import statistics
from pathlib import Path
from typing import Optional

import pandas as pd

from metrica_k import (
    DATASET_PATH,
    normalizar_criterios,
)
import evaluar as ev  # para el monkey-patch rápido


# Grupos de variaciones

# Cada grupo describe el mismo perfil pero con variaciones de superficie.
# El sistema debería extraer los mismos atributos y por tanto el mismo k.

   
GRUPOS = [
    {
        "nombre": "G1 — Investigadora IA Madrid",
        "perfil_esperado": "rol=investigador/a, area=inteligencia artificial, ciudad=Madrid, genero=mujer",
        "variantes": [
            "Soy investigadora en inteligencia artificial y vivo en Madrid.",
            "Vivo en Madrid y trabajo como investigadora en IA.",
            "Soy una mujer dedicada a la investigación en inteligencia artificial en Madrid.",
            "Resido en Madrid y trabajo como investigadora en AI.",
            "En Madrid me dedico como investigadora al campo de la inteligencia artificial.",
        ],
    },
    {
        "nombre": "G2 — Médico cardiólogo Hospital Clínic Barcelona",
        "perfil_esperado": "rol=médico/a especialista, area=cardiología, ciudad=Barcelona, institucion_nombre=Hospital Clínic de Barcelona, institucion_tipo=hospital, genero=hombre",
        "variantes": [
            "Soy médico cardiólogo en el Hospital Clínic de Barcelona.",
            "Trabajo como médico especialista en cardiología en el Hospital Clínic de Barcelona.",
            "Ejerzo de cardiólogo en el Hospital Clínic de Barcelona.",
            "Soy hombre y trabajo como médico especializado en cardiología en el Hospital Clínic de Barcelona.",
            "Mi centro es el Hospital Clínic de Barcelona, donde soy médico especializado en cardiología.",
        ],
    },
    {
        "nombre": "G3 — Ingeniera software ciberseguridad Bilbao",
        "perfil_esperado": "rol=ingeniero/a de software, area=ciberseguridad, ciudad=Bilbao, genero=mujer",
        "variantes": [
            "Soy ingeniera de software especializada en ciberseguridad y vivo en Bilbao.",
            "Trabajo como ingeniera de software en seguridad informática en Bilbao.",
            "En Bilbao desarrollo software como ingeniera en el ámbito de la ciberseguridad.",
            "Soy una mujer que trabaja en ingeniería de software aplicada a ciberseguridad en Bilbao.",
            "Resido en Bilbao y me dedico como ingeniera de software al área de seguridad informática.",
        ],
    },
    {
        "nombre": "G4 — Abogada penalista Sevilla",
        "perfil_esperado": "rol=abogado/a, area=derecho penal, ciudad=Sevilla, genero=mujer",
        "variantes": [
            "Soy abogada penalista en Sevilla.",
            "Trabajo como abogada especializada en derecho penal en Sevilla.",
            "En Sevilla ejerzo como letrada penalista.",
            "Soy una mujer que trabaja como abogada en el ámbito penal en Sevilla.",
            "Resido en Sevilla y me dedico al derecho penal como abogada.",
        ],
    },
    {
        "nombre": "G5 — Profesor secundaria matemáticas Valencia",
        "perfil_esperado": "rol=profesor/a de secundaria, area=matemáticas, ciudad=Valencia, genero=hombre, institucion_tipo=instituto de educación secundaria",
        "variantes": [
            "Soy profesor de secundaria de matemáticas en un instituto de educación secundaria de Valencia.",
            "Doy clase de mates como profesor de secundaria en un instituto de educación secundaria de Valencia.",
            "Soy hombre y trabajo en Valencia como docente de secundaria especializado en matemáticas en un instituto de secundaria.",
            "Soy hombre y enseño matemáticas como profesor de secundaria en un instituto de educación secundaria de Valencia.",
            "En un instituto de educación secundaria de Valencia imparto la asignatura de matemáticas como profesor.",
        ],
    },
    {
        "nombre": "G6 — Ingeniero software IA Barcelona",
        "perfil_esperado": "rol=ingeniero/a de software, area=inteligencia artificial, ciudad=Barcelona, genero=hombre",
        "variantes": [
            "Soy ingeniero de software trabajando en IA en Barcelona.",
            "Trabajo como ingeniero de software en inteligencia artificial en Barcelona.",
            "En Barcelona desarrollo software como ingeniero en proyectos de AI.",
            "Soy un hombre que trabaja en ingeniería de software dentro del área de inteligencia artificial en Barcelona.",
            "Resido en Barcelona y me dedico como ingeniero de software a sistemas de IA.",
        ],
    },
    {
        "nombre": "G7 — Arquitecta urbanista Málaga",
        "perfil_esperado": "rol=arquitecto/a, area=urbanismo, ciudad=Málaga, genero=mujer",
        "variantes": [
            "Soy arquitecta urbanista en Málaga.",
            "Trabajo como arquitecta especializada en urbanismo en Málaga.",
            "En Málaga me dedico al urbanismo como arquitecta.",
            "Soy una mujer arquitecta que trabaja en proyectos urbanísticos en Málaga.",
            "Resido en Málaga y ejerzo como arquitecta en el ámbito del urbanismo.",
        ],
    },
    {
        "nombre": "G8 — Enfermera pediátrica Zaragoza",
        "perfil_esperado": "rol=enfermero/a, area=pediatría, ciudad=Zaragoza, genero=mujer",
        "variantes": [
            "Soy enfermera pediátrica en Zaragoza.",
            "Trabajo como enfermera especializada en pediatría en Zaragoza.",
            "En Zaragoza ejerzo como enfermera en el área pediátrica.",
            "Soy una mujer que trabaja en enfermería pediátrica en Zaragoza.",
            "Resido en Zaragoza y me dedico a la atención pediátrica como enfermera.",
        ],
    },
    {
        "nombre": "G9 — Periodista deportiva A Coruña",
        "perfil_esperado": "rol=periodista, area=deportes, ciudad=A Coruña, genero=mujer",
        "variantes": [
            "Soy periodista deportiva en A Coruña.",
            "Trabajo como periodista especializada en deportes en A Coruña.",
            "En A Coruña me dedico al periodismo deportivo como mujer periodista.",
            "Soy una mujer periodista que cubre información deportiva en A Coruña.",
            "Resido en A Coruña y trabajo en la sección de deportes como mujer periodista.",
        ],
    },
    {
        "nombre": "G10 — Científico de datos banca Madrid",
        "perfil_esperado": "rol=científico/a de datos, area=banca, ciudad=Madrid",
        "variantes": [
            "Soy data scientist especializado en banca y vivo en Madrid.",
            "Trabajo como data scientist en el sector bancario en Madrid.",
            "En Madrid me dedico a la ciencia de datos aplicada a banca como data scientist.",
            "Soy una persona que trabaja como data scientist en el ámbito bancario en Madrid.",
            "Resido en Madrid y desarrollo modelos de datos para el sector bancario como data scientist.",
        ],
    },
]




def extraer_real(prompt_texto: str) -> dict:
    """Llamada al extractor real (API OpenAI)."""
    from extractor import extraer_atributos
    return extraer_atributos(prompt_texto)


# Análisis


def coeficiente_variacion(valores: list[float]) -> float:
    if not valores or statistics.mean(valores) == 0:
        return 0.0
    return statistics.stdev(valores) / statistics.mean(valores) if len(valores) > 1 else 0.0


def perfil_a_tupla(perfil: dict) -> tuple:
    """Convierte un perfil normalizado a una tupla hashable para comparar."""
    return tuple(sorted(
        (k, info["valor"], info["nivel"]) for k, info in perfil.items()
    ))


def comparar_perfiles(perfiles: list[dict]) -> dict:
    """Compara los perfiles normalizados extraídos para un grupo de variantes.

    Devuelve:
        - todos_iguales: bool, si los N perfiles son exactamente iguales.
        - n_distintos: número de perfiles distintos en el grupo.
        - moda: el perfil más frecuente (como tupla).
        - tasa_acuerdo: fracción de variantes que coinciden con la moda.
        - divergencias: por cada variante que difiere, qué atributos faltan
                        o sobran respecto a la moda y con qué valor/nivel.
    """
    from collections import Counter

    tuplas = [perfil_a_tupla(p) for p in perfiles]
    contador = Counter(tuplas)
    moda_tupla, n_moda = contador.most_common(1)[0]

    todos_iguales = (n_moda == len(perfiles))
    moda_perfil = {k: (v, n) for (k, v, n) in moda_tupla}

    divergencias = []
    for i, p in enumerate(perfiles):
        if perfil_a_tupla(p) == moda_tupla:
            continue
        actual = {k: (info["valor"], info["nivel"]) for k, info in p.items()}
        faltan = {k: moda_perfil[k] for k in moda_perfil if k not in actual}
        sobran = {k: actual[k]      for k in actual      if k not in moda_perfil}
        valor_distinto = {
            k: {"moda": moda_perfil[k], "variante": actual[k]}
            for k in moda_perfil.keys() & actual.keys()
            if moda_perfil[k] != actual[k]
        }
        divergencias.append({
            "indice_variante": i,
            "faltan":          faltan,
            "sobran":          sobran,
            "valor_distinto":  valor_distinto,
        })

    return {
        "todos_iguales": todos_iguales,
        "n_distintos":   len(contador),
        "tasa_acuerdo":  n_moda / len(perfiles),
        "divergencias":  divergencias,
    }


def veredicto_perfil(comp: dict) -> str:
    """Veredicto cualitativo basado en el acuerdo de perfiles.

    El extractor es lo único que puede introducir inestabilidad; el matching
    es determinista. Por tanto basamos el veredicto en si los perfiles
    coinciden.
    """
    if comp["todos_iguales"]:
        return "ESTABLE (perfiles idénticos)"
    if comp["tasa_acuerdo"] >= 0.75:
        return "MODERADO (mayoría coincide, 1 variante difiere)"
    if comp["tasa_acuerdo"] >= 0.5:
        return "INESTABLE (perfiles divergen en al menos la mitad)"
    return "MUY INESTABLE (ningún perfil mayoritario claro)"


def ejecutar(modo: str = "mock") -> dict:
    extraer =  extraer_real

    df = pd.read_csv(DATASET_PATH)
    df_pre = ev.precachear_columnas_normalizadas(df.copy())
    ev.instalar_calculo_rapido()

    resumen = []
    print("=" * 78)
    print(f"H2 — Estabilidad ante reformulaciones")
    print("Métrica primaria: igualdad de los perfiles normalizados extraídos.")
    print("=" * 78)
    for grupo in GRUPOS:

        perfiles = []
        for prompt in grupo["variantes"]:
            ext = extraer(prompt)
            perfil = normalizar_criterios(ext)
            perfiles.append(perfil)

        comp = comparar_perfiles(perfiles)
        ver = veredicto_perfil(comp)

        print()
        print(f"### {grupo['nombre']}")
        print(f"    Perfil esperado: {grupo['perfil_esperado']}")
        for i, (p, perf) in enumerate(zip(grupo["variantes"], perfiles), 1):
            resumen_perfil = ", ".join(f"{c}={info['valor']}@n{info['nivel']}" for c, info in sorted(perf.items()))
            print(f"perfil: {resumen_perfil}")
            print(f"        \"{p}\"")
        print(f"    Perfiles distintos: {comp['n_distintos']}/{len(perfiles)}   "
              f"Tasa acuerdo: {comp['tasa_acuerdo']:.2f}   ")
        print(f"    → {ver}")
        if comp["divergencias"]:
            print(f"    Divergencias respecto al perfil moda:")
            for d in comp["divergencias"]:
                idx = d["indice_variante"] + 1
                if d["faltan"]:
                    print(f"      variante [{idx}] no extrae: {list(d['faltan'].keys())}")
                if d["sobran"]:
                    print(f"      variante [{idx}] extrae de más: {list(d['sobran'].keys())}")
                if d["valor_distinto"]:
                    print(f"      variante [{idx}] valor distinto en: "
                          f"{ {k: v['variante'] for k, v in d['valor_distinto'].items()} }")

        resumen.append({
            "grupo":  grupo["nombre"],
            "perfiles_serializados": [perfil_a_tupla(p) for p in perfiles],
            "n_perfiles_distintos":  comp["n_distintos"],
            "tasa_acuerdo":          round(comp["tasa_acuerdo"], 3),
            "divergencias":          comp["divergencias"],
            "veredicto":             ver,
        })

    # Resumen agregado
    tasas = [r["tasa_acuerdo"] for r in resumen]
    print()
    print("=" * 78)
    print(f"RESUMEN AGREGADO: {len(GRUPOS)} grupos")
    print(f"  Tasa de acuerdo media:    {statistics.mean(tasas):.3f}")
    print(f"  Tasa de acuerdo mínima:   {min(tasas):.3f}")
    print(f"  Grupos con perfiles 100% idénticos:     {sum(1 for r in resumen if r['tasa_acuerdo'] == 1.0)}/{len(resumen)}")
    print(f"  Grupos con 1 variante divergente:       {sum(1 for r in resumen if 0.5 < r['tasa_acuerdo'] < 1.0)}/{len(resumen)}")
    print(f"  Grupos sin mayoría clara:               {sum(1 for r in resumen if r['tasa_acuerdo'] <= 0.5)}/{len(resumen)}")
    print("=" * 78)
    return {"modo": modo, "grupos": resumen}


if __name__ == "__main__":
    modo = "real"
    if "--mock" in sys.argv:
        modo = "mock"

    salida = ejecutar(modo)
    out_path = Path(__file__).parent / "resultados_eval" / f"estabilidad_{modo}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {out_path}")