"""
Generador del dataset sintético `personas_sinteticas.csv` para el TFG.

Diseño: diccionarios maestros + muestreo programático con random.seed fijo.
- 1500 perfiles ficticios.
- Distribuciones sesgadas hacia valores comunes (no uniformes).
- Coherencia interna garantizada por construcción (profesión <-> área,
  profesión <-> institución, ciudad <-> región <-> país).
- Jerarquías consistentes: cada valor leaf mapea siempre a los mismos
  valores agregados.
- Género como dimensión propia, desacoplada de la profesión: las profesiones
  se almacenan en forma neutra (`/a` o invariante) para que el género viva
  únicamente en la columna `genero`.

Reproducible: `python generar_dataset.py` siempre produce el mismo CSV.
"""

import random
import csv
from pathlib import Path

SEED = 42
N_FILAS = 1500
SALIDA = Path(__file__).parent / "personas_sinteticas.csv"


 
# Diccionarios maestros
 

# (profesion_rol [neutra], profesion_rol_general, peso_relativo)
PROFESIONES = [
    # Sanitario
    ("médico/a generalista",          "profesional sanitario", 7),
    ("médico/a especialista",         "profesional sanitario", 5),
    ("enfermero/a",                   "profesional sanitario", 8),
    ("fisioterapeuta",                "profesional sanitario", 3),
    ("farmacéutico/a",                "profesional sanitario", 3),
    ("psicólogo/a",                   "profesional sanitario", 3),
    ("veterinario/a",                 "profesional sanitario", 2),
    ("dentista",                      "profesional sanitario", 2),
    # Educativo
    ("profesor/a de primaria",        "profesional educativo", 6),
    ("profesor/a de secundaria",      "profesional educativo", 5),
    ("profesor/a universitario/a",    "profesional educativo", 2),
    ("investigador/a",                "profesional educativo", 3),
    ("maestro/a de educación infantil","profesional educativo", 4),
    # Técnico-ingeniería
    ("ingeniero/a de software",       "profesional técnico", 9),
    ("ingeniero/a industrial",        "profesional técnico", 4),
    ("ingeniero/a civil",             "profesional técnico", 2),
    ("arquitecto/a",                  "profesional técnico", 2),
    ("científico/a de datos",         "profesional técnico", 3),
    ("analista de sistemas",          "profesional técnico", 3),
    ("ingeniero/a aeronáutico/a",     "profesional técnico", 1),
    ("ingeniero/a químico/a",         "profesional técnico", 1),
    # Jurídico-administrativo
    ("abogado/a",                     "profesional jurídico-administrativo", 5),
    ("notario/a",                     "profesional jurídico-administrativo", 1),
    ("contable",                      "profesional jurídico-administrativo", 4),
    ("economista",                    "profesional jurídico-administrativo", 3),
    ("consultor/a",                   "profesional jurídico-administrativo", 4),
    ("administrativo/a",              "profesional jurídico-administrativo", 8),
    # Creativo
    ("diseñador/a gráfico/a",         "profesional creativo", 3),
    ("periodista",                    "profesional creativo", 3),
    ("escritor/a",                    "profesional creativo", 1),
    ("fotógrafo/a",                   "profesional creativo", 1),
    ("músico/a",                      "profesional creativo", 1),
    # Oficio
    ("fontanero/a",                   "oficio", 2),
    ("electricista",                  "oficio", 3),
    ("mecánico/a de automoción",      "oficio", 3),
    ("carpintero/a",                  "oficio", 2),
    ("panadero/a",                    "oficio", 2),
    # Servicios
    ("camarero/a",                    "servicios", 6),
    ("comercial",                     "servicios", 6),
    ("agente inmobiliario/a",         "servicios", 3),
]

# (area_especializacion, area_general, lista de profesion_rol compatibles)
AREAS = [
    ("inteligencia artificial",   "informática y tecnología",
        ["ingeniero/a de software", "científico/a de datos", "investigador/a"]),
    ("aprendizaje automático",    "informática y tecnología",
        ["ingeniero/a de software", "científico/a de datos", "investigador/a"]),
    ("ciberseguridad",            "informática y tecnología",
        ["ingeniero/a de software", "analista de sistemas"]),
    ("desarrollo web",            "informática y tecnología",
        ["ingeniero/a de software"]),
    ("desarrollo móvil",          "informática y tecnología",
        ["ingeniero/a de software"]),
    ("bases de datos",            "informática y tecnología",
        ["ingeniero/a de software", "analista de sistemas", "científico/a de datos"]),
    ("cloud computing",           "informática y tecnología",
        ["ingeniero/a de software", "analista de sistemas"]),
    ("cardiología",               "medicina",
        ["médico/a especialista"]),
    ("oncología",                 "medicina",
        ["médico/a especialista"]),
    ("pediatría",                 "medicina",
        ["médico/a especialista"]),
    ("neurología",                "medicina",
        ["médico/a especialista"]),
    ("medicina interna",          "medicina",
        ["médico/a especialista"]),
    ("traumatología",             "medicina",
        ["médico/a especialista"]),
    ("psiquiatría",               "medicina",
        ["médico/a especialista", "psicólogo/a"]),
    ("derecho penal",             "derecho",
        ["abogado/a"]),
    ("derecho civil",             "derecho",
        ["abogado/a", "notario/a"]),
    ("derecho mercantil",         "derecho",
        ["abogado/a"]),
    ("derecho laboral",           "derecho",
        ["abogado/a"]),
    ("matemáticas",               "educación",
        ["profesor/a de primaria", "profesor/a de secundaria", "profesor/a universitario/a"]),
    ("lengua y literatura",       "educación",
        ["profesor/a de primaria", "profesor/a de secundaria", "profesor/a universitario/a"]),
    ("ciencias naturales",        "educación",
        ["profesor/a de primaria", "profesor/a de secundaria", "profesor/a universitario/a"]),
    ("humanidades",               "educación",
        ["profesor/a de secundaria", "profesor/a universitario/a"]),
    ("arquitectura sostenible",   "arquitectura y construcción",
        ["arquitecto/a", "ingeniero/a civil"]),
    ("urbanismo",                 "arquitectura y construcción",
        ["arquitecto/a"]),
    ("biotecnología",             "biología y ciencias de la vida",
        ["investigador/a", "científico/a de datos"]),
]

# ciudad, region, pais, peso por población aproximada, y por relevancia dentro del dataset ( como el dataset está orientado a españa, tendrá menos peso una ciudad latinoamericana aunque tenga más habitantes)
CIUDADES = [
    ("Madrid",                    "Comunidad de Madrid",  "España", 30),
    ("Barcelona",                 "Cataluña",             "España", 24),
    ("Valencia",                  "Comunidad Valenciana", "España", 12),
    ("Sevilla",                   "Andalucía",            "España", 11),
    ("Zaragoza",                  "Aragón",               "España", 8),
    ("Málaga",                    "Andalucía",            "España", 8),
    ("Murcia",                    "Región de Murcia",     "España", 6),
    ("Palma de Mallorca",         "Islas Baleares",       "España", 5),
    ("Las Palmas de Gran Canaria","Canarias",             "España", 5),
    ("Bilbao",                    "País Vasco",           "España", 5),
    ("Alicante",                  "Comunidad Valenciana", "España", 4),
    ("Córdoba",                   "Andalucía",            "España", 4),
    ("Valladolid",                "Castilla y León",      "España", 3),
    ("Vigo",                      "Galicia",              "España", 3),
    ("Gijón",                     "Asturias",             "España", 3),
    ("Vitoria-Gasteiz",           "País Vasco",           "España", 3),
    ("A Coruña",                  "Galicia",              "España", 3),
    ("Granada",                   "Andalucía",            "España", 3),
    ("Elche",                     "Comunidad Valenciana", "España", 3),
    ("Oviedo",                    "Asturias",             "España", 3),
    ("Pamplona",                  "Comunidad Foral de Navarra", "España", 2),
    ("Donostia-San Sebastián",    "País Vasco",           "España", 2),
    ("Santander",                 "Cantabria",            "España", 2),
    ("Salamanca",                 "Castilla y León",      "España", 2),
    ("Burgos",                    "Castilla y León",      "España", 2),
    ("Albacete",                  "Castilla-La Mancha",   "España", 2),
    ("Toledo",                    "Castilla-La Mancha",   "España", 2),
    ("Logroño",                   "La Rioja",             "España", 2),
    ("Castellón de la Plana",     "Comunidad Valenciana", "España", 2),
    ("Almería",                   "Andalucía",            "España", 2),
    ("Huelva",                    "Andalucía",            "España", 2),
    ("Cádiz",                     "Andalucía",            "España", 2),
    ("Jerez de la Frontera",      "Andalucía",            "España", 2),
    ("Cartagena",                 "Región de Murcia",     "España", 2),
    ("Tarragona",                 "Cataluña",             "España", 2),
    ("Lleida",                    "Cataluña",             "España", 2),
    ("Girona",                    "Cataluña",             "España", 2),
    ("Sabadell",                  "Cataluña",             "España", 2),
    ("Terrassa",                  "Cataluña",             "España", 2),
    ("Badalona",                  "Cataluña",             "España", 2),
    ("León",                      "Castilla y León",      "España", 1),
    ("Cáceres",                   "Extremadura",          "España", 1),
    ("Badajoz",                   "Extremadura",          "España", 1),
    ("Mérida",                    "Extremadura",          "España", 1),
    ("Lugo",                      "Galicia",              "España", 1),
    ("Ourense",                   "Galicia",              "España", 1),
    ("Pontevedra",                "Galicia",              "España", 1),
    ("Santa Cruz de Tenerife",    "Canarias",             "España", 1),
    ("Mataró",                    "Cataluña",             "España", 1),
    ("Reus",                      "Cataluña",             "España", 1),
    ("Marbella",                  "Andalucía",            "España", 1),
    ("Algeciras",                 "Andalucía",            "España", 1),
    ("Ávila",                     "Castilla y León",      "España", 1),
    ("Segovia",                   "Castilla y León",      "España", 1),
    ("Soria",                     "Castilla y León",      "España", 1),
    ("Cuenca",                    "Castilla-La Mancha",   "España", 1),
    ("Guadalajara",               "Castilla-La Mancha",   "España", 1),
    ("Ciudad Real",               "Castilla-La Mancha",   "España", 1),
    ("Teruel",                    "Aragón",               "España", 1),
    ("Huesca",                    "Aragón",               "España", 1),
    ("Buenos Aires",              "Buenos Aires",         "Argentina", 3),
    ("Ciudad de México",          "Ciudad de México",     "México", 3),
    ("Lima",                      "Lima",                 "Perú", 2),
    ("Bogotá",                    "Bogotá D.C.",          "Colombia", 2),
    ("Santiago",                  "Región Metropolitana", "Chile", 2),
]

# (institucion_nombre, institucion_tipo, institucion_general, lista profesion_rol compatibles)
INSTITUCIONES = [
    # Universidades públicas
    ("Universitat Politècnica de Catalunya", "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "científico/a de datos",
         "ingeniero/a de software", "ingeniero/a industrial", "ingeniero/a civil",
         "arquitecto/a", "ingeniero/a aeronáutico/a", "ingeniero/a químico/a"]),
    ("Universidad Politécnica de Madrid",    "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "ingeniero/a de software",
         "ingeniero/a industrial", "arquitecto/a", "ingeniero/a aeronáutico/a"]),
    ("Universitat Politècnica de València",  "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "ingeniero/a de software",
         "ingeniero/a industrial", "arquitecto/a"]),
    ("Universidad Complutense de Madrid",    "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "abogado/a", "economista",
         "psicólogo/a", "periodista"]),
    ("Universitat de Barcelona",             "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "abogado/a", "economista",
         "psicólogo/a", "médico/a especialista"]),
    ("Universidad de Granada",               "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "abogado/a", "economista"]),
    ("Universidad Autónoma de Madrid",       "universidad pública", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "economista", "psicólogo/a"]),
    # Universidades privadas / escuelas de negocio
    ("IE Business School",                   "universidad privada", "educación superior",
        ["profesor/a universitario/a", "economista", "consultor/a"]),
    ("ESADE",                                "universidad privada", "educación superior",
        ["profesor/a universitario/a", "economista", "consultor/a", "abogado/a"]),
    ("Universidad de Navarra",               "universidad privada", "educación superior",
        ["profesor/a universitario/a", "investigador/a", "médico/a especialista",
         "psicólogo/a"]),
    # Hospitales públicos
    ("Hospital Universitario La Paz",        "hospital público", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    ("Hospital Clínic de Barcelona",         "hospital público", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    ("Hospital Universitario y Politécnico La Fe", "hospital público", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    ("Hospital Universitario 12 de Octubre", "hospital público", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    ("Hospital Universitari Vall d'Hebron",  "hospital público", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    # Hospitales privados
    ("Hospital Quirónsalud Madrid",          "hospital privado", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "fisioterapeuta"]),
    ("Hospital HM Sanchinarro",              "hospital privado", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a"]),
    ("Clínica Universidad de Navarra",       "hospital privado", "sanidad",
        ["médico/a generalista", "médico/a especialista", "enfermero/a", "psicólogo/a"]),
    # Banca y finanzas
    ("BBVA",                                 "banco", "finanzas",
        ["economista", "contable", "consultor/a", "administrativo/a", "ingeniero/a de software",
         "científico/a de datos", "analista de sistemas"]),
    ("Banco Santander",                      "banco", "finanzas",
        ["economista", "contable", "consultor/a", "administrativo/a", "ingeniero/a de software"]),
    ("CaixaBank",                            "banco", "finanzas",
        ["economista", "contable", "consultor/a", "administrativo/a", "ingeniero/a de software"]),
    ("KPMG España",                          "consultora", "finanzas",
        ["economista", "contable", "consultor/a", "abogado/a", "administrativo/a"]),
    # Tecnología
    ("Telefónica",                           "empresa tecnológica", "tecnología y telecomunicaciones",
        ["ingeniero/a de software", "ingeniero/a industrial", "analista de sistemas",
         "científico/a de datos", "comercial", "administrativo/a"]),
    ("Indra",                                "consultora tecnológica", "tecnología y telecomunicaciones",
        ["ingeniero/a de software", "analista de sistemas", "consultor/a", "científico/a de datos"]),
    ("Accenture España",                     "consultora", "tecnología y telecomunicaciones",
        ["ingeniero/a de software", "analista de sistemas", "consultor/a", "científico/a de datos",
         "economista"]),
    ("IBM España",                           "empresa tecnológica", "tecnología y telecomunicaciones",
        ["ingeniero/a de software", "analista de sistemas", "científico/a de datos", "consultor/a"]),
    # Despachos jurídicos
    ("Garrigues",                            "despacho de abogados", "servicios jurídicos",
        ["abogado/a", "economista", "consultor/a", "administrativo/a"]),
    ("Cuatrecasas",                          "despacho de abogados", "servicios jurídicos",
        ["abogado/a", "economista", "administrativo/a"]),
    # Otros
    ("Iberia",                               "empresa de transporte aéreo", "servicios",
        ["ingeniero/a aeronáutico/a", "comercial", "administrativo/a"]),
    ("El Corte Inglés",                      "empresa de retail", "servicios",
        ["comercial", "administrativo/a", "diseñador/a gráfico/a"]),
]

# Nombres por género para construir nombre_completo (solo decorativo; no
# participa en el matching de k).
NOMBRES_MUJER = [
    "Laura", "Marta", "Paula", "María", "Ana", "Carmen", "Lucía", "Elena",
    "Sara", "Cristina", "Isabel", "Patricia", "Beatriz", "Andrea", "Natalia",
    "Ainhoa", "Núria", "Clara", "Irene", "Alba", "Rocío", "Esther",
]
NOMBRES_HOMBRE = [
    "Pablo", "Javier", "Daniel", "Carlos", "David", "Manuel", "Alejandro",
    "Jorge", "Adrián", "Diego", "Iván", "Rubén", "Sergio", "Álvaro", "Miguel",
    "Aitor", "Jordi", "Marc", "Roger", "Hugo", "Pedro", "Luis",
]
APELLIDOS = [
    "García", "Rodríguez", "González", "Fernández", "López", "Martínez",
    "Sánchez", "Pérez", "Gómez", "Martín", "Jiménez", "Ruiz", "Hernández",
    "Díaz", "Moreno", "Muñoz", "Álvarez", "Romero", "Alonso", "Gutiérrez",
    "Navarro", "Torres", "Domínguez", "Vázquez", "Ramos", "Gil", "Ramírez",
    "Serrano", "Blanco", "Suárez", "Molina", "Castro", "Ortega", "Delgado",
    "Castillo", "Ortiz", "Marín", "Iglesias",
]

GENEROS = ["mujer", "hombre"]


 
# Helpers
 

def decada_de(anio: int) -> str:
    return f"década de {anio - anio % 10}"


def muestreo_ponderado(rng: random.Random, items, peso_idx: int = -1):
    pesos = [item[peso_idx] for item in items]
    return rng.choices(items, weights=pesos, k=1)[0]


 
# Generación
 

def generar_fila(rng: random.Random, idx: int) -> dict:
    # 1) Género (50/50)
    genero = rng.choice(GENEROS)

    # 2) Profesión (forma neutra; el género ya está separado)
    prof = muestreo_ponderado(rng, PROFESIONES)
    profesion_rol, profesion_rol_general, _ = prof

    # 3) Ciudad
    ciu = muestreo_ponderado(rng, CIUDADES)
    ciudad, region, pais, _ = ciu

    # 4) Año de nacimiento (triangular centrada en 1980, rango 1945-2005)
    anio = int(rng.triangular(1945, 2005, 1980))

    # 5) Área (condicional a la profesión, NULL si no aplica)
    areas_compatibles = [a for a in AREAS if profesion_rol in a[2]]
    if areas_compatibles and rng.random() < 0.7:
        area_especializacion, area_general, _ = rng.choice(areas_compatibles)
    else:
        area_especializacion, area_general = None, None

    # 6) Institución (condicional a la profesión, NULL si no aplica)
    inst_compatibles = [i for i in INSTITUCIONES if profesion_rol in i[3]]
    prob_tiene = 0.70 if inst_compatibles else 0.0
    if inst_compatibles and rng.random() < prob_tiene:
        institucion_nombre, institucion_tipo, institucion_general, _ = rng.choice(inst_compatibles)
    else:
        institucion_nombre, institucion_tipo, institucion_general = None, None, None

    # 7) Nombre completo ficticio. Depende del género.
    pila = rng.choice(NOMBRES_MUJER if genero == "mujer" else NOMBRES_HOMBRE)
    nombre_completo = f"{pila} {rng.choice(APELLIDOS)} {rng.choice(APELLIDOS)}"

    return {
        "person_id": f"P{idx:04d}",
        "nombre_completo": nombre_completo,
        "genero": genero,
        "profesion_rol": profesion_rol,
        "profesion_rol_general": profesion_rol_general,
        "area_especializacion": area_especializacion,
        "area_general": area_general,
        "ciudad": ciudad,
        "region": region,
        "pais": pais,
        "institucion_nombre": institucion_nombre,
        "institucion_tipo": institucion_tipo,
        "institucion_general": institucion_general,
        "año_nacimiento": anio,
        "decada_nacimiento": decada_de(anio),
    }


def generar_dataset(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [generar_fila(rng, i + 1) for i in range(n)]


 
# Validación
 

def validar(filas: list) -> dict:
    from collections import Counter

    n = len(filas)

    ciudad2region = {}
    ciudad2pais = {}
    for f in filas:
        c = f["ciudad"]
        if c in ciudad2region:
            assert ciudad2region[c] == f["region"]
            assert ciudad2pais[c] == f["pais"]
        else:
            ciudad2region[c] = f["region"]
            ciudad2pais[c] = f["pais"]

    inst2tipo = {}
    for f in filas:
        if f["institucion_nombre"]:
            n_ = f["institucion_nombre"]
            if n_ in inst2tipo:
                assert inst2tipo[n_] == f["institucion_tipo"]
            else:
                inst2tipo[n_] = f["institucion_tipo"]

    def top_pct(field, k=5):
        c = Counter(f[field] for f in filas if f[field] is not None)
        return [(v, cnt, round(100 * cnt / n, 1)) for v, cnt in c.most_common(k)]

    return {
        "n_total": n,
        "n_distintos_genero": len({f["genero"] for f in filas}),
        "n_distintos_profesion_rol": len({f["profesion_rol"] for f in filas}),
        "n_distintos_profesion_rol_general": len({f["profesion_rol_general"] for f in filas}),
        "n_distintos_area_especializacion": len({f["area_especializacion"] for f in filas if f["area_especializacion"]}),
        "n_distintos_ciudad": len({f["ciudad"] for f in filas}),
        "n_distintos_region": len({f["region"] for f in filas}),
        "n_distintos_pais": len({f["pais"] for f in filas}),
        "n_distintos_institucion_nombre": len({f["institucion_nombre"] for f in filas if f["institucion_nombre"]}),
        "porc_genero_mujer": round(100 * sum(1 for f in filas if f["genero"] == "mujer") / n, 1),
        "porc_con_area": round(100 * sum(1 for f in filas if f["area_especializacion"]) / n, 1),
        "porc_con_institucion": round(100 * sum(1 for f in filas if f["institucion_nombre"]) / n, 1),
        "rango_año": (min(f["año_nacimiento"] for f in filas), max(f["año_nacimiento"] for f in filas)),
        "top5_profesion_rol":   top_pct("profesion_rol"),
        "top5_ciudad":          top_pct("ciudad"),
        "top5_profesion_rol_general": top_pct("profesion_rol_general"),
    }


 
# Main
 

if __name__ == "__main__":
    filas = generar_dataset(N_FILAS, SEED)

    info = validar(filas)
    print(f"Generadas {info['n_total']} filas. Validación:")
    for k, v in info.items():
        if k.startswith("top5"):
            print(f"  {k}:")
            for valor, cnt, pct in v:
                print(f"     {valor:40s}  {cnt:4d}  ({pct:.1f}%)")
        else:
            print(f"  {k}: {v}")

    columnas = list(filas[0].keys())
    with open(SALIDA, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for f in filas:
            w.writerow(f)
    print(f"\nDataset guardado en: {SALIDA}")
