"""
evaluar.py — Evaluación empírica de la métrica k-anonymity sobre prompts.

Ejecuta los experimentos del capítulo 5 sobre el corpus principal (extracciones
simuladas a partir del dataset sintético):

  E1. Monotonía informativa (H1).
  E3. Eficacia del recomendador (H3).
  E4. Sensibilidad al parámetro alpha (H4).
  E5. Distribución de k sobre el corpus (H5).
  E6. Tiempo medio de cómputo (H6).

H2 (estabilidad ante reformulaciones del prompt en lenguaje natural) requiere
el extractor LLM real y un corpus textual; queda al margen de este script y se
recomienda ejecutarlo manualmente sobre el corpus auxiliar.

Salida: directorio resultados_eval/ con resultados.json y figuras PNG.
"""

from __future__ import annotations

import json
import statistics
import time
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from metrica_k import (
    DATASET_PATH,
    HIERARCHIES,
    calcular_k_estricto,
    normalizar_criterios,
    norm,
)
from modificador_prompt import recomendar_modificaciones


def precachear_columnas_normalizadas(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un df con las columnas usadas por las jerarquías ya
    pre-normalizadas (lowercased + sin diacríticos), para que el cálculo de k
    pueda comparar por igualdad directa sin re-normalizar en cada llamada."""
    df = df.copy()
    columnas_usadas = set()
    for niveles in HIERARCHIES.values():
        columnas_usadas.update(c for c in niveles if c != "*")
    for col in columnas_usadas:
        if col in df.columns:
            df[col] = df[col].apply(norm)
    return df


def calcular_k_rapido(df_pre: pd.DataFrame, perfil: dict):
    """Versión rápida de calcular_k_estricto que asume df_pre ya
    pre-normalizado. Reproduce literalmente la API original (k, candidatos,
    usados) para que se pueda usar como reemplazo transparente, incluso desde
    el recomendador."""
    mascara = pd.Series([True] * len(df_pre), index=df_pre.index)
    usados: list[str] = []
    for clave, info in perfil.items():
        if clave not in HIERARCHIES:
            continue
        niveles = HIERARCHIES[clave]
        nivel = info["nivel"]
        if nivel >= len(niveles):
            continue
        columna = niveles[nivel]
        if columna == "*" or columna not in df_pre.columns:
            continue
        valor = info["valor"]
        if not valor:
            continue
        mascara &= df_pre[columna].eq(norm(valor))
        usados.append(f"{clave}@{columna}={valor}")
    candidatos = df_pre[mascara]
    return len(candidatos), candidatos, usados


def obtener_valor_en_nivel_rapido(df_pre, clave, valor_origen,
                                  nivel_origen, nivel_destino):
    """Versión rápida que asume df_pre pre-normalizado. Devuelve el valor en
    el nivel destino (también pre-normalizado, lo cual basta para los
    experimentos de evaluación que solo computan agregados)."""
    niveles = HIERARCHIES[clave]
    col_destino = niveles[nivel_destino]
    if col_destino == "*":
        return "*"
    col_origen = niveles[nivel_origen]
    if (col_origen == "*" or col_origen not in df_pre.columns
            or col_destino not in df_pre.columns):
        return None
    coincidencias = df_pre[df_pre[col_origen] == norm(valor_origen)]
    if coincidencias.empty:
        return None
    return str(coincidencias[col_destino].iloc[0])


def loss_metric_estructural_rapido(df_pre, clave, valor_origen,
                                   nivel_origen, nivel_destino):
    """Versión rápida que aprovecha el df pre-normalizado."""
    niveles = HIERARCHIES[clave]
    col_base = niveles[nivel_origen]
    col_destino = niveles[nivel_destino]
    if col_base == "*" or col_base not in df_pre.columns:
        return 0.0
    total = df_pre[col_base].nunique()
    if total <= 1:
        return 0.0
    if col_destino == "*":
        return 1.0
    valor_general = obtener_valor_en_nivel_rapido(
        df_pre, clave, valor_origen, nivel_origen, nivel_destino,
    )
    if valor_general is None:
        return 1.0
    cuenta = df_pre[df_pre[col_destino] == norm(valor_general)][col_base].nunique()
    return max(0.0, (cuenta - 1) / (total - 1))


def instalar_calculo_rapido() -> None:
    """Monkey-patch global para que tanto los experimentos como el recomendador
    usen las versiones rápidas que aprovechan el df pre-normalizado."""
    import metrica_k as _mk
    import modificador_prompt as _mp
    _mk.calcular_k_estricto = calcular_k_rapido
    _mp.calcular_k_estricto = calcular_k_rapido
    _mp.obtener_valor_en_nivel = obtener_valor_en_nivel_rapido
    _mp.loss_metric_estructural = loss_metric_estructural_rapido


SEED = 42
DIR_RESULTADOS = Path(__file__).parent / "resultados_eval"
DIR_RESULTADOS.mkdir(exist_ok=True)

# Mapeo dimensión canónica → columna leaf del dataset.
COLUMNA_LEAF = {
    "rol":         "profesion_rol",
    "area":        "area_especializacion",
    "ciudad":      "ciudad",
    "institucion": "institucion_nombre",
    "año":        "año_nacimiento",
    "genero":      "genero",
}

# Nombre que reconoce normalizar_criterios para cada dimensión leaf.
CLAVE_EXTRACTOR = {
    "rol":         "profesion_rol",
    "area":        "area_especializacion",
    "ciudad":      "ciudad",
    "institucion": "institucion_nombre",
    "año":        "año_nacimiento",
    "genero":      "genero",
}

# Orden de revelación: las dimensiones se exponen en este orden para los
# experimentos donde se va revelando información progresivamente.
ORDEN_REVELACION = ["genero", "ciudad", "año", "rol", "area", "institucion"]


 
# Utilidades
 

def construir_extraccion(row: dict, dims: list[str]) -> dict:
    """Extracción simulada: revela las dimensiones indicadas tomando el valor
    leaf directamente de la fila del dataset."""
    ext: dict = {}
    for d in dims:
        col = COLUMNA_LEAF[d]
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        ext[CLAVE_EXTRACTOR[d]] = val
    return ext


def dimensiones_disponibles(row: dict) -> list[str]:
    """Dimensiones del orden de revelación cuyo valor leaf no es nulo."""
    out = []
    for d in ORDEN_REVELACION:
        val = row.get(COLUMNA_LEAF[d])
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        out.append(d)
    return out


 
# E1. Monotonía informativa (H1)
 

def experimento_monotonia(df: pd.DataFrame, n_muestras: int = 150) -> dict:
    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_muestras, len(df)))
    resultados = []
    for idx in indices:
        row = df.iloc[idx].to_dict()
        dims = dimensiones_disponibles(row)
        if len(dims) < 2:
            continue
        ks = []
        for j in range(1, len(dims) + 1):
            ext = construir_extraccion(row, dims[:j])
            perfil = normalizar_criterios(ext)
            k, _, _ = calcular_k_estricto(df, perfil)
            ks.append(k)
        es_monotono = all(ks[i] >= ks[i + 1] for i in range(len(ks) - 1))
        resultados.append({
            "person_id": row.get("person_id"),
            "n_dims":    len(dims),
            "ks":        ks,
            "es_monotono": es_monotono,
        })
    n = len(resultados)
    m = sum(r["es_monotono"] for r in resultados)
    return {
        "n_evaluados":   n,
        "n_monotonos":   m,
        "tasa_monotonia": m / n if n else 0.0,
        "ejemplos":      resultados[:5],
    }


 
# E3. Eficacia del recomendador (H3)
 

def experimento_recomendador(df: pd.DataFrame, n_muestras: int = 20,
                             k_objetivos=(3, 5, 10), alpha: float = 1.0) -> dict:
    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_muestras, len(df)))
    rel = {k: 1.0 for k in COLUMNA_LEAF}

    acumul = {k: {"intentos": 0, "exitos": 0,
                  "perdidas": [], "pasos": []} for k in k_objetivos}

    for idx in indices:
        row = df.iloc[idx].to_dict()
        dims = dimensiones_disponibles(row)
        if not dims:
            continue
        ext = construir_extraccion(row, dims)
        perfil = normalizar_criterios(ext)
        k0, _, _ = calcular_k_estricto(df, perfil)
        for kobj in k_objetivos:
            if k0 >= kobj:
                continue
            r = acumul[kobj]
            r["intentos"] += 1
            rec = recomendar_modificaciones(
                df, perfil, relevancia=rel,
                k_objetivo=kobj, alpha=alpha,
            )
            if rec["k_final"] >= kobj:
                r["exitos"] += 1
            r["perdidas"].append(rec["perdida_utilidad"])
            r["pasos"].append(len(rec["historial_movimientos"]))

    salida = {}
    for kobj, r in acumul.items():
        n = r["intentos"]
        salida[kobj] = {
            "intentos":     n,
            "exitos":       r["exitos"],
            "tasa_exito":   (r["exitos"] / n) if n else None,
            "perdida_media": statistics.mean(r["perdidas"]) if r["perdidas"] else None,
            "pasos_medio":  statistics.mean(r["pasos"]) if r["pasos"] else None,
        }
    return salida


 
# E4. Sensibilidad al parámetro alpha (H4)
 

def experimento_alpha(df: pd.DataFrame, n_muestras: int = 300,
                      k_objetivo: int = 5,
                      alphas=(0.0, 0.5, 1.0, 2.0, 5.0)) -> dict:
    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_muestras, len(df)))
    rel = {k: 1.0 for k in COLUMNA_LEAF}
    salida = {}
    for alpha in alphas:
        k_finales, perdidas = [], []
        for idx in indices:
            row = df.iloc[idx].to_dict()
            dims = dimensiones_disponibles(row)
            if not dims:
                continue
            ext = construir_extraccion(row, dims)
            perfil = normalizar_criterios(ext)
            k0, _, _ = calcular_k_estricto(df, perfil)
            if k0 >= k_objetivo:
                k_finales.append(k0)
                perdidas.append(0.0)
                continue
            rec = recomendar_modificaciones(
                df, perfil, relevancia=rel,
                k_objetivo=k_objetivo, alpha=alpha,
            )
            k_finales.append(rec["k_final"])
            perdidas.append(rec["perdida_utilidad"])
        salida[alpha] = {
            "n":              len(k_finales),
            "k_final_medio":  statistics.mean(k_finales) if k_finales else None,
            "perdida_media":  statistics.mean(perdidas) if perdidas else None,
        }
    return salida


 
# E5. Distribución de k (H5)
 

def experimento_distribucion(df: pd.DataFrame,
                             n_dims_evaluadas=(1, 2, 3, 4, 5, 6),
                             n_muestras: int = 300) -> dict:
    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_muestras, len(df)))
    salida = {}
    for n_dims in n_dims_evaluadas:
        ks = []
        for idx in indices:
            row = df.iloc[idx].to_dict()
            dims = dimensiones_disponibles(row)
            if len(dims) < n_dims:
                continue
            ext = construir_extraccion(row, dims[:n_dims])
            perfil = normalizar_criterios(ext)
            k, _, _ = calcular_k_estricto(df, perfil)
            ks.append(k)
        if not ks:
            continue
        ks_s = sorted(ks)
        n = len(ks_s)
        salida[n_dims] = {
            "n":          n,
            "min":        ks_s[0],
            "p25":        ks_s[n // 4],
            "mediana":    ks_s[n // 2],
            "p75":        ks_s[3 * n // 4],
            "max":        ks_s[-1],
            "p_critico":  sum(1 for k in ks if k <= 1) / n,
            "p_alto":     sum(1 for k in ks if k <= 3) / n,
            "p_medio":    sum(1 for k in ks if k <= 10) / n,
        }
    return salida


 
# E6. Tiempos de cómputo (H6)
 

def experimento_tiempos(df: pd.DataFrame, n_muestras: int = 100,
                        k_objetivo: int = 5, alpha: float = 1.0) -> dict:
    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_muestras, len(df)))
    rel = {k: 1.0 for k in COLUMNA_LEAF}
    t_norm, t_k, t_rec = [], [], []
    for idx in indices:
        row = df.iloc[idx].to_dict()
        dims = dimensiones_disponibles(row)
        if not dims:
            continue
        ext = construir_extraccion(row, dims)

        t0 = time.perf_counter()
        perfil = normalizar_criterios(ext)
        t1 = time.perf_counter()
        t_norm.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        k, _, _ = calcular_k_estricto(df, perfil)
        t1 = time.perf_counter()
        t_k.append((t1 - t0) * 1000)

        if k < k_objetivo:
            t0 = time.perf_counter()
            recomendar_modificaciones(
                df, perfil, relevancia=rel,
                k_objetivo=k_objetivo, alpha=alpha,
            )
            t1 = time.perf_counter()
            t_rec.append((t1 - t0) * 1000)

    def stats(lst):
        if not lst:
            return None
        s = sorted(lst)
        return {
            "n":           len(s),
            "mediana_ms":  round(s[len(s) // 2], 3),
            "p95_ms":      round(s[int(0.95 * (len(s) - 1))], 3),
            "max_ms":      round(s[-1], 3),
        }
    return {
        "normalizacion": stats(t_norm),
        "calculo_k":     stats(t_k),
        "recomendador":  stats(t_rec),
    }


 
# E6b. Tiempo del extractor LLM real (H6 complementario)
 

# Prompts auxiliares representativos para medir la latencia del extractor.
# Cubren distintos niveles de especificidad para que el tiempo no dependa
# de un único tipo de input.
PROMPTS_AUXILIARES_TIEMPOS = [
    # Muy genérico
    "Soy médico y vivo en Madrid.",
    # Rol + área + ciudad
    "Trabajo como ingeniera de software especializada en ciberseguridad en Bilbao.",
    # Con institución
    "Soy médico cardiólogo en el Hospital Clínic de Barcelona.",
    # Con género explícito
    "Soy una investigadora en inteligencia artificial y vivo en Madrid.",
    # Con institución y área
    "Trabajo como abogada especializada en derecho penal en Garrigues, en Sevilla.",
    # Solo rol
    "Soy enfermera y trabajo en Zaragoza.",
    # Con año de nacimiento
    "Soy arquitecto urbanista en Málaga, nací en 1985.",
    # Largo y detallado
    "Soy profesora de matemáticas en un instituto de educación secundaria en Valencia. "
    "Llevo 10 años trabajando en el sector público y me especializo en álgebra.",
    # Con identificador directo (para medir el caso con aviso)
    "Hola, soy Juan García, médico especialista en cardiología en Barcelona.",
    # Prompt mínimo
    "Soy abogado.",
]


def experimento_tiempos_extractor(
    n_repeticiones: int = 3,
    prompts: list[str] | None = None,
) -> dict | None:
    """Mide la latencia de la llamada al extractor LLM real.

    Ejecuta cada prompt ``n_repeticiones`` veces y reporta la mediana y P95
    sobre todas las llamadas. Si el módulo extractor no está disponible o
    falta la API key, devuelve None y avisa sin abortar el script.

    Args:
        n_repeticiones: Número de veces que se repite cada prompt.
        prompts: Lista de prompts a usar. Por defecto PROMPTS_AUXILIARES_TIEMPOS.
    """
    try:
        from extractor import extraer_atributos
    except ImportError:
        print("  [aviso] extractor.py no encontrado; E6b omitido.")
        return None

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [aviso] OPENAI_API_KEY no definida; E6b omitido.")
        return None

    if prompts is None:
        prompts = PROMPTS_AUXILIARES_TIEMPOS

    tiempos_ms: list[float] = []
    errores = 0

    print(f"  Midiendo latencia del extractor ({len(prompts)} prompts × "
          f"{n_repeticiones} repeticiones)...")

    for rep in range(n_repeticiones):
        for i, prompt in enumerate(prompts):
            try:
                t0 = time.perf_counter()
                extraer_atributos(prompt)
                t1 = time.perf_counter()
                tiempos_ms.append((t1 - t0) * 1000)
            except Exception as e:
                errores += 1
                print(f"  [warn] error en prompt {i+1} rep {rep+1}: {e}")

    if not tiempos_ms:
        print("  [aviso] todas las llamadas al extractor fallaron; E6b omitido.")
        return None

    s = sorted(tiempos_ms)
    n = len(s)
    resultado = {
        "n_llamadas":    n,
        "n_errores":     errores,
        "n_prompts":     len(prompts),
        "n_repeticiones": n_repeticiones,
        "mediana_ms":    round(s[n // 2], 1),
        "p95_ms":        round(s[int(0.95 * (n - 1))], 1),
        "min_ms":        round(s[0], 1),
        "max_ms":        round(s[-1], 1),
        "media_ms":      round(sum(s) / n, 1),
    }
    return resultado


 
# Figuras
 

def generar_figuras(distribucion: dict, alpha_res: dict,
                    dir_out: Path = DIR_RESULTADOS) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib no disponible; figuras omitidas.")
        return

    # Figura 1: distribución de k según especificidad
    if distribucion:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ns = sorted(distribucion.keys())
        med = [distribucion[n]["mediana"] for n in ns]
        p25 = [distribucion[n]["p25"] for n in ns]
        p75 = [distribucion[n]["p75"] for n in ns]
        ax.fill_between(ns, p25, p75, alpha=0.3, label="P25–P75")
        ax.plot(ns, med, marker="o", linewidth=2, label="Mediana")
        ax.axhline(y=10, color="orange", linestyle="--", alpha=0.6, label="k=10 (medio)")
        ax.axhline(y=3,  color="red",    linestyle="--", alpha=0.6, label="k=3 (alto)")
        ax.set_xlabel("Número de dimensiones reveladas")
        ax.set_ylabel(r"$\widehat{k}$")
        ax.set_yscale("symlog")
        ax.set_title(r"Distribución de $\widehat{k}$ por especificidad del prompt")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(dir_out / "distribucion_k.png", dpi=150)
        plt.close(fig)

    # Figura 2: trade-off al variar alpha
    if alpha_res:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        alphas = sorted(alpha_res.keys())
        ks = [alpha_res[a]["k_final_medio"] for a in alphas]
        pe = [alpha_res[a]["perdida_media"] for a in alphas]
        ax.plot(pe, ks, marker="o", linewidth=2)
        for a, x, y in zip(alphas, pe, ks):
            ax.annotate(f"α={a}", (x, y), textcoords="offset points",
                        xytext=(7, 4), fontsize=9)
        ax.set_xlabel("Pérdida media de utilidad (LM × relevancia)")
        ax.set_ylabel(r"$\widehat{k}$ final medio")
        ax.set_title(r"Trade-off privacidad–utilidad al variar $\alpha$")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(dir_out / "alpha_tradeoff.png", dpi=150)
        plt.close(fig)


 
# Main
 

if __name__ == "__main__":
    print(f"Cargando dataset desde {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  N = {len(df)} filas")
    print("  Pre-normalizando columnas para acelerar el matching...")
    df = precachear_columnas_normalizadas(df)
    instalar_calculo_rapido()
    print()

    print("=" * 70)
    print("E1. MONOTONÍA INFORMATIVA (H1)")
    print("=" * 70)
    e1 = experimento_monotonia(df)
    print(f"  N evaluados:       {e1['n_evaluados']}")
    print(f"  N monótonos:       {e1['n_monotonos']}")
    print(f"  Tasa de monotonía: {e1['tasa_monotonia']*100:.2f}%\n")

    print("=" * 70)
    print("E3. EFICACIA DEL RECOMENDADOR (H3)")
    print("=" * 70)
    e3 = experimento_recomendador(df)
    print(f"  {'k_obj':>6} {'intentos':>10} {'éxitos':>8} {'%éxito':>10} {'pérdida':>10} {'pasos':>8}")
    for kobj, r in e3.items():
        te = f"{r['tasa_exito']*100:.1f}%" if r['tasa_exito'] is not None else "n/a"
        pm = f"{r['perdida_media']:.3f}" if r['perdida_media'] is not None else "n/a"
        ps = f"{r['pasos_medio']:.1f}"   if r['pasos_medio']   is not None else "n/a"
        print(f"  {kobj:>6} {r['intentos']:>10} {r['exitos']:>8} {te:>10} {pm:>10} {ps:>8}")
    print()

    print("=" * 70)
    print("E4. SENSIBILIDAD A α (H4)")
    print("=" * 70)
    e4 = experimento_alpha(df)
    print(f"  {'α':>6} {'k_final_medio':>16} {'pérdida_media':>16}")
    for a, r in e4.items():
        kf = f"{r['k_final_medio']:.2f}" if r['k_final_medio'] is not None else "n/a"
        pm = f"{r['perdida_media']:.4f}" if r['perdida_media'] is not None else "n/a"
        print(f"  {a:>6} {kf:>16} {pm:>16}")
    print()

    print("=" * 70)
    print("E5. DISTRIBUCIÓN DE k (H5)")
    print("=" * 70)
    e5 = experimento_distribucion(df)
    print(f"  {'n_dims':>6} {'min':>5} {'p25':>5} {'med':>5} {'p75':>5} {'max':>6} "
          f"{'%crítico':>9} {'%alto':>7} {'%medio':>8}")
    for n, r in e5.items():
        print(f"  {n:>6} {r['min']:>5} {r['p25']:>5} {r['mediana']:>5} {r['p75']:>5} "
              f"{r['max']:>6} {r['p_critico']*100:>8.1f}% {r['p_alto']*100:>6.1f}% {r['p_medio']*100:>7.1f}%")
    print()

    print("=" * 70)
    print("E6. TIEMPOS DE CÓMPUTO LOCAL (H6)")
    print("=" * 70)
    e6 = experimento_tiempos(df)
    for comp, st in e6.items():
        if st is None:
            print(f"  {comp:14s}: n/a")
        else:
            print(f"  {comp:14s}: mediana = {st['mediana_ms']:7.3f} ms,  "
                  f"P95 = {st['p95_ms']:7.3f} ms,  max = {st['max_ms']:7.3f} ms  "
                  f"(N = {st['n']})")
    print()

    print("=" * 70)
    print("E6b. LATENCIA DEL EXTRACTOR LLM REAL (H6 complementario)")
    print("=" * 70)
    e6b = experimento_tiempos_extractor()
    if e6b:
        print(f"  N llamadas:  {e6b['n_llamadas']}  "
              f"({e6b['n_prompts']} prompts × {e6b['n_repeticiones']} repeticiones)")
        print(f"  Mediana:     {e6b['mediana_ms']:.1f} ms")
        print(f"  P95:         {e6b['p95_ms']:.1f} ms")
        print(f"  Mín / Máx:   {e6b['min_ms']:.1f} / {e6b['max_ms']:.1f} ms")
        if e6b['n_errores']:
            print(f"  Errores:     {e6b['n_errores']}")
    else:
        print("  (omitido — extractor no disponible o sin API key)")
    print()

    # Volcar JSON
    out_json = DIR_RESULTADOS / "resultados.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "E1_monotonia":    {k: v for k, v in e1.items() if k != "ejemplos"},
            "E3_recomendador": e3,
            "E4_alpha":        e4,
            "E5_distribucion": e5,
            "E6_tiempos":      e6,
            "E6b_extractor":   e6b,
        }, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Resultados guardados en {out_json}")

    generar_figuras(e5, e4)
    print(f"Figuras (si matplotlib) en {DIR_RESULTADOS}/")