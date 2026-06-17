"""
evaluar_tria.py — Validación empírica de la métrica k̂ mediante simulación de TRIA.

Experimento:
    1. Se generan N_PROMPTS prompts sintéticos a partir de filas reales del dataset,
       de modo que existe ground truth (sabemos qué fila generó cada prompt).
    2. Para cada prompt se calcula k̂ con el sistema existente.
    3. Se simula el ataque TRIA: se codifica el prompt y todas las filas del dataset
       con un modelo de embeddings multilingüe y se recupera la fila más similar
       (cosine similarity). El ataque "acierta" si recupera la fila correcta.
    4. Se agrupa por nivel de riesgo cualitativo (crítico/alto/medio/bajo) y se
       calcula la tasa de éxito del ataque por grupo.
    5. Se imprime la tabla de resultados y se guarda en JSON.

Dependencias:
    pip install sentence-transformers pandas numpy

Uso:
    python evaluar_tria.py
    python evaluar_tria.py --n 50 --seed 123 --out resultados_eval/tria.json
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── compatibilidad con versiones antiguas de numpy ──────────────────────────
if not hasattr(np, "float_"):
    np.float_ = np.float64

import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

# ── rutas por defecto (ajusta si tu estructura es distinta) ─────────────────
DATASET_PATH = Path(__file__).parent / "personas_sinteticas.csv"
OUT_DEFAULT  = Path(__file__).parent / "resultados_eval" / "tria_results.json"

# ── modelo multilingüe preentrenado, sin fine-tuning ────────────────────────
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# ── umbrales de clasificación cualitativa (mismos que el sistema) ────────────
def clasificar_riesgo(k: int) -> str:
    if k <= 1:  return "crítico"
    if k <= 3:  return "alto"
    if k <= 10: return "medio"
    return "bajo"


# 1. Generación de prompts sintéticos con ground truth


PLANTILLAS = [
    # rol + area + ciudad
    "Soy {rol} especializado/a en {area} y vivo en {ciudad}.",
    "Trabajo como {rol} en el campo de {area} en {ciudad}.",
    "Me dedico a {area} como {rol} en {ciudad}.",
    # rol + ciudad + institución
    "Soy {rol} en {institucion} en {ciudad}.",
    "Trabajo como {rol} en {institucion}, ubicado en {ciudad}.",
    # rol + area + ciudad + institución
    "Soy {rol} especializado/a en {area} en {institucion} de {ciudad}.",
    "Trabajo como {rol} en {area} en {institucion} ({ciudad}).",
    # con género explícito
    "Soy {genero_frase} {rol} especializado/a en {area} en {ciudad}.",
    "Como {genero_frase} {rol}, trabajo en {area} en {ciudad}.",
    # solo rol + ciudad
    "Soy {rol} y vivo en {ciudad}.",
    "Trabajo como {rol} en {ciudad}.",
]

def _s(v, default="") -> str:
    """Devuelve v si es cadena válida, si no default (maneja NaN del CSV)."""
    return v if isinstance(v, str) and v.strip() else default

def fila_a_prompt(fila: dict, rng: random.Random) -> str:
    """Genera un prompt en lenguaje natural a partir de una fila del dataset."""
    rol       = _s(fila.get("profesion_rol"),       "profesional")
    area      = _s(fila.get("area_especializacion"))
    ciudad    = _s(fila.get("ciudad"),               "su ciudad")
    inst      = _s(fila.get("institucion_nombre"))
    genero    = _s(fila.get("genero"),               "hombre")

    genero_frase = "una" if genero == "mujer" else "un"

    # Filtra plantillas que requieren campos ausentes
    plantillas_validas = []
    for p in PLANTILLAS:
        necesita_area = "{area}" in p
        necesita_inst = "{institucion}" in p
        if necesita_area and not area:
            continue
        if necesita_inst and not inst:
            continue
        plantillas_validas.append(p)

    if not plantillas_validas:
        plantillas_validas = [PLANTILLAS[0]]  # fallback

    plantilla = rng.choice(plantillas_validas)
    return plantilla.format(
        rol=rol,
        area=area or "su especialidad",
        ciudad=ciudad,
        institucion=inst or "su institución",
        genero_frase=genero_frase,
    )


def es_valido(v) -> bool:
    """Devuelve True si el valor es una cadena no vacía (descarta NaN y None)."""
    return isinstance(v, str) and v.strip() != ""


def fila_a_texto_dataset(fila: dict) -> str:
    """Convierte una fila del dataset en texto plano para la codificación."""
    partes = []
    for campo in ("profesion_rol", "area_especializacion", "ciudad",
                  "institucion_nombre", "genero"):
        v = fila.get(campo)
        if es_valido(v):
            partes.append(v)
    return ", ".join(partes)


# 2. Cálculo de k̂ sin llamada a API (matching directo sobre el dataset)


def calcular_k_directo(fila: dict, df: pd.DataFrame) -> int:
    """
    Calcula k̂ filtrando el dataset por los atributos presentes en la fila.
    Equivalente determinista al sistema real, sin necesidad de llamar al extractor.
    Usa los mismos campos que el extractor extraería de un prompt generado.
    """
    mascara = pd.Series([True] * len(df), index=df.index)

    campo_col = [
        ("profesion_rol",       "profesion_rol"),
        ("area_especializacion","area_especializacion"),
        ("ciudad",              "ciudad"),
        ("institucion_nombre",  "institucion_nombre"),
        ("genero",              "genero"),
    ]

    for campo, col in campo_col:
        valor = fila.get(campo)
        if valor and pd.notna(valor):
            mascara &= (df[col].str.lower().str.strip() ==
                        str(valor).lower().strip())

    return int(mascara.sum())


# 3. Ataque TRIA con embeddings


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similitud coseno entre cada vector de `a` y cada vector de `b`."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# 4. Experimento principal


def ejecutar(n_prompts: int, seed: int, out_path: Path) -> None:
    rng = random.Random(seed)

    # Carga dataset
    print(f"Cargando dataset desde {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} individuos cargados.")

    # Muestra N filas aleatorias (sin reemplazo) como ground truth
    indices_gt = rng.sample(range(len(df)), min(n_prompts, len(df)))
    filas_gt   = df.iloc[indices_gt].to_dict(orient="records")

    # Genera prompts
    prompts = [fila_a_prompt(f, rng) for f in filas_gt]

    # Calcula k̂ para cada prompt (matching directo, sin API)
    print("Calculando k̂ para cada prompt ...")
    k_vals   = [calcular_k_directo(f, df) for f in filas_gt]
    riesgos  = [clasificar_riesgo(k) for k in k_vals]

    # Codifica prompts y filas del dataset con sentence-transformers
    print(f"Cargando modelo '{MODEL_NAME}' ...")
    modelo = SentenceTransformer(MODEL_NAME)

    print("Codificando prompts ...")
    emb_prompts = modelo.encode(prompts, show_progress_bar=True,
                                 convert_to_numpy=True, batch_size=32)

    textos_dataset = [fila_a_texto_dataset(r)
                      for r in df.to_dict(orient="records")]
    print("Codificando dataset ...")
    emb_dataset = modelo.encode(textos_dataset, show_progress_bar=True,
                                 convert_to_numpy=True, batch_size=32)

    # Ataque: para cada prompt, recupera la fila más similar
    print("Simulando ataque TRIA ...")
    sim_matrix  = cosine_similarity_matrix(emb_prompts, emb_dataset)
    pred_indices = np.argmax(sim_matrix, axis=1)  # índice predicho en df

    # Evaluación por prompt
    resultados = []
    for i, (fila, prompt, k, riesgo, pred_idx) in enumerate(
            zip(filas_gt, prompts, k_vals, riesgos, pred_indices)):

        gt_idx    = indices_gt[i]
        acierto   = int(pred_idx) == gt_idx
        sim_score = float(sim_matrix[i, pred_idx])

        resultados.append({
            "prompt":       prompt,
            "person_id":    fila.get("person_id", f"P{gt_idx}"),
            "k_hat":        k,
            "riesgo":       riesgo,
            "gt_idx":       gt_idx,
            "pred_idx":     int(pred_idx),
            "acierto":      acierto,
            "sim_score":    round(sim_score, 4),
        })

    # ── Tabla agregada por nivel de riesgo ───────────────────────────────────
    orden_riesgo = ["crítico", "alto", "medio", "bajo"]
    stats: dict[str, dict] = {r: {"n": 0, "aciertos": 0, "k_vals": []}
                               for r in orden_riesgo}

    for r in resultados:
        nivel = r["riesgo"]
        stats[nivel]["n"]        += 1
        stats[nivel]["aciertos"] += r["acierto"]
        stats[nivel]["k_vals"].append(r["k_hat"])

    print()
    print("=" * 68)
    print(f"{'Nivel riesgo':<12} {'N':>4} {'Aciertos':>9} {'Tasa ataque':>12} {'k̂ medio':>9}")
    print("-" * 68)
    for nivel in orden_riesgo:
        s = stats[nivel]
        if s["n"] == 0:
            continue
        tasa   = s["aciertos"] / s["n"]
        k_med  = sum(s["k_vals"]) / len(s["k_vals"])
        print(f"{nivel:<12} {s['n']:>4} {s['aciertos']:>9} {tasa:>11.1%} {k_med:>9.1f}")
    print("=" * 68)

    # Tasa global
    total    = len(resultados)
    aciertos = sum(r["acierto"] for r in resultados)
    print(f"\nTasa de reidentificación global: {aciertos}/{total} = {aciertos/total:.1%}")
    print()

    # ── Guarda resultados ─────────────────────────────────────────────────────
    salida = {
        "config": {
            "n_prompts":  n_prompts,
            "seed":       seed,
            "model":      MODEL_NAME,
            "dataset":    str(DATASET_PATH),
        },
        "resumen": {
            nivel: {
                "n":              stats[nivel]["n"],
                "aciertos":       stats[nivel]["aciertos"],
                "tasa_ataque":    round(stats[nivel]["aciertos"] / stats[nivel]["n"], 4)
                                  if stats[nivel]["n"] > 0 else None,
                "k_medio":        round(sum(stats[nivel]["k_vals"]) /
                                        len(stats[nivel]["k_vals"]), 2)
                                  if stats[nivel]["k_vals"] else None,
            }
            for nivel in orden_riesgo if stats[nivel]["n"] > 0
        },
        "por_prompt": resultados,
    }

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {out_path}")


 
# CLI
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación empírica TRIA vs k̂")
    parser.add_argument("--n",    type=int,  default=60,
                        help="Número de prompts a generar (default: 60)")
    parser.add_argument("--seed", type=int,  default=99,
                        help="Semilla aleatoria (default: 99)")
    parser.add_argument("--out",  type=str,  default=str(OUT_DEFAULT),
                        help="Ruta del JSON de salida")
    args = parser.parse_args()

    ejecutar(
        n_prompts=args.n,
        seed=args.seed,
        out_path=Path(args.out),
    )