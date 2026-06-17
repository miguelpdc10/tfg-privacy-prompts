"""
Modificador de prompts para mejorar la privacidad.

Combina dos métricas para evaluar el coste de generalizar/eliminar un atributo:
  - Loss Metric estructural (Iyengar): cuánto del dominio queda fundido por la
    generalización, medido relativo al nivel del usuario (no al nivel hoja).
  - Relevancia: cuán importante es el atributo para responder a la pregunta
    concreta del usuario, en [0, 1]. La aporta el extractor LLM.

Coste combinado:  C = LM_estructural * relevancia.
Score voraz:      score = Δk / (1 + alpha * C).

Si no se proporciona relevancia, se asume 1.0 para todo (equivale a la LM clásica).
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Optional

import pandas as pd

from metrica_k import HIERARCHIES, calcular_k_estricto, norm


# =====================================================================
# Métricas
# =====================================================================

def obtener_valor_en_nivel(
    df: pd.DataFrame, clave: str, valor_origen: str,
    nivel_origen: int, nivel_destino: int,
) -> Optional[str]:
    """
    Dado un valor a cierto nivel, devuelve el valor equivalente en un nivel
    superior consultando el dataset. Si nivel_destino es '*', devuelve '*'.
    """
    niveles = HIERARCHIES[clave]
    col_destino = niveles[nivel_destino]
    if col_destino == "*":
        return "*"
    col_origen = niveles[nivel_origen]
    if col_origen == "*" or col_origen not in df.columns or col_destino not in df.columns:
        return None
    coincidencias = df[df[col_origen].apply(norm) == norm(valor_origen)]
    if coincidencias.empty:
        return None
    return str(coincidencias[col_destino].iloc[0])


def loss_metric_estructural(
    df: pd.DataFrame, clave: str, valor_origen: str,
    nivel_origen: int, nivel_destino: int,
) -> float:
    """
    Loss Metric (Iyengar) adaptada al escenario de prompts: la base de
    comparación es el nivel que el usuario aportó (nivel_origen), no la hoja
    absoluta. Mide qué fracción del dominio "a ese nivel" queda fundida.

      LM = (n_distintos_bajo_la_generalización - 1) / (n_total_distintos - 1)
    """
    niveles = HIERARCHIES[clave]
    col_base = niveles[nivel_origen]
    col_destino = niveles[nivel_destino]

    if col_base == "*" or col_base not in df.columns:
        return 0.0
    total = df[col_base].nunique()
    if total <= 1:
        return 0.0

    if col_destino == "*":
        return 1.0  # supresión total = pérdida estructural máxima

    valor_general = obtener_valor_en_nivel(df, clave, valor_origen, nivel_origen, nivel_destino)
    if valor_general is None:
        return 1.0
    cuenta = df[df[col_destino].apply(norm) == norm(valor_general)][col_base].nunique()
    return max(0.0, (cuenta - 1) / (total - 1))


def coste_combinado(loss_estructural: float, relevancia: float) -> float:
    """
    Coste de generalizar un atributo combinando estructura y relevancia.

    Si relevancia=1, equivale a la LM clásica.
    Si relevancia=0, el atributo no aporta a la respuesta y generalizar es gratis.
    """
    relevancia = max(0.0, min(1.0, relevancia))
    return loss_estructural * relevancia


def perdida_total_perfil(
    df: pd.DataFrame,
    perfil_original: dict,
    perfil_modificado: dict,
    relevancia: Optional[dict[str, float]] = None,
) -> float:
    """
    Pérdida media del perfil tras los movimientos aplicados, ya combinada con
    relevancia. Sirve para reportar al usuario un único número.
    """
    if not perfil_original:
        return 0.0
    relevancia = relevancia or {}
    perdidas = []
    for clave, info_orig in perfil_original.items():
        nivel_orig = info_orig["nivel"]
        info_mod = perfil_modificado.get(clave, info_orig)
        nivel_mod = info_mod["nivel"]
        if nivel_mod == nivel_orig:
            perdidas.append(0.0)
            continue
        lm = loss_metric_estructural(df, clave, info_orig["valor"], nivel_orig, nivel_mod)
        rel = relevancia.get(clave, 1.0)
        perdidas.append(coste_combinado(lm, rel))
    return sum(perdidas) / len(perdidas)


# =====================================================================
# Búsqueda voraz
# =====================================================================

def evaluar_movimientos(
    df: pd.DataFrame,
    perfil: dict,
    perfil_original: dict,
    relevancia: Optional[dict[str, float]],
    alpha: float,
) -> list[dict]:
    """
    Lista todos los movimientos posibles desde 'perfil': para cada atributo,
    cada nivel superior alcanzable. Calcula Δk, LM estructural, relevancia,
    coste combinado y score, y los ordena de mayor a menor score.
    """
    relevancia = relevancia or {}
    k_actual, _, _ = calcular_k_estricto(df, perfil)
    movimientos = []

    for clave, info in perfil.items():
        if clave not in HIERARCHIES:
            continue
        niveles = HIERARCHIES[clave]
        nivel_actual = info["nivel"]

        for nuevo_nivel in range(nivel_actual + 1, len(niveles)):
            nuevo_valor = obtener_valor_en_nivel(
                df, clave, info["valor"], nivel_actual, nuevo_nivel
            )
            if nuevo_valor is None:
                continue

            perfil_prueba = deepcopy(perfil)
            perfil_prueba[clave] = {"valor": nuevo_valor, "nivel": nuevo_nivel}
            k_nuevo, _, _ = calcular_k_estricto(df, perfil_prueba)
            delta_k = k_nuevo - k_actual

            # Pérdida medida desde el nivel ORIGINAL del perfil de partida.
            nivel_origen = perfil_original.get(clave, info)["nivel"]
            valor_origen = perfil_original.get(clave, info)["valor"]
            lm = loss_metric_estructural(df, clave, valor_origen, nivel_origen, nuevo_nivel)
            rel = relevancia.get(clave, 1.0)
            coste = coste_combinado(lm, rel)

            score = delta_k / (1 + alpha * coste) if delta_k > 0 else 0.0

            movimientos.append({
                "clave": clave,
                "valor_antes": info["valor"],
                "nivel_antes": nivel_actual,
                "valor_despues": nuevo_valor,
                "nivel_despues": nuevo_nivel,
                "columna_despues": niveles[nuevo_nivel],
                "k_antes": k_actual,
                "k_despues": k_nuevo,
                "delta_k": delta_k,
                "lm_estructural": lm,
                "relevancia": rel,
                "coste_utilidad": coste,
                "score": score,
            })

    movimientos.sort(key=lambda m: m["score"], reverse=True)
    return movimientos


def recomendar_modificaciones(
    df: pd.DataFrame,
    perfil: dict,
    relevancia: Optional[dict[str, float]] = None,
    k_objetivo: int = 5,
    alpha: float = 1.0,
    max_pasos: int = 10,
    perdida_utilidad_maxima: Optional[float] = 0.5,
) -> dict:
    """
    Búsqueda voraz: en cada paso aplica el movimiento con mejor score
    (Δk / (1 + alpha * coste_combinado)) hasta alcanzar k_objetivo,
    agotar pasos o superar el presupuesto máximo de pérdida de utilidad.

    Si perdida_utilidad_maxima es None, no se aplica ningún límite adicional
    de pérdida de utilidad.
    """
    perfil_original = deepcopy(perfil)
    perfil_actual = deepcopy(perfil)
    historial = []

    alternativas_iniciales = evaluar_movimientos(
        df, perfil_actual, perfil_original, relevancia, alpha
    )

    def aplicar_temporalmente(perfil_base: dict, movimiento: dict) -> dict:
        """
        Devuelve una copia del perfil tras aplicar un movimiento.
        No modifica el perfil original.
        """
        perfil_tmp = deepcopy(perfil_base)
        perfil_tmp[movimiento["clave"]] = {
            "valor": movimiento["valor_despues"],
            "nivel": movimiento["nivel_despues"],
        }
        return perfil_tmp

    def respeta_presupuesto_utilidad(movimiento: dict) -> bool:
        """
        Comprueba si aplicar el movimiento mantendría la pérdida total
        de utilidad por debajo del máximo permitido.
        """
        if perdida_utilidad_maxima is None:
            return True

        perfil_tmp = aplicar_temporalmente(perfil_actual, movimiento)
        perdida_tmp = perdida_total_perfil(
            df,
            perfil_original,
            perfil_tmp,
            relevancia,
        )

        return perdida_tmp <= perdida_utilidad_maxima

    for _ in range(max_pasos):
        k_actual, _, _ = calcular_k_estricto(df, perfil_actual)

        if k_actual >= k_objetivo:
            break

        movimientos = evaluar_movimientos(
            df,
            perfil_actual,
            perfil_original,
            relevancia,
            alpha,
        )

        # Eliminamos movimientos que empeoran k.
        movimientos = [m for m in movimientos if m["delta_k"] >= 0]

        # Eliminamos movimientos que superarían el presupuesto máximo de utilidad.
        movimientos = [
            m for m in movimientos
            if respeta_presupuesto_utilidad(m)
        ]

        if not movimientos:
            break

        positivos = [m for m in movimientos if m["delta_k"] > 0]

        if positivos:
            # Si hay movimientos con ganancia real, usamos el score voraz.
            movimientos = positivos
            movimientos.sort(key=lambda m: m["score"], reverse=True)
        else:
            # Si no hay ganancia inmediata, permitimos movimientos en plano.
            # Elegimos el de menor coste de utilidad.
            movimientos.sort(key=lambda m: m["coste_utilidad"])

        mejor = movimientos[0]

        perfil_actual[mejor["clave"]] = {
            "valor": mejor["valor_despues"],
            "nivel": mejor["nivel_despues"],
        }

        historial.append(mejor)

    k_inicial, _, _ = calcular_k_estricto(df, perfil_original)
    k_final, _, _ = calcular_k_estricto(df, perfil_actual)
    perdida = perdida_total_perfil(df, perfil_original, perfil_actual, relevancia)

    return {
        "perfil_original": perfil_original,
        "perfil_final": perfil_actual,
        "k_inicial": k_inicial,
        "k_final": k_final,
        "perdida_utilidad": perdida,
        "historial_movimientos": historial,
        "alternativas_iniciales": alternativas_iniciales[:8],
        "objetivo_alcanzado": k_final >= k_objetivo,
        "perdida_utilidad_maxima": perdida_utilidad_maxima,
        "presupuesto_agotado": (
            perdida_utilidad_maxima is not None
            and k_final < k_objetivo
            and perdida >= perdida_utilidad_maxima
        ),
    }


# =====================================================================
# Reescritura del prompt
# =====================================================================

def reescribir_prompt(
    prompt_original: str,
    perfil_inicial: dict,
    perfil_final: dict,
) -> tuple[str, list[dict]]:
    """
    Aplica al texto del prompt las generalizaciones decididas. Para cada
    atributo cuyo nivel haya subido, sustituye el término en el texto por su
    forma generalizada (o por "[OMITIDO]" si el destino es supresión).
    """
    texto = prompt_original
    cambios: list[dict] = []

    for clave, info_orig in perfil_inicial.items():
        info_mod = perfil_final.get(clave, info_orig)
        if info_mod["nivel"] == info_orig["nivel"]:
            continue

        valor_orig = str(info_orig["valor"])
        col_destino = HIERARCHIES[clave][info_mod["nivel"]]
        suprimido = (col_destino == "*")
        valor_nuevo = "[OMITIDO]" if suprimido else str(info_mod["valor"])

        cambios.append({
            "atributo": clave,
            "tipo": "supresion" if suprimido else "generalizacion",
            "valor_original": valor_orig,
            "valor_sugerido": None if suprimido else valor_nuevo,
            "instruccion": (
                f"Eliminar la mención de '{valor_orig}' ({clave}) del prompt."
                if suprimido else
                f"Reemplazar '{valor_orig}' por '{valor_nuevo}' ({clave})."
            ),
        })

        if valor_orig.isalpha() or " " in valor_orig:
            patron = re.compile(rf"\b{re.escape(valor_orig)}\b", re.IGNORECASE)
        else:
            patron = re.compile(re.escape(valor_orig), re.IGNORECASE)
        texto = patron.sub(valor_nuevo, texto)

    return texto, cambios


# =====================================================================
# Clasificación de riesgo
# =====================================================================

UMBRALES_RIESGO = [(1, "crítico"), (3, "alto"), (10, "medio")]

def clasificar_riesgo(k: int) -> str:
    if k <= 0:
        return "indeterminado"
    for tope, etiqueta in UMBRALES_RIESGO:
        if k <= tope:
            return etiqueta
    return "bajo"


# =====================================================================
# Visualización: trade-off privacidad vs utilidad por atributo
# =====================================================================

def graficar_tradeoff(
    df: pd.DataFrame,
    perfil: dict,
    relevancia: Optional[dict[str, float]] = None,
    salida_png: str = "tradeoff_privacidad_utilidad.png",
) -> Optional[str]:
    """
    Genera una figura con dos paneles:
      - Izquierda: por cada atributo del perfil, una curva (coste_utilidad, k)
        que muestra cómo evoluciona el trade-off al subir niveles.
      - Derecha: scatter Pareto con todos los movimientos individuales del
        perfil de partida (Δk vs coste_utilidad), resaltando la frontera.

    Devuelve la ruta al PNG generado, o None si matplotlib no está disponible.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib no instalado. Instala con: pip install matplotlib")
        return None

    if not perfil:
        # Sin QIDs no hay trade-off que graficar (p. ej. prompt con solo un nombre).
        return None

    relevancia = relevancia or {}
    perfil_original = deepcopy(perfil)
    k_inicial, _, _ = calcular_k_estricto(df, perfil_original)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel 1: trayectoria por atributo ----
    colores = plt.cm.tab10.colors
    for idx, (clave, info) in enumerate(perfil_original.items()):
        if clave not in HIERARCHIES:
            continue
        niveles = HIERARCHIES[clave]
        xs, ys, labels = [0.0], [k_inicial], [f"{info['valor']}"]

        for nuevo_nivel in range(info["nivel"] + 1, len(niveles)):
            nuevo_valor = obtener_valor_en_nivel(df, clave, info["valor"], info["nivel"], nuevo_nivel)
            if nuevo_valor is None:
                continue
            perfil_tmp = deepcopy(perfil_original)
            perfil_tmp[clave] = {"valor": nuevo_valor, "nivel": nuevo_nivel}
            k_tmp, _, _ = calcular_k_estricto(df, perfil_tmp)
            lm = loss_metric_estructural(df, clave, info["valor"], info["nivel"], nuevo_nivel)
            rel = relevancia.get(clave, 1.0)
            coste = coste_combinado(lm, rel)
            xs.append(coste)
            ys.append(k_tmp)
            labels.append(nuevo_valor if nuevo_valor != "*" else "[suprimido]")

        if len(xs) > 1:
            color = colores[idx % len(colores)]
            ax1.plot(xs, ys, marker="o", color=color, linewidth=2,
                     label=f"{clave} (rel={relevancia.get(clave, 1.0):.2f})")
            for x, y, lbl in zip(xs, ys, labels):
                ax1.annotate(lbl, (x, y), textcoords="offset points",
                             xytext=(6, 4), fontsize=8, color=color)

    ax1.set_xlabel("Coste de utilidad (LM × relevancia)")
    ax1.set_ylabel("k resultante")
    ax1.set_title("Trade-off por atributo: cómo crece k al generalizarlo")
    ax1.axhline(y=k_inicial, linestyle="--", color="gray", alpha=0.5,
                label=f"k inicial = {k_inicial}")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Pareto de movimientos individuales ----
    movimientos = evaluar_movimientos(df, perfil_original, perfil_original, relevancia, alpha=1.0)
    if movimientos:
        xs = [m["coste_utilidad"] for m in movimientos]
        ys = [m["delta_k"] for m in movimientos]
        labels = [f"{m['clave']}→{m['valor_despues']}" for m in movimientos]

        ax2.scatter(xs, ys, s=80, alpha=0.7,
                    c=[colores[i % len(colores)] for i in range(len(xs))])
        for x, y, lbl in zip(xs, ys, labels):
            ax2.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)

        # Frontera de Pareto (max Δk, min coste)
        puntos = sorted(zip(xs, ys, labels), key=lambda p: (p[0], -p[1]))
        pareto = []
        mejor_y = -1
        for x, y, lbl in puntos:
            if y > mejor_y:
                pareto.append((x, y))
                mejor_y = y
        if pareto:
            px, py = zip(*pareto)
            ax2.plot(px, py, color="red", linestyle="--", alpha=0.6,
                     label="Frontera Pareto")

    ax2.set_xlabel("Coste de utilidad (LM × relevancia)")
    ax2.set_ylabel("Δk (ganancia de privacidad)")
    ax2.set_title("Movimientos individuales: privacidad ganada vs utilidad perdida")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(salida_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return salida_png