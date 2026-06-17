"""
Pipeline global del sistema de privacidad para prompts.

Orquesta los tres módulos:
    extractor.py             -> extracción LLM (atributos + relevancia)
    metrica_k.py             -> cálculo de k y sesión acumulativa
    modificador_prompt.py    -> recomendaciones, reescritura, gráfica

Función principal:
    pipeline_completa(prompt, df, k_objetivo, alpha) -> dict veredicto

CLI:
    python pipeline.py "Soy investigadora en IA en Valencia, nací en 1981"
    python pipeline.py --mock              # un solo prompt, sin API
    python pipeline.py -i                  # modo interactivo multi-prompt
    python pipeline.py -i --mock           # interactivo sin API

EJEMPLOS:
    python pipeline.py "Soy ingeniero de software en Bilbao. ¿Qué frameworks de desarrollo recomiendas aprender en 2025?"
    python pipeline.py "Soy ingeniero de software especializado en ciberseguridad en Bilbao. ¿Qué certificaciones recomiendas?"
    python pipeline.py "Soy un hombre ingeniero de software especializado en ciberseguridad, vivo en Bilbao y trabajo en Accenture España. Nací en 1960. ¿Cómo solicito una excedencia?"
    python pipeline.py "Soy investigadora en aprendizaje automático en la Universidad Autónoma de Madrid, nací en 1999. ¿Cuáles son los grupos de investigación punteros en mi área?"
    python pipeline.py "Soy médica generalista y trabajo en Sevilla. ¿Qué congresos de medicina familiar hay este año?"
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd

from metrica_k import (
    DATASET_PATH,
    SesionPrivacidad,
    calcular_k_estricto,
    normalizar_criterios,
    HIERARCHIES,
)
from modificador_prompt import (
    clasificar_riesgo,
    evaluar_movimientos,
    graficar_tradeoff,
    recomendar_modificaciones,
    reescribir_prompt,
)


# =====================================================================
# Análisis de un prompt ya extraído
# =====================================================================

def analizar_prompt(
    prompt_original: str,
    extraccion: dict,
    df: pd.DataFrame,
    k_objetivo: int = 7,
    alpha: float = 1.0,
    sesion: Optional[SesionPrivacidad] = None,
) -> dict:
    """
    Recibe el resultado del extractor (criterios + relevancia + identificadores)
    y devuelve el veredicto completo.

    Si se pasa una `sesion`, los criterios se acumulan sobre ella.
    """
    relevancia = extraccion.get("relevancia") or {}
    identificadores_directos = extraccion.get("identificadores_directos") or []

    # Construir el dict de criterios para el motor de k.
    criterios = {
        k: v for k, v in extraccion.items()
        if k not in ("relevancia", "identificadores_directos") and v is not None
    }

    if sesion is None:
        perfil = normalizar_criterios(criterios)
    else:
        sesion.añadir_prompt(criterios)
        perfil = deepcopy(sesion.perfil)

    k_actual, candidatos, usados = calcular_k_estricto(df, perfil)
    riesgo = clasificar_riesgo(k_actual)

    veredicto: dict = {
        "k": k_actual,
        "riesgo": riesgo,
        "atributos_efectivos": usados,
        "perfil_canonico": {
            c: {
                "valor": i["valor"],
                "nivel": i["nivel"],
                "columna": HIERARCHIES[c][i["nivel"]],
                "relevancia": relevancia.get(c, 1.0),
            }
            for c, i in perfil.items()
        },
        "identificadores_directos": identificadores_directos,
    }

    if "person_id" in candidatos.columns and 0 < len(candidatos) <= 10:
        cols = [c for c in ["person_id", "nombre_completo"] if c in candidatos.columns]
        veredicto["candidatos_top"] = candidatos[cols].head(10).to_dict(orient="records")

    if k_actual >= k_objetivo and not identificadores_directos:
        veredicto.update({
            "decision": "ok",
            "mensaje": f"k = {k_actual} ≥ k_objetivo = {k_objetivo}. Sin cambios necesarios.",
            "prompt_sugerido": prompt_original,
            "cambios": [],
        })
        return veredicto

    rec = recomendar_modificaciones(
        df, perfil, relevancia=relevancia, k_objetivo=k_objetivo, alpha=alpha
    )
    prompt_reescrito, cambios = reescribir_prompt(
        prompt_original, rec["perfil_original"], rec["perfil_final"]
    )

    if identificadores_directos:
        riesgo = "crítico"
        for ident in identificadores_directos:
            cambios.append({
                "atributo": "identificador_directo",
                "tipo": "supresion",
                "valor_original": ident,
                "valor_sugerido": None,
                "instruccion": f"Eliminar el identificador directo '{ident}' del prompt.",
            })
        # Suprimir también del texto reescrito.
        for ident in identificadores_directos:
            prompt_reescrito = prompt_reescrito.replace(ident, "[OMITIDO]")

    # Mensaje conceptualmente correcto:
    # - DI presente: el identificador directo es lo prioritario; k pierde sentido
    #   porque la reidentificación es directa, no por inferencia.
    # - DI ausente: el mensaje habla solo de k.
    partes_mensaje = []
    if identificadores_directos:
        partes_mensaje.append(
            f"Identificador directo detectado: {identificadores_directos}. "
            f"Debe eliminarse del prompt (k-anonymity asume que NO hay identificadores; "
            f"con uno presente la reidentificación es trivial, independientemente de k)."
        )
    if k_actual < k_objetivo:
        partes_mensaje.append(
            f"Sobre los QIDs restantes: k = {k_actual} < {k_objetivo}. "
            f"Aplicando recomendaciones se alcanza k = {rec['k_final']} con coste "
            f"de utilidad medio {rec['perdida_utilidad']:.2f} (LM × relevancia)."
        )
    elif identificadores_directos:
        partes_mensaje.append(
            f"Tras eliminar el identificador, los QIDs restantes dan k = {k_actual} "
            f"≥ {k_objetivo}: sin riesgo añadido por inferencia."
        )

    veredicto.update({
        "riesgo": riesgo,
        "decision": "modificar" if rec["k_final"] >= k_objetivo else "modificar_parcial",
        "mensaje": " ".join(partes_mensaje),
        "k_objetivo": k_objetivo,
        "k_alcanzado": rec["k_final"],
        "perdida_utilidad": round(rec["perdida_utilidad"], 3),
        "movimientos_aplicados": [
            {
                "atributo": m["clave"],
                "de": m["valor_antes"],
                "a": m["valor_despues"],
                "k_antes": m["k_antes"],
                "k_despues": m["k_despues"],
                "delta_k": m["delta_k"],
                "lm_estructural": round(m["lm_estructural"], 3),
                "relevancia": round(m["relevancia"], 2),
                "coste_utilidad": round(m["coste_utilidad"], 3),
            } for m in rec["historial_movimientos"]
        ],
        "alternativas": [
            {
                "atributo": m["clave"],
                "de": m["valor_antes"],
                "a": m["valor_despues"],
                "delta_k": m["delta_k"],
                "lm_estructural": round(m["lm_estructural"], 3),
                "relevancia": round(m["relevancia"], 2),
                "coste_utilidad": round(m["coste_utilidad"], 3),
                "score": round(m["score"], 2),
            } for m in rec["alternativas_iniciales"]
        ],
        "cambios": cambios,
        "prompt_sugerido": prompt_reescrito,
    })

    return veredicto


# =====================================================================
# Pipeline de extremo a extremo
# =====================================================================

def pipeline_completa(
    prompt_usuario: str,
    df: pd.DataFrame,
    k_objetivo: int = 5,
    alpha: float = 1.0,
    modelo: str = "gpt-4.1-mini",
    sesion: Optional[SesionPrivacidad] = None,
    mock: bool = False,
    generar_grafica: bool = True,
) -> dict:
    """
    Ejecuta extractor → analizador → reescritor → gráfica.

    Si mock=True, evita la llamada al API y usa una extracción simulada
    coherente con el prompt de demo.
    """
    if mock:
        extraccion = _extraccion_mock(prompt_usuario)
    else:
        from extractor import extraer_atributos
        extraccion = extraer_atributos(prompt_usuario, modelo=modelo)

    veredicto = analizar_prompt(
        prompt_original=prompt_usuario,
        extraccion=extraccion,
        df=df,
        k_objetivo=k_objetivo,
        alpha=alpha,
        sesion=sesion,
    )
    veredicto["modelo_extractor"] = "mock" if mock else modelo
    veredicto["extraccion_bruta"] = extraccion

    if generar_grafica:
        # Reconstruimos el perfil del veredicto.
        perfil = normalizar_criterios({
            k: v for k, v in extraccion.items()
            if k not in ("relevancia", "identificadores_directos") and v is not None
        })
        # Sin QIDs no hay nada que dibujar (caso típico: prompt con solo nombre).
        if perfil:
            ruta = graficar_tradeoff(
                df=df,
                perfil=perfil,
                relevancia=extraccion.get("relevancia") or {},
                salida_png="tradeoff_privacidad_utilidad.png",
            )
            if ruta:
                veredicto["grafica"] = ruta

    return veredicto


def _extraccion_mock(prompt_usuario: str) -> dict:
    """
    Extracción simulada para correr el pipeline sin llamar al API.
    Usa un perfil hardcodeado (el del prompt por defecto) para fines de demo.
    """
    return {
        "rol": "investigadora",
        "area": "inteligencia artificial",
        "ciudad": "Valencia",
        "region": None,
        "pais": None,
        "institucion_nombre": None,
        "institucion_tipo": "universidad pública",
        "año_nacimiento": 1981,
        "decada_nacimiento": None,
        "identificadores_directos": [],
        "relevancia": {
            "rol":         1.0,
            "area":        1.0,
            "ciudad":      0.3,
            "institucion": 0.4,
            "año":        0.0,
        },
    }


# =====================================================================
# CLI / Demo
# =====================================================================

def _imprimir_veredicto(v: dict) -> None:
    print("\n" + "-" * 70)
    print(f"Decisión: {v['decision']}   |   "
          f"Riesgo: {v['riesgo']}   |   "
          f"k = {v['k']}")
    print("-" * 70,"\n")
    print(v["mensaje"])

    if v.get("identificadores_directos"):
        print("\n Identificadores directos detectados:")
        for ident in v["identificadores_directos"]:
            print(f"   - {ident}")

    if v.get("perfil_canonico"):
        print("\nPerfil canónico (con relevancia):")
        for clave, info in v["perfil_canonico"].items():
            print(f"   - {clave:<12}  {info['valor']:<30} "
                  f"(nivel {info['nivel']}, rel={info['relevancia']:.2f})")

    if v.get("movimientos_aplicados"):
        print("\nMovimientos aplicados:")
        for m in v["movimientos_aplicados"]:
            print(f"   - {m['atributo']}: '{m['de']}' → '{m['a']}'  "
                  f"k {m['k_antes']}→{m['k_despues']}  "
                  f"(LM={m['lm_estructural']}, rel={m['relevancia']}, "
                  f"coste={m['coste_utilidad']})")

    if v.get("alternativas"):
        print("\nAlternativas (top 5 por score):")
        for m in v["alternativas"][:5]:
            print(f"   - {m['atributo']}: '{m['de']}' → '{m['a']}'  "
                  f"Δk=+{m['delta_k']}, coste={m['coste_utilidad']}, "
                  f"score={m['score']}")

    if v.get("prompt_sugerido"):
        print(f"\nPrompt reescrito:\n   {v['prompt_sugerido']}")

    if v.get("grafica"):
        print(f"\nGráfica generada: {v['grafica']}")


# =====================================================================
# Modo interactivo (sesión multi-prompt)
# =====================================================================

def _aplicar_movimiento(perfil: dict, movimiento: dict) -> dict:
    """Devuelve un perfil nuevo con el movimiento aplicado."""
    nuevo = deepcopy(perfil)
    nuevo[movimiento["clave"]] = {
        "valor": movimiento["valor_despues"],
        "nivel": movimiento["nivel_despues"],
    }
    return nuevo


def _imprimir_perfil(perfil: dict, relevancia: dict) -> None:
    if not perfil:
        print("   (vacío)")
        return
    for c, info in perfil.items():
        col = HIERARCHIES[c][info["nivel"]] if c in HIERARCHIES else "?"
        rel = relevancia.get(c, 1.0)
        print(f"   - {c:<12} = {str(info['valor']):<28} "
              f"(col {col}, nivel {info['nivel']}, rel {rel:.2f})")


def bucle_interactivo(
    df: pd.DataFrame,
    k_objetivo: int = 5,
    alpha: float = 1.0,
    modelo: str = "gpt-4.1-mini",
    mock: bool = False,
    max_alternativas: int = 8,
) -> None:
    """
    Modo interactivo multi-prompt. La sesión vive en memoria; cada prompt se
    extrae, se calcula k acumulado y, si está por debajo del objetivo, se
    ofrecen alternativas numeradas. El usuario elige hasta llegar al objetivo
    o decide aceptar el k actual. Lo que queda en sesion.perfil es siempre la
    versión finalmente enviada (generalizada).

    Comandos: /salir, /reset, /estado, /ayuda.
    """
    sesion = SesionPrivacidad(df=df)
    print("=" * 70)
    print("Modo interactivo. Escribe prompts uno tras otro.")
    print("Comandos: /salir  /reset  /estado  /ayuda")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt in ("/salir", "/exit", "/quit"):
            break
        if prompt == "/reset":
            sesion = SesionPrivacidad(df=df)
            print("Sesión reiniciada.")
            continue
        if prompt == "/estado":
            print(sesion.resumen())
            print("\nPerfil acumulado actual:")
            _imprimir_perfil(sesion.perfil, {})
            continue
        if prompt == "/ayuda":
            print("Escribe un prompt natural, o usa /salir /reset /estado.")
            continue

        # --- 1. Extracción ---
        if mock:
            extraccion = _extraccion_mock(prompt)
        else:
            try:
                from extractor import extraer_atributos
                extraccion = extraer_atributos(prompt, modelo=modelo)
            except (ImportError, RuntimeError) as e:
                print(f"[ERROR] {e}")
                print("Sugerencia: arranca el modo interactivo con --mock para probar sin API.")
                continue

        criterios = {
            k: v for k, v in extraccion.items()
            if k not in ("relevancia", "identificadores_directos") and v is not None
        }
        relevancia = extraccion.get("relevancia") or {}
        identificadores = extraccion.get("identificadores_directos") or []

        # --- 2. Acumular en la sesión (peor caso si se mandase tal cual) ---
        # Snapshot por si el usuario decide cancelar este prompt.
        perfil_previo = deepcopy(sesion.perfil)
        historico_previo = list(sesion.historico)
        sesion.añadir_prompt(criterios)
        perfil_acumulado = deepcopy(sesion.perfil)
        k_actual, _, _ = calcular_k_estricto(df, perfil_acumulado)
        riesgo = clasificar_riesgo(k_actual)

        # Solo es modificable lo que el usuario revela o refina EN ESTE prompt.
        # Lo de prompts anteriores ya está enviado y no podemos retroceder.
        # Techo = el nivel previo (no podemos generalizar más allá de lo ya expuesto).
        nuevos_aqui = normalizar_criterios(criterios)
        nivel_techo: dict[str, int] = {}
        for clave, info_nueva in nuevos_aqui.items():
            info_previa = perfil_previo.get(clave)
            if info_previa is None:
                nivel_techo[clave] = len(HIERARCHIES[clave]) - 1   # clave nueva
            elif info_nueva["nivel"] < info_previa["nivel"]:
                nivel_techo[clave] = info_previa["nivel"]          # refinamiento

        print(f"\n[k acumulado = {k_actual}]   riesgo: {riesgo}")
        print("Perfil acumulado tras este prompt:")
        _imprimir_perfil(perfil_acumulado, relevancia)
        if identificadores:
            print(f" Identificadores directos detectados: {identificadores}")

        if k_actual >= k_objetivo and not identificadores:
            print(f"\nk = {k_actual} ≥ {k_objetivo}. Puedes enviar el prompt tal cual.")
            print(f"Prompt a enviar:\n   {prompt}")
            continue

        # --- 3. Bucle de elección de alternativas ---
        perfil_trabajo = deepcopy(perfil_acumulado)
        movimientos_aplicados = []

        while True:
            k_t, _, _ = calcular_k_estricto(df, perfil_trabajo)
            if k_t >= k_objetivo:
                print(f"\nObjetivo alcanzado: k = {k_t} ≥ {k_objetivo}.")
                break

            movimientos = evaluar_movimientos(
                df, perfil_trabajo, perfil_acumulado, relevancia, alpha
            )
            movimientos = [
                m for m in movimientos
                if m["delta_k"] > 0
                and m["clave"] in nivel_techo
                and m["nivel_despues"] <= nivel_techo[m["clave"]]
            ][:max_alternativas]
            if not movimientos:
                print("\nNo hay movimientos aplicables a este prompt.")
                if k_t < k_objetivo:
                    print(
                        f"k = {k_t} viene del acumulado de prompts anteriores ya enviados;\n"
                        f"este prompt no aporta atributos nuevos que generalizar.\n"
                        f"Opciones: 0 = enviar tal cual, c = cancelar este prompt."
                    )
                    try:
                        eleccion = input("Elige (0/c): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        eleccion = "0"
                    if eleccion in ("c", "cancelar"):
                        sesion.perfil = perfil_previo
                        sesion.historico = historico_previo
                        print("Prompt cancelado.")
                        perfil_trabajo = None
                break

            print(f"\nAlternativas (k actual = {k_t}, objetivo {k_objetivo}):")
            for i, m in enumerate(movimientos, 1):
                print(
                    f"  {i}. {m['clave']:<12} '{m['valor_antes']}' → '{m['valor_despues']}'"
                    f"   Δk=+{m['delta_k']}, coste={m['coste_utilidad']:.3f}, "
                    f"score={m['score']:.2f}"
                )
            print("  0. enviar tal cual (sin aplicar/aplicar más alternativas)")
            print("  c. cancelar este prompt (no lo envío ni se acumula en la sesión)")

            try:
                eleccion = input("Elige: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                eleccion = "0"

            if eleccion in ("", "0"):
                break
            if eleccion in ("c", "cancelar"):
                # Revertir el efecto de añadir_prompt sobre la sesión.
                sesion.perfil = perfil_previo
                sesion.historico = historico_previo
                print("Prompt cancelado. La sesión queda como antes de este prompt.")
                perfil_trabajo = None
                break
            try:
                idx = int(eleccion) - 1
            except ValueError:
                print("Entrada no válida.")
                continue
            if not 0 <= idx < len(movimientos):
                print("Índice fuera de rango.")
                continue

            mov = movimientos[idx]
            perfil_trabajo = _aplicar_movimiento(perfil_trabajo, mov)
            movimientos_aplicados.append(mov)
            print(f"   Aplicado. k: {mov['k_antes']} → {mov['k_despues']}.")

        # Si el usuario canceló dentro del bucle, saltamos a la siguiente vuelta.
        if perfil_trabajo is None:
            continue

        # --- 4. Reescritura del prompt ---
        if movimientos_aplicados:
            prompt_reescrito, cambios = reescribir_prompt(
                prompt, perfil_acumulado, perfil_trabajo
            )
        else:
            prompt_reescrito, cambios = prompt, []

        for ident in identificadores:
            prompt_reescrito = prompt_reescrito.replace(ident, "[OMITIDO]")
            cambios.append({
                "instruccion": f"Eliminar identificador directo '{ident}'."
            })

        # --- 5. La sesión queda con el perfil REALMENTE enviado ---
        sesion.perfil = perfil_trabajo

        print(f"\nPrompt a enviar:\n   {prompt_reescrito}")
        if cambios:
            print("Cambios aplicados:")
            for c in cambios:
                print(f"   - {c['instruccion']}")

    print("\n" + "=" * 70)
    print("Fin de sesión.")
    print("=" * 70)
    print(sesion.resumen())


if __name__ == "__main__":
    args = sys.argv[1:]
    mock = "--mock" in args
    interactivo = ("--interactivo" in args) or ("-i" in args)
    args = [a for a in args if a not in ("--mock", "--interactivo", "-i")]

    df = pd.read_csv(Path(__file__).parent / DATASET_PATH)

    if interactivo:
        bucle_interactivo(df, k_objetivo=5, alpha=1.0, mock=mock)
        sys.exit(0)

    if args:
        prompt_in = " ".join(args)
    else:
        prompt_in = (
            "Hola, soy investigadora en IA en una universidad pública de Valencia, "
            "nací en 1981. ¿Qué congresos del área me recomiendas para este año?"
        )

    print(f"Prompt original:\n   {prompt_in}")

    try:
        veredicto = pipeline_completa(prompt_in, df, k_objetivo=5, alpha=1.0, mock=mock)
    except (ImportError, RuntimeError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    _imprimir_veredicto(veredicto)

