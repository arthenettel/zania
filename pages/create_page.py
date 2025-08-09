"""
Página: Crear receta (2 opciones lado a lado)

• No dibuja el sidebar. Asume que app.py lo crea con streamlit-option-menu.
• Usa GOOGLE_API_KEY desde .env (python-dotenv) o st.secrets como respaldo.
• Flujo: el usuario ingresa ingredientes → botón "Crear" → se generan 2 recetas.
• Cada receta muestra: Título, Tiempo, Nivel, Ingredientes (tabla),
  Procedimiento (pasos detallados), Información nutricional (kcal, proteínas, carbohidratos, grasas),
  y Tips. Todo por PORCIÓN.

Integración en app.py:

    from pages.create_page import render_create
    # ...
    elif st.session_state.nav == "Crear receta":
        render_create()

Requisitos:
    streamlit>=1.33
    google-generativeai>=0.8.0
    python-dotenv>=1.0.1
"""
from __future__ import annotations
import os
import json
import re
import hashlib
from typing import Dict, Any, List

import streamlit as st

# --- .env ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Gemini SDK ---
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


# =====================
# Helpers IA (Gemini)
# =====================

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _ensure_gemini_ready() -> None:
    if genai is None:
        raise RuntimeError("Falta instalar 'google-generativeai' (añade a requirements.txt).")
    api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None)
    if not api_key:
        raise RuntimeError("No se encontró GOOGLE_API_KEY en .env ni en st.secrets.")
    genai.configure(api_key=api_key)


def _call_gemini_recipes(ingredients_text: str) -> Dict[str, Any]:
    """
    Devuelve JSON con dos recetas:
    {
      "recetas": [
        {
          "titulo": str,
          "tiempo_min": number,
          "nivel": "básico"|"intermedio"|"difícil",
          "ingredientes": [{"nombre": str, "cantidad": str}],
          "procedimiento": [str, ...],        # pasos detallados
          "nutricion": {"kcal": number, "proteinas_g": number, "carbohidratos_g": number, "grasas_g": number},
          "tips": [str, ...]
        },
        { ... segunda receta ... }
      ]
    }
    """
    _ensure_gemini_ready()

    # Prompt: experto en nutrición, chef y visión de alimentos. Sin forzar “saludable”.
    prompt = (
        "Actúa como experto en nutrición, chef y visión de alimentos. Con los ingredientes proporcionados, "
        "genera DOS opciones de receta diferentes (si hacen falta básicos como sal, agua o aceite, puedes agregarlos). "
        "Todo debe ser para UNA porción. No transformes el platillo en una versión saludable: respeta el estilo culinario.\n\n"
        "Devuelve SOLO JSON con la forma EXACTA de:\n"
        "{\n"
        '  "recetas": [\n'
        "    {\n"
        '      "titulo": "string",\n'
        '      "tiempo_min": number,\n'
        '      "nivel": "básico" | "intermedio" | "difícil",\n'
        '      "ingredientes": [{"nombre": "string", "cantidad": "string"}],\n'
        '      "procedimiento": ["paso 1 detallado", "paso 2 detallado", "..."],\n'
        '      "nutricion": {"kcal": number, "proteinas_g": number, "carbohidratos_g": number, "grasas_g": number},\n'
        '      "tips": ["tip 1", "tip 2"]\n'
        "    },\n"
        "    { ... segunda receta con misma estructura ... }\n"
        "  ]\n"
        "}\n\n"
        f"Ingredientes del usuario: {ingredients_text}\n"
    )

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content([{"text": prompt}])
    txt = resp.text or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(_strip_code_fences(txt))


# =====================
# UI / Render
# =====================

def _render_recipe(receta: Dict[str, Any]) -> None:
    # Sección 1: Título, tiempo, nivel
    st.markdown(f"### {receta.get('titulo', '—')}")
    col_t, col_n = st.columns(2)
    with col_t:
        st.markdown(f"⏱ **Tiempo:** {receta.get('tiempo_min', '—')} min")
    with col_n:
        st.markdown(f"📊 **Nivel:** {str(receta.get('nivel', '—')).capitalize()}")

    st.divider()

    # Sección 2: Ingredientes (tabla)
    st.markdown("**Ingredientes**")
    ings = receta.get("ingredientes") or []
    if isinstance(ings, list) and ings and isinstance(ings[0], dict):
        st.table(ings)
    else:
        st.write("Sin ingredientes.")

    st.divider()

    # Sección 3: Procedimiento (pasos detallados)
    st.markdown("**Procedimiento**")
    pasos = receta.get("procedimiento") or []
    if pasos:
        for i, paso in enumerate(pasos, start=1):
            st.markdown(f"**{i}.** {paso}")  # sin líneas divisorias
    else:
        st.write("Sin procedimiento.")

    st.divider()

    # Sección 4: Info nutricional
    st.markdown("**Información nutricional (por porción)**")
    nut = receta.get("nutricion") or {}
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("🔥 KCal", f"{nut.get('kcal', '—')}")
    col_b.metric("💪 Proteínas", f"{nut.get('proteinas_g', '—')} g")
    col_c.metric("🍞 Carbohidratos", f"{nut.get('carbohidratos_g', '—')} g")
    col_d.metric("🥑 Grasas", f"{nut.get('grasas_g', '—')} g")

    st.divider()

    # Sección 5: Tips
    st.markdown("**Tips**")
    tips = receta.get("tips") or []
    if tips:
        for t in tips:
            st.markdown(f"• {t}")
    else:
        st.write("Sin tips.")


def render_create():
    # cache por ingredientes para ahorrar cuota
    if "_create_cache" not in st.session_state:
        st.session_state._create_cache = {}

    st.markdown("# Crear receta")
    st.caption(
        "Ingresa los ingredientes que tienes (separados por comas). "
        "Al presionar **Crear**, se generarán **dos opciones de receta** para **una porción**. "
    )

    ingredientes = st.text_area("🧾 Ingredientes", placeholder="ej. pechuga de pollo, arroz, jitomate, cebolla, ajo, aceite", height=120)

    cols = st.columns([1, 1, 2])
    with cols[0]:
        crear = st.button("🍽️ Crear", use_container_width=True)
    with cols[1]:
        force = st.checkbox("Forzar re-generar", value=False)

    # Resultado
    if crear:
        if not ingredientes.strip():
            st.warning("Por favor, ingresa al menos un ingrediente.")
            return

        digest = hashlib.sha256(ingredientes.strip().lower().encode("utf-8")).hexdigest()
        cached = st.session_state._create_cache.get(digest)

        if cached and not force:
            data = cached
        else:
            with st.spinner("Generando recetas con Gemini…"):
                try:
                    data = _call_gemini_recipes(ingredientes.strip())
                    st.session_state._create_cache[digest] = data
                except Exception as e:
                    st.error(f"No fue posible generar recetas: {e}")
                    return

        recetas: List[Dict[str, Any]] = data.get("recetas") or []
        if len(recetas) < 2:
            st.error("La IA no devolvió dos recetas. Intenta de nuevo o usa 'Forzar re-generar'.")
            return

        c1, c2 = st.columns(2)
        with c1:
            _render_recipe(recetas[0])
        with c2:
            _render_recipe(recetas[1])

