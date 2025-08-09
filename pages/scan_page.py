"""
Página: Escanear platillo (modular)

• No dibuja el sidebar. Asume que app.py lo crea con streamlit-option-menu.
• Usa GOOGLE_API_KEY desde .env (python-dotenv) o st.secrets como respaldo.
• Layout: 2 columnas (izq: captura; der: resultado). Abajo: "Analiza tu platillo".

Integración en app.py:

    from pages.scan_page import render_scan
    # ...
    elif st.session_state.nav == "Escanear platillo":
        render_scan()

Requisitos:
    streamlit>=1.33
    pillow>=10.0.0
    google-generativeai>=0.8.0
    python-dotenv>=1.0.1
"""
from __future__ import annotations
import os
import io
import json
import re
from typing import Tuple, List

import streamlit as st
from PIL import Image

# --- Carga .env para entorno local ---
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

GEMINI_MODEL = "gemini-1.5-flash"
SYSTEM_PROMPT = (
    "Eres un asistente de visión experto en alimentos. Recibirás una foto de un platillo. "
    "Devuelve SOLO un JSON con esta forma exacta: "
    "{\"name\": \"<nombre del platillo>\", \"ingredients\": [\"ing1\", \"ing2\", \"ing3\"]}. "
    "Si no reconoces, usa name=\"Platillo\" e ingredients=[]."
)

# =====================
# Gemini helpers
# =====================

def _strip_code_fences(text: str) -> str:
    """Quita ```json ... ``` si el modelo lo agrega."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _ensure_gemini_ready() -> Tuple[bool, str]:
    if genai is None:
        return False, "Falta instalar 'google-generativeai' (añade a requirements.txt)."
    api_key = os.getenv("GOOGLE_API_KEY") or (
        st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
    )
    if not api_key:
        return False, "No se encontró GOOGLE_API_KEY en .env ni en st.secrets."
    try:
        genai.configure(api_key=api_key)
        return True, ""
    except Exception as e:
        return False, f"Error al configurar Gemini: {e}"


def gemini_analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> Tuple[str, List[str]]:
    """Devuelve (name, ingredients) usando respuesta JSON estricta."""
    ok, msg = _ensure_gemini_ready()
    if not ok:
        raise RuntimeError(msg)

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    parts = [
        {"text": SYSTEM_PROMPT},
        {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
    ]
    resp = model.generate_content(parts)

    txt = resp.text or "{}"
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = json.loads(_strip_code_fences(txt))

    name = str(data.get("name") or "Platillo").strip()
    ingredients = [str(x).strip() for x in (data.get("ingredients") or []) if str(x).strip()]
    return name, ingredients


# =====================
# UI / Render
# =====================

def render_scan():
    # Estado local
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = {"name": None, "ingredients": []}
    if "analysis_panel" not in st.session_state:
        st.session_state.analysis_panel = None

    st.markdown("# Escanear platillo")

    # 2 columnas: captura | resultado
    col_left, gap, col_right = st.columns([1, 0.1, 1])

    with col_left:
        st.subheader("Sube o toma una foto")

        metodo = st.radio(
            "Selecciona el método de captura",
            ["Subir imagen", "Tomar foto"],
            index=0,
            key="capture_method",
        )

        image_bytes = None
        mime = None
        file_name = None

        if metodo == "Subir imagen":
            up = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"], key="uploader")
            if up is not None:
                image_bytes = up.getvalue()
                file_name = up.name
                mime = "image/png" if (file_name or "").lower().endswith(".png") else "image/jpeg"
        else:
            cam = st.camera_input("Tomar foto", key="camera")
            if cam is not None:
                image_bytes = cam.getvalue()
                file_name = "camera.jpg"
                mime = "image/jpeg"

        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Vista previa", use_column_width=True)

            with st.spinner("Analizando imagen con Gemini…"):
                try:
                    name, ingredients = gemini_analyze_image(image_bytes, mime or "image/jpeg")
                except Exception as e:
                    st.error(f"No fue posible analizar la imagen. {e}")
                    name, ingredients = None, []
            st.session_state.scan_result = {"name": name, "ingredients": ingredients}
        else:
            st.session_state.scan_result = {"name": None, "ingredients": []}

    with col_right:
        st.subheader("Resultado")
        res = st.session_state.scan_result
        if res["name"]:
            st.markdown(f"### {res['name']}")
            if res["ingredients"]:
                for i, ing in enumerate(res["ingredients"]):
                    st.markdown(f"- {ing}")
                    if i < len(res["ingredients"]) - 1:
                        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.15); margin:6px 0;'>", unsafe_allow_html=True)
            else:
                st.caption("No se detectaron ingredientes con suficiente confianza.")
        else:
            st.info("Cuando subas o tomes una foto, aquí aparecerán el nombre del platillo y sus ingredientes.")

    st.divider()

    # Sección inferior: Analiza tu platillo
    st.markdown("## Analiza tu platillo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🍳 Cómo cocinarlo", use_container_width=True):
            st.session_state.analysis_panel = "Cómo cocinarlo"
    with col2:
        if st.button("📊 Información nutricional", use_container_width=True):
            st.session_state.analysis_panel = "Información nutricional"
    with col3:
        if st.button("🔁 Alternativas similares", use_container_width=True):
            st.session_state.analysis_panel = "Alternativas similares"
    with col4:
        if st.button("📝 Generar reporte", use_container_width=True):
            st.session_state.analysis_panel = "Generar reporte"

    if st.session_state.analysis_panel:
        with st.container(border=True):
            st.markdown(f"### {st.session_state.analysis_panel}")
            st.caption("(Contenido de ejemplo. Más adelante agregaremos la lógica completa.)")
            if st.button("Cerrar", key="close_panel"):
                st.session_state.analysis_panel = None
                st.rerun()
