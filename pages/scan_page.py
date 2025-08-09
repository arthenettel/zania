"""
Página: Escanear platillo (modular, ahorro de peticiones + Análisis con Gemini)

• No dibuja el sidebar. Asume que app.py lo crea con streamlit-option-menu.
• Usa GOOGLE_API_KEY desde .env (python-dotenv) o st.secrets como respaldo.
• Layout: 2 columnas (izq: captura; der: resultado). Abajo: "Analiza tu platillo".
• SOLO llama a Gemini cuando el usuario presiona *Analizar ahora* y cachea el resultado
  por imagen para no repetir llamadas si la imagen es la misma. Las opciones de análisis
  también se ejecutan bajo demanda y se cachean por imagen+opción.
• Las evaluaciones nutricionales y culinarias se hacen *para una porción* y
  se basan SOLO en: *Norma Oficial Mexicana NOM-043* y *Guías Alimentarias para la Población Mexicana*.

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
from _future_ import annotations
import os
import io
import json
import re
import hashlib
from typing import Tuple, List, Dict, Any

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
    """Quita json ...  si el modelo lo agrega."""
    t = (text or "").strip()
    if t.startswith(""):
        t = re.sub(r"^[a-zA-Z0-9]*\n", "", t)
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


def _call_gemini_json(parts: list) -> Dict[str, Any]:
    ok, msg = _ensure_gemini_ready()
    if not ok:
        raise RuntimeError(msg)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content(parts)
    txt = resp.text or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(_strip_code_fences(txt))


def gemini_identify(image_bytes: bytes, mime_type: str = "image/jpeg") -> Tuple[str, List[str]]:
    """Identifica nombre e ingredientes visibles (lista simple)."""
    prompt = (
        f"{SYSTEM_PROMPT}. Devuelve SOLO JSON con: "
        "{\"name\": string, \"ingredients\": [string,...]}.")
    data = _call_gemini_json([
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
    ])
    name = (data.get("name") or "Platillo").strip()
    ingredients = [str(x).strip() for x in (data.get("ingredients") or []) if str(x).strip()]
    return name, ingredients

# ---- Análisis por opción ----

def analyze_cooking(name: str, ingredients: List[str]) -> Dict[str, Any]:
    prompt = (
        f"{SYSTEM_PROMPT}. Para el platillo: '{name}'. Ingredientes detectados: {ingredients}. "
        "Devuelve SOLO JSON con: "
        "{\"ingredientes\":[{\"nombre\":string,\"cantidad\":string}],"
        " \"tiempo_min\":number, \"nivel\":\"básico|intermedio|difícil\","
        " \"procedimiento\":[string,...]}. "
        "Notas: cantidades y unidades MÉTRICAS (g, ml, piezas). Cada elemento de 'procedimiento' debe ser un paso con explicación DETALLADA en un párrafo; NO agregues líneas divisorias."
    )
    return _call_gemini_json([{ "text": prompt }])


def analyze_nutrition(name: str, ingredients: List[str]) -> Dict[str, Any]:
    prompt = (
        f"{SYSTEM_PROMPT}. Calcula para UNA porción del platillo '{name}'. Ingredientes: {ingredients}. "
        "Devuelve SOLO JSON con: "
        "{\"kcal\":number, \"proteinas_g\":number, \"carbohidratos_g\":number, \"grasas_g\":number, "
        " \"tabla_ingredientes\":[{\"ingrediente\":string,\"kcal\":number,\"proteinas_g\":number,\"carbohidratos_g\":number,\"grasas_g\":number}], "
        " \"recomendaciones\":string}. "
        "Apegado a NOM-043 y Guías; si algo excede recomendaciones (azúcares, sodio, grasas saturadas), indícalo en 'recomendaciones'."
    )
    return _call_gemini_json([{ "text": prompt }])


def analyze_alternatives(name: str, ingredients: List[str], kcal_objetivo: float | None) -> Dict[str, Any]:
    target = kcal_objetivo or 0
    prompt = (
        f"{SYSTEM_PROMPT}. Con base en UNA porción estimada para '{name}', kcal={target} (si 0, estímalas tú). "
        "Propón platillos con *cantidad calórica similar* (±10%), dos vegetarianos y dos no vegetarianos. "
        "No necesitan compartir ingredientes. Devuelve SOLO JSON: "
        "{\"kcal_objetivo\":number, \"vegetarianos\":[{\"nombre\":string,\"descripcion\":string,\"kcal\":number}], "
        " \"no_vegetarianos\":[{\"nombre\":string,\"descripcion\":string,\"kcal\":number}]}."
    )
    return _call_gemini_json([{ "text": prompt }])


# =====================
# UI / Render
# =====================

def render_scan():
    # Estado
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = {"name": None, "ingredients": []}
    if "analysis_panel" not in st.session_state:
        st.session_state.analysis_panel = None
    if "_scan_cache" not in st.session_state:
        st.session_state._scan_cache = {"digest": None, "result": None}
    if "_analysis_cache" not in st.session_state:
        st.session_state._analysis_cache = {}  # key: f"{digest}:{panel}" -> data
    if "_last_digest" not in st.session_state:
        st.session_state._last_digest = None

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

            # Digest para cachear por imagen (evita gastar cuota si es la misma)
            digest = hashlib.sha256(image_bytes).hexdigest()
            st.session_state._last_digest = digest

            # Controles de análisis
            colA, colB = st.columns([1, 1])
            with colA:
                do_analyze = st.button("🔍 Analizar ahora", use_container_width=True)
            with colB:
                force = st.checkbox("Forzar re-analizar", value=False)

            if do_analyze:
                use_cache = (
                    st.session_state._scan_cache["digest"] == digest and
                    st.session_state._scan_cache["result"] is not None and
                    not force
                )
                if use_cache:
                    name, ingredients = st.session_state._scan_cache["result"]
                else:
                    with st.spinner("Identificando platillo con Gemini…"):
                        try:
                            name, ingredients = gemini_identify(image_bytes, mime or "image/jpeg")
                            st.session_state._scan_cache = {"digest": digest, "result": (name, ingredients)}
                        except Exception as e:
                            st.error(f"No fue posible analizar la imagen. {e}")
                            name, ingredients = None, []
                st.session_state.scan_result = {"name": name, "ingredients": ingredients}
                # Limpiar caché de análisis al cambiar imagen
                st.session_state._analysis_cache = {}
        else:
            st.session_state.scan_result = {"name": None, "ingredients": []}
            st.session_state._last_digest = None

    with col_right:
        st.subheader("Resultado")
        res = st.session_state.scan_result
        if res["name"]:
            st.markdown(f"### {res['name']}")
            if res["ingredients"]:
                for ing in res["ingredients"]:
                    st.markdown(f"- {ing}")
            else:
                st.caption("No se detectaron ingredientes con suficiente confianza.")
        else:
            st.info("Sube/toma una foto y presiona *Analizar ahora* para obtener el resultado sin gastar llamadas innecesarias.")

    st.divider()

    # Sección inferior: Analiza tu platillo
    st.markdown("## Analiza tu platillo")

    disabled = st.session_state.scan_result.get("name") is None
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🍳 Cómo cocinarlo", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Cómo cocinarlo"
    with col2:
        if st.button("📊 Información nutricional", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Información nutricional"
    with col3:
        if st.button("🔁 Alternativas similares", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Alternativas similares"
    with col4:
        if st.button("📝 Generar reporte", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Generar reporte"

    if st.session_state.analysis_panel:
        panel = st.session_state.analysis_panel
        with st.container(border=True):
            st.markdown(f"### {panel}")

            # Ejecutar análisis bajo demanda y cachear por (digest,panel)
            cache_key = f"{st.session_state._last_digest}:{panel}"
            data = st.session_state._analysis_cache.get(cache_key)

            if data is None and st.session_state._last_digest is not None:
                name = st.session_state.scan_result.get("name")
                ings = st.session_state.scan_result.get("ingredients", [])
                try:
                    if panel == "Cómo cocinarlo":
                        data = analyze_cooking(name, ings)
                    elif panel == "Información nutricional":
                        data = analyze_nutrition(name, ings)
                    elif panel == "Alternativas similares":
                        # Si ya calculamos kcal en el panel de nutrición, úsalo como objetivo
                        nut_key = f"{st.session_state._last_digest}:Información nutricional"
                        kcal_obj = None
                        if nut_key in st.session_state._analysis_cache:
                            kcal_obj = st.session_state._analysis_cache[nut_key].get("kcal")
                        data = analyze_alternatives(name, ings, kcal_obj)
                    else:
                        data = {"msg": "Próximamente"}
                    st.session_state._analysis_cache[cache_key] = data
                except Exception as e:
                    st.error(f"No fue posible completar el análisis: {e}")
                    data = None

            # Renderizado por opción
            if data:
                if panel == "Cómo cocinarlo":
                    # Ingredientes (tabla) + tiempo/nivel + procedimiento enumerado
                    st.subheader("Ingredientes")
                    tabla = data.get("ingredientes") or []
                    if isinstance(tabla, list) and tabla and isinstance(tabla[0], dict):
                        st.table(tabla)
                    else:
                        st.write("Datos insuficientes de ingredientes.")

                    colT, colN = st.columns(2)
                    with colT:
                        st.markdown(f"⏱ *Tiempo:* {data.get('tiempo_min', '—')} min")
                    with colN:
                        lvl = str(data.get("nivel") or "—").capitalize()
                        st.markdown(f"📊 *Nivel:* {lvl}")

                    st.subheader("Procedimiento")
                    pasos = data.get("procedimiento") or []
                    for idx, paso in enumerate(pasos, start=1):
                        st.markdown(f"{idx}.** {paso}")  # sin líneas divisorias

                elif panel == "Información nutricional":
                    colA, colB, colC, colD = st.columns(4)
                    colA.metric("🔥 KCal", f"{data.get('kcal', '—')}")
                    colB.metric("💪 Proteínas", f"{data.get('proteinas_g', '—')} g")
                    colC.metric("🍞 Carbohidratos", f"{data.get('carbohidratos_g', '—')} g")
                    colD.metric("🥑 Grasas", f"{data.get('grasas_g', '—')} g")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Tabla nutricional por ingrediente")
                        tabla = data.get("tabla_ingredientes") or []
                        if isinstance(tabla, list) and (tabla and isinstance(tabla[0], dict)):
                            st.table(tabla)
                        else:
                            st.write("Sin desglose por ingrediente.")
                    with c2:
                        st.subheader("Recomendaciones")
                        st.write(data.get("recomendaciones", "Sin recomendaciones."))

                elif panel == "Alternativas similares":
                    st.caption("Platillos con *cantidad calórica similar* (±10%) a la porción analizada.")
                    veg = data.get("vegetarianos") or []
                    non = data.get("no_vegetarianos") or []
                    items = (veg[:2] + non[:2])[:4]
                    # 4 secciones
                    c1, c2, c3, c4 = st.columns(4)
                    cols = [c1, c2, c3, c4]
                    for col, item in zip(cols, items):
                        with col:
                            st.markdown(f"{item.get('nombre','—')}")
                            if item.get('kcal') is not None:
                                st.caption(f"≈ {item['kcal']} kcal")
                            st.write(item.get('descripcion', ''))

            if st.button("Cerrar", key="close_panel"):
                st.session_state.analysis_panel = None
                st.rerun()

