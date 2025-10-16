"""
• Las evaluaciones nutricionales y culinarias se hacen *para una porción* y
  se basan SOLO en: *Norma Oficial Mexicana NOM-043* y *Guías Alimentarias para la Población Mexicana*.
"""
from __future__ import annotations
import os
import io
import json
import re
import hashlib
from typing import Tuple, List, Dict, Any

import streamlit as st
from PIL import Image

# --- Plotly ---
import plotly.graph_objects as go

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

GEMINI_MODEL = "gemini-2.0-flash"
SYSTEM_PROMPT = (
    "Eres un asistente de visión experto en alimentos. Recibirás una foto de un platillo. "
    "Devuelve SOLO un JSON con esta forma exacta: "
    "{\"name\": \"<nombre del platillo>\", \"ingredients\": [\"ing1\", \"ing2\", \"ing3\"]}. "
    "Todos los valores de texto en el JSON deben estar estrictamente en español. "
    "Si no reconoces, usa name=\"Platillo no reconocido\" e ingredients=[]."
)

# =====================
# Constantes para las gráficas
# =====================

# Recomendaciones Plato del Bien Comer (porcentaje por porción)
TARGETS = {
    "frutas_verduras": 50,
    "granos_cereales": 22,
    "leguminosas": 15,
    "origen_animal": 8,
    "aceites_grasas_saludables": 5,
}

# Definición del orden fijo y colores de las categorías
ORDEN_CATEGORIAS = [
    'frutas_verduras',
    'origen_animal',
    'granos_cereales',
    'leguminosas',
    'aceites_grasas_saludables'
]

# Definición de categorías y sus colores fijos
CATEGORIAS = {
    'frutas_verduras': {
        'label': "🥗 Frutas y verduras",
        'color': "#33EB33",  # Verde claro
    },
    'origen_animal': {
        'label': "🍗 Origen animal",
        'color': "#C70000",  # Rojo
    },
    'granos_cereales': {
        'label': "🌾 Granos y cereales",
        'color': "#F8D92C",  # Amarillo/Dorado
    },
    'leguminosas': {
        'label': "🫘 Leguminosas",
        'color': "#DF650D",  # Marrón
    },
    'aceites_grasas_saludables': {
        'label': "🫒 Aceites y grasas",
        'color': "#B9910F",  # Naranja
    }
}

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
        "{\"name\": string, \"ingredients\": [string,...]}. "
        "Todos los valores de texto deben estar en español."
    )
    data = _call_gemini_json([
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
    ])
    name = (data.get("name") or "Platillo").strip()
    ingredients = [str(x).strip() for x in (data.get("ingredients") or []) if str(x).strip()]
    return name, ingredients


def gemini_get_food_groups(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    """Pide a Gemini un JSON con porcentajes por grupo alimenticio."""
    system = (
        "Actúas como nutriólogo y experto en visión de alimentos. Analiza la imagen del platillo y "
        "estima la composición aproximada por GRUPOS alimenticios del Plato del Bien Comer (México). "
        "Reporta porcentajes para UNA porción que sumen ~100% en un JSON con la siguiente estructura exacta: "
        "{\n \"porcentajes\": {\n \"frutas_verduras\": number,\n \"granos_cereales\": number,\n \"leguminosas\": number,\n \"origen_animal\": number,\n \"aceites_grasas_saludables\": number\n }\n}. "
        "No devuelvas texto fuera del JSON."
    )
    data = _call_gemini_json([
        {"text": system},
        {"inline_data": {"mime_type": mime, "data": image_bytes}},
    ])
    return data.get("porcentajes") or {}

# ---- Análisis por opción ----

def analyze_cooking(name: str, ingredients: List[str]) -> Dict[str, Any]:
    prompt = (
        f"Para el platillo: '{name}'. Ingredientes detectados: {ingredients}. "
        "Devuelve SOLO JSON con: "
        "{\"ingredientes\":[{\"nombre\":string,\"cantidad\":string}],"
        " \"tiempo_min\":number, \"nivel\":\"básico|intermedio|difícil\","
        " \"procedimiento\":[string,...]}. "
        "Notas: cantidades y unidades MÉTRICAS (g, ml, piezas). Cada elemento de 'procedimiento' debe ser un paso con explicación DETALLADA en un párrafo. "
        "Todos los valores de texto (nombres, nivel, procedimiento) deben estar estrictamente en español."
    )
    return _call_gemini_json([{"text": prompt}])


def analyze_nutrition(name: str, ingredients: List[str]) -> Dict[str, Any]:
    prompt = (
        f"Calcula para UNA porción del platillo '{name}'. Ingredientes: {ingredients}. "
        "Devuelve SOLO JSON con: "
        "{\"kcal\":number, \"proteinas_g\":number, \"carbohidratos_g\":number, \"grasas_g\":number, "
        " \"tabla_ingredientes\":[{\"ingrediente\":string,\"kcal\":number,\"proteinas_g\":number,\"carbohidratos_g\":number,\"grasas_g\":number}], "
        " \"recomendaciones\":string}. "
        "Apegado a NOM-043 y Guías Alimentarias para la Población Mexicana. "
        "Si algo excede recomendaciones (azúcares, sodio, grasas saturadas), indícalo en 'recomendaciones'. "
        "Todos los textos (ingredientes, recomendaciones) deben estar estrictamente en español."
    )
    return _call_gemini_json([{"text": prompt}])


def analyze_alternatives(name: str, ingredients: List[str], kcal_objetivo: float | None) -> Dict[str, Any]:
    target = kcal_objetivo or 0
    prompt = (
        f"Con base en UNA porción estimada para '{name}', kcal={target} (si 0, estímalas tú). "
        "Propón platillos con *cantidad calórica similar* (±10%), dos vegetarianos y dos no vegetarianos. "
        "No necesitan compartir ingredientes. Devuelve SOLO JSON: "
        "{\"kcal_objetivo\":number, \"vegetarianos\":[{\"nombre\":string,\"descripcion\":string,\"kcal\":number}], "
        " \"no_vegetarianos\":[{\"nombre\":string,\"descripcion\":string,\"kcal\":number}]}. "
        "Todos los textos (nombre, descripcion) deben estar estrictamente en español."
    )
    return _call_gemini_json([{"text": prompt}])


# =====================
# Helper PDF (lazy import)
# =====================

def _build_report_pdf(nombre, cook, nut, alts):
    import io, textwrap
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
    except Exception as e:
        raise RuntimeError("Falta instalar reportlab. Ejecuta: pip install reportlab") from e

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    x = 2*cm
    y = H - 2*cm

    def page_guard(min_y=2*cm):
        nonlocal y
        if y < min_y:
            c.showPage()
            y = H - 2*cm

    def heading(txt, size=14):
        nonlocal y
        page_guard()
        c.setFont("Helvetica-Bold", size)
        c.drawString(x, y, txt)
        y -= (size + 8)

    def para(txt, size=10, wrap=100, double_spaced=False):
        nonlocal y
        c.setFont("Helvetica", size)
        lines = textwrap.wrap(txt or "", wrap) or [""]
        for line in lines:
            page_guard()
            c.drawString(x, y, line)
            y -= 14
            if double_spaced:
                y -= 6  # espacio extra entre viñetas/pasos
        y -= 6

    # Portada
    heading("Reporte de platillo", 18)
    para(f"Nombre: {nombre or '—'}")

    # ---- Cómo cocinarlo ----
    if cook and isinstance(cook, dict):
        heading("Cómo cocinarlo", 14)
        para(f"Tiempo: {cook.get('tiempo_min','—')} min  |  Nivel: {cook.get('nivel','—')}")

        # Ingredientes en TABLA
        heading("Ingredientes", 12)
        ings = cook.get("ingredientes") or []
        data = [["Ingrediente", "Cantidad"]] + [[i.get("nombre",""), i.get("cantidad"," ")] for i in ings]
        table = Table(data, colWidths=[8*cm, 6*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
        ]))
        # Calcular altura y dibujar
        w, h = table.wrapOn(c, W - 4*cm, y)
        if y - h < 2*cm:
            c.showPage(); y = H - 2*cm
        table.drawOn(c, x, y - h)
        y = y - h - 10

        # Procedimiento con espacio entre pasos
        heading("Procedimiento", 12)
        pasos = cook.get("procedimiento") or []
        for i, p in enumerate(pasos, start=1):
            para(f"{i}. {p}", double_spaced=True)

    # ---- Información nutricional ----
    if nut and isinstance(nut, dict):
        heading("Información nutricional (por porción)", 14)
        macro = (
            f"KCal: {nut.get('kcal','—')}  |  "
            f"Proteínas: {nut.get('proteinas_g','—')} g  |  "
            f"Carbohidratos: {nut.get('carbohidratos_g','—')} g  |  "
            f"Grasas: {nut.get('grasas_g','—')} g"
        )
        para(macro)
        heading("Tabla por ingrediente", 12)
        filas = nut.get("tabla_ingredientes") or []
        # Imprime cada fila como viñeta doble-espaciada para mejorar legibilidad
        for r in filas:
            txt = (f"• {r.get('ingrediente','')}: {r.get('kcal','—')} kcal, "
                   f"P {r.get('proteinas_g','—')} g, C {r.get('carbohidratos_g','—')} g, G {r.get('grasas_g','—')} g")
            para(txt, double_spaced=True)
        heading("Recomendaciones", 12)
        para(nut.get("recomendaciones",""))

    # ---- Alternativas similares ----
    if alts and isinstance(alts, dict):
        heading("Alternativas con calorías similares (±10%)", 14)
        veg = alts.get("vegetarianos") or []
        non = alts.get("no_vegetarianos") or []
        if veg:
            heading("Vegetarianos", 12)
            for a in veg:
                para(f"• {a.get('nombre','')} (≈ {a.get('kcal','—')} kcal): {a.get('descripcion','')}", double_spaced=True)
        if non:
            heading("No vegetarianos", 12)
            for a in non:
                para(f"• {a.get('nombre','')} (≈ {a.get('kcal','—')} kcal): {a.get('descripcion','')}", double_spaced=True)

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

# =====================
# UI / Render
# =====================

def render_scan():
    # Estado
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = {"name": None, "ingredients": []}
    if "food_group_result" not in st.session_state:
        st.session_state.food_group_result = None
    if "analysis_panel" not in st.session_state:
        st.session_state.analysis_panel = None
    if "_scan_cache" not in st.session_state:
        st.session_state._scan_cache = {"digest": None, "result": None}
    if "_groups_cache" not in st.session_state:
        st.session_state._groups_cache = {"digest": None, "result": None}
    if "_analysis_cache" not in st.session_state:
        st.session_state._analysis_cache = {}  # key: f"{digest}:{panel}" -> data
    if "_last_digest" not in st.session_state:
        st.session_state._last_digest = None
    if "_last_image_bytes" not in st.session_state:
        st.session_state._last_image_bytes = None
    if "_last_mime" not in st.session_state:
        st.session_state._last_mime = None

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

        # Reutiliza la última imagen si este rerun no trajo una nueva
        if image_bytes is None and st.session_state._last_image_bytes is not None:
            image_bytes = st.session_state._last_image_bytes
            mime = st.session_state._last_mime

        if image_bytes:
            # Guardar imagen persistente
            st.session_state._last_image_bytes = image_bytes
            st.session_state._last_mime = mime

            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Vista previa", use_container_width=True)

            digest = hashlib.sha256(image_bytes).hexdigest()
            st.session_state._last_digest = digest

            if (
                (not st.session_state.scan_result.get("name"))
                and st.session_state._scan_cache.get("digest") == digest
                and st.session_state._scan_cache.get("result") is not None
            ):
                st.session_state.scan_result = {
                    "name": st.session_state._scan_cache["result"][0],
                    "ingredients": st.session_state._scan_cache["result"][1],
                }

            colA, colB = st.columns([1, 1])
            with colA:
                do_analyze = st.button("🔍 Analizar ahora", use_container_width=True)
            with colB:
                force = st.checkbox("Forzar re-analizar", value=False)

            if do_analyze:
                # --- ANÁLISIS 1: Identificar nombre e ingredientes
                use_cache_scan = (
                    st.session_state._scan_cache["digest"] == digest and
                    st.session_state._scan_cache["result"] is not None and
                    not force
                )
                if use_cache_scan:
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

                # --- ANÁLISIS 2: Obtener grupos para gráficas
                use_cache_groups = (
                    st.session_state._groups_cache["digest"] == digest and
                    st.session_state._groups_cache["result"] is not None and
                    not force
                )
                if use_cache_groups:
                    food_groups = st.session_state._groups_cache["result"]
                else:
                    with st.spinner("Estimando proporciones para la gráfica..."):
                        try:
                            food_groups = gemini_get_food_groups(image_bytes, mime or "image/jpeg")
                            st.session_state._groups_cache = {"digest": digest, "result": food_groups}
                        except Exception as e:
                            st.warning(f"No se pudieron estimar las proporciones: {e}")
                            food_groups = None
                st.session_state.food_group_result = food_groups
                st.session_state._analysis_cache = {} # Limpiar caché de análisis detallado
                st.rerun() # Forzar un rerun para mostrar los resultados de inmediato

        else:
            if st.session_state._last_image_bytes is None:
                st.session_state.scan_result = {"name": None, "ingredients": []}
                st.session_state.food_group_result = None
                st.session_state._last_digest = None

    with col_right:
        st.subheader("Resultado")
        res = st.session_state.scan_result
        if res and res.get("name"):
            st.markdown(f"### {res['name']}")
            if res.get("ingredients"):
                for ing in res["ingredients"]:
                    st.markdown(f"- {ing}")
            else:
                st.caption("No se detectaron ingredientes con suficiente confianza.")
        else:
            st.info("Sube/toma una foto y presiona *Analizar ahora* para obtener el resultado.")

    st.divider()

    # ===============================================
    # NUEVA SECCIÓN: GRÁFICAS DE GRUPOS ALIMENTICIOS
    # ===============================================
    food_groups = st.session_state.get("food_group_result")
    if food_groups and isinstance(food_groups, dict):
        st.markdown("## Proporción de grupos alimenticios")

        def _num(val, default=0.0) -> float:
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    m = re.search(r"[-+]?\d*\.?\d+", val)
                    if m:
                        return float(m.group(0))
            except Exception:
                pass
            return float(default)

        p = {k: _num(v) for k, v in food_groups.items()}
        
        etiquetas = [CATEGORIAS[k]['label'] for k in ORDEN_CATEGORIAS]
        colores = [CATEGORIAS[k]['color'] for k in ORDEN_CATEGORIAS]
        
        valores_est = [p.get(k, 0.0) for k in ORDEN_CATEGORIAS]
        valores_obj = [TARGETS[k] for k in ORDEN_CATEGORIAS]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Tu platillo")
            fig_left = go.Figure(
                data=[go.Pie(
                    labels=etiquetas,
                    values=valores_est,
                    hole=0.4,
                    textinfo="percent",
                    marker=dict(colors=colores),
                    sort=False,
                    direction='clockwise',
                    hoverinfo='label+percent'
                )]
            )
            fig_left.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(traceorder='normal'))
            st.plotly_chart(fig_left, use_container_width=True)
        with col2:
            st.subheader("✅ Recomendación oficial")
            fig_right = go.Figure(
                data=[go.Pie(
                    labels=etiquetas,
                    values=valores_obj,
                    hole=0.4,
                    textinfo="percent",
                    marker=dict(colors=colores),
                    sort=False,
                    direction='clockwise',
                    hoverinfo='label+percent'
                )]
            )
            fig_right.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(traceorder='normal'))
            st.plotly_chart(fig_right, use_container_width=True)
        
        st.divider()

    # ===============================================
    # FIN DE LA SECCIÓN DE GRÁFICAS
    # ===============================================

    # Sección inferior: Analiza tu platillo
    st.markdown("## Analiza tu platillo")

    disabled = st.session_state.scan_result.get("name") is None
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🍳 Cómo cocinarlo", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Cómo cocinarlo"
            st.rerun()
    with col2:
        if st.button("📊 Información nutricional", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Información nutricional"
            st.rerun()
    with col3:
        if st.button("🔁 Alternativas similares", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Alternativas similares"
            st.rerun()
    with col4:
        if st.button("📝 Generar reporte", use_container_width=True, disabled=disabled):
            st.session_state.analysis_panel = "Generar reporte"
            st.rerun()

    if st.session_state.analysis_panel:
        panel = st.session_state.analysis_panel
        with st.container(border=True):
            st.markdown(f"### {panel}")

            cache_key = f"{st.session_state._last_digest}:{panel}"
            data = st.session_state._analysis_cache.get(cache_key)
            
            has_name = bool(st.session_state.scan_result.get("name"))
            if data is None and has_name:
                name = st.session_state.scan_result.get("name")
                ings = st.session_state.scan_result.get("ingredients", [])
                try:
                    with st.spinner(f"Analizando '{panel}'..."):
                        if panel == "Cómo cocinarlo":
                            data = analyze_cooking(name, ings)
                        elif panel == "Información nutricional":
                            data = analyze_nutrition(name, ings)
                        elif panel == "Alternativas similares":
                            nut_key = f"{st.session_state._last_digest}:Información nutricional"
                            kcal_obj = None
                            if nut_key in st.session_state._analysis_cache:
                                nut_data = st.session_state._analysis_cache[nut_key]
                                if isinstance(nut_data, list) and nut_data:
                                    nut_data = nut_data[0]
                                if isinstance(nut_data, dict):
                                    kcal_obj = nut_data.get("kcal")
                            data = analyze_alternatives(name, ings, kcal_obj)
                        else:
                            data = {"msg": "Próximamente"}
                    st.session_state._analysis_cache[cache_key] = data
                except Exception as e:
                    st.error(f"No fue posible completar el análisis: {e}")
                    data = None

            if isinstance(data, list):
                data = data[0] if data else None
            
            # Renderizado por opción
            if data and isinstance(data, dict):
                if panel == "Cómo cocinarlo":
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
                        st.markdown(f"**{idx}.** {paso}")

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
                    
                    if veg:
                        st.subheader("Opciones Vegetarianas")
                        for item in veg:
                            st.markdown(f"**{item.get('nombre','—')}** (≈ {item.get('kcal','—')} kcal)")
                            st.write(item.get('descripcion', ''))
                    
                    if non:
                        st.subheader("Opciones No Vegetarianas")
                        for item in non:
                            st.markdown(f"**{item.get('nombre','—')}** (≈ {item.get('kcal','—')} kcal)")
                            st.write(item.get('descripcion', ''))

                elif panel == "Generar reporte":
                    st.caption("Genera un PDF con el **nombre del platillo** y los resultados de las otras secciones.")
                    
                    if st.button("📄 Generar", use_container_width=True):
                        with st.spinner("Creando reporte PDF..."):
                            digest = st.session_state._last_digest
                            name = st.session_state.scan_result.get("name")
                            ings = st.session_state.scan_result.get("ingredients", [])

                            def get_panel_data(panel_name, analysis_func, *args):
                                key = f"{digest}:{panel_name}"
                                panel_data = st.session_state._analysis_cache.get(key)
                                if panel_data is None and name:
                                    try:
                                        panel_data = analysis_func(*args)
                                        st.session_state._analysis_cache[key] = panel_data
                                    except Exception as e:
                                        st.warning(f"No se pudo obtener '{panel_name}': {e}")
                                        return None
                                if isinstance(panel_data, list) and panel_data:
                                    return panel_data[0]
                                return panel_data if isinstance(panel_data, dict) else None

                            cook = get_panel_data("Cómo cocinarlo", analyze_cooking, name, ings)
                            nut = get_panel_data("Información nutricional", analyze_nutrition, name, ings)
                            kcal_obj = nut.get("kcal") if nut else None
                            alts = get_panel_data("Alternativas similares", analyze_alternatives, name, ings, kcal_obj)

                            try:
                                pdf_bytes = _build_report_pdf(name, cook, nut, alts)
                                st.download_button(
                                    "⬇️ Descargar reporte (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"reporte_{(name or 'platillo').lower().replace(' ','_')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error al generar el PDF: {e}")

            if st.button("Cerrar", key="close_panel"):
                st.session_state.analysis_panel = None
                st.rerun()

if __name__ == "__main__":
    render_scan()