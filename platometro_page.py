"""
Página: Platómetro (análisis por imagen vs. Plato del Bien Comer)

• Flujo: subir/tomar foto → botón "Calcular" → IA (Gemini) estima proporción por grupos
  alimenticios (1: Frutas y verduras; 2: Granos y cereales; 3: Leguminosas; 4: Origen animal;
  5: Aceites y grasas saludables). Luego se compara contra las recomendaciones del
  Plato del Bien Comer (México).

Cambio solicitado: mostrar **dos gráficas lado a lado en una sección aparte**
**debajo** del recordatorio de agua. (No se muestran en la sección de Resultados.)
"""
from __future__ import annotations
import os
import io
import json
import re
import hashlib
from typing import Dict, Any, Tuple

import streamlit as st
from PIL import Image

# --- Plotly ---
import plotly.express as px
import plotly.graph_objects as go

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

GEMINI_MODEL = "gemini-1.5-flash"

# Recomendaciones Plato del Bien Comer (porcentaje por porción)
TARGETS = {
    "frutas_verduras": 50,
    "granos_cereales": 22,
    "leguminosas": 15,
    "origen_animal": 8,
    "aceites_grasas_saludables": 5,
}

# =====================
# Helpers comunes
# =====================

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t


def _ensure_gemini_ready() -> Tuple[bool, str]:
    if genai is None:
        return False, "Falta instalar 'google-generativeai' (añade a requirements.txt)."
    api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None)
    if not api_key:
        return False, "No se encontró GOOGLE_API_KEY en .env ni en st.secrets."
    try:
        genai.configure(api_key=api_key)
        return True, ""
    except Exception as e:
        return False, f"Error al configurar Gemini: {e}"


def _num(val, default=0.0) -> float:
    """Convierte val a float de forma robusta. Acepta int/float o strings como '50', '50.0', '50 %'."""
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


# =====================
# IA: detección de grupos alimenticios y calorías
# =====================

def _call_gemini_groups(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    """Pide a Gemini un JSON con porcentajes por grupo alimenticio."""
    ok, msg = _ensure_gemini_ready()
    if not ok:
        raise RuntimeError(msg)

    system = (
        "Actúas como nutriólogo y experto en visión de alimentos. Analiza la imagen del platillo y "
        "estima la composición aproximada por GRUPOS alimenticios del Plato del Bien Comer (México). "
        "Reporta porcentajes para UNA porción que sumen ~100% en un JSON con la siguiente estructura exacta: "
        "{\n  \"platillo\": string,\n  \"porcentajes\": {\n    \"frutas_verduras\": number,\n    \"granos_cereales\": number,\n    \"leguminosas\": number,\n    \"origen_animal\": number,\n    \"aceites_grasas_saludables\": number\n  },\n  \"notas\": string\n}. "
        "No devuelvas texto fuera del JSON."
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content([
        {"text": system},
        {"inline_data": {"mime_type": mime, "data": image_bytes}},
    ])
    txt = resp.text or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(_strip_code_fences(txt))


def _call_gemini_kcal(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    """Pide a Gemini las **calorías por porción** del platillo visto en la imagen.
    Estructura: { "kcal": number, "notas": string }
    """
    ok, msg = _ensure_gemini_ready()
    if not ok:
        raise RuntimeError(msg)

    prompt = (
        "Eres nutriólogo con visión de alimentos. Estima las **kilocalorías por porción** del platillo "
        "en la imagen considerando porciones típicas en México. Devuelve SOLO JSON con la forma: "
        "{\"kcal\": number, \"notas\": string}. No agregues texto fuera del JSON."
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content([
        {"text": prompt},
        {"inline_data": {"mime_type": mime, "data": image_bytes}},
    ])
    txt = resp.text or "{}"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(_strip_code_fences(txt))


# =====================
# Recomendaciones
# =====================

def _recommendations(p: Dict[str, float]) -> str:
    recs = []
    for key, tgt in TARGETS.items():
        val = float(_num(p.get(key, 0.0)))
        name = key.replace("_", " ")
        if val < tgt * 0.8:
            recs.append(f"Aumenta {name} hasta acercarte al {tgt}% recomendado.")
        elif val > tgt * 1.2:
            recs.append(f"Reduce {name} para aproximarte al {tgt}% recomendado.")
    if not recs:
        recs.append("La distribución es adecuada; mantiene el balance propuesto por el Plato del Bien Comer.")
    recs.append("Tip NOM-043: prefiere agua simple, porciones adecuadas y alimentos naturales.")
    return "\n".join(recs)


# =====================
# UI / Render
# =====================

def render_platometro():
    # Estados
    if "platometro_cache" not in st.session_state:
        st.session_state.platometro_cache = {"digest": None, "result": None}
    if "platometro_data" not in st.session_state:
        st.session_state.platometro_data = None  # último resultado IA
    # caché de kcal
    if "platometro_kcal_cache" not in st.session_state:
        st.session_state.platometro_kcal_cache = {"digest": None, "kcal": None, "notas": None}
    if "platometro_kcal" not in st.session_state:
        st.session_state.platometro_kcal = None
        st.session_state.platometro_kcal_notas = None

    st.markdown("# Platómetro")
    st.caption(
        "Analiza la proporción de **grupos alimenticios** de tu platillo y compárala con el **Plato del Bien Comer**. "
        "Los resultados son estimaciones orientativas basadas en visión por computadora."
    )

    left, gap, right = st.columns([1, 0.08, 1])

    with left:
        st.subheader("Sube o toma una foto")
        metodo = st.radio("Selecciona el método de captura", ["Subir imagen", "Tomar foto"], index=0)

        image_bytes = None
        mime = None
        if metodo == "Subir imagen":
            up = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"], key="plato_uploader")
            if up is not None:
                image_bytes = up.getvalue()
                mime = "image/png" if up.name.lower().endswith(".png") else "image/jpeg"
        else:
            cam = st.camera_input("Tomar foto", key="plato_camera")
            if cam is not None:
                image_bytes = cam.getvalue()
                mime = "image/jpeg"

        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Vista previa", use_container_width=True)

            colA, colB = st.columns([1, 1])
            with colA:
                calcular = st.button("Calcular", use_container_width=True)
            with colB:
                force = st.checkbox("Forzar recalcular", value=False)

            if calcular:
                digest = hashlib.sha256(image_bytes).hexdigest()

                # ---- porcentajes por grupo (con caché) ----
                use_cache = (
                    st.session_state.platometro_cache["digest"] == digest
                    and st.session_state.platometro_cache["result"] is not None
                    and not force
                )
                if use_cache:
                    data = st.session_state.platometro_cache["result"]
                else:
                    with st.spinner("Estimando proporciones con Gemini…"):
                        try:
                            data = _call_gemini_groups(image_bytes, mime or "image/jpeg")
                        except Exception as e:
                            st.error(f"No fue posible analizar la imagen: {e}")
                            data = None
                    st.session_state.platometro_cache = {"digest": digest, "result": data}
                st.session_state.platometro_data = data

                # ---- calorías por porción (con caché) ----
                kcal_cache_ok = (
                    st.session_state.platometro_kcal_cache["digest"] == digest
                    and st.session_state.platometro_kcal_cache["kcal"] is not None
                    and not force
                )
                if kcal_cache_ok:
                    st.session_state.platometro_kcal = st.session_state.platometro_kcal_cache["kcal"]
                    st.session_state.platometro_kcal_notas = st.session_state.platometro_kcal_cache["notas"]
                else:
                    with st.spinner("Estimando calorías por porción…"):
                        try:
                            kcal_data = _call_gemini_kcal(image_bytes, mime or "image/jpeg")
                        except Exception as e:
                            st.warning(f"No se pudieron estimar calorías: {e}")
                            kcal_data = {"kcal": None, "notas": None}
                    st.session_state.platometro_kcal_cache = {
                        "digest": digest,
                        "kcal": kcal_data.get("kcal"),
                        "notas": kcal_data.get("notas"),
                    }
                    st.session_state.platometro_kcal = kcal_data.get("kcal")
                    st.session_state.platometro_kcal_notas = kcal_data.get("notas")
        else:
            st.session_state.platometro_data = None
            st.session_state.platometro_kcal = None
            st.session_state.platometro_kcal_notas = None

    with right:
        st.subheader("Resultados")
        data = st.session_state.platometro_data
        if not data:
            st.info("Sube/toma una foto y pulsa **Calcular** para ver los resultados.")
        else:
            nombre = data.get("platillo") or "Platillo"
            p = data.get("porcentajes") or {}
            # normaliza a floats seguros
            p = {k: _num(v) for k, v in p.items()}

            st.markdown(f"### {nombre}")

            # métricas por grupo
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🥗 Frutas/Verduras", f"{p.get('frutas_verduras', 0):.0f}%", f"obj {TARGETS['frutas_verduras']}%")
            c2.metric("🌾 Granos/Cereales", f"{p.get('granos_cereales', 0):.0f}%", f"obj {TARGETS['granos_cereales']}%")
            c3.metric("🫘 Leguminosas", f"{p.get('leguminosas', 0):.0f}%", f"obj {TARGETS['leguminosas']}%")
            c4.metric("🍗 Origen animal", f"{p.get('origen_animal', 0):.0f}%", f"obj {TARGETS['origen_animal']}%")
            c5.metric("🫒 Aceites/Grasas", f"{p.get('aceites_grasas_saludables', 0):.0f}%", f"obj {TARGETS['aceites_grasas_saludables']}%")

            # -------- Calorías por porción --------
            kcal_val = st.session_state.platometro_kcal
            kcal_notes = st.session_state.platometro_kcal_notas
            if kcal_val is not None:
                st.subheader("🔥 Calorías estimadas (por porción)")
                st.metric("Calorías", f"{_num(kcal_val, 0):.0f} kcal")
                if kcal_notes:
                    st.caption(kcal_notes)
            else:
                st.warning("No se pudieron estimar las calorías para esta imagen.")

            # Recomendaciones (se mantienen)
            st.subheader("Recomendaciones")
            st.write(_recommendations(p))

    st.divider()
    st.info("Recordatorio: consume **6 a 8 vasos (≈2 litros) de agua simple al día**.")

    # =====================
    # Sección de gráficas (abajo del recordatorio)
    # =====================
    data = st.session_state.platometro_data
    if data:
        p = {k: _num(v) for k, v in (data.get("porcentajes") or {}).items()}
        etiquetas = [
            "🥗 Frutas y verduras",
            "🌾 Granos y cereales",
            "🫘 Leguminosas",
            "🍗 Origen animal",
            "🫒 Aceites/grasas saludables",
        ]

        valores_est = [
            p.get("frutas_verduras", 0.0),
            p.get("granos_cereales", 0.0),
            p.get("leguminosas", 0.0),
            p.get("origen_animal", 0.0),
            p.get("aceites_grasas_saludables", 0.0),
        ]

        valores_obj = [
            TARGETS["frutas_verduras"],
            TARGETS["granos_cereales"],
            TARGETS["leguminosas"],
            TARGETS["origen_animal"],
            TARGETS["aceites_grasas_saludables"],
        ]

        # st.markdown("## Comparativa de distribución por grupos alimenticios")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Distribución estimada")
            fig_px = px.pie(values=valores_est, names=etiquetas, hole=0.4)
            fig_px.update_traces(textinfo="percent+label")
            fig_px.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_px, use_container_width=True)

        with col2:
            st.subheader("✅ Recomendación oficial")
            fig_go = go.Figure(data=[go.Pie(labels=etiquetas, values=valores_obj, hole=0.4, textinfo="percent+label")])
            fig_go.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_go, use_container_width=True)
