from __future__ import annotations
"""
Página: Platómetro (análisis por imagen vs. Plato del Bien Comer)

• Flujo: subir/tomar foto → botón "Calcular" → IA (Gemini) estima proporción por grupos
  alimenticios (1: Frutas y verduras; 2: Granos y cereales; 3: Leguminosas; 4: Origen animal;
  5: Aceites y grasas saludables). Luego se compara contra las recomendaciones del
  Plato del Bien Comer (México).
"""

import os
import io
import json
import re
import hashlib
from typing import Dict, Any, Tuple

import streamlit as st
from PIL import Image

# =====================
# MQTT (Streamlit -> ESP32)
# =====================
# pip install paho-mqtt
import paho.mqtt.client as mqtt
import time

MQTT_HOST = os.getenv("MQTT_HOST", "192.168.1.10")  # cambia por la IP de tu broker
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = "nutriapp/platometro/update"

def _ensure_mqtt():
    if "mqtt_client" not in st.session_state:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        # Si necesitas auth:
        # c.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASS"))
        try:
            c.connect(MQTT_HOST, MQTT_PORT, 60)
            c.loop_start()
            st.session_state.mqtt_client = c
        except Exception as e:
            st.warning(f"No se pudo conectar al broker MQTT: {e}")
            st.session_state.mqtt_client = None

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
# Lógica de evaluación (se mantiene por si la vuelves a usar)
# =====================

def _score_from_percentages(p: Dict[str, float]) -> Tuple[int, str]:
    """Calcula calificación 1-5 y devuelve SOLO desviación y mensaje (sin desglose)."""
    if not p:
        return 1, "No se pudo estimar la distribución."
    total_error = 0.0
    for key, tgt in TARGETS.items():
        val = float(_num(p.get(key, 0.0)))
        err = abs(val - tgt) / max(tgt, 1)
        total_error += err
    avg_err_pct = (total_error / len(TARGETS)) * 100
    if avg_err_pct <= 20:
        cal = 5
        msg = "Excelente alineación con el Plato del Bien Comer."
    elif avg_err_pct <= 40:
        cal = 4
        msg = "Buena composición con pequeñas desviaciones."
    elif avg_err_pct <= 60:
        cal = 3
        msg = "Aceptable, hay áreas de mejora en el balance del plato."
    elif avg_err_pct <= 80:
        cal = 2
        msg = "Desbalance notable; conviene ajustar porciones."
    else:
        cal = 1
        msg = "Distribución poco recomendable según el Plato del Bien Comer."
    explicacion = f"Desviación promedio: {avg_err_pct:.1f}%. {msg}"
    return cal, explicacion


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

                # ---- PUBLICAR A ESP32 (MQTT) ----
                _ensure_mqtt()
                try:
                    data = st.session_state.platometro_data or {}
                    porcentajes = (data.get("porcentajes") or {})
                    payload = {
                        "platillo": (data.get("platillo") or "Platillo"),
                        "porcentajes": {
                            "frutas_verduras": float(_num(porcentajes.get("frutas_verduras", 0))),
                            "granos_cereales": float(_num(porcentajes.get("granos_cereales", 0))),
                            "leguminosas": float(_num(porcentajes.get("leguminosas", 0))),
                            "origen_animal": float(_num(porcentajes.get("origen_animal", 0))),
                            "aceites_grasas_saludables": float(_num(porcentajes.get("aceites_grasas_saludables", 0))),
                        },
                        "kcal": (float(_num(st.session_state.platometro_kcal, 0.0)) if st.session_state.platometro_kcal is not None else None),
                    }
                    if st.session_state.mqtt_client:
                        st.session_state.mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                        st.success("Enviado a ESP32 (MQTT).")
                    else:
                        st.warning("Cliente MQTT no disponible; revisa la conexión al broker.")
                except Exception as e:
                    st.error(f"Error publicando MQTT: {e}")

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

            # Recomendaciones
            st.subheader("Recomendaciones")
            st.write(_recommendations(p))

    st.divider()
    st.info("Recordatorio: consume **6 a 8 vasos (≈2 litros) de agua simple al día**.")


if __name__ == "__main__":
    render_platometro()



