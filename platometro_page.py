"""
Página: Platómetro
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
# Session state seguro
# =====================

def _ensure_state():
    ss = st.session_state
    if "platometro_cache" not in ss:
        ss.platometro_cache = {"digest": None, "result": None}
    if "platometro_data" not in ss:
        ss.platometro_data = None  # último resultado IA
    if "platometro_kcal_cache" not in ss:
        ss.platometro_kcal_cache = {"digest": None, "kcal": None, "notas": None}
    if "platometro_kcal" not in ss:
        ss.platometro_kcal = None
    if "platometro_kcal_notas" not in ss:
        ss.platometro_kcal_notas = None

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

def _recommendations_list(p: Dict[str, float]) -> list[str]:
    """Devuelve recomendaciones simples basadas en NOM-043 (lenguaje claro)."""
    recs = []
    for key, tgt in TARGETS.items():
        val = float(_num(p.get(key, 0.0)))
        nombre = {
            "frutas_verduras": "Frutas y verduras",
            "granos_cereales": "Granos y cereales",
            "leguminosas": "Leguminosas",
            "origen_animal": "Alimentos de origen animal",
            "aceites_grasas_saludables": "Aceites y grasas saludables",
        }[key]
        if val < tgt * 0.8:
            recs.append(f"{nombre}: súbele a este grupo.")
        elif val > tgt * 1.2:
            recs.append(f"{nombre}: bájale para equilibrar con los demás.")
    if not recs:
        recs.append("¡Buen balance! Mantén porciones variadas y suficientes.")
    # Mensajes base NOM-043 en lenguaje claro
    recs.extend([
        "Toma agua simple durante el día (6–8 vasos).",
        "Prefiere alimentos naturales y evita el exceso de azúcar y sal.",
        "Mantén horarios regulares de comida y porciones adecuadas a tu apetito.",
    ])
    return recs

# =====================
# UI / Render (UNA SOLA COLUMNA, ORDEN NUEVO)
# =====================

def render_platometro():
    _ensure_state()

    # 1) Título + descripción + subir/tomar foto
    st.markdown("# Platosaurus del bien comer 🦖🍽️")
    st.caption(
        "Analiza la proporción de **grupos alimenticios** de tu platillo y compárala con el **Plato del Bien Comer**. "
        "Los resultados son estimaciones orientativas basadas en visión por computadora."
    )

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
        # NO mostrar vista previa cuando se usa cámara
        if metodo == "Subir imagen":
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Vista previa", use_container_width=True)

        colA, colB = st.columns([1, 1])
        with colA:
            calcular = st.button("⚙️ Calcular", use_container_width=True)
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

    st.divider()

    # 2) Recordatorio de agua
    st.info("Recordatorio: consume **6 a 8 vasos (≈2 litros) de agua simple al día**.")

    # 3) Sección de gráficas
    data = st.session_state.platometro_data
    if data:
        # st.markdown("## Gráficas")
        p = {k: _num(v) for k, v in (data.get("porcentajes") or {}).items()}
        etiquetas = [
            "🥗 Frutas y verduras",
            "🌾 Granos y cereales",
            "🫘 Leguminosas",
            "🍗 Origen animal",
            "🫒 Aceites y grasas saludables",
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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Tu platillo")
            fig_px = px.pie(values=valores_est, names=etiquetas, hole=0.4)
            fig_px.update_traces(textinfo="percent")
            fig_px.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_px, use_container_width=True)
        with col2:
            st.subheader("✅ Recomendación oficial")
            fig_go = go.Figure(data=[go.Pie(labels=etiquetas, values=valores_obj, hole=0.4, textinfo="percent")])
            fig_go.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_go, use_container_width=True)

    st.divider()

    # 4) Enviar a pantalla ESP32 (Web Serial)
    st.subheader("📲 Enviar resultados a la pantalla")
    st.caption("Conecta la pantalla a tu computadora para poder enviar los resultados al Platómetro. Usa Chrome/Edge.")

    data = st.session_state.get("platometro_data")
    payload_mcu = None
    if data:
        p = {k: _num(v) for k, v in (data.get("porcentajes") or {}).items()}
        kcal_val = _num(st.session_state.get("platometro_kcal"), 0)
        payload_mcu = {
            "kcal": int(kcal_val),
            "porcentajes": {
                "frutas_verduras": int(p.get("frutas_verduras", 0)),
                "granos": int(p.get("granos_cereales", 0)),
                "leguminosas": int(p.get("leguminosas", 0)),
                "origen_animal": int(p.get("origen_animal", 0)),
                "grasas": int(p.get("aceites_grasas_saludables", 0)),
            },
        }

    import streamlit.components.v1 as components
    html = f"""
<div style='display:flex;gap:12px;margin:10px 0 14px'>
  <button id='btnConnect' class='st-btn'>Conectar puerto</button>
  <button id='btnSend' class='st-btn' disabled>Ver en pantalla</button>
</div>

<!-- Estado -->
<div id=\"status\" style=\"margin-top:10px;display:none;\" class=\"st-status\"></div>

<style>
  .st-btn {{
    font-family: \"Source Sans Pro\", sans-serif;
    font-size: 1rem;
    padding: 0.6rem 1.4rem;
    border-radius: 0.6rem;
    border: 1px solid rgba(255,255,255,0.25);
    background: #111827;
    color: #f3f4f6;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }}
  .st-btn:hover:enabled {{
    background: #1f2937;
  }}
  .st-btn[disabled] {{
    opacity: 0.5;
    cursor: not-allowed;
  }}
  .st-status {{
    font-family: \"Source Sans Pro\", sans-serif;
    font-size: 1rem;
    padding: 0.6rem 1rem;
    border-radius: 0.6rem;
    border: 1px solid rgba(16,185,129,0.35);
    background: rgba(16,185,129,0.1);
    color: #10b981;
  }}
  .st-status.err {{
    border-color: rgba(239,68,68,0.35);
    background: rgba(239,68,68,0.1);
    color: #ef4444;
  }}
</style>

<script>
  let port, writer;
  const STATUS = document.getElementById('status');

  function setStatus(msg, cls) {{
    STATUS.className = 'st-status ' + (cls || '');
    STATUS.textContent = msg || '';
    STATUS.style.display = msg ? 'block' : 'none';
  }}

  async function connect() {{
    try {{
      if (!('serial' in navigator)) {{
        alert('Web Serial no disponible. Usa Chrome o Edge.');
        return;
      }}
      port = await navigator.serial.requestPort();
      await port.open({{ baudRate: 115200 }});
      writer = port.writable.getWriter();
      document.getElementById('btnSend').disabled = false;
      setStatus('Puerto abierto a 115200', '');
    }} catch (e) {{
      setStatus('Error al abrir: ' + e.message, 'err');
    }}
  }}

  async function send() {{
    try {{
      const payload = {json.dumps(payload_mcu)};
      const txt = JSON.stringify(payload) + "\\n";
      await writer.write(new TextEncoder().encode(txt));
      setStatus('Enviado correctamente', '');
    }} catch (e) {{
      setStatus('Error al escribir: ' + e.message, 'err');
    }}
  }}

  document.getElementById('btnConnect').addEventListener('click', connect);
  document.getElementById('btnSend').addEventListener('click', send);
</script>
"""
    components.html(html, height=160)

    st.divider()

    # 5) Sección de resultados finales (nombre, porcentajes, calorías, recomendaciones)
    st.subheader("Resultados")
    data = st.session_state.platometro_data
    if not data:
        st.info("Sube/toma una foto y pulsa **Calcular** para ver los resultados.")
        return

    nombre = data.get("platillo") or "Platillo"
    p = {k: _num(v) for k, v in (data.get("porcentajes") or {}).items()}  # normaliza a floats

    st.markdown(f"### {nombre}")

    # Porcentajes en formato fácil de entender
    # st.markdown("**Porcentajes por grupo (aprox.):**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🥗 Frutas/Verduras", f"{p.get('frutas_verduras', 0):.0f}%", f"obj {TARGETS['frutas_verduras']}%")
    c2.metric("🌾 Granos/Cereales", f"{p.get('granos_cereales', 0):.0f}%", f"obj {TARGETS['granos_cereales']}%")
    c3.metric("🫘 Leguminosas", f"{p.get('leguminosas', 0):.0f}%", f"obj {TARGETS['leguminosas']}%")
    c4.metric("🍗 Origen animal", f"{p.get('origen_animal', 0):.0f}%", f"obj {TARGETS['origen_animal']}%")
    c5.metric("🫒 Aceites/Grasas", f"{p.get('aceites_grasas_saludables', 0):.0f}%", f"obj {TARGETS['aceites_grasas_saludables']}%")

    # Calorías
    kcal_val = st.session_state.platometro_kcal
    kcal_notes = st.session_state.platometro_kcal_notas
    if kcal_val is not None:
        st.subheader("🔥 Calorías estimadas (por porción)")
        st.metric("Calorías", f"{_num(kcal_val, 0):.0f} kcal")
        if kcal_notes:
            st.caption(kcal_notes)
    else:
        st.warning("No se pudieron estimar las calorías para esta imagen.")

    # Recomendaciones en viñetas (lenguaje claro, basado en NOM-043)
    st.subheader("Recomendaciones")
    recs = _recommendations_list(p)
    st.markdown("\n".join([f"- {r}" for r in recs]))


if __name__ == "__main__":
    render_platometro()
