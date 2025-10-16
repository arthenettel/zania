import streamlit as st
from streamlit_option_menu import option_menu

# --- Configuración general ---
st.set_page_config(page_title="ZanIA", page_icon="🥗", layout="wide")

# --- Estado de navegación ---
if "nav" not in st.session_state:
    st.session_state.nav = "Página principal"

SECCIONES = [
    "Página principal",
    "Escanear platillo",
    "Calculadora nutricional",
    "Crear receta",
    "Platosaurio",
]

# --- Menú lateral con streamlit-option-menu ---
with st.sidebar:
    st.markdown("## 🥗 ZanIA")
    selected = option_menu(
        menu_title="",
        options=SECCIONES,
        icons=["house", "camera", "calculator", "egg-fried", "cpu"],
        default_index=SECCIONES.index(st.session_state.nav),
    )
    st.session_state.nav = selected

# --- Helper navegación ---
def go_to(section: str):
    if section in SECCIONES:
        st.session_state.nav = section
        st.rerun()

# --- Página principal ---
def render_home():
    left, gap, right = st.columns([1, 0.11, 1])

    with left:
        st.markdown(
            """
            <h1 style="margin-bottom:0.25rem; font-size:5rem;">ZanIA</h1>
            <p style="font-size:1.05rem; line-height:1.6; margin-top:0.25rem;">
            <strong>Tu apoyo inteligente de alimentación.</strong><br>
            Usando inteligencia artificial, te ayuda a reconocer platillos con solo una foto.
            Además, podrás calcular tu IMC, saber cuántas calorías necesitas al día, crear recetas y más.
            </p>
            """,
            unsafe_allow_html=True,
        )
        
        # Video de demostración
        st.markdown("### ✨ Comer sano ¡es súper divertido!")
        st.video("https://www.youtube.com/watch?v=amsyeMtqbAg")

    with right:
        st.markdown("## Elige una opción")

        if st.button("📷 Escanear platillo", use_container_width=True):
            go_to("Escanear platillo")
        if st.button("🧮 Calculadora nutricional", use_container_width=True):
            go_to("Calculadora nutricional")
        if st.button("🍳 Crear receta", use_container_width=True):
            go_to("Crear receta")

        st.divider()

        if st.button("🦖 Platosaurio", use_container_width=True):
            go_to("Platosaurio")

        st.caption(
            "Para usar el Platosaurio es necesario conectarlo previamente al dispositivo externo."
        )


def render_placeholder(title: str, note: str = ""):
    st.markdown(f"## {title}")
    st.info(
        "Esta sección se implementará en los siguientes pasos. \n\n"
        "De momento, vuelve a la Página principal para navegar."
    )
    if note:
        st.caption(note)

from scan_page import render_scan
from calc_page import render_calc
from create_page import render_create
from platometro_page import render_platometro
# --- Router ---
if st.session_state.nav == "Página principal":
    render_home()
elif st.session_state.nav == "Escanear platillo":
    render_scan()
elif st.session_state.nav == "Calculadora nutricional":
    render_calc()
elif st.session_state.nav == "Crear receta":
    render_create()
elif st.session_state.nav == "Platosaurio":
    render_platometro()

# --- Estilos globales ---
st.markdown(
    """
    <style>
    div.stButton {margin-bottom: 0.5rem;}
    .stButton > button {border-radius: 12px; padding: 0.9rem 1rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Texto de nota al final de la página
st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:gray; margin-top:30px;">
        Esta información es solo orientativa. Acuda siempre a un especialista antes de tomar decisiones sobre su salud.
    </div>
    """,
    unsafe_allow_html=True
)




