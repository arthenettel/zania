import streamlit as st
from streamlit_option_menu import option_menu

# --- Configuración general ---
st.set_page_config(page_title="Zania", page_icon="🥗", layout="wide")

# --- Estado de navegación ---
if "nav" not in st.session_state:
    st.session_state.nav = "Página principal"

SECCIONES = [
    "Página principal",
    "Escanear platillo",
    "Calculadora nutricional",
    "Crear receta",
    "Platómetro",
]

# --- Menú lateral con streamlit-option-menu ---
with st.sidebar:
    st.markdown("## 🥗 Zania")
    selected = option_menu(
        menu_title="",
        options=SECCIONES,
        icons=["house", "camera", "calculator", "egg-fried", "cpu"],
        default_index=0,
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
            <h1 style="margin-bottom:0.25rem; font-size:5rem;">Zania</h1>
            <p style="font-size:1.05rem; line-height:1.6; margin-top:0.25rem;">
            <strong>Tu apoyo inteligente de alimentación.</strong><br>
            Usando inteligencia artificial, te ayuda a reconocer platillos con solo una foto.
            Además, podrás calcular tu IMC, saber cuántas calorías necesitas al día, crear recetas y más.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("## Elige una opción")

        if st.button("📷 Escanear platillo", use_container_width=True):
            go_to("Escanear platillo")
        if st.button("🧮 Calculadora nutricional", use_container_width=True):
            go_to("Calculadora nutricional")
        if st.button("🍳 Crear receta", use_container_width=True):
            go_to("Crear receta")

        st.divider()

        if st.button("📟 Platómetro", use_container_width=True):
            go_to("Platómetro")

        st.caption(
            "Para usar el Platómetro es necesario conectarlo previamente al dispositivo externo."
        )


def render_placeholder(title: str, note: str = ""):
    st.markdown(f"## {title}")
    st.info(
        "Esta sección se implementará en los siguientes pasos. \n\n"
        "De momento, vuelve a la Página principal para navegar."
    )
    if note:
        st.caption(note)

# --- Router ---
if st.session_state.nav == "Página principal":
    render_home()
elif st.session_state.nav == "Escanear platillo":
    render_placeholder("Escanear platillo", "Aquí podrás subir una foto o usar la cámara para reconocer el platillo.")
elif st.session_state.nav == "Calculadora nutricional":
    render_placeholder("Calculadora nutricional", "Cálculo de IMC, TMB y calorías diarias recomendadas.")
elif st.session_state.nav == "Crear receta":
    render_placeholder("Crear receta", "Genera recetas personalizadas a partir de ingredientes y preferencias.")
elif st.session_state.nav == "Platómetro":
    render_placeholder("Platómetro", "Recuerda: requiere conexión previa al dispositivo externo para funcionar.")

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





