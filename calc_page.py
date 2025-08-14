"""
Página: Calculadora nutricional

• No dibuja el sidebar. Asume que app.py lo crea con streamlit-option-menu.
• Layout: 2 columnas (izq: formulario; der: resultados).
• Fórmula de gasto basal: Mifflin–St Jeor. IMC con clasificación OMS.

Integración en app.py:

    from pages.calc_page import render_calc
    # ...
    elif st.session_state.nav == "Calculadora nutricional":
        render_calc()

Requisitos: streamlit>=1.33
"""
import streamlit as st

ACTIVITY_OPTIONS = [
    {"key": "Sedentario", "label": "Sedentario: Poco o nada de ejercicio", "factor": 1.2},
    {"key": "Ligeramente Activo", "label": "Ligeramente Activo: Ejercicio 2 a 3 días por semana", "factor": 1.375},
    {"key": "Moderadamente Activo", "label": "Moderadamente Activo: Ejercicio 4 a 5 días por semana", "factor": 1.55},
    {"key": "Muy Activo", "label": "Muy Activo: Ejercicio 6 a 7 días por semana", "factor": 1.725},
    {"key": "Atleta Profesional", "label": "Atleta Profesional: Ejercicio intenso 6 a 7 días por semana", "factor": 1.9},
]


def _bmr_mifflin(sexo: str, edad: int, peso_kg: float, altura_cm: float) -> float:
    if sexo == "Hombre":
        return 10 * peso_kg + 6.25 * altura_cm - 5 * edad + 5
    else:
        return 10 * peso_kg + 6.25 * altura_cm - 5 * edad - 161


def _bmi(peso_kg: float, altura_cm: float) -> float:
    return peso_kg / ((altura_cm / 100) ** 2)


def _bmi_class(bmi: float) -> str:
    if bmi < 18.5:
        return "Bajo peso"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Sobrepeso"
    else:
        return "Obesidad"


def render_calc():
    if "calc_result" not in st.session_state:
        st.session_state.calc_result = None

    st.markdown("# Calculadora nutricional")

    left, gap, right = st.columns([1, 0.1, 1])

    with left:
        st.subheader("Rellena los datos")
        sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        edad = st.number_input("Edad (años)", min_value=10, max_value=100, value=25)
        altura = st.number_input("Altura (cm)", min_value=100, max_value=250, value=170)
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)

        opcion_etiquetas = [o["label"] for o in ACTIVITY_OPTIONS]
        etiqueta_sel = st.selectbox("Nivel de actividad física", options=opcion_etiquetas)
        factor = next(o["factor"] for o in ACTIVITY_OPTIONS if o["label"] == etiqueta_sel)

        if st.button("Calcular", use_container_width=True):
            bmr = _bmr_mifflin(sexo, edad, peso, altura)
            tdee = round(bmr * factor)
            bmi = round(_bmi(peso, altura), 2)
            bmi_c = _bmi_class(bmi)
            st.session_state.calc_result = {"imc": bmi, "imc_estado": bmi_c, "kcal_dia": tdee}

    with right:
        res = st.session_state.calc_result
        if res is None:
            st.info("Aquí verás tu **IMC** (es una medida que relaciona el peso y la estatura de una persona para evaluar su estado nutricional y determinar si se encuentra en un rango de peso saludable) y las **calorías recomendadas por día** después de calcular.")
        else:
            st.subheader("Resultados")
            # Sección IMC
            st.markdown("### IMC")
            st.markdown(f"**{res['imc']}** — {res['imc_estado']}")
            st.divider()
            # Sección Calorías
            st.markdown("### Calorías recomendadas al día")
            st.markdown(f"**{res['kcal_dia']} kcal**")
