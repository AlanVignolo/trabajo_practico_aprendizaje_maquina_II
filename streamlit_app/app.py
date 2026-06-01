import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Predicción de Precio de Autos",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Predicción de Precio de Autos")
st.markdown(
    "Completá las características del vehículo y presioná **Predecir** "
    "para obtener el precio estimado."
)

# Estado del modelo (barra lateral)
with st.sidebar:
    st.header("Estado del servicio")
    if st.button("Verificar conexión"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            data = resp.json()
            if data.get("model_loaded"):
                st.success("Modelo cargado ✓")
            else:
                st.warning("Conectado, pero el modelo aún no está disponible. Ejecutá el pipeline de entrenamiento.")
        except Exception:
            st.error(f"No se puede conectar con la API en {API_URL}")

    st.markdown("---")
    st.caption(f"API: `{API_URL}`")

# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------

with st.form("car_form"):

    st.subheader("Información general")
    col1, col2, col3 = st.columns(3)
    make = col1.text_input("Marca", value="Maruti")
    year = col2.number_input("Año", min_value=1990, max_value=2025, value=2019, step=1)
    location = col3.text_input("Ciudad / Ubicación", value="Mumbai")

    col1, col2, col3 = st.columns(3)
    fuel_type = col1.selectbox(
        "Combustible",
        ["Petrol", "Diesel", "CNG", "Electric", "LPG"],
    )
    transmission = col2.selectbox("Transmisión", ["Manual", "Automatic"])
    drivetrain_options = ["FWD", "RWD", "AWD", "4WD", "Ninguno"]
    drivetrain_sel = col3.selectbox("Tracción", drivetrain_options, index=0)

    col1, col2, col3 = st.columns(3)
    color = col1.text_input("Color", value="White")
    owner = col2.selectbox(
        "Propietario",
        ["First", "Second", "Third", "Fourth", "4 or More"],
    )
    seller_type = col3.selectbox(
        "Tipo de vendedor",
        ["Individual", "Dealer", "Trustmark Dealer"],
    )

    st.subheader("Motor")
    col1, col2, col3 = st.columns(3)
    engine_cc = col1.number_input(
        "Cilindrada (cc)", min_value=500.0, max_value=8000.0, value=1197.0, step=1.0
    )
    max_power_bhp = col2.number_input(
        "Potencia máx. (bhp)", min_value=20.0, max_value=1000.0, value=82.0, step=0.1
    )
    max_torque_nm = col3.number_input(
        "Torque máx. (Nm)", min_value=20.0, max_value=2000.0, value=113.0, step=0.1
    )

    st.subheader("Uso y dimensiones")
    col1, col2, col3 = st.columns(3)
    kilometer = col1.number_input(
        "Kilómetros recorridos", min_value=0.0, max_value=1_000_000.0, value=45_000.0, step=500.0
    )
    seating_capacity = col2.selectbox(
        "Asientos", [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], index=2
    )
    fuel_tank_capacity = col3.number_input(
        "Tanque (L) — opcional", min_value=0.0, max_value=200.0, value=37.0, step=0.5
    )

    col1, col2, col3 = st.columns(3)
    length = col1.number_input(
        "Longitud (mm)", min_value=2000.0, max_value=7000.0, value=3845.0, step=1.0
    )
    width = col2.number_input(
        "Ancho (mm)", min_value=1000.0, max_value=3000.0, value=1695.0, step=1.0
    )
    height = col3.number_input(
        "Altura (mm)", min_value=1000.0, max_value=3000.0, value=1520.0, step=1.0
    )

    submitted = st.form_submit_button("🔍 Predecir precio", use_container_width=True)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

if submitted:
    payload = {
        "make": make,
        "year": int(year),
        "fuel_type": fuel_type,
        "transmission": transmission,
        "location": location,
        "color": color,
        "owner": owner,
        "seller_type": seller_type,
        "drivetrain": drivetrain_sel if drivetrain_sel != "Ninguno" else None,
        "engine_cc": float(engine_cc),
        "max_power_bhp": float(max_power_bhp),
        "max_torque_nm": float(max_torque_nm),
        "kilometer": float(kilometer),
        "length": float(length),
        "width": float(width),
        "height": float(height),
        "seating_capacity": float(seating_capacity),
        "fuel_tank_capacity": float(fuel_tank_capacity) if fuel_tank_capacity > 0 else None,
    }

    with st.spinner("Consultando modelo..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                price = result["predicted_price_inr"]
                st.markdown("---")
                st.metric(
                    label="Precio estimado",
                    value=f"₹ {price:,.0f}",
                    help="Precio en rupias indias (INR). Dataset basado en el mercado indio.",
                )
                st.caption(f"Modelo utilizado: `{result['model_alias']}`")

            elif response.status_code == 503:
                st.error(
                    "El modelo no está disponible. "
                    "Ejecutá primero el DAG **training_pipeline** en Airflow (puerto 8080)."
                )
            elif response.status_code == 422:
                detail = response.json().get("detail", "Error de validación.")
                st.error(f"Error de validación: {detail}")
            else:
                st.error(f"Error del servidor ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"No se pudo conectar con la API en `{API_URL}`. "
                "Verificá que el servicio **fastapi** esté corriendo."
            )
        except requests.exceptions.Timeout:
            st.error("La API tardó demasiado en responder. Intentá de nuevo.")
        except Exception as exc:
            st.error(f"Error inesperado: {exc}")
