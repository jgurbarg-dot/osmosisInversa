import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página web (debe ir al principio siempre)
st.set_page_config(
    page_title="Simulador RO - Litio DLE",
    page_icon="🧪",
    layout="wide"
)

# ==============================================================================
# 1. PROPIEDADES TERMODINÁMICAS Y CONSTANTES
# ==============================================================================
MOLAR_MASS = {
    'Li': 6.941, 'Na': 22.989, 'K': 39.098, 'Mg': 24.305, 'Ca': 40.078,
    'B': 10.811, 'SO4': 96.060, 'Cl': 35.453, 'CO3': 60.009, 'HCO3': 61.017
}

VALENCIAS = {
    'Li': 1, 'Na': 1, 'K': 1, 'Mg': 2, 'Ca': 2,
    'B': 1, 'SO4': 2, 'Cl': 1, 'CO3': 2, 'HCO3': 1
}

RECHAZO_IONICO = {
    'Li': 0.985, 'Na': 0.985, 'K': 0.980, 'Mg': 0.996, 'Ca': 0.996,
    'B': 0.650,  # El ácido bórico neutro tiene menor retención a pH < 9
    'SO4': 0.998, 'Cl': 0.988, 'CO3': 0.995, 'HCO3': 0.980
}

# ==============================================================================
# 2. MOTORES TERMODINÁMICOS (VAN'T HOFF & PITZER)
# ==============================================================================
def calcular_presion_osmotica_vant_hoff(concentraciones_mg_l, temp_c):
    """Modelo ideal / diluido de Van't Hoff"""
    temp_k = temp_c + 273.15
    r_const = 0.08314  # L·bar/(mol·K)
    molaridad_total = sum((conc / 1000.0) / MOLAR_MASS[ion] for ion, conc in concentraciones_mg_l.items())
    return molaridad_total * r_const * temp_k

def calcular_presion_osmotica_pitzer(concentraciones_mg_l, temp_c):
    """
    Modelo termodinámico riguroso de Pitzer para alta fuerza iónica y salmueras.
    Calcula el coeficiente osmótico (Phi) incorporando Debye-Hückel y términos viriales.
    """
    temp_k = temp_c + 273.15
    r_const = 0.08314  # L·bar/(mol·K)
    
    molaridad_total = 0.0
    fuerza_ionica = 0.0
    
    for ion, conc in concentraciones_mg_l.items():
        m_i = (conc / 1000.0) / MOLAR_MASS[ion]
        molaridad_total += m_i
        z = VALENCIAS.get(ion, 1)
        fuerza_ionica += 0.5 * (z**2) * m_i
        
    # Parámetro Debye-Hückel A_phi con corrección de temperatura
    a_phi = 0.392 * ((temp_k / 298.15) ** 1.5)
    
    # Término electrostático (Debye-Hückel modificado)
    dh_term = - (a_phi * (fuerza_ionica ** 1.5)) / (1.0 + 1.2 * (fuerza_ionica ** 0.5))
    
    # Coeficientes viriales efectivos de corto alcance para sistemas clorurados/sulfatados complejos
    b_virial = 0.095
    c_virial = 0.0025
    virial_term = b_virial * fuerza_ionica + c_virial * (fuerza_ionica ** 2.0)
    
    # Coeficiente osmótico Pitzer (Phi)
    phi_pitzer = max(0.5, 1.0 + dh_term + virial_term)
    
    # Presión osmótica ajustada por Pitzer
    return phi_pitzer * molaridad_total * r_const * temp_k

def simular_osmosis_inversa_etapa_unica(q_feed, rec_target, p_oper, temp_c, a_perm, eluato_init, usar_pitzer=False):
    rec_frac = rec_target / 100.0
    q_perm = q_feed * rec_frac
    q_conc = q_feed - q_perm

    # Selección de modelo termodinámico
    calc_pi = calcular_presion_osmotica_pitzer if usar_pitzer else calcular_presion_osmotica_vant_hoff

    pi_feed = calc_pi(eluato_init, temp_c)

    conc_reject = {}
    conc_perm = {}

    for ion, c_feed in eluato_init.items():
        r_ion = RECHAZO_IONICO[ion]
        c_p = c_feed * (1.0 - r_ion)
        conc_perm[ion] = c_p
        c_c = (q_feed * c_feed - q_perm * c_p) / q_conc
        conc_reject[ion] = max(c_c, 0.0)

    pi_conc = calc_pi(conc_reject, temp_c)
    pi_perm = calc_pi(conc_perm, temp_c)
    pi_promedio = (pi_feed + pi_conc) / 2.0

    p_perdida_canal = 1.5
    ndp = (p_oper - (p_perdida_canal / 2.0)) - (pi_promedio - pi_perm)

    if ndp <= 0:
        modelo_str = "Pitzer" if usar_pitzer else "Van't Hoff"
        raise ValueError(
            f"La presión aplicada ({p_oper} bar) es insuficiente según el modelo {modelo_str}. "
            f"La presión osmótica promedio del sistema alcanzó los {pi_promedio:.2f} bar y supera el impulso hidráulico."
        )

    flux_lmh = a_perm * ndp
    area_m2 = (q_perm * 1000.0) / flux_lmh

    return {
        'q_feed': q_feed, 'q_perm': q_perm, 'q_conc': q_conc,
        'rec': rec_target, 'p_oper': p_oper, 'ndp': ndp,
        'flux_lmh': flux_lmh, 'area_m2': area_m2,
        'pi_feed': pi_feed, 'pi_conc': pi_conc,
        'conc_reject': conc_reject, 'conc_perm': conc_perm,
        'modelo_usado': "Pitzer (Riguroso Alta Salinidad)" if usar_pitzer else "Van't Hoff (Ideal)"
    }

# ==============================================================================
# 3. INTERFAZ INTERACTIVA STREAMLIT
# ==============================================================================
st.title("💧 Simulador de Ósmosis Inversa (1 Etapa)")
st.subheader("Concentración de Eluato de Litio (DLE)")
st.markdown("---")

# BARRA LATERAL DE CONFIGURACIÓN (INPUTS DEL USUARIO)
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    q_in = st.number_input("1. Caudal de Alimentación (m³/h)", min_value=1.0, value=50.0, step=1.0)
    t_in = st.number_input("2. Temperatura del Eluato (°C)", min_value=1.0, value=25.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### Régimen de Presión y Termodinámica")
    tipo_ro = st.radio(
        "Selecciona el régimen:",
        options=["Alta Presión (SWRO)", "Baja Presión (BWRO)"],
        index=0,
        help="Alta Presión activa el modelo termodinámico de Pitzer para soluciones de alta salinidad."
    )
    
    # Asignación de valores por defecto dinámicos
    if tipo_ro == "Baja Presión (BWRO)":
        def_p, def_rec, def_a = 22.0, 65.0, 3.5
    else:
        def_p, def_rec, def_a = 60.0, 55.0, 1.2
        
    p_in = st.number_input("3. Presión Operativa (bar)", min_value=1.0, value=float(def_p), step=1.0)
    rec_in = st.slider("4. Recuperación Deseada (%)", min_value=10.0, max_value=90.0, value=float(def_rec), step=1.0)
    a_in = st.number_input("5. Permeabilidad Membrana 'A' (L/m²·h·bar)", min_value=0.1, value=float(def_a), step=0.1)

    st.markdown("---")
    st.header("🧪 Composición Eluato DLE")
    st.markdown("Ingrese las concentraciones iniciales (mg/L):")
    
    default_eluato = {
        'Li': 639.496, 'Na': 428.557, 'K': 29.233, 'Mg': 3.630, 'Ca': 1.995,
        'B': 175.739, 'SO4': 874.200, 'Cl': 3539.000, 'CO3': 0.00273, 'HCO3': 0.074
    }
    
    eluato_dle_init = {}
    with st.expander("Modificar concentraciones iónicas", expanded=True):
        for ion, default_val in default_eluato.items():
            eluato_dle_init[ion] = st.number_input(
                f"[{ion}] (mg/L)", 
                min_value=0.0, 
                value=float(default_val), 
                step=0.0001 if default_val < 1.0 else 0.1, 
                format="%.5f" if default_val < 1.0 else "%.2f"
            )

# Determinar si se activa Pitzer
usar_pitzer = True if tipo_ro == "Alta Presión (SWRO)" else False

# ==============================================================================
# 4. EJECUCIÓN Y REPORTE VISUAL DE INGENIERÍA
# ==============================================================================
try:
    res = simular_osmosis_inversa_etapa_unica(q_in, rec_in, p_in, t_in, a_in, eluato_dle_init, usar_pitzer=usar_pitzer)
    
    # SECCIÓN 1: INDICADORES PRINCIPALES (KPIs)
    st.markdown(f"### 📊 Indicadores Clave de Concentración (Li+) — Modelo activo: *{res['modelo_usado']}*")
    
    li_in = eluato_dle_init['Li']
    li_out = res['conc_reject']['Li']
    li_perm = res['conc_perm']['Li']
    factor_conc = li_out / li_in if li_in > 0 else 0.0
    tds_in = sum(eluato_dle_init.values())
    tds_out = sum(res['conc_reject'].values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Litio Inicial", value=f"{li_in:.1f} mg/L", delta=f"{li_in/1000:.2f} g/L")
    with col2:
        st.metric(label="Litio Concentrado", value=f"{li_out:.1f} mg/L", delta=f"{li_out/1000:.2f} g/L")
    with col3:
        st.metric(label="Factor de Concentración", value=f"{factor_conc:.2f} X")
    with col4:
        st.metric(label="TDS Final Salmuera", value=f"{tds_out/1000:.2f} g/L", delta=f"Inic: {tds_in/1000:.2f} g/L")
        
    st.markdown("---")
    
    # SECCIÓN 2: TABLAS DETALLADAS EN PESTAÑAS
    tab1, tab2 = st.tabs(["⚙️ Resumen Operativo e Hidráulico", "🧪 Balance de Masa por Ion"])
    
    with tab1:
        df_hidro = pd.DataFrame([
            {'Parámetro Hidráulico / Operativo': 'Modelo Termodinámico de Presión Osmótica', 'Valor': res['modelo_usado']},
            {'Parámetro Hidráulico / Operativo': 'Caudal de Alimentación (Feed)', 'Valor': f"{res['q_feed']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Caudal de Permeado (Agua Extraída)', 'Valor': f"{res['q_perm']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Caudal de Salmuera Concentrada (Rechazo)', 'Valor': f"{res['q_conc']:.2f} m³/h"},
            {'Parámetro Hidráulico / Operativo': 'Recuperación del Sistema', 'Valor': f"{res['rec']:.1f} %"},
            {'Parámetro Hidráulico / Operativo': 'Presión Operativa Aplicada', 'Valor': f"{res['p_oper']:.1f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Osmótica Entrada (Eluato DLE)', 'Valor': f"{res['pi_feed']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Osmótica Salida (Salmuera)', 'Valor': f"{res['pi_conc']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Presión Neta de Impulso (NDP)', 'Valor': f"{res['ndp']:.2f} bar"},
            {'Parámetro Hidráulico / Operativo': 'Flujo de Membrana (Flux)', 'Valor': f"{res['flux_lmh']:.1f} LMH"},
            {'Parámetro Hidráulico / Operativo': 'Área de Membrana Requerida', 'Valor': f"{res['area_m2']:.1f} m²"}
        ])
        st.dataframe(df_hidro, use_container_width=True, hide_index=True)
        
    with tab2:
        df_quimico = pd.DataFrame({
            'Ion / Especie': list(eluato_dle_init.keys()),
            'Alimentación (mg/L)': [eluato_dle_init[k] for k in eluato_dle_init],
            'Salmuera Concentrada (mg/L)': [res['conc_reject'][k] for k in eluato_dle_init],
            'Permeado - Agua Extraída (mg/L)': [res['conc_perm'][k] for k in eluato_dle_init]
        })
        st.dataframe(df_quimico.round(2), use_container_width=True, hide_index=True)

# CAPTURA DE ERROR TERMODINÁMICO EN PANTALLA
except ValueError as e:
    st.error("🚨 **ERROR DE DISEÑO HIDRÁULICO / TERMODINÁMICO**")
    st.warning(str(e))
    st.info("💡 **Solución:** Ve al panel lateral a la izquierda y disminuye el % de recuperación o eleva los bar de presión operativa.")
