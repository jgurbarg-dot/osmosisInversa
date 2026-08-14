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
# 0. SISTEMA DE SEGURIDAD (CONTRASEÑA)
# ==============================================================================
def check_password():
    """Verifica la contraseña antes de cargar el resto de la aplicación."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.warning("🔒 Por favor, ingresa la contraseña para acceder al simulador.")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Ingresar"):
            if password == "AdeIri61Azu":
                st.session_state["password_correct"] = True
                st.rerun()  # Recarga la página para mostrar el contenido
            else:
                st.error("❌ Contraseña incorrecta.")
        
        # Detiene la ejecución del script aquí si no está autenticado
        st.stop()

# Activamos el candado de seguridad
check_password()

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
    pi = molaridad_total * r_const * temp_k
    
    # Retornamos también los parámetros internos para el reporte
    return pi, {
        "molaridad_total": molaridad_total,
        "temp_k": temp_k,
        "phi": 1.0
    }

def calcular_presion_osmotica_pitzer(concentraciones_mg_l, temp_c):
    """Modelo termodinámico de Pitzer para alta fuerza iónica"""
    temp_k = temp_c + 273.15
    r_const = 0.08314  # L·bar/(mol·K)
    
    molaridad_total = 0.0
    fuerza_ionica = 0.0
    
    for ion, conc in concentraciones_mg_l.items():
        m_i = (conc / 1000.0) / MOLAR_MASS[ion]
        molaridad_total += m_i
        z = VALENCIAS.get(ion, 1)
        fuerza_ionica += 0.5 * (z**2) * m_i
        
    # Parámetro Debye-Hückel A_phi
    a_phi = 0.392 * ((temp_k / 298.15) ** 1.5)
    
    # Término electrostático (Debye-Hückel modificado)
    dh_term = - (a_phi * (fuerza_ionica ** 1.5)) / (1.0 + 1.2 * (fuerza_ionica ** 0.5))
    
    # Coeficientes viriales efectivos
    b_virial = 0.095
    c_virial = 0.0025
    virial_term = b_virial * fuerza_ionica + c_virial * (fuerza_ionica ** 2.0)
    
    # Coeficiente osmótico Pitzer (Phi)
    phi_pitzer = max(0.5, 1.0 + dh_term + virial_term)
    
    # Presión osmótica
    pi = phi_pitzer * molaridad_total * r_const * temp_k
    
    return pi, {
        "molaridad_total": molaridad_total,
        "fuerza_ionica": fuerza_ionica,
        "a_phi": a_phi,
        "dh_term": dh_term,
        "virial_term": virial_term,
        "phi": phi_pitzer,
        "temp_k": temp_k
    }

def simular_osmosis_inversa_etapa_unica(q_feed, rec_target, p_oper, temp_c, a_perm, eluato_init, usar_pitzer=False):
    rec_frac = rec_target / 100.0
    q_perm = q_feed * rec_frac
    q_conc = q_feed - q_perm

    calc_pi = calcular_presion_osmotica_pitzer if usar_pitzer else calcular_presion_osmotica_vant_hoff

    pi_feed, stats_feed = calc_pi(eluato_init, temp_c)

    conc_reject = {}
    conc_perm = {}

    for ion, c_feed in eluato_init.items():
        r_ion = RECHAZO_IONICO[ion]
        c_p = c_feed * (1.0 - r_ion)
        conc_perm[ion] = c_p
        c_c = (q_feed * c_feed - q_perm * c_p) / q_conc
        conc_reject[ion] = max(c_c, 0.0)

    pi_conc, stats_conc = calc_pi(conc_reject, temp_c)
    pi_perm, stats_perm = calc_pi(conc_perm, temp_c)
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
        'flux_lmh': flux_lmh, 'area_m2': area_m2, 'p_perdida': p_perdida_canal,
        'pi_feed': pi_feed, 'pi_conc': pi_conc, 'pi_perm': pi_perm, 'pi_promedio': pi_promedio,
        'conc_reject': conc_reject, 'conc_perm': conc_perm,
        'stats_feed': stats_feed, 'stats_conc': stats_conc, 'stats_perm': stats_perm,
        'modelo_usado': "Pitzer (Riguroso Alta Salinidad)" if usar_pitzer else "Van't Hoff (Ideal)",
        'usar_pitzer': usar_pitzer, 'temp_k': stats_feed['temp_k']
    }

# ==============================================================================
# 3. INTERFAZ INTERACTIVA STREAMLIT
# ==============================================================================
st.title("💧 Simulador de Ósmosis Inversa (1 Etapa)")
st.subheader("Concentración de Eluato de Litio (DLE)")
st.markdown("---")

# BARRA LATERAL DE CONFIGURACIÓN
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

usar_pitzer = True if tipo_ro == "Alta Presión (SWRO)" else False

# ==============================================================================
# 4. EJECUCIÓN Y REPORTE VISUAL DE INGENIERÍA
# ==============================================================================
try:
    res = simular_osmosis_inversa_etapa_unica(q_in, rec_in, p_in, t_in, a_in, eluato_dle_init, usar_pitzer=usar_pitzer)
    
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
    
    # SECCIÓN 2: PESTAÑAS
    tab1, tab2, tab3 = st.tabs(["⚙️ Resumen Operativo", "🧪 Balance de Masa por Ion", "📖 Memoria de Cálculo (Paso a Paso)"])
    
    with tab1:
        df_hidro = pd.DataFrame([
            {'Parámetro': 'Modelo Termodinámico', 'Valor': res['modelo_usado']},
            {'Parámetro': 'Caudal Alimentación', 'Valor': f"{res['q_feed']:.2f} m³/h"},
            {'Parámetro': 'Caudal Permeado', 'Valor': f"{res['q_perm']:.2f} m³/h"},
            {'Parámetro': 'Caudal Salmuera', 'Valor': f"{res['q_conc']:.2f} m³/h"},
            {'Parámetro': 'Recuperación', 'Valor': f"{res['rec']:.1f} %"},
            {'Parámetro': 'Presión Operativa', 'Valor': f"{res['p_oper']:.1f} bar"},
            {'Parámetro': 'Presión Osmótica Entrada', 'Valor': f"{res['pi_feed']:.2f} bar"},
            {'Parámetro': 'Presión Osmótica Salida', 'Valor': f"{res['pi_conc']:.2f} bar"},
            {'Parámetro': 'NDP', 'Valor': f"{res['ndp']:.2f} bar"},
            {'Parámetro': 'Flux', 'Valor': f"{res['flux_lmh']:.1f} LMH"},
            {'Parámetro': 'Área Requerida', 'Valor': f"{res['area_m2']:.1f} m²"}
        ])
        st.dataframe(df_hidro, use_container_width=True, hide_index=True)
        
    with tab2:
        df_quimico = pd.DataFrame({
            'Ion / Especie': list(eluato_dle_init.keys()),
            'Alimentación (mg/L)': [eluato_dle_init[k] for k in eluato_dle_init],
            'Salmuera Concentrada (mg/L)': [res['conc_reject'][k] for k in eluato_dle_init],
            'Permeado - Agua (mg/L)': [res['conc_perm'][k] for k in eluato_dle_init]
        })
        st.dataframe(df_quimico.round(2), use_container_width=True, hide_index=True)

    # NUEVA PESTAÑA: MEMORIA DE CÁLCULO
    with tab3:
        st.header("1. Balances de Materia Globales e Iónicos")
        st.markdown("Se determina el caudal de permeado ($Q_{perm}$) basado en la recuperación ($Y$) y, por diferencia, el caudal de rechazo ($Q_{conc}$).")
        
        st.latex(r"Q_{perm} = Q_{feed} \times \left(\frac{Y}{100}\right)")
        st.markdown(f"**Ejecución:** $Q_{{perm}} = {res['q_feed']} \\times ({res['rec']}/100) = \\mathbf{{{res['q_perm']:.2f} \\text{{ m}}^3\\text{{/h}}}}$")
        
        st.latex(r"Q_{conc} = Q_{feed} - Q_{perm}")
        st.markdown(f"**Ejecución:** $Q_{{conc}} = {res['q_feed']} - {res['q_perm']:.2f} = \\mathbf{{{res['q_conc']:.2f} \\text{{ m}}^3\\text{{/h}}}}$")

        st.markdown("Para cada ion $i$, la concentración en el permeado depende del rechazo iónico ($R_i$), y el rechazo se calcula por balance de masa:")
        st.latex(r"C_{perm,i} = C_{feed,i} \times (1 - R_i)")
        st.latex(r"C_{conc,i} = \frac{Q_{feed} \cdot C_{feed,i} - Q_{perm} \cdot C_{perm,i}}{Q_{conc}}")
        st.info("💡 *El simulador itera estas dos últimas fórmulas sobre todos los iones ingresados (ver pestaña 'Balance de Masa por Ion' para los resultados).*")

        st.divider()

        st.header("2. Termodinámica: Presión Osmótica ($\\pi$)")
        if not res['usar_pitzer']:
            st.markdown("### Modelo Ideal: Ecuación de Van't Hoff")
            st.markdown("Asume soluciones diluidas donde las interacciones iónicas son despreciables. El coeficiente osmótico es $\Phi = 1$.")
            st.latex(r"\pi = \sum \left( \frac{C_i}{MW_i} \right) \cdot R \cdot T")
            
            st.markdown("**Valores calculados (Corriente de Alimentación):**")
            st.markdown(f"- $\\sum Molaridad = {res['stats_feed']['molaridad_total']:.4f} \\text{{ mol/L}}$")
            st.markdown(f"- $R = 0.08314 \\text{{ L·bar/(mol·K)}}$")
            st.markdown(f"- $T = {res['temp_k']:.2f} \\text{{ K}}$")
            st.latex(fr"\pi_{{feed}} = {res['stats_feed']['molaridad_total']:.4f} \times 0.08314 \times {res['temp_k']:.2f} = \mathbf{{{res['pi_feed']:.2f} \text{{ bar}}}}")
        
        else:
            st.markdown("### Modelo Riguroso: Ecuación de Pitzer")
            st.markdown("Calcula un Coeficiente Osmótico ($\\Phi$) para corregir desviaciones por alta salinidad mediante la Fuerza Iónica ($I$), términos de Debye-Hückel electrostáticos y coeficientes viriales específicos.")
            
            st.latex(r"I = \frac{1}{2} \sum z_i^2 \cdot m_i")
            st.latex(r"DH = - \frac{A_\phi \cdot I^{1.5}}{1 + 1.2 \cdot I^{0.5}}")
            st.latex(r"Virial = B \cdot I + C \cdot I^2")
            st.latex(r"\Phi = 1 + DH + Virial")
            st.latex(r"\pi = \Phi \cdot \sum m_i \cdot R \cdot T")

            st.markdown("**Valores calculados (Corriente de Salmuera Concentrada / Rechazo):**")
            stats = res['stats_conc']
            st.markdown(f"- $\\sum Molaridad = {stats['molaridad_total']:.4f} \\text{{ mol/L}}$")
            st.markdown(f"- $I \\text{{ (Fuerza Iónica)}} = {stats['fuerza_ionica']:.4f}$")
            st.markdown(f"- $A_\\phi \\text{{ (Parámetro DH a }} {res['temp_k']:.1f} \\text{{ K)}} = {stats['a_phi']:.4f}$")
            st.markdown(f"- $DH \\text{{ (Término Electrostático)}} = {stats['dh_term']:.4f}$")
            st.markdown(f"- $Virial \\text{{ (Interacciones de corto alcance)}} = {stats['virial_term']:.4f}$")
            
            st.latex(fr"\Phi_{{conc}} = 1 + ({stats['dh_term']:.4f}) + ({stats['virial_term']:.4f}) = \mathbf{{{stats['phi']:.4f}}}")
            st.latex(fr"\pi_{{conc}} = {stats['phi']:.4f} \times {stats['molaridad_total']:.4f} \times 0.08314 \times {res['temp_k']:.2f} = \mathbf{{{res['pi_conc']:.2f} \text{{ bar}}}}")

        st.divider()

        st.header("3. Hidráulica de Membrana (Diseño RO)")
        st.markdown("La Fuerza Motriz Neta (NDP) define si el sistema puede vencer la ósmosis natural y empujar agua limpia. Requiere promediar la presión osmótica de entrada y salida.")
        
        st.latex(r"\pi_{avg} = \frac{\pi_{feed} + \pi_{conc}}{2}")
        st.markdown(f"**Ejecución:** $\\pi_{{avg}} = \\frac{{{res['pi_feed']:.2f} + {res['pi_conc']:.2f}}}{{2}} = \\mathbf{{{res['pi_promedio']:.2f} \\text{{ bar}}}}$")

        st.latex(r"NDP = \left(P_{oper} - \frac{\Delta P_{canal}}{2}\right) - (\pi_{avg} - \pi_{perm})")
        st.markdown(f"**Ejecución:** $NDP = \\left({res['p_oper']} - \\frac{{{res['p_perdida']}}}{{2}}\\right) - ({res['pi_promedio']:.2f} - {res['pi_perm']:.2f}) = \\mathbf{{{res['ndp']:.2f} \\text{{ bar}}}}$")

        st.markdown("Finalmente, se calcula el Flujo de agua a través de los poros (Flux) y el Área de membrana total requerida:")
        st.latex(r"Flux (J) = A_{perm} \times NDP")
        st.markdown(f"**Ejecución:** $Flux = {a_in} \\times {res['ndp']:.2f} = \\mathbf{{{res['flux_lmh']:.2f} \\text{{ L/m}}^2\\cdot\\text{{h}}}}$")

        st.latex(r"Area = \frac{Q_{perm} \cdot 1000}{Flux}")
        st.markdown(f"**Ejecución:** $Area = \\frac{{{res['q_perm']:.2f} \cdot 1000}}{{{res['flux_lmh']:.2f}}} = \\mathbf{{{res['area_m2']:.2f} \\text{{ m}}^2}}$")

# CAPTURA DE ERROR TERMODINÁMICO EN PANTALLA
except ValueError as e:
    st.error("🚨 **ERROR DE DISEÑO HIDRÁULICO / TERMODINÁMICO**")
    st.warning(str(e))
    st.info("💡 **Solución:** Ve al panel lateral a la izquierda y disminuye el % de recuperación o eleva los bar de presión operativa.")
