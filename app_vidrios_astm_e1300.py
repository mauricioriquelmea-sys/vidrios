# -*- coding: utf-8 -*-
"""
=====================================================================
CÁLCULO ESTRUCTURAL DE VIDRIOS Y TERMOPANELES (DVH)
Norma de referencia: ASTM E1300-24
"Standard Practice for Determining Load Resistance of Glass in Buildings"

Autor: Proyectos Estructurales EIRL
Ejecución: streamlit run app_vidrios_astm_e1300.py

-------------------------------------------------------------------
BASES DE CÁLCULO IMPLEMENTADAS
-------------------------------------------------------------------
1. Espesor efectivo laminado : Modelo de Wölfel-Bennison (ASTM E1300, Anexo X).
2. Deflexión                 : Formulación no lineal de gran deformación
                               (ASTM E1300, Anexo X), con fallback a la
                               solución lineal de Timoshenko fuera del
                               dominio de validez del ajuste.
3. Tensión de trabajo        : Coeficientes de Timoshenko para placa
                               rectangular simplemente apoyada en 4 bordes
                               bajo carga uniforme (teoría lineal,
                               conservadora frente a la no lineal).
4. Tensión admisible         : Valores base a 3 s con pb = 8/1000, ajustados
                               por duración mediante el exponente de Weibull
                               (3/d)^(1/16) de ASTM E1300.
5. Reparto de carga (LSF)    : Proporcional a la rigidez relativa (t_ef^3).
=====================================================================
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA CORPORATIVA
# =====================================================================
st.set_page_config(
    page_title="ASTM E1300-24 | Cálculo de Vidrios y Termopaneles",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-left: 2rem; padding-right: 2rem; max-width: 100%; }
    .stMetric {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .verdict-ok {
        background-color: #e7f6ec; border-left: 8px solid #1e7e34;
        padding: 22px; border-radius: 8px; margin: 15px 0;
        font-size: 1.55em; font-weight: 700; color: #14532d;
    }
    .verdict-fail {
        background-color: #fdecea; border-left: 8px solid #b02a37;
        padding: 22px; border-radius: 8px; margin: 15px 0;
        font-size: 1.55em; font-weight: 700; color: #6b1219;
    }
    .info-box {
        background-color: #eef2f7; border-left: 6px solid #0056b3;
        padding: 18px; border-radius: 6px; margin: 15px 0;
        font-size: 0.92em; line-height: 1.5;
    }
    .lite-header {
        background-color: #343a40; color: #ffffff; padding: 8px 14px;
        border-radius: 5px; font-weight: 600; margin-bottom: 10px;
    }
    .sidebar-help { font-size: 0.83em; color: #555; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 2. CONSTANTES NORMATIVAS Y TABLAS DE REFERENCIA
# =====================================================================

E_GLASS = 71_700e6          # Módulo de elasticidad del vidrio [Pa] (ASTM E1300)
NU_GLASS = 0.22             # Coeficiente de Poisson del vidrio [-]
RHO_GLASS = 2500.0          # Densidad del vidrio [kg/m3]
WEIBULL_EXP = 1.0 / 16.0    # Exponente de duración de carga (ASTM E1300)
REF_DURATION_S = 3.0        # Duración de referencia de la tensión base [s]

# Tensión admisible base para duración de 3 s y probabilidad de rotura
# pb = 8 lites / 1000 (ASTM E1300, vidrio vertical) [Pa]
ALLOWABLE_STRESS_3S = {
    "Crudo (Annealed)": 23.3e6,
    "Termoendurecido (Heat Strengthened)": 46.6e6,
    "Templado (Tempered)": 93.1e6,
}

# Colores de representación gráfica según tratamiento térmico (convención Saflex)
GLASS_COLORS = {
    "Crudo (Annealed)": "#F2E205",              # Amarillo
    "Termoendurecido (Heat Strengthened)": "#F28C0F",  # Naranjo
    "Templado (Tempered)": "#D62828",           # Rojo
}

# Espesores nominales comerciales [mm] y su espesor mínimo de cálculo
# (ASTM E1300, Tabla 4 - nominal vs. minimum thickness)
NOMINAL_TO_MIN_THICKNESS = {
    2.5: 2.16, 3.0: 2.92, 4.0: 3.78, 5.0: 4.57, 6.0: 5.56,
    8.0: 7.42, 10.0: 9.02, 12.0: 11.91, 16.0: 15.09,
    19.0: 18.26, 22.0: 21.44, 25.0: 24.61,
}

# Módulo de corte G del interlayer [Pa]. Valores indicativos: SIEMPRE verificar
# contra la tabla del fabricante para la combinación (temperatura, duración).
INTERLAYER_PRESETS = {
    "PVB - 3 s @ 20 °C": 8.06e6,
    "PVB - 3 s @ 30 °C": 1.60e6,
    "PVB - 3 s @ 40 °C": 0.62e6,
    "PVB - 3 s @ 50 °C": 0.44e6,
    "PVB - 60 s @ 30 °C": 0.86e6,
    "PVB - 60 s @ 40 °C": 0.44e6,
    "SGP (Ionoplast) - 3 s @ 30 °C": 141.0e6,
    "SGP (Ionoplast) - 3 s @ 50 °C": 25.0e6,
    "Definido por el usuario": None,
}

# Tablas de Timoshenko: placa rectangular simplemente apoyada en 4 bordes,
# carga uniforme.   w_max = alpha * q * b^4 / D    ;   sigma_max = beta * q * b^2 / t^2
# (b = lado corto, a = lado largo, nu = 0.3)
TIMO_RATIO = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 1e3])
TIMO_ALPHA = np.array([0.00406, 0.00564, 0.00705, 0.00830, 0.00931,
                       0.01013, 0.01223, 0.01282, 0.01297, 0.01302])
TIMO_BETA = np.array([0.28740, 0.37620, 0.45300, 0.51720, 0.56880,
                      0.61020, 0.71340, 0.74100, 0.74760, 0.75000])


# =====================================================================
# 3. MOTOR MATEMÁTICO
# =====================================================================

def timoshenko_coefficients(aspect_ratio: float) -> tuple:
    """
    Interpola los coeficientes alpha (deflexión) y beta (tensión) de Timoshenko
    para una placa simplemente apoyada en sus 4 bordes.

    aspect_ratio : a/b (lado largo / lado corto), >= 1.0
    """
    ar = max(1.0, float(aspect_ratio))
    alpha = float(np.interp(ar, TIMO_RATIO, TIMO_ALPHA))
    beta = float(np.interp(ar, TIMO_RATIO, TIMO_BETA))
    return alpha, beta


def plate_flexural_rigidity(t: float) -> float:
    """Rigidez flexural D = E*t^3 / (12*(1 - nu^2))  [N*m]."""
    return E_GLASS * t ** 3 / (12.0 * (1.0 - NU_GLASS ** 2))


def linear_deflection(q: float, a: float, b: float, t: float) -> float:
    """Deflexión lineal de Timoshenko [m]. q [Pa], a/b/t [m]."""
    if t <= 0:
        return 0.0
    alpha, _ = timoshenko_coefficients(a / b)
    return alpha * q * b ** 4 / plate_flexural_rigidity(t)


def astm_nonlinear_deflection(q: float, a: float, b: float, t: float) -> tuple:
    """
    Deflexión de gran deformación según ASTM E1300 (Anexo X).

        x     = ln( ln( q*(a*b)^2 / (E*t^4) ) )
        w     = t * exp( r0 + r1*x + r2*x^2 )

    con r_i función de la relación de aspecto (válido para 1 <= a/b <= 5).

    Retorna (deflexión [m], método utilizado [str]).
    """
    if t <= 0:
        return 0.0, "N/A"

    ar = a / b
    q_hat = q * (a * b) ** 2 / (E_GLASS * t ** 4)  # carga adimensional

    # Dominio de validez del ajuste logarítmico doble
    if q_hat <= 1.0 or not (1.0 <= ar <= 5.0):
        return linear_deflection(q, a, b, t), "Timoshenko lineal (fuera del ajuste E1300)"

    x = math.log(math.log(q_hat))

    r0 = 0.553 - 3.830 * ar + 1.110 * ar ** 2 - 0.0969 * ar ** 3
    r1 = -2.290 + 5.830 * ar - 2.170 * ar ** 2 + 0.2067 * ar ** 3
    r2 = 1.485 - 1.908 * ar + 0.815 * ar ** 2 - 0.0822 * ar ** 3

    w = t * math.exp(r0 + r1 * x + r2 * x ** 2)
    return w, "ASTM E1300 no lineal"


def timoshenko_stress(q: float, a: float, b: float, t: float) -> float:
    """Tensión principal máxima de flexión [Pa] (teoría lineal, conservadora)."""
    if t <= 0:
        return 0.0
    _, beta = timoshenko_coefficients(a / b)
    return beta * q * b ** 2 / t ** 2


def allowable_stress(treatment: str, duration_s: float) -> float:
    """
    Tensión admisible [Pa] ajustada por duración de carga.

        sigma_adm(d) = sigma_adm(3 s) * (3 / d)^(1/16)

    El exponente 1/16 corresponde al parámetro de Weibull adoptado por E1300.
    """
    base = ALLOWABLE_STRESS_3S[treatment]
    return base * (REF_DURATION_S / duration_s) ** WEIBULL_EXP


def laminated_effective_thickness(h1: float, h2: float, hv: float,
                                  g_int: float, a_min: float,
                                  force_monolithic: bool = False) -> dict:
    """
    Espesor efectivo de un vidrio laminado de 2 láminas.
    Modelo de Wölfel-Bennison (ASTM E1300, Anexo X). Todas las unidades en [m],
    salvo g_int en [Pa].

        hs   = 0.5*(h1 + h2) + hv
        hs1  = hs * h1 / (h1 + h2)   ;   hs2 = hs * h2 / (h1 + h2)
        Is   = h1*hs2^2 + h2*hs1^2
        Gamma= 1 / (1 + 9.6 * E * Is * hv / (G * hs^2 * a_min^2))
        h_ef,w     = (h1^3 + h2^3 + 12*Gamma*Is)^(1/3)
        h_ef,sigma,i = sqrt( h_ef,w^3 / (h_i + 2*Gamma*hs_j) )

    Retorna dict con h_ef_w, h_ef_s1, h_ef_s2 y Gamma.
    """
    if force_monolithic:
        h_mono = h1 + h2
        return {"h_ef_w": h_mono, "h_ef_s1": h_mono, "h_ef_s2": h_mono, "gamma": 1.0}

    hs = 0.5 * (h1 + h2) + hv
    hs1 = hs * h1 / (h1 + h2)
    hs2 = hs * h2 / (h1 + h2)
    i_s = h1 * hs2 ** 2 + h2 * hs1 ** 2

    denom = g_int * hs ** 2 * a_min ** 2
    gamma = 0.0 if denom <= 0 else 1.0 / (1.0 + 9.6 * E_GLASS * i_s * hv / denom)
    gamma = min(max(gamma, 0.0), 1.0)

    h_ef_w = (h1 ** 3 + h2 ** 3 + 12.0 * gamma * i_s) ** (1.0 / 3.0)
    h_ef_s1 = math.sqrt(h_ef_w ** 3 / (h1 + 2.0 * gamma * hs2))
    h_ef_s2 = math.sqrt(h_ef_w ** 3 / (h2 + 2.0 * gamma * hs1))

    return {"h_ef_w": h_ef_w, "h_ef_s1": h_ef_s1, "h_ef_s2": h_ef_s2, "gamma": gamma}


def build_lite(cfg: dict, a_min: float) -> dict:
    """
    Construye las propiedades geométricas de una lámina (lite) a partir de su
    configuración de entrada. Devuelve espesores efectivos en [m].
    """
    lite = dict(cfg)

    if cfg["construction"] == "Monolítico":
        t_min = NOMINAL_TO_MIN_THICKNESS[cfg["t_nom"]] / 1000.0
        lite.update({
            "h_ef_w": t_min,
            "h_ef_s": t_min,
            "gamma": None,
            "t_total_nom": cfg["t_nom"],
            "t_glass_total": t_min,
        })
    else:  # Laminado (2 láminas del mismo espesor nominal)
        h_ply = NOMINAL_TO_MIN_THICKNESS[cfg["t_nom"]] / 1000.0
        hv = cfg["t_interlayer"] / 1000.0
        res = laminated_effective_thickness(
            h_ply, h_ply, hv, cfg["g_int"], a_min, cfg["force_monolithic"]
        )
        lite.update({
            "h_ef_w": res["h_ef_w"],
            "h_ef_s": res["h_ef_s1"],  # láminas simétricas -> s1 = s2
            "gamma": res["gamma"],
            "t_total_nom": 2 * cfg["t_nom"] + cfg["t_interlayer"],
            "t_glass_total": 2 * h_ply,
        })

    return lite


def load_share_factors(t1_ef: float, t2_ef: float) -> tuple:
    """
    Factores de reparto de carga (Load Share Factor) de un termopanel,
    proporcionales a la rigidez relativa de cada lámina (t_ef^3).

        f1 = t1^3 / (t1^3 + t2^3)   ;   f2 = 1 - f1
    """
    s1, s2 = t1_ef ** 3, t2_ef ** 3
    total = s1 + s2
    if total <= 0:
        return 0.5, 0.5
    return s1 / total, s2 / total


def analyze_lite(lite: dict, q_lite: float, a: float, b: float,
                 duration_s: float) -> dict:
    """
    Verificación completa de una lámina: tensión y deflexión.
    q_lite [Pa]; a, b [m] (a = lado largo, b = lado corto).
    """
    sigma = timoshenko_stress(q_lite, a, b, lite["h_ef_s"])
    sigma *= 1.0 / max(lite.get("lsf_strength", 1.0), 1e-6)  # Laminate Strength Factor

    sigma_adm = allowable_stress(lite["treatment"], duration_s)
    delta, method = astm_nonlinear_deflection(q_lite, a, b, lite["h_ef_w"])

    return {
        "q_lite": q_lite,
        "sigma": sigma,
        "sigma_adm": sigma_adm,
        "fu_stress": sigma / sigma_adm if sigma_adm > 0 else np.inf,
        "delta": delta,
        "method": method,
        "weight": RHO_GLASS * lite["t_glass_total"] * a * b,
    }


# =====================================================================
# 4. REPRESENTACIÓN GRÁFICA DE LA SECCIÓN TRANSVERSAL
# =====================================================================

def draw_section(lite1: dict, lite2: dict = None, air_gap: float = 0.0):
    """
    Dibuja el esquema transversal del vidrio / termopanel, análogo a la
    interfaz de Saflex: rectángulos verticales, PVB como línea negra,
    cámara de aire en blanco y flecha de dirección de carga.
    """
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    height = 100.0
    x = 0.0
    labels = []

    def draw_lite(x0: float, lite: dict, tag: str) -> float:
        """Dibuja una lámina y retorna la coordenada x final."""
        color = GLASS_COLORS[lite["treatment"]]
        if lite["construction"] == "Monolítico":
            t = lite["t_nom"]
            ax.add_patch(Rectangle((x0, 0), t, height, facecolor=color,
                                   edgecolor="black", linewidth=1.4))
            x_end = x0 + t
        else:
            t_ply = lite["t_nom"]
            t_int = lite["t_interlayer"]
            ax.add_patch(Rectangle((x0, 0), t_ply, height, facecolor=color,
                                   edgecolor="black", linewidth=1.4))
            # Interlayer PVB / SGP representado como línea negra gruesa
            ax.add_patch(Rectangle((x0 + t_ply, 0), t_int, height,
                                   facecolor="black", edgecolor="black"))
            ax.add_patch(Rectangle((x0 + t_ply + t_int, 0), t_ply, height,
                                   facecolor=color, edgecolor="black", linewidth=1.4))
            x_end = x0 + 2 * t_ply + t_int
        labels.append(((x0 + x_end) / 2.0, tag))
        return x_end

    x = draw_lite(x, lite1, "Lite 1")

    if lite2 is not None:
        # Cámara de aire: espacio en blanco delimitado por línea punteada
        ax.add_patch(Rectangle((x, 0), air_gap, height, facecolor="white",
                               edgecolor="gray", linestyle="--", linewidth=1.0))
        ax.text(x + air_gap / 2.0, height + 4, f"Cámara\n{air_gap:.0f} mm",
                ha="center", va="bottom", fontsize=8, color="gray")
        x += air_gap
        x = draw_lite(x, lite2, "Lite 2")

    total = x
    margin = max(total * 0.55, 15.0)

    # Etiquetas de cada lámina
    for xc, tag in labels:
        ax.text(xc, -7, tag, ha="center", va="top", fontsize=9, fontweight="bold")

    # Flecha de dirección de carga (siempre incide sobre el Lite 1 / exterior)
    ax.annotate("", xy=(-2, height / 2), xytext=(-margin * 0.75, height / 2),
                arrowprops=dict(arrowstyle="-|>", linewidth=2.2, color="#0056b3"))
    ax.text(-margin * 0.78, height / 2 + 5, "Load →", fontsize=11,
            fontweight="bold", color="#0056b3")

    ax.set_xlim(-margin, total + margin * 0.35)
    ax.set_ylim(-20, height + 22)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(f"Sección transversal — Espesor total: {total:.2f} mm",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


# =====================================================================
# 5. BARRA LATERAL — ENTRADA DE DATOS
# =====================================================================

def lite_input_panel(prefix: str, label: str) -> dict:
    """Genera los controles de configuración de una lámina en la sidebar."""
    st.sidebar.markdown(f'<div class="lite-header">{label}</div>',
                        unsafe_allow_html=True)

    construction = st.sidebar.selectbox(
        "Tipo de construcción", ["Monolítico", "Laminado"], key=f"{prefix}_constr"
    )
    treatment = st.sidebar.selectbox(
        "Tratamiento térmico", list(ALLOWABLE_STRESS_3S.keys()), key=f"{prefix}_treat"
    )
    t_nom = st.sidebar.selectbox(
        "Espesor nominal de la lámina [mm]",
        list(NOMINAL_TO_MIN_THICKNESS.keys()),
        index=4, key=f"{prefix}_tnom",
        help="Para vidrio laminado corresponde al espesor de CADA lámina (ply).",
    )

    cfg = {
        "construction": construction,
        "treatment": treatment,
        "t_nom": float(t_nom),
        "t_interlayer": 0.0,
        "g_int": 0.0,
        "lsf_strength": 1.0,
        "force_monolithic": False,
    }

    if construction == "Laminado":
        cfg["t_interlayer"] = st.sidebar.number_input(
            "Espesor del interlayer [mm]", min_value=0.10, max_value=6.00,
            value=0.76, step=0.38, format="%.2f", key=f"{prefix}_tint",
        )
        preset = st.sidebar.selectbox(
            "Interlayer (módulo de corte G)", list(INTERLAYER_PRESETS.keys()),
            index=1, key=f"{prefix}_preset",
            help="Valores indicativos. Verificar contra la tabla del fabricante "
                 "para la combinación temperatura/duración del proyecto.",
        )
        g_preset = INTERLAYER_PRESETS[preset]
        if g_preset is None:
            g_mpa = st.sidebar.number_input(
                "G del interlayer [MPa]", min_value=0.01, max_value=500.0,
                value=1.60, step=0.10, format="%.2f", key=f"{prefix}_g",
            )
            cfg["g_int"] = g_mpa * 1e6
        else:
            cfg["g_int"] = g_preset
            st.sidebar.caption(f"G = {g_preset / 1e6:.2f} MPa")

        cfg["lsf_strength"] = st.sidebar.number_input(
            "Laminate Strength Factor [-]", min_value=0.10, max_value=1.00,
            value=1.00, step=0.05, format="%.2f", key=f"{prefix}_lsf",
            help="Factor de reducción de resistencia del laminado (Saflex). "
                 "Divide la resistencia disponible: FU = sigma / (LSF * sigma_adm).",
        )
        cfg["force_monolithic"] = st.sidebar.checkbox(
            "Check for Monolithic (h_ef = h1 + h2)", value=False,
            key=f"{prefix}_mono",
            help="Fuerza el comportamiento monolítico del laminado "
                 "(acoplamiento total, Gamma = 1). Verificar aplicabilidad.",
        )

    return cfg


st.sidebar.title("⚙️ Datos de Entrada")
st.sidebar.markdown(
    '<div class="sidebar-help">Verificación de vidrios y unidades de doble '
    'vidriado hermético (DVH) sometidos a presión de viento, según '
    'ASTM E1300-24. Apoyo en 4 bordes.</div>', unsafe_allow_html=True
)
st.sidebar.divider()

# ---- Datos generales -------------------------------------------------
st.sidebar.subheader("1. Geometría del vano")
a_long_mm = st.sidebar.number_input("Largo del vano, a [mm]", min_value=100.0,
                                    max_value=6000.0, value=2000.0, step=50.0)
b_short_mm = st.sidebar.number_input("Ancho del vano, b [mm]", min_value=100.0,
                                     max_value=6000.0, value=1200.0, step=50.0)

st.sidebar.subheader("2. Solicitación")
q_kpa = st.sidebar.number_input("Presión de viento de diseño, q [kPa]",
                                min_value=0.05, max_value=20.0, value=1.50,
                                step=0.05, format="%.2f")
duration_label = st.sidebar.selectbox("Duración de la carga", ["3 seg", "60 seg"])
duration_s = 3.0 if duration_label == "3 seg" else 60.0

st.sidebar.subheader("3. Deformación admisible")
defl_criterion = st.sidebar.radio(
    "Criterio", ["Valor fijo [mm]", "L/60 (lado corto)", "L/175 (lado corto)"],
    horizontal=False,
)
if defl_criterion == "Valor fijo [mm]":
    defl_adm_mm = st.sidebar.number_input("Deflexión admisible [mm]",
                                          min_value=1.0, max_value=100.0,
                                          value=19.05, step=0.5, format="%.2f")
elif defl_criterion == "L/60 (lado corto)":
    defl_adm_mm = min(a_long_mm, b_short_mm) / 60.0
else:
    defl_adm_mm = min(a_long_mm, b_short_mm) / 175.0

if defl_criterion != "Valor fijo [mm]":
    st.sidebar.caption(f"Δ_adm = {defl_adm_mm:.2f} mm")

st.sidebar.divider()

# ---- Tipo de sistema -------------------------------------------------
st.sidebar.subheader("4. Tipo de sistema")
system_type = st.sidebar.radio(
    "Glass Construction",
    ["Vidrio Simple (Single Lite)", "Termopanel (Insulating Unit)"],
)
is_igu = system_type.startswith("Termopanel")
st.sidebar.divider()

# ---- Configuración de láminas ---------------------------------------
cfg1 = lite_input_panel("l1", "Vidrio 1 — Exterior (cargado)")
air_gap_mm = 0.0
cfg2 = None

if is_igu:
    st.sidebar.divider()
    air_gap_mm = st.sidebar.number_input("Espesor de cámara de aire [mm]",
                                         min_value=6.0, max_value=30.0,
                                         value=12.0, step=1.0)
    st.sidebar.divider()
    cfg2 = lite_input_panel("l2", "Vidrio 2 — Interior")


# =====================================================================
# 6. CÁLCULO
# =====================================================================

# Normalización geométrica: a = lado largo, b = lado corto [m]
a_m = max(a_long_mm, b_short_mm) / 1000.0
b_m = min(a_long_mm, b_short_mm) / 1000.0
aspect_ratio = a_m / b_m
q_pa = q_kpa * 1000.0
defl_adm_m = defl_adm_mm / 1000.0

lite1 = build_lite(cfg1, b_m)
lite2 = build_lite(cfg2, b_m) if is_igu else None

if is_igu:
    f1, f2 = load_share_factors(lite1["h_ef_w"], lite2["h_ef_w"])
    res1 = analyze_lite(lite1, q_pa * f1, a_m, b_m, duration_s)
    res2 = analyze_lite(lite2, q_pa * f2, a_m, b_m, duration_s)
    results = [("Lite 1 (Exterior)", lite1, res1, f1),
               ("Lite 2 (Interior)", lite2, res2, f2)]
else:
    res1 = analyze_lite(lite1, q_pa, a_m, b_m, duration_s)
    results = [("Lite 1", lite1, res1, 1.0)]

fu_stress_max = max(r[2]["fu_stress"] for r in results)
delta_max = max(r[2]["delta"] for r in results)
fu_defl = delta_max / defl_adm_m if defl_adm_m > 0 else np.inf
design_ok = (fu_stress_max <= 1.0) and (fu_defl <= 1.0)
weight_total = sum(r[2]["weight"] for r in results)


# =====================================================================
# 7. PANEL PRINCIPAL — RESULTADOS
# =====================================================================

st.title("🪟 Cálculo Estructural de Vidrios y Termopaneles")
st.caption("ASTM E1300-24 — Determinación de la resistencia de vidrios en "
           "edificaciones | Proyectos Estructurales EIRL")

# ---- Veredicto técnico ----
if design_ok:
    st.markdown('<div class="verdict-ok">✅ DISEÑO ACEPTABLE — CUMPLE</div>',
                unsafe_allow_html=True)
else:
    motivos = []
    if fu_stress_max > 1.0:
        motivos.append("tensión")
    if fu_defl > 1.0:
        motivos.append("deflexión")
    st.markdown(
        f'<div class="verdict-fail">❌ DISEÑO NO ACEPTABLE — NO CUMPLE '
        f'({" y ".join(motivos)})</div>', unsafe_allow_html=True
    )

col_main, col_fig = st.columns([1.75, 1.0], gap="large")

with col_main:
    st.subheader("Factores de Utilización")
    c1, c2 = st.columns(2)
    c1.metric("FU a flexión", f"{fu_stress_max:.3f}",
              delta="OK" if fu_stress_max <= 1.0 else "EXCEDE",
              delta_color="normal" if fu_stress_max <= 1.0 else "inverse")
    c2.metric("FU a deflexión", f"{fu_defl:.3f}",
              delta="OK" if fu_defl <= 1.0 else "EXCEDE",
              delta_color="normal" if fu_defl <= 1.0 else "inverse")

    st.subheader("Verificación por Lámina")
    for name, lite, res, share in results:
        with st.container(border=True):
            st.markdown(f"**{name} — {lite['construction']} / {lite['treatment']}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("σ máxima", f"{res['sigma'] / 1e6:.2f} MPa",
                      help="Coeficientes de Timoshenko, placa apoyada en 4 bordes.")
            m2.metric("σ admisible", f"{res['sigma_adm'] / 1e6:.2f} MPa",
                      help=f"Base 3 s ajustada por duración de {duration_s:.0f} s.")
            m3.metric("FU flexión", f"{res['fu_stress']:.3f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Δ máxima", f"{res['delta'] * 1000:.2f} mm")
            d2.metric("Δ admisible", f"{defl_adm_mm:.2f} mm")
            d3.metric("Carga en la lámina", f"{res['q_lite'] / 1000:.3f} kPa",
                      help=f"Load Share Factor = {share:.3f}")

            detalle = [
                f"Espesor efectivo a deflexión, h_ef,w = **{lite['h_ef_w'] * 1000:.2f} mm**",
                f"Espesor efectivo a tensión, h_ef,σ = **{lite['h_ef_s'] * 1000:.2f} mm**",
                f"Método de deflexión: *{res['method']}*",
            ]
            if lite["gamma"] is not None:
                detalle.insert(0, f"Coeficiente de transferencia de corte, "
                                  f"Γ = **{lite['gamma']:.4f}**")
            if lite["construction"] == "Laminado" and lite["lsf_strength"] < 1.0:
                detalle.append(f"Laminate Strength Factor = **{lite['lsf_strength']:.2f}**")
            st.markdown("- " + "\n- ".join(detalle))

    # ---- Tabla resumen ----
    st.subheader("Resumen de Cálculo")
    resumen = pd.DataFrame([{
        "Lámina": name,
        "Construcción": lite["construction"],
        "Tratamiento": lite["treatment"].split(" (")[0],
        "h_ef,w [mm]": round(lite["h_ef_w"] * 1000, 2),
        "h_ef,σ [mm]": round(lite["h_ef_s"] * 1000, 2),
        "LSF [-]": round(share, 3),
        "q lámina [kPa]": round(res["q_lite"] / 1000, 3),
        "σ [MPa]": round(res["sigma"] / 1e6, 2),
        "σ adm [MPa]": round(res["sigma_adm"] / 1e6, 2),
        "FU σ": round(res["fu_stress"], 3),
        "Δ [mm]": round(res["delta"] * 1000, 2),
        "FU Δ": round(res["delta"] / defl_adm_m, 3),
    } for name, lite, res, share in results])
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar resumen (CSV)",
        data=resumen.to_csv(index=False).encode("utf-8-sig"),
        file_name="verificacion_vidrio_astm_e1300.csv",
        mime="text/csv",
    )

with col_fig:
    st.subheader("Sección Transversal")
    fig = draw_section(cfg1, cfg2 if is_igu else None, air_gap_mm)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Datos generales**")
    st.markdown(
        f"""
        - Dimensiones: **{a_long_mm:.0f} × {b_short_mm:.0f} mm**
        - Relación de aspecto a/b: **{aspect_ratio:.2f}**
        - Presión de viento: **{q_kpa:.2f} kPa** ({duration_label})
        - Apoyo: **4 bordes simplemente apoyados**
        - Peso del vidrio: **{weight_total:.1f} kg**
        - E = **{E_GLASS / 1e6:.0f} MPa** · ν = **{NU_GLASS}**
        """
    )

# ---- Advertencias de validez ----
if aspect_ratio > 5.0:
    st.warning(
        f"Relación de aspecto a/b = {aspect_ratio:.2f} > 5.0. El ajuste no lineal "
        "de deflexión de ASTM E1300 está fuera de su dominio de validez; se aplicó "
        "la solución lineal de Timoshenko."
    )

st.markdown(
    """
    <div class="info-box">
    <b>Bases y limitaciones del cálculo</b><br>
    • <b>Tensión:</b> coeficientes de Timoshenko (teoría lineal de placas,
      simplemente apoyada en 4 bordes). Es conservadora respecto de la
      formulación no lineal, ya que no considera el efecto membrana.<br>
    • <b>Deflexión:</b> formulación de gran deformación de ASTM E1300 (Anexo X),
      válida para 1 ≤ a/b ≤ 5 y q(ab)²/Et⁴ &gt; 1.<br>
    • <b>Tensión admisible:</b> valores base a 3 s con p<sub>b</sub> = 8/1000
      (23.3 / 46.6 / 93.1 MPa), ajustados por (3/d)<sup>1/16</sup>. Esta app
      <u>no reproduce las cartas NFL</u> de la norma; el chequeo se realiza por
      tensión admisible, criterio equivalente pero no idéntico.<br>
    • <b>Laminado:</b> modelo de Wölfel-Bennison, dos láminas simétricas.
      El módulo G del interlayer depende de temperatura y duración: verificar
      contra la tabla del fabricante.<br>
    • <b>Termopanel:</b> reparto de carga por rigidez relativa (t<sub>ef</sub>³).
      No se incluyen cargas climáticas de la cámara (presión isócora,
      variación de temperatura o de altitud) ni el efecto de la lámina no cargada
      bajo carga de larga duración.<br>
    Los resultados requieren revisión y validación por un ingeniero responsable.
    </div>
    """,
    unsafe_allow_html=True,
)
