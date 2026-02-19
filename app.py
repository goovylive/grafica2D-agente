import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import time
import json
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# --- CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="IA Pilot App - Pro", layout="wide")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 1. ESTADO DE SESIÓN: El "Cerebro" de la App
variables_control = {
    'equation': "x^2 + y^2 - 25",
    'x_min': -10.0, 'x_max': 10.0,
    'y_min': -10.0, 'y_max': 10.0,
    't_val': 5.0,
    'show_plot': False,
    'needs_scroll': False
}

for key, val in variables_control.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 2. EL AGENTE PILOTO (Control Total) ---
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    with st.sidebar:
        st.header("🤖 Agente Autónomo")
        orden = st.text_area("Ordena al piloto:", placeholder="Ej: Grafica un círculo de radio 10 y ajusta los ejes")
        
        if st.button("🚀 Ejecutar Acción Completa", use_container_width=True):
            status = st.status("🧠 IA operando la aplicación...")
            
            prompt = f"""
            Actúas como un ASISTENTE EXPERTO en matemáticas y ploteo de gráficas.
            Tu objetivo es configurar las variables de visualización basándote en la solicitud del usuario: "{orden}".

            Las variables que puedes controlar son:
            - equation (str): La ecuación a graficar en formato Python/SymPy (ej: "x**2 + y**2 - 25", "sin(x)*cos(y)"). IMPORTANTE: Usa sintaxis válida de Python/SymPy.
            - x_min, x_max, y_min, y_max (float): Límites de los ejes.
            - t_val (float): Valor del parámetro tiempo 't' si la ecuación lo usa.
            - show_plot (bool): Debe ser true para graficar.
            - needs_scroll (bool): Debe ser true para desplazar la vista.

            Responde ÚNICAMENTE con un objeto JSON válido que contenga las claves que deseas modificar. NO incluyas markdown, explicaciones ni texto adicional fuera del JSON.

            Ejemplo de respuesta válida:
            {{
                "equation": "x**2 + y**2 - 100",
                "x_min": -15.0,
                "x_max": 15.0,
                "y_min": -15.0,
                "y_max": 15.0,
                "show_plot": true,
                "needs_scroll": true
            }}
            """
            
            try:
                respuesta = model.generate_content(prompt).text
                # Limpiar la respuesta por si el modelo incluye bloques de código
                respuesta_limpia = respuesta.replace('```json', '').replace('```', '').strip()
                
                datos_accion = json.loads(respuesta_limpia)
                
                status.write(f"Interpretación: {datos_accion}")
                
                for key, val in datos_accion.items():
                    if key in variables_control:
                        st.session_state[key] = val
                
                status.update(label="✅ Acción ejecutada", state="complete")
                time.sleep(1) # Dar tiempo para leer el estado
                st.rerun()
                
            except json.JSONDecodeError as e:
                st.error(f"Error interpretando la respuesta del agente (JSON inválido): {e}")
                st.code(respuesta) # Mostrar la respuesta cruda para depuración
            except Exception as e:
                st.error(f"Error general del agente: {e}")

# --- 3. LÓGICA DE GRAFICACIÓN (Corregida para evitar deformación) ---
def generar_grafica(eq_str, x_lims, y_lims, t_val):
    try:
        if not eq_str or not eq_str.strip():
            return ValueError("La ecuación no puede estar vacía.")
            
        clean_eq = eq_str.replace("^", "**")
        x, y, t_sym = sp.symbols('x y t')
        expr = parse_expr(clean_eq, transformations=(standard_transformations + (implicit_multiplication_application,) + (convert_xor,)))
        
        f_num = sp.lambdify((x, y, t_sym), expr, modules=['numpy'])
        
        x_vals = np.linspace(x_lims[0], x_lims[1], 400) # Más resolución
        y_vals = np.linspace(y_lims[0], y_lims[1], 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_num(X, Y, t_val)
        
        # Configuración de la figura
        fig, ax = plt.subplots(figsize=(8, 8)) # Tamaño cuadrado
        
        # --- CORRECCIÓN DE DEFORMACIÓN ---
        ax.set_aspect('equal', adjustable='box') # Fuerza el aspecto 1:1
        
        # Dibujar curva
        contour = ax.contour(x_vals, y_vals, Z, levels=[0], colors='#00d4ff', linewidths=2)
        
        # --- ENUMERACIÓN Y COORDENADAS ---
        ax.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
        ax.axhline(0, color='white', lw=1.5)
        ax.axvline(0, color='white', lw=1.5)
        
        # Etiquetas y ticks visibles
        ax.set_xlabel("Eje X", color="white")
        ax.set_ylabel("Eje Y", color="white")
        ax.tick_params(colors='white', which='both') # Números en blanco
        
        # Títulos
        plt.title(f"Ecuación: {eq_str}", color="white", pad=20)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        return fig
    except Exception as e:
        return e

# --- 4. INTERFAZ ---
st.title("📈 Graficadora de Precisión con IA")

st.session_state.equation = st.text_input("Ecuación (x, y, t):", value=st.session_state.equation)

c1, c2 = st.columns(2)
with c1:
    st.session_state.x_min = st.number_input("X Mín", value=float(st.session_state.x_min))
    st.session_state.x_max = st.number_input("X Máx", value=float(st.session_state.x_max))
with c2:
    st.session_state.y_min = st.number_input("Y Mín", value=float(st.session_state.y_min))
    st.session_state.y_max = st.number_input("Y Máx", value=float(st.session_state.y_max))

st.session_state.t_val = st.slider("Valor de t", -100.0, 100.0, value=float(st.session_state.t_val))

if st.button("📊 Graficar", use_container_width=True):
    st.session_state.show_plot = True

# --- 5. RENDERIZADO Y SCROLL ---
st.markdown("<div id='plot_zone'></div>", unsafe_allow_html=True)

if st.session_state.show_plot:
    res = generar_grafica(
        st.session_state.equation, 
        (st.session_state.x_min, st.session_state.x_max), 
        (st.session_state.y_min, st.session_state.y_max), 
        st.session_state.t_val
    )
    
    if isinstance(res, Exception):
        st.error(f"Error: {res}")
    else:
        st.pyplot(res)
        
        if st.session_state.needs_scroll:
            st.components.v1.html("<script>window.parent.document.getElementById('plot_zone').scrollIntoView({behavior: 'smooth'});</script>", height=0)
            st.session_state.needs_scroll = False