# streamlit run /home/martingut27/WineRecommendation/app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from models.synthetic_user import SyntheticUserSimulator
from src.utils import utils as ut

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Proyecto Final - Data Science",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsubtitle-header {
        font-size: 1.2rem;
        color: #555; /* gris oscuro para contraste */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-style: italic; /* opcional, le da un toque distinto */
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .conclusion-box {
        background-color: #f0f8e8;
        border: 2px solid #2ca02c;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar para navegación
    st.sidebar.markdown("## 🧭 Navegación")
    page = st.sidebar.selectbox(
        "Selecciona una sección:",
        [
            "🏠 Introducción",
            "📊 Análisis Exploratorio (EDA)",
            "🎭 Modelado de Usuario Sintético",
            "🤖 Modelado y Optimización de ML",
            "📈 Selección del Mejor Modelo",
            "📋 Conclusiones",
            "⚡ Aplicación Interactiva"
        ]
    )
    
    if page == "🏠 Introducción":
        show_introduction()
    elif page == "📊 Análisis Exploratorio (EDA)":
        show_eda()
    elif page == "🎭 Modelado de Usuario Sintético":
        show_synthetic_user_modeling()
    elif page == "🤖 Modelado y Optimización de ML":
        show_modeling()
    elif page == "📈 Selección del Mejor Modelo":
        show_model_selection()
    elif page == "📋 Conclusiones":
        show_conclusions()
    elif page == "⚡ Aplicación Interactiva":
        show_interactive_app()





#=====================================#
# INTRODUCCIÓN
#=====================================#





def show_introduction():
    st.markdown('<div class="main-header">📊 Proyecto Final - Diplomatura en Data Science</div>', unsafe_allow_html=True)
    
    # Información del proyecto
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📋 Información del Proyecto</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <h3>🍷 Wine Sommelier</h3>
        
        <h3>📚 Dplo. Data Science con Python y R</h3>
        
        <h3>👩‍🏫 Participantes</h3>
        <p><strong>Alumno:</strong> Martín Augusto Gutiérrez</p>
        <p><strong>Tutores:</strong> Ignacio Urteaga, Julio Paredes, Anahí Romo Santagostino</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Resumen Ejecutivo</div>', unsafe_allow_html=True)
        
        # Aquí puedes agregar métricas clave de tu proyecto
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Dataset", "[Tamaño]", "[N° registros]")
            st.metric("Variables", "[N° features]", "[N° target classes]")
        with col_metric2:
            st.metric("Mejor Modelo", "RandomForestClassifier", "[Ganancia %]")
            st.metric("ROI Estimado", "[Valor]", "[Incremento %]")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Objetivos e Hipótesis
    st.markdown('<div class="section-header">🎯 Objetivos e Hipótesis</div>', unsafe_allow_html=True)
    
    col_obj, col_hip = st.columns(2)
    
    with col_obj:
        st.markdown("""
        ### 📌 Objetivos
        
        **Objetivo General:**
        - Generar una aplicación analítica que permita obtener recomendaciones de vinos personalizadas para acompañar comidas según las preferencias del usuario.
        
        **Objetivos Específicos:**
        1. Permitir al usuario seleccionar un comida y preferencias de sabor, precio, uvas y obtener al menos 3 recomendaciones.
        2. Evaluar al menos 2 modelos de ML para la recomendación y mejorar su performance a través de técnicas de finetunning.
        3. Generar una ganancia a la empresa de $X por usuario tras la aplicacion del Wine Sommelier.
        """)
    
    with col_hip:
        st.markdown("""
        ### 🔬 Hipótesis
        
        **Hipótesis Principal:**
        - Existe una mayor probabilidad de que a un usuario le guste un vino si este cumple con sus demandas de sabor y calidad.
        
        **Hipótesis Secundarias:**
        1. El usuario valora vinos con buen rating percibido por otros usuarios.
        2. El usuario valora vinos que tengan sabores similares a los que demanda.
        3. Un usuario al que se le recomienda un vino que le gusta acaba comprando más y generando más rentabilidad para la bodega.
        """)
    
    # Metodología
    st.markdown('<div class="section-header">⚙️ Metodología</div>', unsafe_allow_html=True)
    
    methodology_cols = st.columns(4)
    
    with methodology_cols[0]:
        st.markdown("""
        #### 📥 1. Recolección
        - Fuente: www.vivino.com
        - Web Scrapping
        - Procesamiento inicial (ETL)
        """)
    
    with methodology_cols[1]:
        st.markdown("""
        #### 🔍 2. Exploración (EDA)
        - Análisis de Nulos
        - Análisis descriptivos
        - Análisis de relaciones
        - Otros análisis avanzados
        """)
    
    with methodology_cols[2]:
        st.markdown("""
        #### 🤖 3. Modelado
        - Usuario Sintético + Simulación
        - Feature Engineering + Feature Selection
        - Algoritmos + Finetunning
        """)
    
    with methodology_cols[3]:
        st.markdown("""
        #### 💰 4. Evaluación
        - Análisis de performance de modelo
        - Análisis económico entre modelos
        - Selección por rentabilidad
        - Implementación
        """)





#=====================================#
# ANÁLISIS EXPLORATORIO (EDA)
#=====================================#






def show_eda():
    st.markdown('<div class="section-header">📊 Análisis Exploratorio de Datos (EDA)</div>', unsafe_allow_html=True)
    
    # Aquí cargarías tus datos reales
    wines_tra = pd.read_csv("src/data/transformed/wines_transformed.csv")
    
    # Información general del dataset
    st.markdown('<div class="subsection-header">📋 Información General del Dataset</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Registros ", f"{len(wines_tra):,}")
    with col2:
        st.metric("Variables", f"{len(wines_tra.columns)}")
    with col3:
        st.metric("Variables Numéricas", f"{len(wines_tra.select_dtypes(include=[np.number]).columns)}")
    with col4:
        st.metric("Variables Categóricas", f"{len(wines_tra.select_dtypes(include=['object']).columns)}")

    # Muestra del dataset con scroll
    st.markdown('<div class="subsubtitle-header">🔍 Vista previa de los datos</div>', unsafe_allow_html=True)

    st.dataframe(
        wines_tra.head(50),  # muestra 50 filas, podés ajustar
        height=300,      # alto fijo -> scroll vertical
        use_container_width=True  # ocupa el ancho y mete scroll horizontal si hace falta
)

    # --------------------------- #
    # Análisis de Datos Faltantes #
    # --------------------------- #

    st.markdown('<div class="subsection-header">👻 Análisis de Datos Faltantes</div>', unsafe_allow_html=True)
    
    # Gráfico interactivo de datos faltantes
    missing_data = (
        pd.DataFrame(wines_tra.isna().sum(), columns=["nulls"])
        .sort_values("nulls", ascending=True)
    )

    # Filtrar solo las columnas con nulls
    missing_data_filtered = missing_data[missing_data["nulls"] > 0]

    if not missing_data_filtered.empty:
        fig_missing = px.bar(
            x=missing_data_filtered["nulls"],
            y=missing_data_filtered.index,
            title="Datos Faltantes por Variable",
            labels={'x': 'Cantidad de Datos Faltantes', 'y': 'Variables'},
            orientation="h",
            color_discrete_sequence=["purple"]
        )
        fig_missing.update_layout(
            height=40 * len(missing_data_filtered),  # alto dinámico
            margin=dict(l=150, r=50, t=50, b=50),
            yaxis=dict(title_standoff=20)  # separa el label "Variables" del eje
        )

        # Insertar el gráfico con scroll vertical en un área fija
        components.html(
            fig_missing.to_html(full_html=False, include_plotlyjs='cdn'),
            height=400,  # alto fijo del contenedor
            width=1000,
            scrolling=True
        )
    else:
        st.success("✅ No se encontraron datos faltantes en el dataset")

    st.markdown('<div class="subsubtitle-header">Gestión de Faltantes</div>', unsafe_allow_html=True)
    st.markdown("""
    - `alcohol y tastes`: tomamos el promedio de alcohol o taste (body, sweetness, tannins, acidity) por uva para imputar el campo.
    - `rating_qty`: hay bastantes vinos con muy pocos ratings. En ese caso, podríamos hacer una compleción de nulos con el mínimo.
    - `pairing`: hay varios vinos que no tienen maridajes. Deberíamos eliminarlos, ya que no son útiles en nuestra app.
    - `grapes`: si no tiene ninguna uva, podemos dropear los vinos.
    - `price`: los vinos con precio nulo son dudosos en vivino. Lo más seguro creo que es dropearlo (además, son pocos).
    - `year`: tomamos la mediana de año del vino por bodega (asumimos que la misma bodega tiene vinos cercanos a la media en el catálogo).
    """)

    
    # -------------------------- #
    # Análisis de Distribuciones #
    # -------------------------- #

    st.markdown('<div class="subsection-header">📈 Análisis de Distribuciones</div>', unsafe_allow_html=True)
    st.markdown("*(variables one-hot y outliers quitados para una mejor visualización)*")
    
    # Carga de datos limpios
    wines_clean = pd.read_csv("src/data/transformed/wines_clean.csv")

    # Selección interactiva de variable
    numeric_columns = wines_clean.select_dtypes(include=[np.number]).columns.tolist()
    valid_numeric = numeric_columns[:22] # Deja variables one-hot fuera
    selected_var = st.selectbox("Selecciona una variable numérica:", valid_numeric)
    wines_clean_num = ut.manage_outlier_IQR(data=wines_clean[valid_numeric], func="remove")

    if selected_var:
        col_hist, col_box = st.columns(2)
        
        with col_hist:
            sel_x = wines_clean_num[selected_var].dropna()

            fig_hist = go.Figure()

            # Histograma
            fig_hist.add_trace(go.Histogram(
                x=sel_x,
                nbinsx=50,
                marker_color="salmon",
                opacity=0.7,
                name="Frecuencia"
            ))

            # KDE (curva suavizada)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(sel_x)
            x_range = np.linspace(sel_x.min(), sel_x.max(), 200)
            fig_hist.add_trace(go.Scatter(
                x=x_range,
                y=kde(x_range) * len(sel_x) * (sel_x.max()-sel_x.min())/50,  # escalar al histograma
                mode="lines",
                line=dict(color="purple", width=2),
                name="Densidad"
            ))

            fig_hist.update_layout(
                title=f"Distribución de {selected_var}",
                bargap=0.05
            )

            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_box:
            fig_box = px.box(
                wines_clean_num, 
                y=selected_var,
                title=f"Box Plot de {selected_var}",
                color_discrete_sequence=["purple"]
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Análisis de correlaciones
    st.markdown('<div class="subsection-header">🔗 Matriz de Correlaciones</div>', unsafe_allow_html=True)
    
    # Características del Vino
    wine_features = ["rating", "year", "rating_qty", "price", "body", "tannins", "sweetness", "acidity", "alcohol"]

    # Uvas
    grapes_cols = pd.read_csv("src/data/processed/aux/grapes.csv")
    grapes_cols = grapes_cols["grapes"].to_list()
    grapes_cols.insert(0,"rating")
    # Elimino columnas sin correlación (NaN)
    grapes_with_corr = wines_clean[grapes_cols].loc[:, (wines_clean[grapes_cols].sum() > 0)].columns.to_list()
    
    # Notas de Sabor
    notes_cols = pd.read_csv("src/data/processed/aux/notes.csv")
    notes_cols = notes_cols["notes"].to_list()
    notes_cols.insert(0,"rating")

    # Maridajes
    pairings_cols = pd.read_csv("src/data/processed/aux/pairings.csv")
    pairings_cols = pairings_cols["pairings"].to_list()
    pairings_cols.insert(0, "rating")

    # Regiones
    region_cols = pd.read_csv("src/data/processed/aux/region.csv")
    region_cols = region_cols["region"].to_list()
    region_cols.insert(0,"rating")


    corr_selector_map = {
        "Características del Vino": wine_features,
        "Uvas": grapes_with_corr,
        "Notas de Sabor": notes_cols,
        "Maridajes": pairings_cols,
        "Regiones": region_cols
    }

    # Matriz de correlación
    sel_group_col, sel_meth_col = st.columns(2)

    with sel_group_col:
        selected_group = st.selectbox("Selecciona un grupo de análisis:", corr_selector_map.keys())
    
    with sel_meth_col:
        selected_method = st.selectbox("Selecciona un grupo de análisis:", ["pearson", "spearman"])
    
    correlation_matrix = wines_clean[corr_selector_map[selected_group]].corr(method=selected_method)
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        title=f"Matriz de Correlaciones",
        color_continuous_scale="burgyl"
    )
    
    # Insertar el gráfico con scroll vertical en un área fija
    fig_corr.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Aumentar tamaño del texto de las anotaciones (los números)
    fig_corr.update_traces(textfont={"size":12})  # tamaño de los números

    st.plotly_chart(fig_corr, use_container_width=True)

    # Correlación con variable objetivo
    corr_matrix_rating = correlation_matrix["rating"].sort_values(ascending=False)

    fig_corr_rating = px.bar(
        x=corr_matrix_rating.index,
        y=corr_matrix_rating.values,
        title=f"Correlación de {selected_method} con Rating",
        labels={'x': '', 'y': 'Correlación con rating'},
        orientation="v",
        color=corr_matrix_rating.values,  # <- esto define el color de cada barra
        color_continuous_scale="burgyl"
    )

    fig_corr_rating.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Insertar el gráfico con scroll vertical en un área fija
    components.html(
        fig_corr_rating.to_html(full_html=False, include_plotlyjs='cdn'),
        height=350,  # alto fijo del contenedor
        width=1000
    )


    # --------------------------- #
    # Co-Ocurrrencia de Variables #
    # --------------------------- #

    st.markdown('<div class="subsection-header">👥 Co-Ocurrencia de Variables</div>', unsafe_allow_html=True)

    coocurrencias_options = [
        "Maridaje + Uvas",
        "Maridaje + Perfil de Sabor",
        "Maridaje + Notas de Sabor",
        "Uvas + Perfil de Sabor",
        "Uvas + Notas de Sabor",
        "Perfil de Sabor + Notas de Sabor"
    ]

    grapes_cols.remove("rating")
    notes_cols.remove("rating")
    pairings_cols.remove("rating")
    region_cols.remove("rating")

    selected_coocu = st.selectbox("Seleccionar coocurrencia de variables:", coocurrencias_options)

    # Agrupación de maridajes para coocurrencias
    pairing_groups = {
        "Aperitivos y Chatarra": [
            "any junk food will do",
            "aperitif",
            "appetizers and snacks"
        ],
        "Quesos": [
            "blue cheese",
            "goat's milk cheese",
            "mature and hard cheese",
            "mild and soft cheese"
        ],
        "Carnes Rojas": [
            "beef",
            "lamb",
            "veal",
            "game (deer, venison)"
        ],
        "Carnes Blancas": [
            "pork",
            "poultry"
        ],
        "Pescados y Mariscos": [
            "lean fish",
            "rich fish (salmon, tuna etc)",
            "shellfish"
        ],
        "Vegetales y Hongos": [
            "mushrooms",
            "vegetarian"
        ],
        "Pasta": [
            "pasta"
        ],
        "Embutidos": [
            "cured meat"
        ],
        "Comida Picante": [
            "spicy food"
        ]
    }

    # Sabores para coocurrencia
    tastes_dict = {
        "body": "body_scld",
        "tannins": "tannins_scld",
        "sweetness": "sweetness_scld",
        "acidity": "acidity_scld"
    }

    tastes_plain = list(tastes_dict.keys())
    tastes_scld = list(tastes_dict.values())

    # Scaler
    mm_scaler = MinMaxScaler()


    # <-- Maridaje + Uvas -->
    if selected_coocu == "Maridaje + Uvas":

        # Calcular el % de vinos de cada pairing que caen en cada uva
        grouped_pairing_grape_count = pd.DataFrame(0, index=list(pairing_groups.keys()), columns=grapes_cols)

        for pairing_group, columns in pairing_groups.items():
            wine_in_group = wines_clean[wines_clean[columns].any(axis=1) == 1]
            wines_by_group = wine_in_group[grapes_cols].sum()
            grouped_pairing_grape_count.loc[pairing_group] = wines_by_group

        # Eliminar uvas sin vinos
        wines_by_grape = grouped_pairing_grape_count[grapes_cols].sum()
        grapes_with_no_wines = wines_by_grape.index[wines_by_grape==0]
        grouped_pairing_grape_count = grouped_pairing_grape_count.drop(columns=grapes_with_no_wines)

        # Calcular porcentajes
        grouped_pairing_grape_perc = grouped_pairing_grape_count.div(
            grouped_pairing_grape_count.sum(axis=1),
            axis=0
        )

        # Crear heatmap interactivo con Plotly
        fig_coocu = px.imshow(
            grouped_pairing_grape_perc,
            text_auto=".2f",
            height=600,
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Uvas", y="Grupos de Maridaje", color="Porcentaje de Vinos"),
            title="🍷 Distribución de Uvas agrupado por Grupo de Maridaje"
        )
    
    
    # <-- Maridaje + Perfil de Sabor -->
    elif selected_coocu == "Maridaje + Perfil de Sabor":

        mean_taste_by_pairing = pd.DataFrame(0, index=pairings_cols, columns=tastes_scld)
        mean_taste_by_pairing[tastes_scld] = mean_taste_by_pairing[tastes_scld].astype(float)

        columns_for_scaling = tastes_plain + pairings_cols
        scaled_wines = wines_clean[columns_for_scaling].copy()

        scaled_wines[tastes_scld] = mm_scaler.fit_transform(scaled_wines[tastes_plain])

        grouped_pairing_taste_count = pd.DataFrame(0, index=list(pairing_groups.keys()), columns=tastes_scld)
        grouped_pairing_taste_count[tastes_scld] = grouped_pairing_taste_count[tastes_scld].astype(float)

        for pairing_group, columns in pairing_groups.items():
            wine_in_group = scaled_wines[scaled_wines[columns].any(axis=1)==1]
            mean_taste_by_group = wine_in_group[tastes_scld].mean()
            grouped_pairing_taste_count.loc[pairing_group] = mean_taste_by_group
        
        fig_coocu = px.imshow(
            grouped_pairing_taste_count,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Perfil de Sabor", y="Grupos de Maridaje", color="Promedio"),
            title="🍷 Perfil de Sabor promedio agrupado por Grupo de Maridaje"
        )


    # <-- Maridaje + Notas de Sabor -->
    elif selected_coocu == "Maridaje + Notas de Sabor":

        grouped_pairing_note_count = pd.DataFrame(0, index=list(pairing_groups.keys()), columns=notes_cols)
        grouped_pairing_note_count[notes_cols] = grouped_pairing_note_count[notes_cols].astype(float)

        for pairing_group, pairings in pairing_groups.items():
            wine_with_pairing_group = wines_clean[wines_clean[pairings].any(axis=1)==1]
            mean_by_note = wine_with_pairing_group[notes_cols].mean()
            grouped_pairing_note_count.loc[pairing_group] = mean_by_note

        fig_coocu = px.imshow(
            grouped_pairing_note_count,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Nota de Sabor", y="Grupos de Maridaje", color="Promedio"),
            title="🍷 Notas de Sabor promedio agrupado por Grupo de Maridaje"
        )


    # <-- Uvas + Perfil de Sabor -->
    elif selected_coocu == "Uvas + Perfil de Sabor":

        mean_taste_by_grape = pd.DataFrame(0, index=grapes_cols, columns=tastes_scld)
        mean_taste_by_grape[tastes_scld] = mean_taste_by_grape[tastes_scld].astype(float)

        columns_for_scaling = tastes_plain + grapes_cols
        scaled_wines = wines_clean[columns_for_scaling].copy()

        scaled_wines[tastes_scld] = mm_scaler.fit_transform(scaled_wines[tastes_plain])

        for grape in grapes_cols:
            wines_with_grape = scaled_wines[scaled_wines[grape]==1]
            mean_grape_tastes = wines_with_grape[tastes_scld].mean()
            mean_taste_by_grape.loc[grape] = mean_grape_tastes

        grapes_with_wines = mean_taste_by_grape.index[mean_taste_by_grape.sum(axis=1) > 0]
        mean_taste_by_grape = mean_taste_by_grape.loc[grapes_with_wines]

        fig_coocu = px.imshow(
            mean_taste_by_grape,
            text_auto=".2f",
            height=600,
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Perfil de Sabor", y="Uvas", color="Promedio"),
            title="🍷 Perfil de Sabor promedio agrupado por Uvas"
        )


    # <-- Uvas + Notas de Sabor -->
    elif selected_coocu == "Uvas + Notas de Sabor":

        grape_note_mean = pd.DataFrame(0, index=grapes_cols, columns=notes_cols)
        grape_note_mean[notes_cols] = grape_note_mean[notes_cols].astype(float)

        for grape in grapes_cols:
            wines_with_grape = wines_clean[wines_clean[grape]==1]
            mean_notes_in_grape = wines_with_grape[notes_cols].mean()
            grape_note_mean.loc[grape] = mean_notes_in_grape

        grapes_with_wines = grape_note_mean.index[grape_note_mean.sum(axis=1) > 0]
        grape_note_mean = grape_note_mean.loc[grapes_with_wines]

        fig_coocu = px.imshow(
            grape_note_mean,
            text_auto=".2f",
            height=700,
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Notas de Sabor", y="Uvas", color="Promedio"),
            title="🍷 Notas de Sabor promedio agrupado por Uvas"
        )

    
    # <-- Perfil de Sabor + Notas de Sabor -->
    elif selected_coocu == "Perfil de Sabor + Notas de Sabor":

        columns_for_scaling = tastes_plain + notes_cols
        scaled_wines = wines_clean[columns_for_scaling].copy()

        scaled_wines[tastes_scld] = mm_scaler.fit_transform(scaled_wines[tastes_plain])

        complete_corr_matrix = scaled_wines.corr()
        sliced_corr_matrix = complete_corr_matrix.loc[notes_cols, tastes_scld]

        fig_coocu = px.imshow(
            sliced_corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="burgyl",
            labels=dict(x="Perfil de Sabor", y="Notas de Sabor", color="Correlación"),
            title="🍷 Correlación entre Perfil de Sabor y Notas de Sabor"
        )

    if fig_coocu is None:
        st.error("❌ No seleccionaste una opción válida.")
    else:
        st.plotly_chart(fig_coocu, use_container_width=True)



    # --------------------------- #
    # Análisis por Taste + Rating #
    # --------------------------- #

    st.markdown('<div class="subsection-header">⭐ Relación entre Sabores y Rating (objetivo)</div>', unsafe_allow_html=True)

    # Selector en streamlit
    selected_taste = st.selectbox("Elegí un perfil de sabor para analizar:", tastes_dict.keys())

    # Crear bins
    hist_taste_rating_df = wines_clean.copy()
    bin_edges = np.histogram_bin_edges(hist_taste_rating_df[selected_taste], bins=20)
    hist_taste_rating_df[selected_taste + "_bin"] = pd.cut(
        hist_taste_rating_df[selected_taste], bins=bin_edges, include_lowest=True
    )

    # Calcular promedio de ratings por bin
    avg_ratings = hist_taste_rating_df.groupby(selected_taste + "_bin", observed=False)["rating"].mean()

    # Crear un DataFrame para plotly
    plot_df = pd.DataFrame({
        selected_taste: [interval.mid for interval in avg_ratings.index],
        "avg_rating": avg_ratings.values,
        "count": hist_taste_rating_df.groupby(selected_taste + "_bin").size().values
    })

    labels = plot_df["avg_rating"].round(2)

    # Calcular mínimo y máximo de los labels
    rango = labels.max() - labels.min()
    min_label = labels.min() + (rango * 0.3)  # Mínimo + 20% del rango
    max_label = labels.max() 

    # Crear gráfico interactivo
    fig = px.bar(
        plot_df,
        x=selected_taste,
        y="count",
        color="avg_rating",
        color_continuous_scale="burgyl",
        range_color=[min_label, max_label],
        text=plot_df["avg_rating"].round(2),
        labels={selected_taste: selected_taste.capitalize(), "count": "# Vinos", "avg_rating": "Rating Promedio"},
        title=f"✨🍇 Distribución de {selected_taste.capitalize()} y Rating Promedio"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(height=500, margin=dict(l=50, r=50, t=50, b=50))

    st.plotly_chart(fig, use_container_width=True)


    # -------------------- #
    # Key Insights del EDA #
    # -------------------- #
    
    # Insights clave
    st.markdown('<div class="subsection-header">💡 Insights Clave del EDA</div>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    

    
    with insights_col1:
        st.markdown("""
        <div class="highlight-box">
        <h4>🔍 Hallazgos Principales:</h4>
        <ol>
        <li><strong>Perfil óptimo:</strong> Alto cuerpo (>0.6), tánicos moderados (~0.4), dulzor promedio (0.2) y acidez balanceada (0.3-0.6) correlacionan con mejor rating.</li>
        <li><strong>Maridajes específicos:</strong> Cada maridaje tiene combinaciones distintivas de perfiles de sabor bien definidas.</li>
        <li><strong>Precio-calidad:</strong> Correlación positiva entre precio alto y rating superior del vino.</li>
        <li><strong>Uvas y notas:</strong> Correlación baja general, pero ciertas uvas específicas (Malbec, Cabernet Franc) y notas (oaky) sí impactan la calidad.</li>
        <li><strong>Popularidad no lineal:</strong> La correlación popularidad-calidad es curvilínea - beneficiosa hasta cierto punto, luego se estabiliza.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="highlight-box">
        <h4>⚠️ Consideraciones para el Modelado:</h4>
        <ol>
        <li><strong>Target continuo:</strong> Usar rating como proxy de probabilidad de gusto del usuario basado en rangos de sabor personalizados y precio.</li>
        <li><strong>Simplificación por maridaje:</strong> Preconfigurar rangos de sabor según el maridaje seleccionado para garantizar coherencia.</li>
        <li><strong>Factores principales:</strong> Priorizar precio (con tope), sabores coherentes con maridaje y uvas principales para determinar probabilidad de gusto (like).</li>
        <li><strong>Factores secundarios:</strong> Uvas y notas como variables de apoyo, no determinantes.</li>
        <li><strong>Efecto social:</strong> Incorporar popularidad como factor de influencia - vinos más consumidos generan mayor probabilidad de adopción.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)





#=====================================#
# LÓGICA DE USUARIO SINTÉTICO
#=====================================#






def show_synthetic_user_modeling():
    st.markdown('<div class="section-header">🎭 Modelado de Usuario Sintético</div>', unsafe_allow_html=True)
    
    # Introducción
    st.markdown("""
    <div class="highlight-box">
    <p>🎯 <strong>Objetivo:</strong> Crear un modelo de comportamiento de consumidor que simule decisiones reales de compra de vinos basado en preferencias de maridaje, presupuesto, sabor y uvas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Lógica de Decisión del Usuario
    st.markdown('<div class="subsection-header">🧠 Lógica de Decisión del Usuario</div>', unsafe_allow_html=True)
    
    logic_col1, logic_col2 = st.columns([1, 1])
    
    with logic_col1:
        st.markdown("""
        #### 🍽️ Flujo de Decisión:
        1. **Selección de Comida/Maridaje**
           - Usuario elige tipo de comida
           - Sistema ajusta rangos se sabor disponibles según ingredientes de la comida
        
        2. **Filtros de Preferencia**
           - Rango de precios personalizado
           - Preferencias de uvas 
           - Perfil de sabor del vino
        
        3. **Evaluación de Opciones**
           - Scoring basado en fit de vino con preferencias personalizadas del usuario
           - Adición de factor aleatorio
                    
        4. **Like y Recompra**
            - Simulación de like según puntaje de fit
            - Simulación de recompra según experiencia anterior
        """)
    
    with logic_col2:
        # Diagrama de flujo simplificado
        st.markdown("""
        ```
        🍽️ Comida Elegida
              ↓
        🎯 Ajuste de Rangos de Sabor
              ↓
        💰 Filtro Presupuesto
              ↓
        🍇 Filtro Uvas Preferidas
              ↓
        🍷 Elección de Perfil de Sabor
              ↓
        📊 Scoring de Opciones
              ↓
        🎲 Decisión Final
              ↓
        👍👎 Like/Dislike
              ↓
        🔄 Abandono o Recompra
        ```
        """)
    
    # Elección de comidas
    meals_df = pd.read_excel(ut.get_project_file_path("src", "data", "raw", "meals", "Meals.xlsx"))

    # === SECCIÓN ELECCIÓN DE COMIDA ===
    st.markdown("---")
    st.markdown('<div class="subsection-header">🍽️ Selección de Comida</div>', unsafe_allow_html=True)

    st.markdown("### 🥘 Elige tu Comida Favorita")
    st.markdown("Selecciona una comida para ver sus características de maridaje y encontrar el vino perfecto.")

    # Crear selector de comida
    available_meals = sorted(meals_df.iloc[:, 0].unique())  # Asumiendo que la primera columna tiene las comidas
    selected_meal = st.selectbox(
        "🍴 Selecciona una comida:",
        options=available_meals,
        index=0,
        help="Elige la comida para la cual quieres encontrar el vino ideal"
    )

    if selected_meal:
        # Obtener información de la comida seleccionada
        meal_info = meals_df[meals_df.iloc[:, 0] == selected_meal].iloc[0]
        
        # Obtener el main_pairing (asumiendo que está en la columna 1)
        main_pairing = meal_info.iloc[1] if len(meal_info) > 1 else "No especificado"
        
        # Obtener todas las columnas de pairings (one-hot encoded)
        # Asumiendo que las columnas de pairings empiezan desde la columna 2
        pairing_columns = meals_df.columns[2:]  # Columnas one-hot de pairings
        active_pairings = [col for col in pairing_columns if meal_info[col] == 1]
        
        # === VISUALIZACIÓN DE LA COMIDA ===
        food_col1, food_col2 = st.columns([1, 2])
        
        with food_col1:
            # Generar imagen placeholder basada en el nombre de la comida
            # En una implementación real, podrías usar URLs reales de imágenes
            food_emoji_map = {
            # Carnes rojas y derivados
                'paté de hígado': '🥩', 'carne con salsa de champiñones': '🥩🍄', 'empanadas de carne': '🥟',
                'guiso de arroz con carne': '🍲', 'hamburguesa con queso': '🍔', 'locro': '🍲',
                'lomo a la mostaza': '🥩', 'milanesa de carne': '🍖', 'pastel de papa': '🥧',
                'tacos de res': '🌮', 'ciervo estofado': '🦌', 'cordero asado': '🐑',
                'milanesa de ternera': '🥩', "asado": "🍖",
                
                # Quesos y lácteos
                'roquefort con nueces': '🧀🥜', 'tabla de quesos y embutidos': '🍱', 'fondue de queso': '🧀',
                
                # Pescados y mariscos
                'trucha / salmón a la mantequilla': '🐟', 'salmón a la plancha': '🍣', 'sushi de atún': '🍣',
                'ceviche mixto': '🍤', 'gambas al ajillo': '🦐', 'ostras frescas': '🦪', 'paella de mariscos': '🥘',
                
                # Pasta y risotto
                'pasta con salsa roquefort': '🍝', 'risotto de champiñones': '🍚', 'lasagna de jamón y queso': '🍝',
                'lasagna de verduras': '🍝', 'pasta con salsa mixta': '🍝', 'pasta con salsa de tomate': '🍝',
                
                # Cerdo
                'burrito de cerdo': '🌯', 'chucrut con salchichas': '🌭', 'costillas de cerdo bbq': '🍖',
                'matambre a la pizza': '🍕', 'tacos de cerdo': '🌮',
                
                # Pollo
                'empanadas de pollo': '🥟', 'milanesa de pollo': '🍗', 'pechuga de pollo rellena': '🍗',
                'pollo al curry': '🍛', 'pollo al horno con papas': '🍗',
                
                # Jamón y embutidos
                'snacks con dips': '🧀', 'sandwich de jamón y queso': '🥪', 'empanadas de jamón y queso': '🥟',
                
                # Vegetarianos
                'empanadas de verdura': '🥟', 'ensalada': '🥗', 'guiso de lentejas': '🍲',
                'pizza napolitana': '🍕', 'sopa de cebolla': '🍲', 'sopa de calabaza': '🍲',
                'tarta de vegetales y queso': '🥧', 'tortilla de papa': '🥚'
            }
            
            # Buscar emoji apropiado
            food_emoji = '🍽️'  # emoji por defecto
            for key, emoji in food_emoji_map.items():
                if key.lower() in selected_meal.lower():
                    food_emoji = emoji
                    break
            
            # Card de la comida con imagen grande
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 30px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                margin: 20px 0;
            ">
                <div style="font-size: 4em; margin-bottom: 15px;">{food_emoji}</div>
                <h3 style="margin: 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    {selected_meal.title()}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        with food_col2:
            # Información detallada de la comida
            st.markdown("#### 🎯 Características de Maridaje")
            
            # Main pairing destacado
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border-left: 5px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            ">
                <h5 style="margin: 0; color: #28a745;">🌟 Ingrediente Principal</h5>
                <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0; color: #333;">
                    {main_pairing}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar todos los pairings activos
            if active_pairings:
                st.markdown("#### 🧩 Ingredientes y Sabores")
                
                # Crear tags para los pairings usando st.columns para mejor control
                tags_per_row = 3
                for i in range(0, len(active_pairings), tags_per_row):
                    cols = st.columns(tags_per_row)
                    row_pairings = active_pairings[i:i + tags_per_row]
                    
                    for j, pairing in enumerate(row_pairings):
                        clean_pairing = pairing.replace('_', ' ').title()
                        with cols[j]:
                            st.markdown(f"""
                            <div style="
                                background-color: #e3f2fd;
                                color: #1976d2;
                                padding: 8px 12px;
                                border-radius: 20px;
                                font-size: 0.9em;
                                font-weight: 500;
                                margin: 5px 0;
                                text-align: center;
                                border: 1px solid #bbdefb;
                            ">{clean_pairing}</div>
                            """, unsafe_allow_html=True)
                    
                    # Llenar columnas vacías si es necesario
                    for k in range(len(row_pairings), tags_per_row):
                        with cols[k]:
                            st.empty()
            else:
                st.warning("No se encontraron ingredientes específicos para esta comida.")
        
        # === ESTADÍSTICAS DE LA COMIDA ===
        st.markdown("#### 📊 Análisis de Complejidad")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            total_ingredients = len(active_pairings)
            st.metric(
                "🧩 Total Ingredientes", 
                total_ingredients,
                help="Cantidad total de ingredientes/sabores identificados"
            )
        
        with stats_col2:
            # Clasificar complejidad basada en número de ingredientes
            if total_ingredients <= 1:
                complexity = "Simple"
                complexity_color = "🟢"
            elif total_ingredients <= 2:
                complexity = "Moderada"
                complexity_color = "🟡"
            else:
                complexity = "Compleja"
                complexity_color = "🔴"
                
            st.metric(
                "⚖️ Complejidad",
                f"{complexity_color} {complexity}",
                help="Complejidad basada en la cantidad de sabores"
            )
        
        with stats_col3:
            # Categoría de comida (basada en palabras clave)
            category_map = {
                "🥩 Carnes Rojas": ["beef", "lamb", "veal", "game (deer, venison)"],
                "🐷 Cerdo": ["pork"],
                "🍗 Aves": ["poultry"],
                "🐟 Pescados": ["lean fish", "rich fish (salmon, tuna etc)"],
                "🦐 Mariscos": ["shellfish"],
                "🧀 Quesos": ["blue cheese", "goat's milk cheese", "mature and hard cheese", "mild and soft cheese"],
                "🥓 Embutidos": ["cured meat"],
                "🍄 Vegetales": ["mushrooms", "vegetarian"],
                "🍝 Pasta": ["pasta"],
                "🌶️ Picante": ["spicy food"],
                "🥨 Aperitivos": ["aperitif", "appetizers and snacks", "any junk food will do"]
            }
            
            # Buscar categoría basada en el main_pairing
            food_category = "🍽️ General"
            main_pairing_lower = main_pairing.lower()
            
            for category, keywords in category_map.items():
                if any(keyword.lower() in main_pairing_lower for keyword in keywords):
                    food_category = category
                    break
            
            st.metric(
                "🏷️ Categoría",
                food_category,
                help=f"Categoría basada en el ingrediente principal: {main_pairing}"
            )

    # 2. Curvas de Decisión
    st.markdown('---')
    st.markdown('<div class="subsection-header">📈 Curvas de Decisión</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="highlight-box">
        <p>El usuario aplica filtros, selecciona un vino y decide si le gustó o no en base a una lógica dinámica determinada por diferentes variables: 1) Precio Calidad, 2) Similitud con sus gustos, 3) Popularidad del Vino, 4) Maridaje con su Comida.</p>
        </div>
        """, unsafe_allow_html=True)

    wines_clean = pd.read_csv(ut.get_project_file_path("src", "data", "transformed", "wines_clean.csv"))
    meals_df = pd.read_excel(ut.get_project_file_path("src", "data", "raw", "meals", "Meals.xlsx"))
    synthetic_user = SyntheticUserSimulator(wine_df=wines_clean, meals_df=meals_df)
    default_weights = synthetic_user.weights

    # Selector de visualización de curvas de decisión
    viz_type = st.selectbox(
        "🎯 Selecciona el tipo de análisis:",
        ["Precio-Calidad", "Popularidad del Vino", "Similitud con Gustos del Usuario", "Efecto Main Pairing", "Score Conjunto 3D"]
    )

    # Crear tabs para mejor organización
    tab1, tab2 = st.tabs(["📊 Visualización", "⚙️ Parámetros"])

    with tab2:

        if viz_type == "Precio-Calidad":
            # Parámetros específicos para este análisis
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            test_user_max_price_enabled = st.checkbox("Establecer Precio Máximo", value=False)
            with analysis_col1:
                st.markdown("###### Características del Vino Analizado")
                fixed_rating = st.slider("Rating Del Vino Analizado (para análisis de precio)", 1.0, 5.0, 4.0, 0.1)
                fixed_price = st.slider("Precio Del Vino Analizado (para análisis de rating)", 5, 100, 25)
            with analysis_col2:
                st.markdown("###### Parámetros del Usuario")
                test_user_min_price = st.slider("Precio Mínimo Usuario ($)", 5, 50, 15)
                test_user_max_price = st.slider("Precio Máximo Usuario ($)", test_user_min_price+5, 200, 50) if test_user_max_price_enabled else None
            with analysis_col3:
                st.markdown("###### Hiperparámetros de Curvas")
                test_rating_threshold = st.slider("Umbral de Rating", 2.0, 4.5, 4.0, 0.1)
                test_price_sensitivity = st.slider("Sensibilidad al Precio", 0.1, 2.0, 0.5, 0.1)

        elif viz_type == "Similitud con Gustos del Usuario":
            # Parámetros específicos para este análisis
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

            with analysis_col1:
                q_low = st.slider("Rango de Preferencia - Mínimo", 0.0, 1.0, 0.5, 0.01)
            with analysis_col2:
                q_high = st.slider("Rango de Preferencia - Máximo", q_low, 1.0, 0.6, 0.01)
            with analysis_col3:
                slope_factor = st.slider("Pendiente de la Curva", 1, 50, 13)

        elif viz_type == "Score Conjunto 3D":
            # Parámetros específicos para este análisis
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            test_user_max_price_enabled = st.checkbox("Establecer Precio Máximo", value=False)
            has_main_pairing_3d = st.checkbox("Con Main Pairing", value=True)
            with analysis_col1:
                st.markdown("###### Características del Vino Analizado")
                fixed_popularity = st.slider("Popularidad Fija (qty ratings)", 0, 1000, 300, 50)
                fixed_similarity = st.slider("Similitud Fija", 0.0, 1.0, 0.8, 0.1)
            with analysis_col2:
                st.markdown("###### Parámetros del Usuario")
                test_user_min_price = st.slider("Precio Mínimo Usuario ($)", 5, 50, 15)
                test_user_max_price = st.slider("Precio Máximo Usuario ($)", test_user_min_price+5, 200, 50) if test_user_max_price_enabled else None
            with analysis_col3:
                st.markdown("###### Hiperparámetros de Curvas")
                test_rating_threshold = st.slider("Umbral de Rating", 2.0, 4.5, 4.0, 0.1)
                test_price_sensitivity = st.slider("Sensibilidad al Precio", 0.1, 2.0, 0.5, 0.1)

      

    with tab1:
        if viz_type == "Precio-Calidad":
            st.markdown("#### 💰 Análisis de Precio-Calidad")
            
            selection_col1, selection_col2, selection_col3 = st.columns(3)
            with selection_col1:
                st.info(f"""
                        **Características del Vino Analizado**
                        - Rating={fixed_rating}
                        - Precio del Vino={fixed_price}
                        """)
            with selection_col2:
                st.info(f"""
                        **Parámetros del Usuario**
                        - Precio Mínimo={test_user_min_price}
                        - Precio Máximo={test_user_max_price}
                        """)
            with selection_col3:
                st.info(f"""
                        **Hiperparámetros de Curvas:**
                        - Umbral Rating={test_rating_threshold}
                        - Sensibilidad al Precio={test_price_sensitivity}
                        """)

            # Generar datos para gráficos
            prices = np.linspace(5, 100, 200)
            ratings = np.linspace(1.0, 5.0, 200)
            
            # Análisis de variación de precio
            price_scores = []
            price_quality_components = []
            price_price_components = []
            
            for p in prices:
                score = synthetic_user._calc_qual_price_score(
                    fixed_rating, p, test_user_min_price, test_user_max_price, 
                    test_rating_threshold, test_price_sensitivity
                )
                price_scores.append(score)
            
            # Análisis de variación de rating
            rating_scores = []
            
            for r in ratings:
                score = synthetic_user._calc_qual_price_score(
                    r, fixed_price, test_user_min_price, test_user_max_price,
                    test_rating_threshold, test_price_sensitivity
                )
                rating_scores.append(score)
            
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    f'Score vs Precio (Rating={fixed_rating})',
                    f'Score vs Rating (Precio=${fixed_price})'
                ]
            )
            
            # Gráfico de precio
            fig.add_trace(
                go.Scatter(x=prices, y=price_scores, name="Score Precio-Calidad", 
                        line=dict(color='#70284A', width=3)),
                row=1, col=1
            )
            
            # Gráfico de rating
            fig.add_trace(
                go.Scatter(x=ratings, y=rating_scores, name="Score Precio-Calidad", 
                        line=dict(color='#DC7178', width=3), showlegend=False),
                row=2, col=1
            )
            
            # Líneas de referencia
            if test_user_max_price_enabled and test_user_max_price:
                fig.add_vline(x=test_user_max_price, line_dash="dash", line_color="gray", 
                            annotation_text="Precio Máximo", row=1, col=1)
            
            fig.add_vline(x=test_user_min_price, line_dash="dash", line_color="gray", 
                        annotation_text="Precio Mínimo", row=1, col=1)
            fig.add_vline(x=test_rating_threshold, line_dash="dash", line_color="gray", 
                        annotation_text="Umbral Rating", row=2, col=1)
            
            fig.update_layout(height=700, title_text="Análisis de Componente Precio-Calidad")
            fig.update_xaxes(title_text="Precio ($)", row=1, col=1)
            fig.update_xaxes(title_text="Rating", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Score", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculadora interactiva
            st.markdown("##### 🧮 Calculadora Interactiva")
            calc_col1, calc_col2, calc_col3 = st.columns(3)
            
            with calc_col1:
                calc_rating = st.number_input("Rating del Vino", 1.0, 5.0, 4.0, 0.1)
                calc_price = st.number_input("Precio del Vino ($)", 1, 200, 25)
            
            calc_score = synthetic_user._calc_qual_price_score(
                calc_rating, calc_price, test_user_min_price, test_user_max_price,
                test_rating_threshold, test_price_sensitivity
            )
            
            with calc_col2:
                st.metric("Score Precio-Calidad", f"{calc_score:.3f}")
                
            with calc_col3:
                st.metric("Contribución al Score Total", f"{calc_score * default_weights['price_quality']:.3f}")

        elif viz_type == "Popularidad del Vino":
            st.markdown("#### ⭐ Análisis de Popularidad")
            
            # Generar datos de popularidad
            rating_quantities = np.arange(0, 1200, 10)
            popularity_scores = [synthetic_user._calc_popularity_score(qty) for qty in rating_quantities]
            
            # Gráfico principal
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rating_quantities, 
                y=popularity_scores, 
                mode='lines',
                line=dict(color='purple', width=3),
                name='Score de Popularidad',
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ))
            
            # Líneas de umbral
            thresholds = [40, 70, 100, 150, 200, 300, 400, 500, 700, 1000]
            threshold_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            for thresh, score in zip(thresholds, threshold_scores):
                fig.add_vline(x=thresh, line_dash="dot", line_color="gray", opacity=0.5)
                fig.add_annotation(x=thresh, y=score + 0.05, text=f"{thresh}", 
                                showarrow=False, font=dict(size=10))
            
            fig.update_layout(
                title='Score de Popularidad vs Cantidad de Ratings',
                xaxis_title='Número de Ratings',
                yaxis_title='Score de Popularidad',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculadora interactiva
            st.markdown("##### 🧮 Calculadora de Popularidad")
            
            calc_col1, calc_col2, calc_col3 = st.columns(3)
            
            with calc_col1:
                selected_qty = st.slider("Número de Ratings", 0, 1200, 300)
                
            calculated_score = synthetic_user._calc_popularity_score(selected_qty)
            
            with calc_col2:
                st.metric("Score de Popularidad", f"{calculated_score:.1f}")
                
            with calc_col3:
                st.metric("Contribución al Score Total", f"{calculated_score * default_weights['wine_popularity']:.3f}")

        elif viz_type == "Similitud con Gustos del Usuario":
            st.markdown("#### 🎯 Análisis de Similitud Fuzzy")
            
            selection_col1, selection_col2, selection_col3 = st.columns(3)
            with selection_col1:
                st.info(f"**Rango Preferido:** [{q_low:.2f}, {q_high:.2f}]")
            with selection_col2:
                st.info(f"**Ancho del Rango:** {q_high - q_low:.2f}")
            with selection_col3:
                st.info(f"**Pendiente de la Curva:** {slope_factor}")


            # Generar datos de similitud
            values = np.linspace(0, 1.01, 100)
            similarities = [synthetic_user._fuzzy_smooth_similarity(v, q_low, q_high, slope_factor) 
                        for v in values]
            
            # Gráfico principal
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=values, 
                y=similarities, 
                mode='lines',
                line=dict(color='#70284A', width=3),
                name='Score de Similitud',
                fill='tozeroy',
                fillcolor='#fbe6c5'
            ))
            
            # Destacar rango preferido
            fig.add_vrect(
                x0=q_low, x1=q_high,
                fillcolor="green", opacity=0.2,
                layer="below", line_width=0
            )
            
            # Líneas de referencia
            fig.add_vline(x=q_low, line_dash="dash", line_color="green", 
                        annotation_text="Mín Preferencia")
            fig.add_vline(x=q_high, line_dash="dash", line_color="green", 
                        annotation_text="Máx Preferencia")
            
            fig.update_layout(
                title='Score de Similitud vs Valor de Sabor del Vino',
                xaxis_title='Valor de Sabor del Vino',
                yaxis_title='Score de Similitud (fuzzy distance)',
                height=500,
                yaxis_range=[0, 1.1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculadora interactiva
            st.markdown("##### 🧮 Calculadora de Similitud")
            
            calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
            
            with calc_col1:
                test_value = st.slider("Valor de Prueba", 0.0, 1.0, 0.5, 0.01)
                
            similarity_score = synthetic_user._fuzzy_smooth_similarity(test_value, q_low, q_high, slope_factor)
            distance_from_range = max(0, q_low - test_value, test_value - q_high)
            in_range = q_low <= test_value <= q_high
            
            with calc_col2:
                st.metric("Score de Similitud", f"{similarity_score:.3f}")
                
            with calc_col3:
                st.metric("Distancia del Rango", f"{distance_from_range:.1f}")
                
            with calc_col4:
                st.metric("¿En Rango?", "✅ Sí" if in_range else "❌ No")

        elif viz_type == "Efecto Main Pairing":
            st.markdown("#### 🍽️ Análisis de Efecto Main Pairing")
            
            st.info("**Lógica:** Si el vino tiene como maridaje el ingrediente principal de la comida, suma +0.1 al score final.")
            
            # Simulación de scores base
            base_scores = np.linspace(0.1, 0.9, 100)
            scores_with_pairing = np.clip(base_scores + 0.1, 0, 1)
            
            # Gráfico principal
            fig = go.Figure()
            
            # Scores sin pairing
            fig.add_trace(go.Scatter(
                x=list(range(len(base_scores))), 
                y=base_scores,
                mode='lines',
                line=dict(color='red', width=2),
                name='Sin Main Pairing'
            ))
            
            # Scores con pairing
            fig.add_trace(go.Scatter(
                x=list(range(len(scores_with_pairing))), 
                y=scores_with_pairing,
                mode='lines',
                line=dict(color='green', width=2),
                name='Con Main Pairing (+0.1)'
            ))
            
            # Área de diferencia
            fig.add_trace(go.Scatter(
                x=list(range(len(base_scores))) + list(range(len(base_scores)))[::-1],
                y=list(base_scores) + list(scores_with_pairing)[::-1],
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Bonus de Pairing',
                showlegend=False
            ))
            
            fig.update_layout(
                title='Efecto de Main Pairing en Scores de Vinos',
                xaxis_title='Muestra de Vino',
                yaxis_title='Score Final',
                height=500,
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculadora de impacto
            st.markdown("##### 🧮 Calculadora de Impacto de Pairing")
            
            calc_col1, calc_col2 = st.columns(2)
            
            with calc_col1:
                base_score = st.slider("Score Base del Vino", 0.0, 1.0, 0.65, 0.01)
                has_pairing = st.checkbox("¿Tiene Main Ingredient Pairing?", value=False)
                
            final_score = min(1.0, base_score + (0.1 if has_pairing else 0.0))
            improvement = (0.1 if has_pairing and base_score <= 0.9 else max(0, 1.0 - base_score)) if has_pairing else 0
            
            with calc_col2:
                st.metric("Score Final", f"{final_score:.3f}", 
                        f"+{improvement:.3f}" if improvement > 0 else None)
                st.metric("Mejora Relativa", f"{improvement*100:.1f}%")
            
            # Tabla de resumen
            st.markdown("##### 📊 Resumen de Impacto")
            sample_scores = [0.3, 0.5, 0.7, 0.85, 0.95]
            df_summary = pd.DataFrame({
                'Score Base': sample_scores,
                'Con Pairing': [min(1.0, s + 0.1) for s in sample_scores],
                'Ganancia Absoluta': [min(0.1, 1.0 - s) for s in sample_scores],
                'Ganancia Relativa %': [f"{min(0.1, 1.0 - s) / s * 100:.1f}%" for s in sample_scores]
            })
            
            st.dataframe(df_summary, use_container_width=True)

        elif viz_type == "Score Conjunto 3D":
            st.markdown("#### 📈 Visualización de Score Conjunto 3D")
            
            # Crear meshgrid
            price_range = np.linspace(5, min(200, test_user_max_price*1.5), 50)
            rating_range = np.linspace(1.0, 5.0, 50)
            P, R = np.meshgrid(price_range, rating_range)
            Z = np.zeros_like(P)
            
            # Calcular scores para cada combinación
            popularity_score = synthetic_user._calc_popularity_score(fixed_popularity)
            main_pairing_bonus = 1.0 if has_main_pairing_3d else 0.0

            for i in range(len(rating_range)):
                for j in range(len(price_range)):
                    # Score precio-calidad
                    pq_score = synthetic_user._calc_qual_price_score(
                        R[i,j], P[i,j], test_user_min_price, test_user_max_price,
                        test_rating_threshold, test_price_sensitivity
                    )
                    
                    # Score conjunto
                    combined_score = (
                        pq_score * default_weights['price_quality'] +
                        popularity_score * default_weights['wine_popularity'] +
                        fixed_similarity * default_weights['user_similarity'] +
                        main_pairing_bonus * default_weights['main_pairing']
                    )
                    
                    Z[i,j] = min(1.0, combined_score)

            # Información de la selección de parámetros
            selection_col1, selection_col2, selection_col3 = st.columns(3)
            with selection_col1:
                st.info(f"""
                        **Características del Vino Analizado**
                        - Score Popularidad = {popularity_score}
                        - Score Similitud Gustos = {fixed_similarity}
                        - Contiene Maridaje Principal = {"Si" if has_main_pairing_3d else "No"}
                        """)
            with selection_col2:
                st.info(f"""
                        **Parámetros del Usuario**
                        - Precio Mínimo = {test_user_min_price}
                        - Precio Máximo = {test_user_max_price}
                        """)
            with selection_col3:
                st.info(f"""
                        **Hiperparámetros de Curvas:**
                        - Umbral Rating = {test_rating_threshold}
                        - Sensibilidad al Precio = {test_price_sensitivity}
                        """)
            
            # Crear gráfico 3D
            fig_3d = go.Figure(data=[go.Surface(
                x=P, y=R, z=Z, 
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score Conjunto")
            )])
            
            fig_3d.update_layout(
                title=f'Score Conjunto 3D<br><sub>Popularidad={popularity_score:.1f}, Similitud={fixed_similarity:.1f}, Main Pairing={"Sí" if has_main_pairing_3d else "No"}</sub>',
                scene=dict(
                    xaxis_title='Precio ($)',
                    yaxis_title='Rating',
                    zaxis_title='Score Conjunto',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Métricas del punto óptimo
            max_score_idx = np.unravel_index(np.argmax(Z), Z.shape)
            optimal_price = P[max_score_idx]
            optimal_rating = R[max_score_idx]
            optimal_score = Z[max_score_idx]
            
            st.markdown("##### 🏆 Punto Óptimo Encontrado")
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            
            with opt_col1:
                st.metric("Precio Óptimo", f"${optimal_price:.0f}")
                
            with opt_col2:
                st.metric("Rating Óptimo", f"{optimal_rating:.1f}")
                
            with opt_col3:
                st.metric("Score Máximo", f"{optimal_score:.3f}")

    
    # Mostrar los pesos por defecto de la clase
    st.markdown("---")
    st.markdown('<div class="subsection-header">📋 Parámetros de la Simulación</div>', unsafe_allow_html=True)

    # === PESOS DEL SISTEMA ===
    st.markdown("#### 🎛️ Configuración del Motor de Recomendación")

    # Crear cards más atractivos para los pesos
    weight_container = st.container()
    with weight_container:
        # Crear una grid de 2x2 para los componentes
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")
        
        components_info = [
            {
                "name": "precio_calidad",
                "display_name": "🏆 Precio-Calidad",
                "weight": default_weights["price_quality"],
                "description": "Equilibrio entre calidad percibida y precio accesible",
                "details": "Incremento decreciente para calidad + función binomial en precio",
                "col": row1_col1
            },
            {
                "name": "similitud_usuario", 
                "display_name": "🎯 Similitud Usuario",
                "weight": default_weights["user_similarity"],
                "description": "Matching con preferencias personales del usuario",
                "details": "Decaimiento exponencial fuera del rango preferido",
                "col": row1_col2
            },
            {
                "name": "popularidad_vino",
                "display_name": "⭐ Popularidad",
                "weight": default_weights["wine_popularity"],
                "description": "Basado en cantidad de ratings de otros usuarios",
                "details": "Función escalonada",
                "col": row2_col1
            },
            {
                "name": "main_pairing",
                "display_name": "🍽️ Maridaje Principal", 
                "weight": default_weights["main_pairing"],
                "description": "Bonus por compatibilidad con ingrediente principal",
                "details": "Bonus simple de +0.1 cuando aplica",
                "col": row2_col2
            }
        ]
        
        for comp in components_info:
            with comp["col"]:
                # Card personalizado con mejor styling
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-left: 4px solid #007bff;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0 0 10px 0; color: #212529;">{comp["display_name"]}</h4>
                    <div style="
                        font-size: 2.5em; 
                        font-weight: bold; 
                        color: #007bff;
                        margin: 10px 0;
                    ">{comp["weight"]:.0%}</div>
                    <p style="margin: 0 0 8px 0; color: #6c757d; font-size: 0.9em;">
                        {comp["description"]}
                    </p>
                    <small style="color: #868e96; font-style: italic;">
                        {comp["details"]}
                    </small>
                </div>
                """, unsafe_allow_html=True)

    # === SIMULACIÓN DE RECOMPRAS ===
    st.markdown("---")
    st.markdown('<div class="subsection-header">🔄 Simulación del Ciclo de Vida del Usuario</div>', unsafe_allow_html=True)

    st.markdown("""
    #### 🧪 Modelo de Recompras de Usuario

    El sistema simula el comportamiento real de recompra de usuarios basado en su experiencia previa:
    """)

    tab3, tab4 = st.tabs(["🔄 Flujo", "⚙️ Parámetros"])

    
    with tab4:
        # Controles interactivos para los parámetros de simulación
        sim_col1, sim_col2 = st.columns([1, 1])

        with sim_col1:
            st.markdown("##### ⚙️ Parámetros de Simulación")
            
            avg_days = st.slider(
                "📅 Días promedio entre compras", 
                min_value=7, max_value=60, value=30,
                help="Tiempo promedio que espera un usuario entre compras"
            )
            
            std_days = st.slider(
                "📊 Variabilidad (desviación estándar)", 
                min_value=1, max_value=15, value=5,
                help="Qué tan variable es el tiempo entre compras"
            )
            
            max_repurchases = st.slider(
                "🔄 Máximo de recompras por usuario", 
                min_value=5, max_value=30, value=20,
                help="Límite máximo de recompras que puede hacer un usuario"
            )

        with sim_col2:
            st.markdown("##### 🎯 Tasas de Abandono (Churn)")
            
            liked_churn = st.slider(
                "😊 Churn si le gustó el vino", 
                min_value=0.000, max_value=0.200, value=0.008, step=0.001,
                help="Probabilidad de abandonar después de una experiencia positiva"
            )
            
            disliked_churn = st.slider(
                "😞 Churn si NO le gustó", 
                min_value=0.2, max_value=1.0, value=0.65, step=0.05,
                help="Probabilidad de abandonar después de una experiencia negativa"
            )
            
            # Mostrar la diferencia como insight
            churn_difference = disliked_churn - liked_churn
            st.markdown(f"""
            **💡 Insight:** Los usuarios insatisfechos tienen 
            **{churn_difference:.1%}** más probabilidad de abandonar
            """)

    with tab3:
        # === VISUALIZACIÓN DEL PROCESO ===
        st.markdown("##### 🔄 Flujo del Proceso de Simulación")

        # Crear un diagrama de flujo visual
        process_steps = [
            "🎯 Usuario hace primera compra",
            "⏱️ Tiempo de espera (distribución normal)",
            "🤔 ¿Le gustó la experiencia anterior?",
            "🎲 Evaluación de churn probabilístico",
            "🛒 Nueva compra o 👋 Abandono"
        ]

        cols = st.columns(len(process_steps))
        for i, (step, col) in enumerate(zip(process_steps, cols)):
            with col:
                # Diferentes colores para diferentes tipos de pasos
                if "?" in step:
                    color = "#ffc107"  # amarillo para decisiones
                elif "Abandono" in step:
                    color = "#dc3545"  # rojo para abandono
                elif "Nueva compra" in step:
                    color = "#28a745"  # verde para éxito
                else:
                    color = "#007bff"  # azul para procesos
                    
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 5px;
                    font-weight: bold;
                    font-size: 0.9em;
                    min-height: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    {step}
                </div>
                """, unsafe_allow_html=True)
                
                # Agregar flechas entre pasos (excepto el último)
                if i < len(process_steps) - 1:
                    st.markdown(
                        '<div style="text-align: center; font-size: 1.5em; margin: 5px;">⬇️</div>', 
                        unsafe_allow_html=True
                    )

    # === SIMULACIÓN INTERACTIVA ===
    st.markdown("---")
    st.markdown("#### 📊 Simulación Interactiva")

    # Simular un ejemplo con los parámetros actuales
    if st.button("🎲 Ejecutar Simulación de Ejemplo", type="primary"):
        
        # Simular serie temporal de decisiones
        days = []
        decisions = []
        current_day = 0
        likes_history = []
        purchase_count = 0
        
        for purchase in range(max_repurchases):
            # Simular si le gustó (70% probabilidad)
            liked = np.random.random() < 0.7
            likes_history.append(liked)
            
            # Decidir si continúa basado en experiencia
            churn_prob = liked_churn if liked else disliked_churn
            continues = np.random.random() > churn_prob
            
            days.append(current_day)
            decisions.append("Continúa" if continues else "Abandona")
            purchase_count += 1
            
            if not continues:
                break
                
            # Calcular próximo día de compra
            days_to_wait = max(1, int(np.random.normal(avg_days, std_days)))
            current_day += days_to_wait
        
        # Crear visualización con múltiples trazas
        fig = go.Figure()
        
        # Preparar datos para visualización
        purchase_numbers = list(range(len(days)))

        # Línea conectando todas las compras
        fig.add_trace(go.Scatter(
            x=days,
            y=purchase_numbers,
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            name="Trayectoria",
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Separar puntos por estado (continúa vs abandona)
        continuing_indices = [i for i, d in enumerate(decisions) if d == "Continúa"]
        abandoning_indices = [i for i, d in enumerate(decisions) if d == "Abandona"]
        
        # Traza para usuarios que continúan
        if continuing_indices:
            continuing_days = [days[i] for i in continuing_indices]
            continuing_purchases = [purchase_numbers[i] for i in continuing_indices]
            continuing_likes = [likes_history[i] for i in continuing_indices]
            
            # Colores basados en si le gustó: verde oscuro (gustó), verde claro (no gustó)
            continuing_colors = ['#28a745' if liked else '#90EE90' for liked in continuing_likes]
            continuing_symbols = ['circle' if liked else 'triangle-up' for liked in continuing_likes]
            continuing_text = [f"Compra {i+1}<br>Le gustó: {'Sí' if likes_history[i] else 'No'}<br>Continúa ✅" 
                            for i in continuing_indices]
            
            fig.add_trace(go.Scatter(
                x=continuing_days,
                y=continuing_purchases,
                mode='markers',
                marker=dict(
                    size=15, 
                    color=continuing_colors,
                    symbol=continuing_symbols,
                    line=dict(width=2, color='darkgreen')
                ),
                text=continuing_text,
                textposition="middle right",
                name="Usuario Continúa",
                hovertemplate='<b>%{text}</b><br>Día: %{x}<br>Compra #: %{y}<extra></extra>'
            ))
        
        # Traza para usuarios que abandonan
        if abandoning_indices:
            abandoning_days = [days[i] for i in abandoning_indices]
            abandoning_purchases = [purchase_numbers[i] for i in abandoning_indices]
            abandoning_likes = [likes_history[i] for i in abandoning_indices]
            
            # Colores basados en si le gustó: rojo oscuro (gustó), rojo claro (no gustó)
            abandoning_colors = ['#dc3545' if liked else '#FFB6C1' for liked in abandoning_likes]
            abandoning_symbols = ['circle' if liked else 'triangle-up' for liked in abandoning_likes]
            abandoning_text = [f"Compra {i+1}<br>Le gustó: {'Sí' if likes_history[i] else 'No'}<br>Abandona ❌" 
                            for i in abandoning_indices]
            
            fig.add_trace(go.Scatter(
                x=abandoning_days,
                y=abandoning_purchases,
                mode='markers',
                marker=dict(
                    size=18, 
                    color=abandoning_colors,
                    symbol=abandoning_symbols,
                    line=dict(width=3, color='darkred')
                ),
                text=abandoning_text,
                textposition="middle right",
                name="Usuario Abandona",
                hovertemplate='<b>%{text}</b><br>Día: %{x}<br>Compra #: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="📈 Simulación de Comportamiento de Usuario",
            xaxis_title="Días desde Primera Compra",
            yaxis_title="Número de Compra",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Leyenda explicativa
        st.markdown("""
        **📖 Leyenda del Gráfico:**
        - 🟢 **Círculo Verde**: Le gustó el vino y continúa
        - 🔼 **Triángulo Verde**: NO le gustó pero continúa
        - 🔴 **Círculo Rojo**: Le gustó el vino pero abandona
        - 🔺 **Triángulo Rojo**: NO le gustó y abandona
        """)
        
        # Mostrar estadísticas de la simulación
        total_purchases = len(days)
        abandoned = any("Abandona" in d for d in decisions)
        liked_count = sum(likes_history)
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("🛒 Total de Compras", total_purchases)
        
        with stats_col2:
            if total_purchases > 1:
                avg_time_between = np.mean(np.diff(days))
                st.metric("⏱️ Promedio entre compras", f"{avg_time_between:.1f} días")
            else:
                st.metric("⏱️ Promedio entre compras", "N/A")
        
        with stats_col3:
            satisfaction_rate = (liked_count / total_purchases) * 100 if total_purchases > 0 else 0
            st.metric("😊 Tasa de Satisfacción", f"{satisfaction_rate:.1f}%")

    # === EJEMPLO DE DATOS GENERADOS ===
    st.markdown("---")
    st.markdown('<div class="subsection-header">🎲 Ejemplos de Datos de la Simulación</div>', unsafe_allow_html=True)
    users_data = pd.read_pickle(ut.get_project_file_path("src", "data", "synthetic", "simulation_13.pkl"))
    users_data = users_data.dropna()
    users_data = users_data.head()
    st.dataframe(users_data)




def show_modeling():
    st.markdown('<div class="section-header">🤖 Modelado y Optimización</div>', unsafe_allow_html=True)
    
    # Feature Engineering
    st.markdown('<div class="subsection-header">⚙️ Feature Engineering</div>', unsafe_allow_html=True)
    
    feature_eng_col1, feature_eng_col2 = st.columns(2)
    
    with feature_eng_col1:
        st.markdown("""
        #### 🔧 Transformaciones Aplicadas:
        - [Transformación 1 que aplicaste]
        - [Transformación 2 que aplicaste]
        - [Transformación 3 que aplicaste]
        """)
    
    with feature_eng_col2:
        st.markdown("""
        #### 📊 Variables Creadas:
        - [Variable nueva 1]
        - [Variable nueva 2]
        - [Variable nueva 3]
        """)
    
    # Selección de Features
    st.markdown('<div class="subsection-header">🎯 Selección de Features</div>', unsafe_allow_html=True)
    
    # Aquí mostrarías los resultados de tu feature selection
    feature_importance_data = {
        'Feature': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
        'Importance': [0.25, 0.22, 0.18, 0.15, 0.10],
        'Selected': [True, True, True, False, False]
    }
    
    fig_features = px.bar(
        feature_importance_data,
        x='Importance',
        y='Feature',
        color='Selected',
        orientation='h',
        title="Importancia de Features y Selección"
    )
    st.plotly_chart(fig_features, use_container_width=True)
    
    # Modelos implementados
    st.markdown('<div class="subsection-header">🤖 Modelos Implementados</div>', unsafe_allow_html=True)
    
    model_tabs = st.tabs(["🌳 Random Forest", "📈 Logistic Regression", "🔄 Modelo 3", "⚡ Modelo 4"])
    
    with model_tabs[0]:
        st.markdown("#### Random Forest Classifier")
        
        col_rf1, col_rf2 = st.columns(2)
        
        with col_rf1:
            st.markdown("""
            **Hiperparámetros Optimizados:**
            - n_estimators: [valor]
            - max_depth: [valor]
            - min_samples_split: [valor]
            - min_samples_leaf: [valor]
            """)
        
        with col_rf2:
            # Simulación de curva de validación
            param_range = [10, 50, 100, 200, 300]
            train_scores = [0.85, 0.88, 0.90, 0.89, 0.88]
            val_scores = [0.82, 0.85, 0.87, 0.85, 0.83]
            
            fig_validation = go.Figure()
            fig_validation.add_trace(go.Scatter(
                x=param_range, y=train_scores,
                mode='lines+markers', name='Train Score'
            ))
            fig_validation.add_trace(go.Scatter(
                x=param_range, y=val_scores,
                mode='lines+markers', name='Validation Score'
            ))
            fig_validation.update_layout(title="Curva de Validación - n_estimators")
            st.plotly_chart(fig_validation, use_container_width=True)
    
    with model_tabs[1]:
        st.markdown("#### Logistic Regression")
        
        col_lr1, col_lr2 = st.columns(2)
        
        with col_lr1:
            st.markdown("""
            **Hiperparámetros Optimizados:**
            - C: [valor]
            - penalty: [valor]
            - solver: [valor]
            - max_iter: [valor]
            """)
        
        with col_lr2:
            # Aquí pondrías tus gráficos de regularización
            st.info("Gráfico de regularización y coeficientes")
    
    with model_tabs[2]:
        st.markdown("#### [Nombre de tu tercer modelo]")
        # Repite la estructura para tus otros modelos
    
    with model_tabs[3]:
        st.markdown("#### [Nombre de tu cuarto modelo (opcional)]")
        # Si tienes un cuarto modelo
    
    # Cross-Validation
    st.markdown('<div class="subsection-header">🔄 Validación Cruzada</div>', unsafe_allow_html=True)
    
    # Ejemplo de resultados de CV
    cv_results = {
        'Model': ['Random Forest', 'Logistic Regression', 'Modelo 3'],
        'CV_Mean': [0.85, 0.82, 0.80],
        'CV_Std': [0.03, 0.04, 0.05],
        'Train_Time': [2.3, 0.5, 1.8]
    }
    
    cv_df = pd.DataFrame(cv_results)
    
    fig_cv = px.bar(
        cv_df, x='Model', y='CV_Mean', 
        error_y='CV_Std',
        title="Resultados de Validación Cruzada",
        labels={'CV_Mean': 'Score Promedio', 'Model': 'Modelo'}
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    
    # Métricas de performance
    st.markdown('<div class="subsection-header">📊 Comparación de Métricas</div>', unsafe_allow_html=True)
    
    metrics_data = {
        'Modelo': ['Random Forest', 'Logistic Regression', 'Modelo 3'],
        'Precision': [0.85, 0.82, 0.80],
        'Recall': [0.88, 0.85, 0.83],
        'F1-Score': [0.86, 0.83, 0.81],
        'ROC-AUC': [0.91, 0.88, 0.85]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Gráfico radar para comparar métricas
    fig_radar = go.Figure()
    
    for i, model in enumerate(metrics_df['Modelo']):
        fig_radar.add_trace(go.Scatterpolar(
            r=[metrics_df.iloc[i]['Precision'], metrics_df.iloc[i]['Recall'], 
               metrics_df.iloc[i]['F1-Score'], metrics_df.iloc[i]['ROC-AUC']],
            theta=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(title="Comparación de Métricas por Modelo")
    st.plotly_chart(fig_radar, use_container_width=True)

def show_model_selection():
    st.markdown('<div class="section-header">📈 Selección del Mejor Modelo</div>', unsafe_allow_html=True)
    
    # Análisis económico
    st.markdown('<div class="subsection-header">💰 Análisis Económico</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <h4>📊 Métricas de Negocio Consideradas:</h4>
    <p>En lugar de usar accuracy para la selección del modelo, se evaluaron las siguientes métricas de impacto económico:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Matriz de confusión económica
    economic_col1, economic_col2 = st.columns(2)
    
    with economic_col1:
        st.markdown("""
        #### 💵 Impacto Económico por Predicción:
        - **Verdadero Positivo**: +$[valor] (beneficio por detección correcta)
        - **Verdadero Negativo**: +$[valor] (ahorro por no intervenir incorrectamente)
        - **Falso Positivo**: -$[valor] (costo de intervención innecesaria)
        - **Falso Negativo**: -$[valor] (costo de oportunidad perdida)
        """)
    
    with economic_col2:
        # Matriz de confusión con valores económicos
        confusion_economic = np.array([[150, 25], [30, 200]])  # Ejemplo
        fig_confusion_econ = px.imshow(
            confusion_economic,
            text_auto=True,
            title="Matriz de Confusión - Mejor Modelo",
            labels=dict(x="Predicción", y="Real"),
            x=['Negativo', 'Positivo'],
            y=['Negativo', 'Positivo']
        )
        st.plotly_chart(fig_confusion_econ, use_container_width=True)
    
    # Comparación de rentabilidad
    st.markdown('<div class="subsection-header">📊 Comparación de Rentabilidad por Modelo</div>', unsafe_allow_html=True)
    
    # Datos de ejemplo - REEMPLAZA CON TUS RESULTADOS REALES
    profitability_data = {
        'Modelo': ['Random Forest', 'Logistic Regression', 'Modelo 3'],
        'Ganancia_Total': [125000, 98000, 87000],
        'ROI_Porcentaje': [18.5, 14.2, 12.1],
        'Costo_Implementacion': [15000, 12000, 13000],
        'Ganancia_Neta': [110000, 86000, 74000]
    }
    
    profit_df = pd.DataFrame(profitability_data)
    
    # Gráfico de ganancia por modelo
    fig_profit = px.bar(
        profit_df, 
        x='Modelo', 
        y='Ganancia_Total',
        title="Ganancia Total por Modelo",
        color='ROI_Porcentaje',
        text='Ganancia_Total'
    )
    fig_profit.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # ROI Comparison
    fig_roi = px.scatter(
        profit_df,
        x='Costo_Implementacion',
        y='Ganancia_Neta',
        size='ROI_Porcentaje',
        color='Modelo',
        title="ROI vs Costo de Implementación",
        labels={'Costo_Implementacion': 'Costo de Implementación ($)', 
                'Ganancia_Neta': 'Ganancia Neta ($)'}
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Selección del mejor modelo
    st.markdown('<div class="subsection-header">🏆 Modelo Seleccionado</div>', unsafe_allow_html=True)
    
    best_model_col1, best_model_col2 = st.columns([2, 1])
    
    with best_model_col1:
        st.markdown("""
        <div class="conclusion-box">
        <h3>🥇 Modelo Ganador: [Nombre del Modelo]</h3>
        
        <h4>📈 Justificación de la Selección:</h4>
        <ul>
        <li><strong>Ganancia Total:</strong> $[valor] (mayor que los otros modelos)</li>
        <li><strong>ROI:</strong> [X]% (retorno sobre inversión más alto)</li>
        <li><strong>Estabilidad:</strong> [Descripción de la estabilidad del modelo]</li>
        <li><strong>Implementación:</strong> [Facilidad de implementación]</li>
        </ul>
        
        <h4>⚙️ Parámetros Finales:</h4>
        <ul>
        <li>Parámetro 1: [valor]</li>
        <li>Parámetro 2: [valor]</li>
        <li>Parámetro 3: [valor]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with best_model_col2:
        # Métricas del mejor modelo
        st.markdown("#### 📊 Métricas Clave")
        st.metric("Ganancia Anual", "$125,000", "↑18.5%")
        st.metric("Precisión", "85%", "↑3%")
        st.metric("Recall", "88%", "↑5%")
        st.metric("F1-Score", "86%", "↑4%")
    
    # Curva ROC del mejor modelo
    st.markdown('<div class="subsection-header">📈 Curva ROC - Modelo Seleccionado</div>', unsafe_allow_html=True)
    
    # Simulación de curva ROC - REEMPLAZA CON TUS DATOS REALES
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Curva de ejemplo
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = 0.91)',
        line=dict(width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='red')
    ))
    fig_roc.update_layout(
        title="Curva ROC - Modelo Seleccionado",
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

def show_conclusions():
    st.markdown('<div class="section-header">📋 Conclusiones</div>', unsafe_allow_html=True)
    
    # Resumen ejecutivo
    st.markdown('<div class="subsection-header">📊 Resumen Ejecutivo</div>', unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown("""
        <div class="metric-container">
        <h4>🎯 Objetivos Cumplidos</h4>
        <ul>
        <li>✅ Objetivo 1</li>
        <li>✅ Objetivo 2</li>
        <li>✅ Objetivo 3</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown("""
        <div class="metric-container">
        <h4>📈 Resultados Clave</h4>
        <ul>
        <li>Ganancia: $[valor]</li>
        <li>ROI: [X]%</li>
        <li>Precisión: [X]%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        st.markdown("""
        <div class="metric-container">
        <h4>🔍 Hipótesis</h4>
        <ul>
        <li>✅ Confirmada</li>
        <li>❌ Rechazada</li>
        <li>⚠️ Parcial</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Implicaciones económicas
    st.markdown('<div class="subsection-header">💰 Implicaciones Económicas</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-box">
    <h4>💵 Impacto Económico Proyectado:</h4>
    
    <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
        <div style="text-align: center;">
            <h3 style="color: #2ca02c;">$[Valor]</h3>
            <p>Ganancia Anual Estimada</p>
        </div>
        <div style="text-align: center;">
            <h3 style="color: #1f77b4;">[X]%</h3>
            <p>ROI Proyectado</p>
        </div>
        <div style="text-align: center;">
            <h3 style="color: #ff7f0e;">[X] meses</h3>
            <p>Tiempo de Recuperación</p>
        </div>
    </div>
    
    <h4>📊 Análisis de Sensibilidad:</h4>
    <ul>
    <li><strong>Escenario Optimista:</strong> $[valor] (+[X]% vs base)</li>
    <li><strong>Escenario Base:</strong> $[valor]</li>
    <li><strong>Escenario Pesimista:</strong> $[valor] (-[X]% vs base)</li>
    </ul>
    
    <h4>💡 Factores de Riesgo Económico:</h4>
    <ul>
    <li>[Factor de riesgo 1 y su impacto]</li>
    <li>[Factor de riesgo 2 y su impacto]</li>
    <li>[Factor de riesgo 3 y su impacto]</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Lecciones aprendidas
    st.markdown('<div class="subsection-header">🎓 Lecciones Aprendidas</div>', unsafe_allow_html=True)
    
    lessons_col1, lessons_col2 = st.columns(2)
    
    with lessons_col1:
        st.markdown("""
        <div class="highlight-box">
        <h4>✅ Lo que Funcionó Bien:</h4>
        <ul>
        <li>[Aspecto exitoso 1 de tu proyecto]</li>
        <li>[Aspecto exitoso 2 de tu proyecto]</li>
        <li>[Aspecto exitoso 3 de tu proyecto]</li>
        <li>[Técnica o metodología que dio buenos resultados]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with lessons_col2:
        st.markdown("""
        <div class="highlight-box">
        <h4>⚠️ Desafíos Encontrados:</h4>
        <ul>
        <li>[Desafío 1 y cómo lo resolviste]</li>
        <li>[Desafío 2 y cómo lo resolviste]</li>
        <li>[Limitación técnica encontrada]</li>
        <li>[Problema de datos que enfrentaste]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Áreas de mejora
    st.markdown('<div class="subsection-header">🔧 Áreas de Mejora y Trabajo Futuro</div>', unsafe_allow_html=True)
    
    improvements_tabs = st.tabs(["🔮 Mejoras Técnicas", "📊 Datos y Features", "⚡ Implementación", "🌐 Escalabilidad"])
    
    with improvements_tabs[0]:
        st.markdown("""
        #### 🛠️ Mejoras Técnicas Propuestas:
        
        **Algoritmos:**
        - [Algoritmo avanzado que podrías probar]
        - [Técnica de ensemble que podrías implementar]
        - [Método de deep learning aplicable]
        
        **Optimización:**
        - [Técnica de optimización de hiperparámetros más avanzada]
        - [Método de selección de features más sofisticado]
        - [Técnica de validación más robusta]
        """)
    
    with improvements_tabs[1]:
        st.markdown("""
        #### 📈 Datos y Feature Engineering:
        
        **Nuevas Fuentes de Datos:**
        - [Fuente de datos adicional 1]
        - [Fuente de datos adicional 2]
        - [Datos externos que podrían enriquecer el análisis]
        
        **Feature Engineering Avanzado:**
        - [Feature compleja que podrías crear]
        - [Transformación no lineal a explorar]
        - [Interacciones entre variables a investigar]
        """)
    
    with improvements_tabs[2]:
        st.markdown("""
        #### 🚀 Implementación y Deployment:
        
        **Infraestructura:**
        - [Plataforma de deployment sugerida]
        - [Sistema de monitoreo a implementar]
        - [Pipeline de MLOps a desarrollar]
        
        **Automatización:**
        - [Proceso a automatizar 1]
        - [Proceso a automatizar 2]
        - [Sistema de reentrenamiento automático]
        """)
    
    with improvements_tabs[3]:
        st.markdown("""
        #### 📈 Escalabilidad y Expansión:
        
        **Escalabilidad Técnica:**
        - [Mejora de performance para más datos]
        - [Optimización para tiempo real]
        - [Distribución de procesamiento]
        
        **Expansión de Negocio:**
        - [Aplicación a otros segmentos]
        - [Extensión a otros mercados]
        - [Integración con otros sistemas]
        """)
    
    # Impacto y valor agregado
    st.markdown('<div class="subsection-header">🌟 Impacto y Valor Agregado</div>', unsafe_allow_html=True)
    
    impact_metrics = st.columns(4)
    
    with impact_metrics[0]:
        st.metric(
            "Eficiencia Ganada", 
            "[X]%", 
            "vs proceso manual"
        )
    
    with impact_metrics[1]:
        st.metric(
            "Reducción de Errores", 
            "[X]%", 
            "vs método anterior"
        )
    
    with impact_metrics[2]:
        st.metric(
            "Tiempo Ahorrado", 
            "[X] horas/semana", 
            "por automatización"
        )
    
    with impact_metrics[3]:
        st.metric(
            "Satisfacción Usuario", 
            "[X]/10", 
            "score de adopción"
        )
    
    # Recomendaciones finales
    st.markdown('<div class="subsection-header">💡 Recomendaciones Finales</div>', unsafe_allow_html=True)
    
    recommendations_col1, recommendations_col2 = st.columns(2)
    
    with recommendations_col1:
        st.markdown("""
        <div class="conclusion-box">
        <h4>🎯 Recomendaciones Inmediatas (0-3 meses):</h4>
        <ol>
        <li>[Recomendación inmediata 1]</li>
        <li>[Recomendación inmediata 2]</li>
        <li>[Recomendación inmediata 3]</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with recommendations_col2:
        st.markdown("""
        <div class="conclusion-box">
        <h4>🚀 Recomendaciones a Mediano Plazo (3-12 meses):</h4>
        <ol>
        <li>[Recomendación mediano plazo 1]</li>
        <li>[Recomendación mediano plazo 2]</li>
        <li>[Recomendación mediano plazo 3]</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Reflexión personal
    st.markdown('<div class="subsection-header">🤔 Reflexión Personal del Proyecto</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <h4>📝 Aprendizajes Personales:</h4>
    <p>[Escribe aquí tus reflexiones personales sobre el proyecto, qué aprendiste, qué te resultó más desafiante, qué te gustó más, etc.]</p>
    
    <h4>🎯 Aplicación en el Futuro:</h4>
    <p>[Describe cómo planeas aplicar estos conocimientos en tu carrera profesional o en futuros proyectos]</p>
    
    <h4>🙏 Agradecimientos:</h4>
    <p>[Agradecimientos a tutores, compañeros, fuentes de datos, etc.]</p>
    </div>
    """, unsafe_allow_html=True)

def show_interactive_app():
    st.markdown('<div class="section-header">⚡ Aplicación Interactiva</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <h3>🎯 Demo del Modelo en Producción</h3>
    <p>Esta sección permite a los usuarios interactuar directamente con tu modelo entrenado, 
    ingresando valores y obteniendo predicciones en tiempo real.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interfaz de predicción
    st.markdown('<div class="subsection-header">🔮 Realizar Predicción</div>', unsafe_allow_html=True)
    
    # Dividir en columnas para el input
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("#### 📊 Variables de Entrada")
        
        # REEMPLAZA ESTOS INPUTS CON TUS VARIABLES REALES
        var1 = st.slider("Variable 1", min_value=0, max_value=100, value=50)
        var2 = st.selectbox("Variable 2", options=["Opción A", "Opción B", "Opción C"])
        var3 = st.number_input("Variable 3", min_value=0.0, max_value=1000.0, value=100.0)
        var4 = st.radio("Variable 4", options=["Sí", "No"])
    
    with input_col2:
        st.markdown("#### ⚙️ Configuración del Modelo")
        
        modelo_seleccionado = st.selectbox(
            "Modelo a usar:", 
            ["Mejor Modelo (Random Forest)", "Modelo Alternativo", "Ensemble"]
        )
        
        confianza_threshold = st.slider(
            "Umbral de Confianza", 
            min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )
        
        mostrar_probabilidades = st.checkbox("Mostrar probabilidades detalladas", value=True)
        mostrar_explicacion = st.checkbox("Mostrar explicación de la predicción", value=True)
    
    # Botón de predicción
    if st.button("🚀 Realizar Predicción", type="primary"):
        
        # AQUÍ INTEGRARÍAS TU MODELO REAL
        # prediction = tu_modelo.predict([[var1, var2_encoded, var3, var4_encoded]])
        # prediction_proba = tu_modelo.predict_proba([[var1, var2_encoded, var3, var4_encoded]])
        
        # Simulación de predicción - REEMPLAZAR CON TU CÓDIGO REAL
        import random
        prediction = random.choice([0, 1])
        prediction_proba = [random.random(), random.random()]
        prediction_proba = [p/sum(prediction_proba) for p in prediction_proba]  # Normalizar
        
        # Mostrar resultados
        st.markdown('<div class="subsection-header">📊 Resultado de la Predicción</div>', unsafe_allow_html=True)
        
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            if prediction == 1:
                st.success(f"✅ **Predicción: POSITIVO**")
                st.metric("Confianza", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.error(f"❌ **Predicción: NEGATIVO**")
                st.metric("Confianza", f"{prediction_proba[0]*100:.1f}%")
        
        if mostrar_probabilidades:
            st.markdown("#### 📈 Distribución de Probabilidades")
            
            prob_data = pd.DataFrame({
                'Clase': ['Negativo', 'Positivo'],
                'Probabilidad': prediction_proba
            })
            
            fig_prob = px.bar(
                prob_data, 
                x='Clase', 
                y='Probabilidad',
                title="Probabilidades por Clase",
                color='Probabilidad',
                text='Probabilidad'
            )
            fig_prob.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_prob, use_container_width=True)
        
        if mostrar_explicacion:
            st.markdown("#### 🔍 Explicación de la Predicción")
            
            # Simulación de feature importance para esta predicción
            feature_impact = {
                'Variable': ['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4'],
                'Impacto': [0.35, -0.12, 0.28, -0.05],
                'Valor_Ingresado': [var1, var2, var3, var4]
            }
            
            impact_df = pd.DataFrame(feature_impact)
            
            fig_impact = px.bar(
                impact_df,
                x='Variable',
                y='Impacto',
                color='Impacto',
                title="Impacto de cada Variable en la Predicción",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Explicación textual
            st.markdown("""
            **Interpretación:**
            - Las variables con impacto positivo (verde) favorecen la predicción positiva
            - Las variables con impacto negativo (rojo) favorecen la predicción negativa
            - El tamaño del impacto indica la importancia de cada variable para esta predicción específica
            """)
    
    # Análisis de sensibilidad
    st.markdown('<div class="subsection-header">📊 Análisis de Sensibilidad</div>', unsafe_allow_html=True)
    
    sens_col1, sens_col2 = st.columns(2)
    
    with sens_col1:
        st.markdown("#### 🎛️ Variable a Analizar")
        variable_sensibilidad = st.selectbox(
            "Selecciona variable para análisis de sensibilidad:",
            ["Variable 1", "Variable 3"]  # Solo variables numéricas
        )
    
    with sens_col2:
        st.markdown("#### ⚙️ Configuración")
        rango_analisis = st.slider("Rango de variación (%)", 10, 50, 20)
    
    if st.button("🔬 Realizar Análisis de Sensibilidad"):
        # Simulación de análisis de sensibilidad
        if variable_sensibilidad == "Variable 1":
            base_value = var1
            variation_range = np.linspace(
                base_value * (1 - rango_analisis/100),
                base_value * (1 + rango_analisis/100),
                20
            )
        else:
            base_value = var3
            variation_range = np.linspace(
                base_value * (1 - rango_analisis/100),
                base_value * (1 + rango_analisis/100),
                20
            )
        
        # Simulación de predicciones para cada valor
        predictions_sens = [random.random() for _ in variation_range]
        
        sens_data = pd.DataFrame({
            'Valor': variation_range,
            'Probabilidad_Positiva': predictions_sens
        })
        
        fig_sens = px.line(
            sens_data,
            x='Valor',
            y='Probabilidad_Positiva',
            title=f"Sensibilidad de la Predicción a {variable_sensibilidad}",
            markers=True
        )
        fig_sens.add_vline(x=base_value, line_dash="dash", line_color="red", 
                          annotation_text="Valor Actual")
        st.plotly_chart(fig_sens, use_container_width=True)
    
    # Comparación de modelos en vivo
    st.markdown('<div class="subsection-header">⚖️ Comparación de Modelos</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Comparar Todos los Modelos"):
        
        # Simulación de predicciones de múltiples modelos
        modelos_comparacion = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
        predicciones_comp = [random.random() for _ in modelos_comparacion]
        
        comp_data = pd.DataFrame({
            'Modelo': modelos_comparacion,
            'Probabilidad_Positiva': predicciones_comp,
            'Confianza': [p if p > 0.5 else 1-p for p in predicciones_comp]
        })
        
        fig_comp = px.bar(
            comp_data,
            x='Modelo',
            y='Probabilidad_Positiva',
            color='Confianza',
            title="Comparación de Predicciones entre Modelos",
            text='Probabilidad_Positiva'
        )
        fig_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_comp.add_hline(y=0.5, line_dash="dash", line_color="black", 
                          annotation_text="Umbral de Decisión")
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # Simulador de impacto económico
    st.markdown('<div class="subsection-header">💰 Simulador de Impacto Económico</div>', unsafe_allow_html=True)
    
    eco_col1, eco_col2 = st.columns(2)
    
    with eco_col1:
        st.markdown("#### 💵 Parámetros Económicos")
        valor_tp = st.number_input("Valor Verdadero Positivo ($)", value=1000)
        costo_fp = st.number_input("Costo Falso Positivo ($)", value=200)
        costo_fn = st.number_input("Costo Falso Negativo ($)", value=800)
        volumen_mensual = st.number_input("Volumen Mensual de Casos", value=1000)
    
    with eco_col2:
        if st.button("💹 Calcular Impacto Económico"):
            
            # Simulación de matriz de confusión
            tp = int(volumen_mensual * 0.3 * 0.85)  # 30% positivos, 85% detected
            fn = int(volumen_mensual * 0.3 * 0.15)  # 15% missed
            fp = int(volumen_mensual * 0.7 * 0.10)  # 10% false alarms
            tn = int(volumen_mensual * 0.7 * 0.90)  # 90% correct negatives
            
            ganancia_mensual = (tp * valor_tp) - (fp * costo_fp) - (fn * costo_fn)
            ganancia_anual = ganancia_mensual * 12
            
            st.success(f"💰 **Ganancia Mensual: ${ganancia_mensual:,.2f}**")
            st.info(f"📊 **Ganancia Anual: ${ganancia_anual:,.2f}**")
            
            # Desglose
            st.markdown("#### 📊 Desglose:")
            st.write(f"- Verdaderos Positivos: {tp} × ${valor_tp} = ${tp * valor_tp:,}")
            st.write(f"- Falsos Positivos: {fp} × ${costo_fp} = -${fp * costo_fp:,}")
            st.write(f"- Falsos Negativos: {fn} × ${costo_fn} = -${fn * costo_fn:,}")
    
    # Información de contacto y próximos pasos
    st.markdown('<div class="subsection-header">📞 Información Adicional</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-box">
    <h4>📈 Próximos Pasos para Implementación:</h4>
    <ol>
    <li><strong>Validación Adicional:</strong> Probar con datos más recientes</li>
    <li><strong>Integración:</strong> Conectar con sistemas existentes</li>
    <li><strong>Monitoreo:</strong> Establecer métricas de seguimiento</li>
    <li><strong>Escalamiento:</strong> Preparar para mayor volumen de datos</li>
    </ol>
    
    <h4>👥 Contacto del Proyecto:</h4>
    <p><strong>Alumno:</strong> [Tu Nombre] - [tu.email@email.com]</p>
    <p><strong>Tutor:</strong> [Nombre Tutor] - [tutor.email@email.com]</p>
    <p><strong>Institución:</strong> [Nombre de la Institución]</p>
    </div>
    """, unsafe_allow_html=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()