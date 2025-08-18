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
            "🤖 Modelado y Optimización",
            "📈 Selección del Mejor Modelo",
            "📋 Conclusiones",
            "⚡ Aplicación Interactiva"
        ]
    )
    
    if page == "🏠 Introducción":
        show_introduction()
    elif page == "📊 Análisis Exploratorio (EDA)":
        show_eda()
    elif page == "🤖 Modelado y Optimización":
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

    
    # Análisis de datos faltantes
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

    # Análisis de distribuciones
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







    # TODO:  PASAR ACÁ EL ANÁLISIS POR SABOR Y RATING. LUEGO ESCRIBIR CONCLUSIONES (CREO QUE EL DE CLUSTER NO TIENE SENTIDO)
    # Una vez tengas eso, pasar al usuario sintético en el modelado directamente (mostrar gráficos de decision del usuario)







    # Análisis por target
    st.markdown('<div class="subsection-header">🎯 Análisis por Variable Objetivo</div>', unsafe_allow_html=True)
    
    if 'target' in wines_clean.columns:
        target_analysis_var = st.selectbox(
            "Selecciona una variable para analizar vs target:", 
            [col for col in numeric_columns if col != 'target']
        )
        
        if target_analysis_var:
            fig_target = px.box(
                wines_clean, 
                x='target', 
                y=target_analysis_var,
                title=f"Distribución de {target_analysis_var} por Target"
            )
            st.plotly_chart(fig_target, use_container_width=True)
    
    # Insights clave
    st.markdown('<div class="subsection-header">💡 Insights Clave del EDA</div>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class="highlight-box">
        <h4>🔍 Hallazgos Principales:</h4>
        <ul>
        <li>[Insight 1 de tu análisis]</li>
        <li>[Insight 2 de tu análisis]</li>
        <li>[Insight 3 de tu análisis]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="highlight-box">
        <h4>⚠️ Consideraciones para el Modelado:</h4>
        <ul>
        <li>[Consideración 1]</li>
        <li>[Consideración 2]</li>
        <li>[Consideración 3]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

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