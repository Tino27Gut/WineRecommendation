import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
from src.utils import utils as ut
from models.synthetic_user import SyntheticUserSimulator

# TODO: INCLUIR IMÁGENES DE VINOS, MEJORAR PREFORMANCE, CORREGIR OPCIONES DE "AJUSTAR FILTROS", MEJORAR CONTADOR DE INTENSIDAD EN VINOS, PONER ÍCONOS DE SABOR EN PÁGINA DE AGRADECIMIENTO.

wine_df = pd.read_csv(ut.get_project_file_path("src", "data", "transformed", "wines_clean.csv"))
wine_df = wine_df.reset_index(drop=False)
wine_df = wine_df.rename(columns={"index": "wine_id"})

# Cargar datos de usuarios
users_data = pd.read_pickle(ut.get_project_file_path("src", "data", "synthetic", "simulation_13.pkl"))
users_data = users_data.dropna()

# Cargar listas auxiliares
grapes = pd.read_csv(ut.get_project_file_path("src", "data", "processed", "aux", "grapes.csv"))
grapes_list = list(grapes["grapes"])

region = pd.read_csv(ut.get_project_file_path("src", "data", "processed", "aux", "region.csv"))
region_list = list(region["region"])

pairings = pd.read_csv(ut.get_project_file_path("src", "data", "processed", "aux", "pairings.csv"))
pairings_list = list(pairings["pairings"])

taste_columns = ["body", "tannins", "sweetness", "acidity"]

# Datos de comidas
meals_df = pd.read_excel(ut.get_project_file_path("src", "data", "raw", "meals", "Meals.xlsx"))

# Instanciar Clase de Usuario Sintético
user = SyntheticUserSimulator(
    wine_df=wine_df,
    meals_df=meals_df,
    pairing_cols=pairings_list,
    taste_cols=taste_columns,
    grape_cols=grapes_list
)

# Configurar el estado de la sesión
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_meal' not in st.session_state:
        st.session_state.selected_meal = None
    if 'price_range' not in st.session_state:
        st.session_state.price_range = (1, 150)
    if 'taste_preferences' not in st.session_state:
        st.session_state.taste_preferences = {}
    if 'selected_grapes' not in st.session_state:
        st.session_state.selected_grapes = []
    if 'filtered_wines' not in st.session_state:
        st.session_state.filtered_wines = None
    if 'wine_predictions' not in st.session_state:
        st.session_state.wine_predictions = None
    if 'current_wine_page' not in st.session_state:
        st.session_state.current_wine_page = 0
    if 'selected_wine' not in st.session_state:
        st.session_state.selected_wine = None
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []

def reset_session():
    """Resetea el estado de la sesión para empezar de nuevo"""
    keys_to_reset = ['selected_meal', 'price_range', 'taste_preferences', 
                    'selected_grapes', 'filtered_wines', 'wine_predictions', 
                    'current_wine_page', 'selected_wine']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.page = 'home'
    st.session_state.price_range = (1, 150)
    st.session_state.taste_preferences = {}
    st.session_state.selected_grapes = []
    st.session_state.current_wine_page = 0

# Simulación de datos (REEMPLAZAR CON TUS DATAFRAMES REALES)
@st.cache_data
def load_sample_data():
    # MEALS DATA - Reemplazar con tu meals_df real
    meals_data = {
        'meal': ['asado', 'pasta con salsa roquefort', 'salmón a la plancha', 'pizza napolitana', 
                'milanesa de carne', 'risotto de champiñones', 'empanadas de carne', 'ensalada'],
        'main_pairing': ['beef', 'blue cheese', 'rich fish (salmon, tuna etc)', 'vegetarian',
                        'beef', 'mushrooms', 'beef', 'vegetarian'],
        # Simulación de columnas one-hot para pairings
        'beef': [1, 0, 0, 0, 1, 0, 1, 0],
        'blue_cheese': [0, 1, 0, 0, 0, 0, 0, 0],
        'rich_fish': [0, 0, 1, 0, 0, 0, 0, 0],
        'vegetarian': [0, 0, 0, 1, 0, 0, 0, 1],
        'mushrooms': [0, 0, 0, 0, 0, 1, 0, 0]
    }
    meals_df = pd.DataFrame(meals_data)
    
    # WINE DATA - Reemplazar con tu wine_df real
    np.random.seed(42)
    wine_data = {
        'wine_name': [f'Vino {i+1}' for i in range(100)],
        'winery': [f'Bodega {chr(65 + i%10)}' for i in range(100)],
        'region': [f'Región {i%5 + 1}' for i in range(100)],
        'price': np.random.uniform(10, 200, 100),
        'image_link': [f'https://example.com/wine{i+1}.jpg' for i in range(100)],
        'body': np.random.choice(['leve', 'moderado', 'marcado', 'intenso'], 100),
        'tannins': np.random.choice(['leve', 'moderado', 'marcado', 'intenso'], 100),
        'sweetness': np.random.choice(['leve', 'moderado', 'marcado', 'intenso'], 100),
        'acidity': np.random.choice(['leve', 'moderado', 'marcado', 'intenso'], 100),
        # Simulación de columnas de uvas
        'malbec': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'cabernet_sauvignon': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'merlot': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'pinot_noir': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'syrah': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'chardonnay': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'sauvignon_blanc': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'torrontes': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'otras_uvas': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        # Simulación de columnas de pairings (mismas que meals)
        'beef': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
        'blue_cheese': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'rich_fish': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'vegetarian': np.random.choice([0, 1], 100, p=[0.5, 0.5]),
        'mushrooms': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    }
    wine_df = pd.DataFrame(wine_data)
    
    return meals_df, wine_df

def simulate_wine_predictions(wine_df, taste_preferences):
    """Simulación del modelo de ML - REEMPLAZAR CON TU MODELO REAL"""
    np.random.seed(42)
    
    # Simulación: mayor probabilidad si coinciden más características de sabor
    predictions = []
    for _, wine in wine_df.iterrows():
        base_prob = 0.3
        
        # Bonus por coincidencia de sabores
        taste_bonus = 0
        for taste, preference in taste_preferences.items():
            if wine[taste] == preference:
                taste_bonus += 0.15
        
        # Agregar algo de ruido aleatorio
        noise = np.random.normal(0, 0.1)
        
        final_prob = np.clip(base_prob + taste_bonus + noise, 0, 1)
        predictions.append(final_prob)
    
    return predictions

def save_feedback_to_csv(wine_data, liked):
    """Guarda el feedback en un CSV - IMPLEMENTAR según tus necesidades"""
    feedback_row = wine_data.copy()
    feedback_row['liked'] = 1 if liked else 0
    feedback_row['timestamp'] = datetime.now().isoformat()
    
    st.session_state.feedback_data.append(feedback_row)
    
    # En implementación real, guardarías en un CSV:
    # feedback_df = pd.DataFrame(st.session_state.feedback_data)
    # feedback_df.to_csv('wine_feedback.csv', index=False)
    
    return True

# PÁGINA PRINCIPAL
def show_home_page():
    st.markdown('<div class="section-header">⚡ Aplicación Interactiva</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <h3>🎯 Recomendador Inteligente de Vinos</h3>
    <p>Descubre el vino perfecto para cada ocasión usando inteligencia artificial. 
    Elige cómo quieres explorar nuestras recomendaciones personalizadas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 ¿Cómo quieres comenzar?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🍽️ Encontrar Maridaje Perfecto", 
                    help="Encuentra el vino ideal para tu comida", 
                    type="primary"):
            st.session_state.page = 'meal_selection'
            st.rerun()
        
    
    with col2:
        if st.button("🍷 Descubrir Nuevos Vinos", 
                    help="Explora vinos basándose en tus preferencias de sabor", 
                    type="secondary"):
            st.info("🚧 Esta funcionalidad estará disponible próximamente.")
            st.markdown("""
            **Próximamente podrás:**
            - Explorá vinos por región y bodega
            - Descubrí vinos con popularidad en alza
            - Conocé los mejores vinos para cada momento del día
            - Recibí nuevas recomendaciones adaptadas a tu paladar
            """)
        
    
    # Estadísticas rápidas
    st.markdown("---")
    st.markdown("### 📊 Estadísticas de la Aplicación")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("🍷 Vinos en Base", f"{wine_df["wine_id"].count()}", help="Total de vinos disponibles")
    
    with stats_col2:
        st.metric("🍽️ Comidas Analizadas", f"{meals_df["Comida"].count()}", help="Platos con perfiles de maridaje")
    
    with stats_col3:
        st.metric("🎯 Precisión del Modelo", "~86%", help="Precisión en recomendaciones")
    
    with stats_col4:
        feedback_count = len(st.session_state.feedback_data)
        st.metric("👥 Evaluaciones", feedback_count, help="Feedback de usuarios")

# SELECCIÓN DE COMIDA
def show_meal_selection():
       
    st.markdown('<div class="subsection-header">🍽️ Selección de Comida</div>', unsafe_allow_html=True)
    
    # Botón para volver
    if st.button("← Volver al Inicio"):
        reset_session()
        st.rerun()
    
    st.markdown("### 🥘 Elige tu Comida")
    st.markdown("Selecciona la comida para la cual quieres encontrar el vino perfecto.")
    
    # Crear selector de comida
    available_meals = sorted(meals_df.iloc[:, 0].unique())
    selected_meal = st.selectbox(
        "🍴 Selecciona una comida:",
        options=available_meals,
        index=0,
        help="Elige la comida para la cual quieres encontrar el vino ideal"
    )
    
    if selected_meal:
        # Obtener información de la comida seleccionada
        meal_info = meals_df[meals_df.iloc[:, 0] == selected_meal].iloc[0]
        main_pairing = meal_info.iloc[1] if len(meal_info) > 1 else "No especificado"
        
        # Obtener pairings activos
        pairing_columns = meals_df.columns[2:]
        active_pairings = [col for col in pairing_columns if meal_info[col] == 1]
        
        # Visualización de la comida
        food_col1, food_col2 = st.columns([1, 2])
        
        with food_col1:
            # Mapa de emojis para comidas
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
            
            food_emoji = food_emoji_map.get(selected_meal.lower(), '🍽️')
            
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
            st.markdown("#### 🎯 Características de Maridaje")
            
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
            
            if active_pairings:
                st.markdown("#### 🧩 Ingredientes Compatibles")
                
                # Mostrar pairings como tags
                tags_html = "".join([
                    f'<span style="background-color: #e3f2fd; '
                    f'color: #1976d2; '
                    f'padding: 5px 10px; '
                    f'border-radius: 15px; '
                    f'font-size: 0.9em; '
                    f'margin: 3px; '
                    f'display: inline-block; '
                    f'border: 1px solid #bbdefb;">{pairing.replace("_", " ").title()}</span>'
                    for pairing in active_pairings
                ])

                st.markdown(f'<div style="margin: 10px 0;">{tags_html}</div>', unsafe_allow_html=True)

        
        # Botón para continuar
        if st.button("🎯 Continuar con esta Comida", type="primary"):
            st.session_state.selected_meal = selected_meal
            st.session_state.page = 'price_selection'
            st.rerun()

# SELECCIÓN DE PRECIO
def show_price_selection():
    st.markdown('<div class="subsection-header">💰 Selección de Precio</div>', unsafe_allow_html=True)
    
    # Navegación
    nav_col1, nav_col2 = st.columns([1, 4])
    with nav_col1:
        if st.button("← Cambiar Comida"):
            st.session_state.page = 'meal_selection'
            st.rerun()
    
    with nav_col2:
        st.markdown(
            f"""
            - **Comida: 🍽️** {st.session_state.selected_meal}
            """
        )
    
    st.markdown("### 💵 Define tu Presupuesto")
    st.markdown("Selecciona el rango de precios para filtrar las recomendaciones de vinos.")
    
    # Slider de rango para precio
    precio_min, precio_max = st.slider(
        "💰 Rango de Precio ($)",
        min_value=0,
        max_value=51,
        value=(5, 20),
        help="Selecciona el rango de precio por botella. 0 = sin mínimo, 51 = sin máximo"
    )
    
    # Validación de precios
    if precio_min >= precio_max:
        st.error("⚠️ El precio mínimo debe ser menor al precio máximo")
        return
    
    # Visualización del rango
    st.markdown("#### 📊 Rango de Precios Seleccionado")
    
    range_text = f"\\${precio_min}"
    if precio_min == 0:
        range_text = "Sin mínimo"
    
    range_text += " - "
    
    if precio_max > 50:
        range_text += "Sin máximo"
        precio_max = None
    else:
        range_text += f"\\${precio_max}"
    
    st.info(f"🎯 **Rango seleccionado:** {range_text}")
    
    # Estadísticas simuladas del rango
    wines_in_range = wine_df[
        (wine_df['price'] >= precio_min) &
        (wine_df['price'] <= (9999 if precio_max == None else precio_max))
    ]
    
    st.markdown(f"📈 **Vinos disponibles en este rango:** {len(wines_in_range)} vinos")
    
    # Botón para continuar
    if st.button("🎯 Continuar con este Presupuesto", type="primary"):
        st.session_state.price_range = (precio_min, precio_max)
        st.session_state.page = 'taste_selection'
        st.rerun()

# SELECCIÓN DE SABORES
def show_taste_selection():
    st.markdown('<div class="subsection-header">👅 Selección de Intensidades</div>', unsafe_allow_html=True)
    
    # Navegación
    nav_col1, nav_col2 = st.columns([1, 4])
    with nav_col1:
        if st.button("← Cambiar Precio"):
            st.session_state.page = 'price_selection'
            st.rerun()
    
    with nav_col2:
        precio_min, precio_max = st.session_state.price_range
        range_text = f"\\${precio_min}" if precio_min != 0 else "Sin mín"
        range_text += " - "
        range_text += f"\\${precio_max}" if precio_max != None else "Sin máx"
        st.markdown(
            f"""
            - **Comida: 🍽️** {st.session_state.selected_meal}
            - **Precio: 💰** {range_text}
            """
        )
    
    st.markdown("### 🎭 Define tus Preferencias de Sabor")
    st.markdown("Selecciona la intensidad deseada para cada característica del vino.")
    st.markdown("---")

    intensity_options = ["leve", "moderado", "marcado", "intenso"]
    
    # Definir las características con sus descripciones
    taste_characteristics = {
        "body": {
            "emoji": "🏋️",
            "name": "Cuerpo",
            "description": "Sensación de peso y plenitud en boca",
            "examples": {
                "leve": "Vinos ligeros, fáciles de beber",
                "moderado": "Equilibrio entre ligereza y estructura",
                "marcado": "Vinos con presencia notable",
                "intenso": "Vinos robustos y concentrados"
            }
        },
        "tannins": {
            "emoji": "🌿",
            "name": "Taninos",
            "description": "Sensación de sequedad y astringencia",
            "examples": {
                "leve": "Suaves, casi imperceptibles",
                "moderado": "Presentes pero balanceados",
                "marcado": "Notables, con estructura",
                "intenso": "Dominantes, muy secos"
            }
        },
        "sweetness": {
            "emoji": "🍯",
            "name": "Dulzura",
            "description": "Nivel de azúcar residual percibido",
            "examples": {
                "leve": "Seco, sin dulzura",
                "moderado": "Ligeramente dulce",
                "marcado": "Notablemente dulce",
                "intenso": "Muy dulce, tipo postre"
            }
        },
        "acidity": {
            "emoji": "🍋",
            "name": "Acidez",
            "description": "Frescura y vivacidad del vino",
            "examples": {
                "leve": "Suave, relajada",
                "moderado": "Refrescante, equilibrada",
                "marcado": "Vibrante, notable",
                "intenso": "Muy ácida, punzante"
            }
        }
    }
    
    taste_preferences = {}
    
    for i, (key, info) in enumerate(taste_characteristics.items()):
        st.markdown(f"#### {info['emoji']} {info['name']}")
        st.markdown(f"*{info['description']}*")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            selected_intensity = st.selectbox(
                f"Intensidad de {info['name'].lower()}:",
                options=intensity_options,
                index=1,  # Default a "moderado"
                key=f"taste_{key}"
            )
            taste_preferences[key] = selected_intensity
        
        with col2:
            # Mostrar descripción de la intensidad seleccionada
            st.markdown(f"**{selected_intensity.title()}:** {info['examples'][selected_intensity]}")
        
        if i < len(taste_characteristics) - 1:
            st.markdown("---")
    
    # Resumen de selecciones
    st.markdown("---")
    st.markdown("#### 📋 Resumen de Preferencias")
    
    summary_cols = st.columns(4)
    for i, (key, info) in enumerate(taste_characteristics.items()):
        with summary_cols[i]:
            intensity = taste_preferences[key]
            # Crear indicador visual de intensidad
            intensity_level = intensity_options.index(intensity) + 1
            bars = "█" * intensity_level + "░" * (4 - intensity_level)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 1.5em;">{info['emoji']}</div>
                <div style="font-weight: bold;">{info['name']}</div>
                <div style="font-family: monospace; color: #666;">{bars}</div>
                <div style="font-size: 0.9em; color: #888;">{intensity}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Botón para continuar
    if st.button("🍇 Continuar a Selección de Uvas", type="primary"):
        st.session_state.taste_preferences = taste_preferences
        st.session_state.page = 'grape_selection'
        st.rerun()

# SELECCIÓN DE UVAS
def show_grape_selection():
    st.markdown('<div class="subsection-header">🍇 Selección de Uvas</div>', unsafe_allow_html=True)
    

    # Navegación
    nav_col1, nav_col2 = st.columns([1, 4])
    with nav_col1:
        if st.button("← Cambiar Preferencias de Sabor"):
            st.session_state.page = 'taste_selection'
            st.rerun()

    with nav_col2:
        # Precio
        precio_min, precio_max = st.session_state.price_range
        range_text = f"\\${precio_min}" if precio_min != 0 else "Sin mín"
        range_text += " - "
        range_text += f"\\${precio_max}" if precio_max != None else "Sin máx"

        # Preferencias de sabor
        taste_prefs = st.session_state.taste_preferences
        taste_text = " | ".join(
            [f"{k.capitalize()}: {v}" for k, v in taste_prefs.items()]
        )

        st.markdown(
            f"""
            - **Comida: 🍽️** {st.session_state.selected_meal}
            - **Precio: 💰** {range_text}
            - **Sabores: 😋** {taste_text}
            """
        )
    
    st.markdown("### 🍇 Elige las Variedades de Uva")
    st.markdown("Selecciona una o más variedades de uva que te gustaría probar.")
    
    # Definir las uvas con información
    grape_info = {
        'malbec': {
            'name': 'Malbec',
            'emoji': '🍇',
            'description': 'Tinta argentina por excelencia, intensa y frutal',
            'characteristics': 'Cuerpo medio-alto, taninos suaves, notas de ciruela'
        },
        'cabernet_sauvignon': {
            'name': 'Cabernet Sauvignon',
            'emoji': '🍷',
            'description': 'Rey de los tintos, estructura y elegancia',
            'characteristics': 'Cuerpo alto, taninos firmes, notas de cassis'
        },
        'cabernet_franc': {
            'name': 'Cabernet Franc',
            'emoji': '🍇',
            'description': 'Aromática y elegante, más ligera que el Cabernet Sauvignon',
            'characteristics': 'Cuerpo medio, taninos finos, notas de pimiento y frambuesa'
        },
        'petit_verdot': {
            'name': 'Petit Verdot',
            'emoji': '🍷',
            'description': 'Potente y concentrada, aporta color y estructura a blends',
            'characteristics': 'Cuerpo alto, taninos intensos, notas florales y especiadas'
        },
        'merlot': {
            'name': 'Merlot',
            'emoji': '🍇',
            'description': 'Suave y accesible, ideal para iniciarse',
            'characteristics': 'Cuerpo medio, taninos suaves, notas de cereza'
        },
        'pinot_noir': {
            'name': 'Pinot Noir',
            'emoji': '🍷',
            'description': 'Elegante y compleja, la diva de los tintos',
            'characteristics': 'Cuerpo ligero-medio, taninos delicados, notas terrosas'
        },
        'syrah': {
            'name': 'Syrah/Shiraz',
            'emoji': '🍇',
            'description': 'Especiada y potente, gran personalidad',
            'characteristics': 'Cuerpo alto, taninos medios, notas especiadas'
        },
        'chardonnay': {
            'name': 'Chardonnay',
            'emoji': '🥂',
            'description': 'Blanco versátil, desde fresco a cremoso',
            'characteristics': 'Cuerpo variable, acidez balanceada, notas de manzana'
        },
        'otras_uvas': {
            'name': 'Otras Variedades',
            'emoji': '🎭',
            'description': 'Variedades menos comunes pero interesantes',
            'characteristics': 'Diversidad de estilos y características únicas'
        }
    }

    st.markdown("#### 🎯 Selección Múltiple")
    st.markdown("*Puedes elegir una o varias variedades. Si no seleccionas ninguna, se considerarán todas.*")
    
    # Crear checkboxes en una grilla
    cols_per_row = 3
    grape_keys = list(grape_info.keys())
    selected_grapes = []
    
    for i in range(0, len(grape_keys), cols_per_row):
        cols = st.columns(cols_per_row)
        row_grapes = grape_keys[i:i + cols_per_row]
        
        for j, grape_key in enumerate(row_grapes):
            with cols[j]:
                info = grape_info[grape_key]
                
                # Checkbox para la uva
                is_selected = st.checkbox(
                    f"{info['emoji']} {info['name']}",
                    key=f"grape_{grape_key}",
                    help=info['description']
                )
                
                if is_selected:
                    selected_grapes.append(grape_key)
                
                # Información adicional
                with st.expander(f"ℹ️ Más sobre {info['name']}", expanded=False):
                    st.markdown(f"**{info['description']}**")
                    st.markdown(f"*{info['characteristics']}*")
    
    # Mostrar selección actual
    if selected_grapes:
        st.markdown("#### ✅ Variedades Seleccionadas")
        selected_names = [grape_info[grape]['name'] for grape in selected_grapes]
        selected_emojis = [grape_info[grape]['emoji'] for grape in selected_grapes]
        
        display_text = " • ".join([f"{emoji} {name}" for emoji, name in zip(selected_emojis, selected_names)])
        st.success(f"🍇 **{len(selected_grapes)} variedades:** {display_text}")
    else:
        st.info("ℹ️ **Sin selección específica** - Se considerarán todas las variedades disponibles")
    
    # Botón para continuar
    if st.button("🔍 Buscar Vinos", type="primary"):
        st.session_state.selected_grapes = selected_grapes
        st.session_state.page = 'wine_results'
        st.rerun()

# RESULTADOS DE VINOS
def show_wine_results(trained_model):
    st.markdown('<div class="subsection-header">🍷 Recomendaciones Personalizadas</div>', unsafe_allow_html=True)
    
    # Navegación
    nav_col1, nav_col2 = st.columns([1, 4])
    with nav_col1:
        if st.button("← Cambiar Uvas"):
            st.session_state.page = 'grape_selection'
            st.rerun()
    
    # Procesar filtros solo una vez
    if st.session_state.filtered_wines is None or st.session_state.wine_predictions is None:
        with st.spinner("🔍 Buscando los vinos perfectos para ti..."):
            
            # Obtener pairings de la comida seleccionada
            meal_row = meals_df[meals_df.iloc[:, 0] == st.session_state.selected_meal].iloc[0]
            pairing_columns = meals_df.columns[2:]  # Columnas de pairings
            user_pairings = [col for col in pairing_columns if meal_row[col] == 1]
            
            # Aplicar filtros
            wine_base = wine_df.copy()
            wine_base.columns = wine_base.columns.str.lower() # Estandariza columnas (lower case)
            
            # Filtro por pairings
            if user_pairings:
                pairing_filtered = wine_base[wine_base[user_pairings].sum(axis=1) > 0]
                if len(pairing_filtered) > 0:
                    wine_base = pairing_filtered
            
            # Filtro por precio
            precio_min, precio_max = st.session_state.price_range
            if precio_min != 0:
                precio_min_filtered = wine_base[wine_base["price"] >= precio_min]
                if len(precio_min_filtered) > 0:
                    wine_base = precio_min_filtered
            
            if precio_max:
                precio_max_filtered = wine_base[wine_base["price"] <= precio_max]
                if len(precio_max_filtered) > 0:
                    wine_base = precio_max_filtered
            
            # Filtro por uvas
            if st.session_state.selected_grapes:
                selected_grapes = [g.lower() for g in st.session_state.selected_grapes]
                grape_filtered = wine_base[wine_base[selected_grapes].sum(axis=1) > 0]
                if len(grape_filtered) > 0:
                    wine_base = grape_filtered
            
            # Agregar columnas que faltan que el modelo utiliza como input:
            # Pairing: 'has_user_main_pairing'.
            # Taste: 'user_body_diff', 'user_acidity_diff', 'user_sweetness_diff', 'user_tannins_diff'.
            # Price: 'user_price_diff', 'user_price_center'.

            def calc_user_taste_diff(x, min_taste, max_taste):
                if min_taste <= x <= max_taste:
                    return 0
                elif x > max_taste:
                    return x - max_taste
                else:
                    return x - min_taste

            def calc_user_price_diff(x, min_price, max_price):
                center = 0
                if max_price > 50:
                    center = min_price
                else:
                    center = (min_price + max_price) / 2
                
                return x - center
            

            # Traducir labels de sabor del usuario a valores
            meal_info = meals_df[meals_df.iloc[:, 0] == st.session_state.selected_meal].iloc[0]
            main_pairing = meal_info.iloc[1].lower() if len(meal_info) > 1 else "No especificado"
            taste_profiles, _ = user._build_pairing_profile()
            food_quantiles = taste_profiles[main_pairing] # Diccionario de sabores con serie de su cuantil, valor.

            intensity_options = ["leve", "moderado", "marcado", "intenso"]
            tastes = ["body", "tannins", "sweetness", "acidity"]

            for t in tastes:
                taste_quantile_values = food_quantiles[t].values
                taste_pref = st.session_state.taste_preferences[t]
                intensity_value = intensity_options.index(taste_pref)
                selected_taste_range = [taste_quantile_values[intensity_value], taste_quantile_values[intensity_value+1]]
                wine_base["user_" + t + "_diff"] = wine_base.apply(
                    lambda row: calc_user_taste_diff(
                        x=row[t],
                        min_taste=selected_taste_range[0],
                        max_taste=selected_taste_range[1]
                    ),
                    axis=1
                )

            wine_base["has_user_main_pairing"] = wine_base.apply(lambda row: 1 if row[main_pairing]==1 else 0, axis=1)

            price_min, price_max = st.session_state.price_range

            wine_base["user_price_min"] = price_min
            wine_base["user_price_max"] = price_max if price_max != None else 2000 # replaces nulls for a high value (same as in model analysis)
            wine_base["user_price_center"] = (wine_base["user_price_max"] + wine_base["user_price_min"]) / 2
            wine_base["user_price_diff"] = wine_base.apply(
                lambda row: calc_user_price_diff(
                    x=row["price"],
                    min_price=row["user_price_min"],
                    max_price=row["user_price_max"]
                ),
                axis=1
            )

            # Generar predicciones reales usando el pipeline entrenado
            try:
                # Preparar las features para el modelo (usar solo las features que el modelo espera)
                model_features = wine_base[trained_model.feature_names_in_]
                
                # Generar predicciones usando el pipeline
                predictions = trained_model.predict_proba(model_features)[:, 1]  # Probabilidad de clase positiva
                
            except Exception as e:
                st.error(f"Error al generar predicciones: {str(e)}")
                # Fallback a predicciones simuladas si hay error
                # predictions = simulate_wine_predictions(wine_base, st.session_state.taste_preferences)
            
            # Combinar vinos con predicciones y ordenar
            wine_base_with_predictions = wine_base.copy()
            wine_base_with_predictions['prediction_score'] = predictions
            wine_base_with_predictions = wine_base_with_predictions.sort_values('prediction_score', ascending=False)
            
            st.session_state.filtered_wines = wine_base_with_predictions
            st.session_state.wine_predictions = predictions
    
    filtered_wines = st.session_state.filtered_wines
    
    # Mostrar resumen de filtros aplicados
    with nav_col2:
        st.markdown("**🎯 Filtros aplicados:**")
        precio_min, precio_max = st.session_state.price_range
        price_text = f"\\${precio_min}" if precio_min != 0 else "Sin mín"
        price_text += f"-\\${precio_max}" if precio_max != None else "-Sin máx"
        
        grape_text = f"{len(st.session_state.selected_grapes)} uvas" if st.session_state.selected_grapes else "Todas"
        st.markdown(f"🍽️ {st.session_state.selected_meal} | 💰 {price_text} | 🍇 {grape_text} | 🍷 {len(filtered_wines)} vinos")
    
    if len(filtered_wines) == 0:
        st.error("😔 No se encontraron vinos que coincidan con tus criterios.")
        st.markdown("### 💡 Sugerencias:")
        st.markdown("- Amplía tu rango de precios")
        st.markdown("- Selecciona más variedades de uva")
        st.markdown("- Prueba con una comida diferente")
        
        if st.button("🔄 Comenzar de Nuevo"):
            reset_session()
            st.rerun()
        return
    
    # Paginación - mostrar 3 vinos por página
    wines_per_page = 3
    total_pages = (len(filtered_wines) + wines_per_page - 1) // wines_per_page
    current_page = st.session_state.current_wine_page
    
    start_idx = current_page * wines_per_page
    end_idx = min(start_idx + wines_per_page, len(filtered_wines))
    current_wines = filtered_wines.iloc[start_idx:end_idx]
    
    # Header con estadísticas
    st.markdown(f"### 🎯 Top {len(filtered_wines)} Recomendaciones")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        avg_score = filtered_wines['prediction_score'].mean()
        st.metric("📊 Puntuación Promedio", f"{avg_score:.1%}")
    
    with stats_col2:
        avg_price = filtered_wines['price'].mean()
        st.metric("💰 Precio Promedio", f"${avg_price:.0f}")
    
    with stats_col3:
        st.metric("📄 Página Actual", f"{current_page + 1} de {total_pages}")
    
    with stats_col4:
        best_score = filtered_wines['prediction_score'].iloc[0]
        st.metric("⭐ Mejor Match", f"{best_score:.1%}")
    
    st.markdown("---")
    
    # Mostrar vinos de la página actual
    for idx, (_, wine) in enumerate(current_wines.iterrows()):
        wine_col1, wine_col2, wine_col3 = st.columns([1, 2, 1])
        
        with wine_col1:
            # Imagen del vino (placeholder)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #8B0000, #DC143C);
                border-radius: 15px;
                height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 3em;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                margin: 10px 0;
            ">🍷</div>
            """, unsafe_allow_html=True)
            
            # Puntuación
            score = wine['prediction_score']
            score_color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div style="
                background-color: {score_color};
                color: white;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                margin: 10px 0;
            ">
                🎯 Match: {score:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        with wine_col2:
            # Información del vino
            st.markdown(f"#### 🍷 {wine['name']}")
            st.markdown(f"**🏭 Bodega:** {wine['winery']}")
            #st.markdown(f"**📍 Región:** {wine['region']}") -> No puedo traerlo así porque está en formato one-hot.
            st.markdown(f"**💰 Precio:** ${wine['price']:.0f}")
            
            # Características de sabor
            st.markdown("**👅 Perfil de Sabor:**")
            taste_cols = st.columns(4)
            
            taste_characteristics = ['body', 'tannins', 'sweetness', 'acidity']
            taste_emojis = ['🏋️', '🌿', '🍯', '🍋']
            
            for i, (char, emoji) in enumerate(zip(taste_characteristics, taste_emojis)):
                with taste_cols[i]:
                    intensity = wine[char]
                    # Crear indicador visual
                    intensity_levels = {"leve": 1, "moderado": 2, "marcado": 3, "intenso": 4}
                    level = intensity_levels.get(intensity, 2)
                    bars = "█" * level + "░" * (4 - level)
                    
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 0.8em;">
                        <div>{emoji}</div>
                        <div style="font-family: monospace; color: #666; font-size: 0.7em;">{bars}</div>
                        <div style="color: #888;">{intensity}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Variedades de uva
            grape_columns = ['malbec', 'cabernet sauvignon', 'cabernet franc', 'petit verdot',
                             'merlot', 'pinot noir', 'shiraz/syrah', 'chardonnay']

    
            wine_grapes = [col.title() for col in grape_columns if wine[col] == 1]

            other_grapes = [col for col in wine.index if col not in grape_columns and wine[col] == 1]

            if other_grapes:
                wine_grapes.append("Otras Variedades")

            if wine_grapes:
                grape_text = " • ".join(wine_grapes)
                st.markdown(f"**🍇 Variedades:** {grape_text}")
        
        with wine_col3:
            st.markdown("### ")  # Espaciado
            st.markdown("### ")  # Espaciado
            
            # Botón para seleccionar este vino
            if st.button(f"✅ Elegir Este Vino", key=f"select_wine_{start_idx + idx}", type="primary"):
                st.session_state.selected_wine = wine.to_dict()
                st.session_state.page = 'wine_feedback'
                st.rerun()
            
            # Información adicional
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                font-size: 0.9em;
                margin: 10px 0;
            ">
                <div><strong>🔢 Ranking:</strong> #{start_idx + idx + 1}</div>
                <div><strong>🎯 Confianza:</strong> {score:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Separador entre vinos
        if idx < len(current_wines) - 1:
            st.markdown("---")
    
    # Controles de paginación
    st.markdown("### 📄 Navegación")
    
    pagination_col1, pagination_col2, pagination_col3 = st.columns([1, 2, 1])
    
    with pagination_col1:
        if current_page > 0:
            if st.button("⬅️ Anteriores 3 Vinos"):
                st.session_state.current_wine_page -= 1
                st.rerun()
    
    with pagination_col2:
        st.markdown(f"**Mostrando vinos {start_idx + 1}-{end_idx} de {len(filtered_wines)}**")
        
        # Barra de progreso visual
        progress = (current_page + 1) / total_pages
        st.progress(progress)
    
    with pagination_col3:
        if current_page < total_pages - 1:
            if st.button("Siguientes 3 Vinos ➡️"):
                st.session_state.current_wine_page += 1
                st.rerun()
    
    # Opciones adicionales
    st.markdown("---")
    st.markdown("### 🔄 Opciones Adicionales")
    
    option_col1, option_col2, option_col3 = st.columns(3)
    
    with option_col1:
        if st.button("🔄 Comenzar Nueva Búsqueda"):
            reset_session()
            st.rerun()
    
    with option_col2:
        if st.button("⚙️ Ajustar Filtros"):
            st.session_state.page = 'meal_selection'
            # Mantener algunos datos para facilitar la edición
            st.rerun()
    
    with option_col3:
        if st.button("📊 Ver Estadísticas Detalladas"):
            show_detailed_statistics(filtered_wines)

def show_detailed_statistics(filtered_wines):
    """Muestra estadísticas detalladas de los vinos filtrados"""
    st.markdown("### 📊 Análisis Detallado de Resultados")
    
    # Distribución de puntuaciones
    st.markdown("#### 🎯 Distribución de Puntuaciones de Match")
    
    fig_scores = px.histogram(
        filtered_wines, 
        x='prediction_score',
        nbins=20,
        title="Distribución de Probabilidades de Match",
        labels={'prediction_score': 'Probabilidad de Match', 'count': 'Cantidad de Vinos'}
    )
    fig_scores.update_layout(showlegend=False)
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # Análisis por precio
    st.markdown("#### 💰 Relación Precio vs Match")
    
    fig_price = px.scatter(
        filtered_wines,
        x='price',
        y='prediction_score',
        title="Precio vs Probabilidad de Match",
        labels={'price': 'Precio ($)', 'prediction_score': 'Probabilidad de Match'},
        hover_data=['wine_name', 'winery']
    )
    st.plotly_chart(fig_price, use_container_width=True)

# FEEDBACK DEL VINO
def show_wine_feedback():
    st.markdown('<div class="subsection-header">💭 Tu Opinión Cuenta</div>', unsafe_allow_html=True)
    
    wine = st.session_state.selected_wine
    
    st.markdown("### 🍷 Vino Seleccionado")
    
    # Mostrar el vino elegido
    wine_col1, wine_col2 = st.columns([1, 2])
    
    with wine_col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #8B0000, #DC143C);
            border-radius: 15px;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 4em;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            margin: 20px 0;
        ">🍷</div>
        """, unsafe_allow_html=True)
    
    with wine_col2:
        st.markdown(f"#### 🍷 {wine['name']}")
        st.markdown(f"**🏭 Bodega:** {wine['winery']}")
        #st.markdown(f"**📍 Región:** {wine['region']}")
        st.markdown(f"**💰 Precio:** ${wine['price']:.0f}")
        st.markdown(f"**🎯 Match Predicho:** {wine['prediction_score']:.1%}")
        
        # Características de sabor
        st.markdown("**👅 Perfil de Sabor:**")
        taste_info = f"Cuerpo {wine['body']}, Taninos {wine['tannins']}, Dulzura {wine['sweetness']}, Acidez {wine['acidity']}"
        st.markdown(f"*{taste_info}*")
    
    st.markdown("---")
    
    # Pregunta de feedback
    st.markdown("### ❓ ¿Te gustó este vino?")
    st.markdown("Tu respuesta nos ayuda a mejorar nuestras recomendaciones futuras.")
    
    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 1])
    
    with feedback_col1:
        if st.button("👍 ¡Sí, me gustó!", type="primary", help="El vino cumplió o superó tus expectativas"):
            if save_feedback_to_csv(wine, True):
                st.session_state.page = 'feedback_thanks'
                st.session_state.user_liked_wine = True
                st.rerun()
    
    with feedback_col2:
        if st.button("👎 No me gustó", type="secondary", help="El vino no fue de tu agrado"):
            if save_feedback_to_csv(wine, False):
                st.session_state.page = 'feedback_thanks'
                st.session_state.user_liked_wine = False
                st.rerun()
    
    with feedback_col3:
        if st.button("🔙 Volver a Resultados", help="Elegir otro vino"):
            st.session_state.page = 'wine_results'
            st.rerun()
    
    # Información sobre el feedback
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Sobre tu Feedback
    
    **¿Para qué usamos tu respuesta?**
    - 📈 Mejoramos la precisión de nuestro modelo de recomendaciones
    - 🎯 Personalizamos futuras sugerencias para usuarios con gustos similares
    - 📊 Analizamos patrones de preferencias por región, precio y tipo de vino
    
    **Tu privacidad:**
    - No guardamos información personal identificable
    - Los datos se usan únicamente para mejorar el sistema
    - Puedes solicitar la eliminación de tus datos en cualquier momento
    """)

# PÁGINA DE AGRADECIMIENTO
def show_feedback_thanks():
    st.markdown('<div class="subsection-header">🙏 ¡Gracias por tu Feedback!</div>', unsafe_allow_html=True)
    
    liked = st.session_state.get('user_liked_wine', True)
    wine_name = st.session_state.selected_wine['name']
    
    if liked:
        st.success(f"🎉 ¡Excelente! Nos alegra que hayas disfrutado el {wine_name}")
        st.balloons()
        
        st.markdown("""
        ### 🌟 ¡Tu gusto es excelente!
        
        Tu feedback positivo nos ayuda a:
        - ✅ Confirmar que nuestro modelo está funcionando bien
        - 🎯 Recomendar vinos similares a otros usuarios
        - 📈 Mejorar la precisión de futuras recomendaciones
        """)
    else:
        st.info(f"😔 Lamentamos que el {wine_name} no haya sido de tu agrado")
        
        st.markdown("""
        ### 🔍 Tu feedback es valioso
        
        Aunque este vino no fue perfecto, tu respuesta nos ayuda a:
        - 🎯 Ajustar nuestro modelo para mejores recomendaciones
        - 📊 Entender mejor los patrones de preferencias
        - ⚡ Ofrecerte mejores opciones en el futuro
        """)
    
    # Estadísticas de feedback acumulado
    st.markdown("---")
    st.markdown("### 📊 Impacto de tu Contribución")
    
    total_feedback = len(st.session_state.feedback_data)
    if total_feedback > 0:
        positive_feedback = sum(1 for feedback in st.session_state.feedback_data if feedback.get('liked', 0) == 1)
        satisfaction_rate = (positive_feedback / total_feedback) * 100
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("📝 Total de Evaluaciones", total_feedback)
        
        with stats_col2:
            st.metric("👍 Evaluaciones Positivas", positive_feedback)
        
        with stats_col3:
            st.metric("📈 Tasa de Satisfacción", f"{satisfaction_rate:.1f}%")
    
    # Botones de acción
    st.markdown("---")
    st.markdown("### 🚀 ¿Qué quieres hacer ahora?")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🍷 Buscar Otro Vino", type="primary"):
            # Volver a los resultados
            st.session_state.page = 'wine_results'
            st.rerun()
    
    with action_col2:
        if st.button("🔄 Nueva Búsqueda Completa"):
            reset_session()
            st.rerun()
    
    with action_col3:
        if st.button("🏠 Volver al Inicio"):
            reset_session()
            st.rerun()
    
    # Mensaje de despedida personalizado
    st.markdown("---")
    if liked:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
        ">
            <h4>🍷 ¡Salud! 🥂</h4>
            <p>Esperamos que disfrutes mucho este vino. No olvides maridar con tu comida elegida: 
            <strong>{}</strong></p>
            <p>¡Vuelve pronto para descubrir más vinos perfectos para ti!</p>
        </div>
        """.format(st.session_state.selected_meal), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #6c757d, #495057);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
        ">
            <h4>🎯 ¡No te rindas!</h4>
            <p>El mundo del vino es muy diverso. Seguramente hay un vino perfecto esperándote.</p>
            <p>¡Prueba con diferentes filtros o comidas para encontrar tu match ideal!</p>
        </div>
        """, unsafe_allow_html=True)

# APLICACIÓN PRINCIPAL
def show_interactive_recommender(trained_model):
    init_session_state()
    
    st.header("🍷 Recomendador Interactivo de Vinos")

    #CSS personalizado (usando el estilo de tu aplicación)
    st.markdown("""
    <style>
    .section-header {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 20px 0;
        padding: 20px;
    }
    
    .subsection-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #2E86AB;
        border-bottom: 3px solid #F18F01;
        padding: 15px 0;
        margin: 25px 0 20px 0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .highlight-box h3 {
        margin-top: 0;
        color: white !important;
    }
    
    .conclusion-box {
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 15px;
        padding: 25px;
        margin: 30px 0;
    }
    
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enrutador de páginas
    page = st.session_state.get('page', 'home')
    
    if page == 'home':
        show_home_page()
    elif page == 'meal_selection':
        show_meal_selection()
    elif page == 'price_selection':
        show_price_selection()
    elif page == 'taste_selection':
        show_taste_selection()
    elif page == 'grape_selection':
        show_grape_selection()
    elif page == 'wine_results':
        show_wine_results(trained_model)
    elif page == 'wine_feedback':
        show_wine_feedback()
    elif page == 'feedback_thanks':
        show_feedback_thanks()

