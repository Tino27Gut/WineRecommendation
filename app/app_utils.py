# app_utils.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve, auc

import src.utils.model_prep_eval as mod_prep


def plot_learning_curve(pipeline, X, y, cv, model_name="Classifier", scoring="roc_auc"):
    """
    Genera y muestra una learning curve interactiva en Streamlit con Plotly
    
    Parameters:
    -----------
    pipeline : sklearn pipeline
        Pipeline del modelo entrenado
    X : array-like
        Features
    y : array-like
        Target variable
    cv : cross-validation object
        Objeto de validación cruzada (ej: StratifiedKFold)
    model_name : str
        Nombre del modelo para el título
    scoring : str
        Métrica para evaluar (default: 'roc_auc')
    """
    
    with st.spinner(f'Generando learning curve para {model_name}...'):
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=pipeline,
                X=X,
                y=y,
                cv=cv,
                scoring=scoring,
                train_sizes=np.linspace(0.1, 1.0, 5),
                n_jobs=-1
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_mean = test_scores.mean(axis=1)
            test_std = test_scores.std(axis=1)

            # Crear figura interactiva
            fig = go.Figure()

            # Train
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean,
                mode="lines+markers",
                name=f"Train {scoring.upper()}",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
                error_y=dict(type="data", array=train_std, visible=True)
            ))

            # Test
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_mean,
                mode="lines+markers",
                name=f"Test {scoring.upper()}",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=8),
                error_y=dict(type="data", array=test_std, visible=True)
            ))

            # Layout
            fig.update_layout(
                title=f"Learning Curve - {model_name}",
                xaxis_title="Training Set Size",
                yaxis_title=f"{scoring.upper()} Score",
                template="plotly_white",
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Métricas finales
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Train {scoring.upper()} Final", 
                         f"{train_mean[-1]:.3f}", 
                         f"±{train_std[-1]:.3f}")
            with col2:
                st.metric(f"Test {scoring.upper()} Final", 
                         f"{test_mean[-1]:.3f}", 
                         f"±{test_std[-1]:.3f}")
            st.success("Curva generada ✅")
            return train_sizes, train_scores, test_scores
        except Exception as e:
            st.error(f"Error al generar la learning curve: {str(e)}")
            return None, None, None
    


def display_feature_tags(features, tags_per_row=5):
    """
    Muestra las features seleccionadas como tags estilizados
    
    Parameters:
    -----------
    features : list
        Lista de nombres de features
    tags_per_row : int
        Número de tags por fila
    """
    
    for i in range(0, len(features), tags_per_row):
        cols = st.columns(tags_per_row)
        row_feats = features[i:i + tags_per_row]

        for j, feat in enumerate(row_feats):
            clean_feat = feat.replace('_', ' ').title()
            with cols[j]:
                st.markdown(f"""
                <div style="
                    background-color: #e3f2fd;
                    color: #1976d2;
                    padding: 8px 12px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: 500;
                    margin: 5px 0;
                    text-align: center;
                    border: 1px solid #bbdefb;
                ">{clean_feat}</div>
                """, unsafe_allow_html=True)

        # Rellenar columnas vacías
        for k in range(len(row_feats), tags_per_row):
            with cols[k]:
                st.empty()

def display_hyperparameters(params_dict):
    """
    Muestra los hiperparámetros de forma estilizada
    
    Parameters:
    -----------
    params_dict : dict
        Diccionario con los hiperparámetros
    """
    
    params_text = ""
    for param, value in params_dict.items():
        params_text += f"- **{param}**: **`{value}`**\n"
    
    st.markdown(params_text)

def plot_validation_curve_plotly(pipeline, X, y, cv, param_name, param_range, 
                                model_name="Classifier", scoring="roc_auc"):
    """
    Genera y muestra una curva de validación usando Plotly
    
    Parameters:
    -----------
    pipeline : sklearn pipeline
        Pipeline del modelo
    X : array-like
        Features
    y : array-like
        Target variable
    cv : cross-validation object
        Objeto de validación cruzada
    param_name : str
        Nombre del parámetro a validar (ej: 'rf__n_estimators')
    param_range : list
        Rango de valores del parámetro
    model_name : str
        Nombre del modelo para el título
    scoring : str
        Métrica para evaluar (default: 'roc_auc')
    """
    
    with st.spinner(f'Generando curva de validación para {model_name}...'):
        try:
            train_scores, test_scores = validation_curve(
                estimator=pipeline,
                X=X,
                y=y,
                param_name=param_name,
                param_range=param_range,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_mean = test_scores.mean(axis=1)
            test_std = test_scores.std(axis=1)

            # Crear gráfico con Plotly
            fig = go.Figure()

            # Training scores
            fig.add_trace(go.Scatter(
                x=param_range,
                y=train_mean,
                mode='lines+markers',
                name=f'Train {scoring.upper()}',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=train_std,
                    visible=True,
                    color='#1f77b4',
                    thickness=1.5,
                    width=3
                )
            ))

            # Test scores
            fig.add_trace(go.Scatter(
                x=param_range,
                y=test_mean,
                mode='lines+markers',
                name=f'Test {scoring.upper()}',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8, symbol='square'),
                error_y=dict(
                    type='data',
                    array=test_std,
                    visible=True,
                    color='#ff7f0e',
                    thickness=1.5,
                    width=3
                )
            ))

            # Configurar layout
            param_display = param_name.split('__')[-1] if '__' in param_name else param_name
            fig.update_layout(
                title=f"Curva de Validación - {param_display} ({model_name})",
                xaxis_title=param_display,
                yaxis_title=f"{scoring.upper()} Score",
                template="plotly_white",
                height=400,
                showlegend=True,
                legend=dict(x=0.7, y=0.1)
            )

            fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')

            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar métricas del mejor parámetro
            best_idx = np.argmax(test_mean)
            best_param_val = param_range[best_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mejor Parámetro", f"{best_param_val}")
            with col2:
                st.metric(f"Train {scoring.upper()}", f"{train_mean[best_idx]:.3f}")
            with col3:
                st.metric(f"Test {scoring.upper()}", f"{test_mean[best_idx]:.3f}")
            
            return train_scores, test_scores
            
        except Exception as e:
            st.error(f"Error al generar la curva de validación: {str(e)}")
            return None, None


def plot_feature_importance(model, feature_names, model_name="Random Forest", top_n=15):
    """
    Muestra la importancia de features para modelos tree-based
    """
    try:
        # Obtener importancias
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            st.warning("El modelo no tiene importancias de features disponibles")
            return

        # Crear DataFrame y ordenar
        import pandas as pd
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)

        # Crear gráfico
        fig = px.bar(
            feature_imp, 
            x='importance', 
            y='feature',
            orientation='h',
            title=f"Top {top_n} Features más importantes - {model_name}",
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al mostrar importancia de features: {str(e)}")

@st.cache_data
def plot_selectkbest(features_scores: pd.DataFrame, highlight_features: list = None):
    """
    Genera un gráfico interactivo de barras para mostrar la importancia de features.
    
    Parameters
    ----------
    features_scores : pd.DataFrame
        DataFrame con columnas ["features", "scores", "pvalues"].
    highlight_features : list, optional
        Lista de features a resaltar en otro color. Default = None.
    
    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Figura de Plotly lista para mostrar en Streamlit con st.plotly_chart.
    """

    # Definir columna auxiliar para colores
    features_scores = features_scores.copy()
    if highlight_features:
        features_scores["selected"] = features_scores["features"].isin(highlight_features)
    else:
        features_scores["selected"] = False

    # Crear gráfico
    fig = px.bar(
        features_scores,
        y="features",
        x="scores",
        color="selected",
        color_discrete_map={True: "crimson", False: "steelblue"},
        hover_data=["pvalues"],
        orientation="h"
    )

    fig.update_layout(
        title="📊 Poder predictivo de features (f_classif)",
        xaxis_title="Score (f_classif)",
        yaxis_title="Features",
        legend_title="Seleccionadas",
        height=800
    )

    return fig


# =============================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================

@st.cache_data
def load_base_datasets(ut):
    """
    Carga los datasets base necesarios para el análisis
    
    Parameters:
    -----------
    ut : module
        Módulo de utilidades con get_project_file_path()
        
    Returns:
    --------
    tuple: (wine_df, users_data, grapes_list, region_list, pairings_list)
    """
    try:
        # Cargar datos de vinos
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

        return wine_df, users_data, grapes_list, region_list, pairings_list
        
    except Exception as e:
        st.error(f"Error cargando datasets base: {str(e)}")
        return None, None, None, None, None

def taste_dist_to_optimal(x, min_val, max_val):
    """
    Calcula la distancia de un valor de sabor al rango óptimo del usuario
    """
    if min_val <= x <= max_val:
        return 0
    elif x > max_val:
        return x - max_val
    else:
        return x - min_val

def price_dist_to_optimal(x, min_val, max_val, center):
    """
    Calcula la distancia de un precio al punto óptimo del usuario
    """
    if max_val == 2000:
        # Sin precio máximo: el punto óptimo es el mínimo del rango
        return x - min_val
    else:
        # Con precio máximo: el punto óptimo es el centro del rango
        return x - center

@st.cache_data
def create_recommendation_dataset(wine_df, users_data):
    """
    Crea el dataset de recomendaciones con features de usuario
    
    Parameters:
    -----------
    wine_df : DataFrame
        Dataset de vinos
    users_data : DataFrame
        Dataset de usuarios sintéticos
        
    Returns:
    --------
    DataFrame: Dataset con features de recomendación
    """
    try:
        # Merge de datos
        joined_data = pd.merge(users_data, wine_df, how="inner", on="wine_id")
        
        # Crear dataset de recomendación
        recommend_df = joined_data.copy()
        
        # Agregar información de main pairing
        recommend_df["user_main_pairing"] = recommend_df["user_input"].apply(lambda x: x["main_pairing"])
        recommend_df["has_user_main_pairing"] = recommend_df.apply(
            lambda row: 1 if row[row["user_main_pairing"]] == 1 else 0, axis=1
        )

        # Procesar features de sabor
        taste_list = ["body", "tannins", "sweetness", "acidity"]
        for taste in taste_list:
            recommend_df[f"user_{taste}_min"] = recommend_df["user_input"].apply(
                lambda x: x["tastes"][taste][0]
            )
            recommend_df[f"user_{taste}_max"] = recommend_df["user_input"].apply(
                lambda x: x["tastes"][taste][1]
            )
            recommend_df[f"user_{taste}_center"] = (
                recommend_df[f"user_{taste}_max"] + recommend_df[f"user_{taste}_min"]
            ) / 2
            recommend_df[f"user_{taste}_range"] = (
                recommend_df[f"user_{taste}_max"] - recommend_df[f"user_{taste}_min"]
            )
            recommend_df[f"user_{taste}_diff"] = recommend_df.apply(
                lambda row: taste_dist_to_optimal(
                    x=row[taste],
                    min_val=row[f"user_{taste}_min"],
                    max_val=row[f"user_{taste}_max"]
                ),
                axis=1
            )

        # Procesar features de precio
        recommend_df["user_price_min"] = recommend_df["user_input"].apply(lambda x: x["precio_min"])
        recommend_df["user_price_max"] = recommend_df["user_input"].apply(lambda x: x["precio_max"])
        recommend_df["user_price_max"] = recommend_df["user_price_max"].fillna(2000)
        recommend_df["user_price_range"] = recommend_df["user_price_max"] - recommend_df["user_price_min"]
        recommend_df["user_price_center"] = (
            recommend_df["user_price_max"] + recommend_df["user_price_min"]
        ) / 2
        recommend_df["user_price_diff"] = recommend_df.apply(
            lambda row: price_dist_to_optimal(
                x=row["price"],
                min_val=row["user_price_min"],
                max_val=row["user_price_max"],
                center=row["user_price_center"]
            ),
            axis=1
        )

        return recommend_df
        
    except Exception as e:
        st.error(f"Error creando dataset de recomendaciones: {str(e)}")
        return None

@st.cache_data
def create_model_datasets(recommend_df, grapes_list, region_list, pairings_list):
    """
    Crea los diferentes datasets para modelado
    
    Parameters:
    -----------
    recommend_df : DataFrame
        Dataset de recomendaciones
    grapes_list : list
        Lista de uvas
    region_list : list
        Lista de regiones
    pairings_list : list
        Lista de maridajes
        
    Returns:
    --------
    tuple: (model_df, model_df_nooh, model_df_nocat)
    """
    try:
        # Dataset base para modelado (quita columnas no útiles)
        columns_to_drop = [
            "event_timestamp", "wine_id", "wine_link", "name", "winery", 
            "style", "image",  # Columnas originales del df
            "user_id", "user_input", "prob_like", "user_main_pairing"  # Columnas de datos sintéticos
        ]
        
        model_df = recommend_df.drop(columns=columns_to_drop).copy()

        # Dataset sin one-hot encoding (quita uvas y regiones)
        model_df_nooh = model_df.drop(columns=(grapes_list + region_list)).copy()

        # Dataset sin columnas categóricas (quita maridajes también)
        model_df_nocat = model_df_nooh.drop(columns=pairings_list).copy()

        return model_df, model_df_nooh, model_df_nocat
        
    except Exception as e:
        st.error(f"Error creando datasets de modelado: {str(e)}")
        return None, None, None

def get_complete_datasets(ut):
    """
    Función principal que carga y prepara todos los datasets
    
    Parameters:
    -----------
    ut : module
        Módulo de utilidades
        
    Returns:
    --------
    dict: Diccionario con todos los datasets preparados
    """
    
    with st.spinner('Cargando y preparando datasets...'):
        # Cargar datasets base
        wine_df, users_data, grapes_list, region_list, pairings_list = load_base_datasets(ut)
        
        if wine_df is None:
            st.error("No se pudieron cargar los datasets base")
            return None
        
        # Crear dataset de recomendaciones
        recommend_df = create_recommendation_dataset(wine_df, users_data)
        
        if recommend_df is None:
            st.error("No se pudo crear el dataset de recomendaciones")
            return None
        
        # Crear datasets de modelado
        model_df, model_df_nooh, model_df_nocat = create_model_datasets(
            recommend_df, grapes_list, region_list, pairings_list
        )
        
        if model_df is None:
            st.error("No se pudieron crear los datasets de modelado")
            return None
        
        # Devolver diccionario con todos los datasets
        datasets = {
            'wine_df': wine_df,
            'users_data': users_data,
            'recommend_df': recommend_df,
            'model_df': model_df,
            'model_df_nooh': model_df_nooh,
            'model_df_nocat': model_df_nocat,
            'grapes_list': grapes_list,
            'region_list': region_list,
            'pairings_list': pairings_list
        }
        
        return datasets


# =============================================
# MODEL EVALUATION FUNCTIONS
# =============================================


def evaluate_pipeline(pipeline, X, y, scoring_metrics=None, n_splits=5, random_state=0):
    """
    Evalúa un pipeline con validación cruzada estratificada y devuelve un DataFrame
    con las métricas promedio, desviación estándar y coeficiente de variación.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline a evaluar (ej: con scaler y modelo).
    X : pd.DataFrame o np.ndarray
        Features de entrenamiento.
    y : pd.Series o np.ndarray
        Target binario o multiclase.
    scoring_metrics : list, optional
        Lista de métricas de sklearn para cross_validate.
        Default = ["roc_auc", "accuracy", "precision_macro", "recall_macro", "f1_macro"].
    n_splits : int, optional
        Cantidad de folds para StratifiedKFold. Default = 5.
    random_state : int, optional
        Semilla para reproducibilidad. Default = 0.
    
    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame con columnas ["Mean", "Std", "Coef. Var"] para cada métrica.
    """
    
    if scoring_metrics is None:
        scoring_metrics = ["roc_auc", "accuracy", "precision_macro", "recall_macro", "f1_macro"]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Cross-validation
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring_metrics)

    # Agrupar resultados
    metrics_summary = {metric: scores[f"test_{metric}"] for metric in scoring_metrics}

    metrics_df = pd.DataFrame({
        "Mean": {k: v.mean() for k, v in metrics_summary.items()},
        "Std": {k: v.std() for k, v in metrics_summary.items()},
        "Coef. Var": {k: v.std() / v.mean() for k, v in metrics_summary.items()}
    })

    return metrics_df



# Función auxiliar de evaluate model - Confussion Matrix Interactivo
def plot_cm_interactive(cm, labels, title="", normalize=False):
    """
    Confusion Matrix interactiva con opción de normalización.
    
    Parameters
    ----------
    cm : array-like
        Matriz de confusión (2D).
    labels : list
        Lista de clases (orden de índices/columnas).
    title : str
        Título del gráfico.
    normalize : bool
        Si True, muestra proporciones (%) en vez de valores absolutos.
    """
    cm = np.array(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum()
        cm_df = pd.DataFrame(
            np.round(cm * 100, 1),  # porcentaje con 1 decimal
            index=labels,
            columns=labels
        )
        color_label = "Percentage (%)"
    else:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        color_label = "Count"

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
        aspect="auto",
        labels=dict(x="Predicted label", y="True label", color=color_label)
    )
    
    # ticks solo en enteros (0, 1, 2, ...)
    fig.update_xaxes(
        side="bottom",
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=[str(l) for l in labels]
    )
    
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=[str(l) for l in labels]
    )
    return fig

# Función auxiliar de evaluate model - Classification Report Interactivo
def classification_report_df(y_true, y_pred, labels):
    report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=labels)
    df_report = pd.DataFrame(report_dict).transpose().round(3)
    return df_report

# Función para evaluar modelos
def evaluate_model(pipeline, X, y, features, avg_tkt, churn_like, churn_dislike, test_size=0.2, stratify=True):

    # Split
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y, test_size=test_size, stratify=strat
    )
    
    # Fit
    pipeline.fit(X_train, y_train)

    # --- TRAIN ---
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    cm_train = confusion_matrix(y_train, y_train_pred)
    fig_cm_train = plot_cm_interactive(cm_train, pipeline.classes_, "", normalize=True)
    report_train_df = classification_report_df(y_train, y_train_pred, pipeline.classes_)
    auc_train = roc_auc_score(y_train, y_train_proba)
    eco_train = mod_prep.economic_score(cm_train, avg_tkt, churn_like, churn_dislike)

    # --- TEST ---
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    cm_test = confusion_matrix(y_test, y_test_pred)
    fig_cm_test = plot_cm_interactive(cm_test, pipeline.classes_, "", normalize=True)
    report_test_df = classification_report_df(y_test, y_test_pred, pipeline.classes_)
    auc_test = roc_auc_score(y_test, y_test_proba)
    eco_test = mod_prep.economic_score(cm_test, avg_tkt, churn_like, churn_dislike)

    return {
        "fig_cm_train": fig_cm_train,
        "fig_cm_test": fig_cm_test,
        "report_train": report_train_df,
        "report_test": report_test_df,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "eco_train": eco_train,
        "eco_test": eco_test,
        # - Datos para la curva roc-auc
        "y_test_true": y_test,
        "y_test_proba": y_test_proba,
        "y_train_true": y_train,
        "y_train_proba": y_train_proba,
        "pipeline": pipeline
    }




# Función para generar y plotear la curva ROC
def plot_roc_curve(y_true, y_proba, model_name="Random Forest", dataset="Test"):
    """
    Genera y plotea la curva ROC usando Plotly
    
    Args:
        y_true: array con las etiquetas verdaderas (0s y 1s)
        y_proba: array con las probabilidades predichas por el modelo
        model_name: nombre del modelo para mostrar en el gráfico
        dataset: tipo de dataset (Train/Test)
    """
    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Crear el gráfico
    fig_roc = go.Figure()
    
    # Agregar la curva ROC del modelo
    fig_roc.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(width=3, color='#1f77b4'),
        hovertemplate='<b>Threshold:</b> %{customdata:.3f}<br>' +
                      '<b>FPR:</b> %{x:.3f}<br>' +
                      '<b>TPR:</b> %{y:.3f}<extra></extra>',
        customdata=thresholds
    ))
    
    # Agregar línea de referencia (clasificador aleatorio)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio (AUC = 0.50)',
        line=dict(dash='dash', color='red', width=2),
        hoverinfo='skip'
    ))
    
    # Configurar el layout
    fig_roc.update_layout(
        title=f"Curva ROC - {model_name} ({dataset})",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        height=600,
        legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.95,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        template="plotly_white",
        hovermode='closest'
    )
    
    # Asegurar que el gráfico sea cuadrado y bien proporcionado
    fig_roc.update_xaxes(range=[0, 1], constrain="domain")
    fig_roc.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    
    # Agregar grid
    fig_roc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_roc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig_roc, roc_auc
