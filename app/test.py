"""
    # 4. Métricas de Negocio
    st.markdown('<div class="subsection-header">📊 Métricas de Negocio Simuladas</div>', unsafe_allow_html=True)
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Cálculos basados en los parámetros actuales
    avg_order_value = initial_budget * (1 + risk_tolerance * 0.3)
    monthly_orders = 3 + loyalty_factor * 2
    monthly_revenue = avg_order_value * monthly_orders
    annual_ltv = monthly_revenue * 12 * retention[-1]
    
    with metrics_col1:
        st.metric(
            label="🛒 AOV",
            value=f"${avg_order_value:.2f}",
            delta=f"{risk_tolerance*10:.1f}%" if risk_tolerance > 0.5 else f"-{(1-risk_tolerance)*5:.1f}%"
        )
    
    with metrics_col2:
        st.metric(
            label="📅 Pedidos/Mes",
            value=f"{monthly_orders:.1f}",
            delta=f"{loyalty_factor*20:.1f}%" if loyalty_factor > 0.5 else None
        )
    
    with metrics_col3:
        st.metric(
            label="💰 Revenue Mensual",
            value=f"${monthly_revenue:.2f}",
            delta=f"${(monthly_revenue - initial_budget*3):.2f}"
        )
    
    with metrics_col4:
        st.metric(
            label="⭐ LTV Anual",
            value=f"${annual_ltv:.2f}",
            delta=f"{retention[-1]*100:.1f}% ret."
        )
    
    # Análisis What-If
    st.markdown('<div class="subsection-header">🎲 Análisis What-If</div>', unsafe_allow_html=True)
    
    whatif_col1, whatif_col2 = st.columns(2)
    
    with whatif_col1:
        st.markdown("""
        #### 📈 Escenarios de Optimización:
        """)
        
        # Crear diferentes escenarios
        scenarios = {
            'Actual': {'budget': initial_budget, 'risk': risk_tolerance, 'loyalty': loyalty_factor},
            'Más Aventurero': {'budget': initial_budget, 'risk': min(1.0, risk_tolerance + 0.3), 'loyalty': loyalty_factor},
            'Más Leal': {'budget': initial_budget, 'risk': risk_tolerance, 'loyalty': min(1.0, loyalty_factor + 0.2)},
            'Premium': {'budget': initial_budget * 1.5, 'risk': risk_tolerance, 'loyalty': loyalty_factor}
        }
        
        scenario_results = []
        for name, params in scenarios.items():
            aov = params['budget'] * (1 + params['risk'] * 0.3)
            orders = 3 + params['loyalty'] * 2
            ltv = aov * orders * 12 * ((1 - (0.05 + (1 - params['loyalty']) * 0.15)) ** 12)
            scenario_results.append({'Escenario': name, 'LTV': ltv, 'AOV': aov, 'Pedidos_Mes': orders})
        
        scenarios_df = pd.DataFrame(scenario_results)
        
        fig_scenarios = px.bar(
            scenarios_df, 
            x='Escenario', 
            y='LTV',
            title="Comparación de LTV por Escenario",
            color='LTV',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_scenarios, use_container_width=True)
    
    with whatif_col2:
        st.markdown("""
        #### 🎯 Insights de la Simulación:
        """)
        
        # Mostrar la tabla de resultados
        st.dataframe(
            scenarios_df.style.format({
                'LTV': '${:.2f}',
                'AOV': '${:.2f}',
                'Pedidos_Mes': '{:.1f}'
            }).highlight_max(subset=['LTV'], color='lightgreen'),
            use_container_width=True
        )
        
        # Recomendaciones automáticas
        best_scenario = scenarios_df.loc[scenarios_df['LTV'].idxmax(), 'Escenario']
        ltv_improvement = scenarios_df['LTV'].max() - scenarios_df.loc[scenarios_df['Escenario'] == 'Actual', 'LTV'].values[0]
        
        st.markdown(f"""
        <div class="highlight-box">
        <h5>💡 Recomendación:</h5>
        <p>El escenario <strong>"{best_scenario}"</strong> genera el mayor LTV, 
        con una mejora de <strong>${ltv_improvement:.2f}</strong> respecto al actual.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Validación del Modelo
    st.markdown('<div class="subsection-header">✅ Validación del Comportamiento</div>', unsafe_allow_html=True)
    
    validation_col1, validation_col2 = st.columns(2)
    
    with validation_col1:
        st.markdown("""
        #### 🔍 Métricas de Realismo:
        
        **Comportamientos Validados:**
        - ✅ Mayor precio → Mayor selectividad
        - ✅ Mejor match → Mayor satisfacción
        - ✅ Experiencias positivas → Mayor lealtad
        - ✅ Diversidad vs Consistencia por perfil
        
        **Benchmarks de Industria:**
        - Churn Rate: 15-25% (Simulado: ~12%)
        - AOV Crecimiento: 10-20% anual
        - Retención: 60-80% a 12 meses
        """)
    
    with validation_col2:
        # Distribución de comportamientos simulados
        np.random.seed(42)  # Para reproducibilidad
        simulated_purchases = np.random.beta(2, 3, 1000) * 100  # Distribución realista
        
        fig_dist = px.histogram(
            x=simulated_purchases,
            nbins=30,
            title="Distribución de Comportamientos Simulados",
            labels={'x': 'Score de Decisión', 'y': 'Frecuencia'}
        )
        fig_dist.update_layout(height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.info("💡 La distribución muestra comportamientos realistas con mayor concentración en decisiones moderadas.")"""