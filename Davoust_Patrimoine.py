import pandas as pd
import os
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Suivi et Analyse du Portefeuille", layout="centered")

# Définition du chemin des données
data_folder = "resultats_analyses/donnees"
metrics_folder = "resultats_analyses/metriques"

# Chargement des données
@st.cache_data
def load_data():
    # Données existantes
    df_weight = pd.read_csv(os.path.join(data_folder, "df_weight.csv")).drop(columns=["Unnamed: 0"], errors="ignore")
    indicateur_ptf = pd.read_csv(os.path.join(data_folder, "indicateur_ptf.csv"), index_col=0, parse_dates=True)
    indicateur_benchmark = pd.read_csv(os.path.join(data_folder, "indicateur_benchmark.csv"), index_col=0, parse_dates=True)
    eur_price_matrix = pd.read_csv(os.path.join(data_folder, "eur_price_matrix.csv"), index_col=0, parse_dates=True)
    metriques_ptf_initial = pd.read_excel(os.path.join(metrics_folder, "metriques_ptf_initial.xlsx"))
    
    # Données du portefeuille arbitré
    df_poids_offensif = pd.read_csv(os.path.join(data_folder, "df_poids_offensif.csv"))
    indicateurs_offensifs = pd.read_csv(os.path.join(data_folder, "indicateurs_offensifs.csv"), index_col=0, parse_dates=True)
    
    return df_weight, indicateur_ptf, indicateur_benchmark, eur_price_matrix, metriques_ptf_initial, df_poids_offensif, indicateurs_offensifs

df_weight, indicateur_ptf, indicateur_benchmark, eur_price_matrix, metriques_ptf_initial, df_poids_offensif, indicateurs_offensifs = load_data()

# Après le chargement des données, ajoutez :
print("Colonnes disponibles dans indicateurs_offensifs:")
print(indicateurs_offensifs.columns)

# Définition des périodes de crise
crises = [
    ("2020-02", "2020-05"),  # COVID-19 crash
    ("2022-02", "2022-07"),  # Conflit Ukraine
    ("2023-03", "2023-06")   # Crise bancaire US
]

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Accueil", "Analyse détaillée", "Portefeuille Arbitré"])

# Affichage de la page
if page == "Accueil":
    st.image("Davoust Patrimoine.png", use_column_width=True)
    st.markdown("# 📊 **Suivi et Analyse du Portefeuille**")
    st.write("Bienvenue sur la plateforme d'analyse du portefeuille de Davoust Patrimoine.")
    
    # Menu déroulant pour choisir le portefeuille
    choix_portefeuille = st.selectbox("Sélectionner le portefeuille :", ["Voir mon portefeuille actuel", "Voir une seconde proposition"])
    
    # Affichage en fonction du choix
    if choix_portefeuille == "Voir mon portefeuille actuel":
        df_to_show = indicateur_ptf
        title = "Évolution du Portefeuille Initial"
        value_column = "Ptf_value"  # Colonne pour le portefeuille initial
    else:
        df_to_show = indicateurs_offensifs
        title = "Évolution de la Seconde Proposition"
        value_column = "Valeur_Ptf"  # Colonne pour le portefeuille arbitré
    
    # Graphique interactif avec Plotly et ajout des zones de crise
    st.markdown(f"## {title}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_to_show.index, 
        y=df_to_show[value_column], 
        mode='lines', 
        name="Valeur du portefeuille", 
        line=dict(color='blue')
    ))
    
    for start, end in crises:
        fig.add_shape(type="rect", x0=start, x1=end, y0=1e6, y1=2.4e6, fillcolor="gray", opacity=0.3, layer="below", line_width=0)
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Valeur (€)", yaxis_range=[1e6, 2.4e6], xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # Affichage des indicateurs de risque et performance avec un meilleur style
    st.markdown("### 📊 Indicateurs de Risque et Performance")
    st.markdown(
        f"""
        <style>
            .metric-container {{
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                font-size: 18px;
            }}
        </style>
        <div class="metric-container">
            <b>Rendement Annuel :</b> {metriques_ptf_initial.iloc[0, 1]}<br>
            <b>Volatilité Annuelle :</b> {metriques_ptf_initial.iloc[0, 2]}<br>
            <b>Tracking Error :</b> {metriques_ptf_initial.iloc[0, 3]}
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Analyse détaillée":
    st.markdown("# 📈 **Analyse détaillée du portefeuille**")
    st.write("Ici vous trouverez des analyses plus approfondies sur la composition et les métriques de risque du portefeuille.")
    
    # Graphiques des fonds composant le portefeuille
    st.markdown("### Évolution des fonds du portefeuille")
    fig_fonds = px.line(eur_price_matrix, x=eur_price_matrix.index, y=eur_price_matrix.columns, title="Évolution des fonds", labels={"value": "Valeur (€)"})
    st.plotly_chart(fig_fonds, use_container_width=True)
    
    # Volatilité
    st.markdown("### Volatilité du portefeuille")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=indicateur_ptf.index, y=indicateur_ptf["roll_vol"], mode='lines', name="Volatilité", line=dict(color='red')))
    
    fig_vol.update_layout(title="Volatilité", xaxis_title="Date", yaxis_title="Volatilité", yaxis_range=[0.058, 0.18])
    st.plotly_chart(fig_vol, use_container_width=True)

    # Value at Risk
    st.markdown("### Value at Risk (VaR)")
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=indicateur_ptf.index, y=indicateur_ptf["VaR Norm(95%, 1Y)"], mode='lines', name="VaR", line=dict(color='blue')))
    
    fig_var.update_layout(title="VaR", xaxis_title="Date", yaxis_title="VaR", yaxis_range=[-0.30, 0.0])
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Drawdown en pourcentage
    st.markdown("### Drawdown du portefeuille")
    Daily_Drawdown = (indicateur_ptf["Ptf_value"].cummax() - indicateur_ptf["Ptf_value"]) / indicateur_ptf["Ptf_value"].cummax() * 100
    fig_drawdown = px.area(x=indicateur_ptf.index, y=Daily_Drawdown, title="Drawdown du Portefeuille", labels={"y": "Drawdown (%)"})
    st.plotly_chart(fig_drawdown, use_container_width=True)

# Ajout de la section pour le portefeuille arbitré
if page == "Portefeuille Arbitré":
    st.markdown("# 📈 **Analyse du Portefeuille Arbitré**")
    
    # Vérification des colonnes et utilisation de "Valeur_Ptf" si disponible
    value_column = "Valeur_Ptf" if "Valeur_Ptf" in indicateurs_offensifs.columns else "Ptf_value"
    
    fig_arb = go.Figure()
    fig_arb.add_trace(go.Scatter(
        x=indicateurs_offensifs.index,
        y=indicateurs_offensifs[value_column],
        mode='lines',
        name="Valeur du portefeuille",
        line=dict(color='orange')
    ))
    
    # Mise à jour des limites y selon les données
    y_min = indicateurs_offensifs[value_column].min() * 0.95
    y_max = indicateurs_offensifs[value_column].max() * 1.05
    
    for start, end in crises:
        fig_arb.add_shape(
            type="rect",
            x0=start, x1=end,
            y0=y_min,
            y1=y_max,
            fillcolor="gray",
            opacity=0.3,
            layer="below",
            line_width=0
        )
    
    fig_arb.update_layout(
        title="Évolution du Portefeuille Arbitré",
        xaxis_title="Date",
        yaxis_title="Valeur (€)",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig_arb, use_container_width=True)
    
    # Composition du portefeuille
    st.markdown("### Composition du portefeuille arbitré")
    fig_comp = px.pie(
        df_poids_offensif,
        values='Poids%',
        names='Isin',
        title="Répartition du portefeuille arbitré"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Volatilité glissante
    st.markdown("### Volatilité glissante")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=indicateurs_offensifs.index,
        y=indicateurs_offensifs["Volatilite_glissante"],
        mode='lines',
        name="Volatilité",
        line=dict(color='red')
    ))
    fig_vol.update_layout(
        title="Volatilité glissante",
        xaxis_title="Date",
        yaxis_title="Volatilité"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # VaR
    st.markdown("### Value at Risk (VaR)")
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=indicateurs_offensifs.index,
        y=indicateurs_offensifs["VaR Norm(95%, 1Y)"],
        mode='lines',
        name="VaR",
        line=dict(color='purple')
    ))
    fig_var.update_layout(
        title="Value at Risk (95%)",
        xaxis_title="Date",
        yaxis_title="VaR"
    )
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Drawdown
    st.markdown("### Drawdown du portefeuille")
    drawdown = (indicateurs_offensifs["Valeur_Ptf"].cummax() - indicateurs_offensifs["Valeur_Ptf"]) / indicateurs_offensifs["Valeur_Ptf"].cummax() * 100
    fig_dd = px.area(
        x=indicateurs_offensifs.index,
        y=drawdown,
        title="Drawdown du Portefeuille Arbitré",
        labels={"y": "Drawdown (%)"}
    )
    st.plotly_chart(fig_dd, use_container_width=True)

st.success("Analyse terminée !")
