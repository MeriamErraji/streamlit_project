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
    indicateur_offensif = pd.read_csv(os.path.join(data_folder, "indicateurs_offensifs.csv"), index_col=0, parse_dates=True)
    metriques_ptf_arbitre = pd.read_excel(os.path.join(metrics_folder, "metriques_ptf_arbitre.xlsx"))
    indicateur_comparaison = pd.read_csv(os.path.join(data_folder, "indicateur_comparaison.csv"), index_col=0, parse_dates=True)
    metriques_initial_comparable = pd.read_excel(os.path.join(metrics_folder, "metriques_initial_comparable.xlsx"))
    bench_matrix = pd.read_csv(os.path.join(data_folder, "bench_matrix.csv"), index_col=0, parse_dates=True)
    bench_indicateur = pd.read_csv(os.path.join(data_folder, "indicateur_benchmark.csv"), index_col=0, parse_dates=True)
   
    # Données du portefeuille arbitré
    df_poids_offensif = pd.read_csv(os.path.join(data_folder, "df_poids_offensif.csv"))
    indicateurs_offensifs = pd.read_csv(os.path.join(data_folder, "indicateurs_offensifs.csv"), index_col=0, parse_dates=True)
    
    return bench_indicateur, bench_matrix, df_weight,metriques_initial_comparable,indicateur_comparaison, indicateur_ptf, indicateur_benchmark, eur_price_matrix, metriques_ptf_initial, indicateur_offensif, metriques_ptf_arbitre, indicateurs_offensifs, df_poids_offensif

bench_indicateur, bench_matrix, df_weight, metriques_initial_comparable, indicateur_comparaison, indicateur_ptf, indicateur_benchmark, eur_price_matrix, metriques_ptf_initial, indicateurs_offensif, metriques_ptf_arbitre, indicateurs_offensifs, df_poids_offensif= load_data()

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
page = st.sidebar.radio("Aller à :", ["Accueil", "Analyse détaillée", "Portefeuille Arbitré", "Comparaison Portefeuilles"])

# Affichage de la page
if page == "Accueil":
    st.image(os.path.join(data_folder, "Davoust_Patrimoine.png"), use_container_width=True) 
    st.markdown("# 📊 **Suivi et Analyse du Portefeuille**")
    st.write("Bienvenue sur la plateforme d'analyse de portefeuille de Davoust Patrimoine.")
    
    # Menu déroulant pour choisir le portefeuille
    choix_portefeuille = st.selectbox("Sélectionner le portefeuille :", ["Voir mon portefeuille actuel", "Proposition d'arbitrage"])
    
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
        fig.add_shape(type="rect", x0=start, x1=end, y0=1e6, y1=2.7e6, fillcolor="gray", opacity=0.3, layer="below", line_width=0)
     
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Valeur (€)", yaxis_range=[1e6, 2.7e6], xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    if choix_portefeuille=="Voir mon portefeuille actuel":
        # Affichage des indicateurs de risque et performance avec un meilleur style
        st.markdown("### 📊 Indicateurs de Risque et Performance du portefeuille initial")
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
                <b>Tracking Error (VS benchmark):</b> {metriques_ptf_initial.iloc[0, 3]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else : 
        st.markdown("### 📊 Indicateurs de Risque et Performance du portefeuille offensif")
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
                <b>Rendement Annuel :</b> {metriques_ptf_arbitre.iloc[0, 1]}<br>
                <b>Volatilité Annuelle :</b> {metriques_ptf_arbitre.iloc[0, 2]}<br>
                <b>Tracking Error (VS le portefeuille initial) :</b> {metriques_ptf_arbitre.iloc[0, 3]}
            </div>
            """,
            unsafe_allow_html=True
        )

     
elif page == "Analyse détaillée":
    st.markdown("# 📈 **Analyse détaillée du portefeuille**")
    
    # Partie 1 : Introduction et répartition initiale
    st.write("Après avoir extrait les données de Bloomberg, nous constatons l'analyse de la composition et de la performance du portefeuille.")
    st.write("Tout d’abord, nous avons recherché la répartition initiale des fonds :")
    st.write("En regroupant ces fonds par classe d’actifs, on observe une répartition fidèle au profil de risque souscrit initialement.")
    st.write("D’après le graphique et la répartition par classe d’actifs, le portefeuille a bénéficié de la bonne performance des marchés européens, dont l’évolution reste globalement positive, avec 29,98 % du portefeuille investis en actions européennes. Les actions américaines affichent également des courbes très haussières, ce qui a été très positif.")
    
    # Graphique 1 : Évolution des fonds du portefeuille
    st.markdown("### Évolution des fonds du portefeuille")
    st.write("Le graphique ci-dessous présente l'évolution des valeurs des fonds, mettant en lumière la performance globale des actions européennes et américaines.")
    fig_fonds = px.line(eur_price_matrix, x=eur_price_matrix.index, y=eur_price_matrix.columns, 
                        title="Évolution des fonds", labels={"value": "Valeur (€)"})
    st.plotly_chart(fig_fonds, use_container_width=True)
    
    # Partie 2 : Analyse complémentaire de la répartition
    st.write("L’évolution des obligations montre un impact contrasté, notamment avec des obligations émergentes affichant une forte volatilité. Par exemple, le fonds EdRF Emerging Bonds A EUR H a connu plusieurs baisses significatives, pénalisant ainsi la rentabilité globale.")
    st.write("Les actions japonaises présentent une performance mitigée avec des fluctuations fréquentes, tandis que les fonds diversifiés affichent une tendance plus modérée, ne contribuant que partiellement à la performance globale.")
    st.write("Dans l’ensemble, les actions européennes et américaines dynamisent le portefeuille. En revanche, certaines obligations, malgré leur poids conséquent, ont souffert des variations des taux d’intérêt et des incertitudes économiques, tandis que les fonds diversifiés ont joué un rôle d’amortisseur lors des phases de volatilité.")
    
    # Graphique 2 : Composition du portefeuille initial
    st.markdown("### Composition du portefeuille initial")
    st.write("Le diagramme ci-après illustre la répartition initiale des fonds, avec une part importante d’actions européennes et une contribution notable des fonds américains.")
    fig_comp = px.pie(df_weight, values='Poids%', names='Isin', title="Répartition du portefeuille initial")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Partie 3 : Analyse des rendements et de la volatilité
    st.write("Pour analyser le portefeuille, nous avons calculé les rendements logarithmiques et les volatilités des différents fonds à partir de leurs prix. Toutes les matrices nécessaires se trouvent dans le fichier donnees_complete.xlsx.")
    st.write("Le rendement annualisé du portefeuille est de 6,51 %, ce qui traduit une performance positive sur la période. Avec un profil dynamique, les actions américaines et européennes ont contribué à cette progression.")
    st.write("La méthode des rendements logarithmiques permet de capturer les variations relatives sans être influencée par le niveau absolu des prix, sur une base de 252 jours de trading. La volatilité annualisée de 10,7 % confirme une exposition significative aux marchés actions.")
    
    # Graphique 3 : Volatilité glissante du portefeuille
    st.markdown("### Volatilité glissante du portefeuille")
    st.write("Le graphique suivant présente la volatilité glissante, calculée sur une fenêtre de 252 jours, qui illustre l’évolution du risque au fil du temps.")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=indicateur_ptf.index, y=indicateur_ptf["roll_vol"], mode='lines',
                                 name="Volatilité", line=dict(color='red')))
    fig_vol.update_layout(title="Volatilité", xaxis_title="Date", yaxis_title="Volatilité glissante",
                          yaxis_range=[0.058, 0.18])
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Graphique 4 : Volatilité EWMA du portefeuille
    st.markdown("### Volatilité EWMA du portefeuille")
    st.write("La volatilité EWMA, qui applique des pondérations décroissantes aux rendements historiques, offre une estimation de la volatilité future.")
    fig_vol_ewma = go.Figure()
    fig_vol_ewma.add_trace(go.Scatter(x=indicateur_ptf.index, y=indicateur_ptf["vol_ewma"], mode='lines',
                                      name="Volatilité EWMA", line=dict(color='red')))
    fig_vol_ewma.update_layout(title="Volatilité EWMA", xaxis_title="Date", yaxis_title="Volatilité EWMA",
                               yaxis_range=[0, 0.20])
    st.plotly_chart(fig_vol_ewma, use_container_width=True)
    
    # Partie 4 : Interprétation de la volatilité et de la VaR
    st.write("Interprétation du graphique de la volatilité et de la VaR (à partir de 2021) :")
    st.write("Au début de 2021, la volatilité reste stable autour de 0,12 à 0,14, indiquant un marché apaisé après la reprise post-COVID. Progressivement, dès fin 2021-début 2022, une remontée de la VaR et de la volatilité signale un retournement haussier.")
    st.write("Cette augmentation du risque correspond aux premières tensions inflationnistes de 2022, lorsque la Fed et la BCE amorcent un resserrement monétaire, entraînant une correction des marchés.")
    
    # Graphique 5 : Value at Risk (VaR)
    st.markdown("### Value at Risk (VaR)")
    st.write("Le graphique ci-dessous illustre la montée du risque, avec une augmentation progressive de la VaR en phase avec les tensions économiques et géopolitiques.")
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=indicateur_ptf.index, y=indicateur_ptf["VaR Norm(95%, 1Y)"], mode='lines',
                                 name="VaR", line=dict(color='blue')))
    fig_var.update_layout(title="VaR", xaxis_title="Date", yaxis_title="VaR", yaxis_range=[-0.30, 0.0])
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Partie 5 : Impact de la guerre en Ukraine avec bouton d'affichage des détails
    st.write("Le choc de 2022 et l’impact de la guerre en Ukraine :")
    with st.expander("Impact de la guerre en Ukraine"):
        st.write(
            "L’invasion de l’Ukraine par la Russie en février 2022 a généré une incertitude majeure, entraînant une crise énergétique en Europe, des pressions inflationnistes supplémentaires et une détérioration du climat économique mondial. "
            "Sur le graphique, la volatilité atteint un pic autour de mi-2022 avant de diminuer progressivement, tandis que le VaR augmente, traduisant un risque de pertes extrêmes accru face à l'incertitude."
        )
    
    # Partie 6 : Stabilisation et performance post-crise
    st.write("Stabilisation progressive fin 2022 - 2023 :")
    st.write("Après le pic, la volatilité décroît progressivement fin 2022 et début 2023, alors que le marché s’ajuste aux politiques monétaires restrictives et digère les impacts de l’inflation. Des rebonds ponctuels apparaissent en lien avec les décisions des banques centrales et les craintes de récession mondiale.")
    st.write("Un creux de volatilité observé mi-2023 suggère une stabilisation des marchés, accompagné d’une diminution de la VaR.")
    st.header("Analyse des performances :")
    st.write("Votre portefeuille affiche une volatilité annualisée de 10,7 %, supérieure à celle du benchmark (7,45 %), et un rendement annualisé de 6,51 % contre 2,26 % pour le benchmark. La Tracking Error de 6,76 % indique une divergence sensible par rapport au benchmark.")
    
    # Benchmark composite
    st.write("Le benchmark composite a été construit à partir de deux indices référents :")
    st.write("- **MXWD Index** : représentant le segment actions, utilisé avec une pondération de 60 %. Cet indice suit la performance des actions internationales et est largement utilisé pour mesurer la dynamique du marché actions.")
    st.write("- **LEGATRUU** : représentant le segment obligataire, utilisé avec une pondération de 40 %. Cet indice reflète la performance du marché obligataire, incluant obligations d'État et d'entreprise.")
    st.write("Cette répartition (60 % actions / 40 % obligations) permet de comparer la performance du portefeuille à un benchmark équilibré qui intègre à la fois le potentiel de croissance des actions et la stabilité des obligations.")
    
    # Affichage du benchmark composite
    st.markdown("### Évolution du Benchmark Composite")
    st.write("Le graphique ci-dessous présente l'évolution du benchmark composite.")
    fig_bench = px.line(bench_matrix, x=bench_matrix.index, y=bench_matrix.columns, 
                        title="Évolution du Benchmark Composite", labels={"value": "Valeur"})
    st.plotly_chart(fig_bench, use_container_width=True)
    
    
    # Partie 7 : Analyse des risques et périodes de crise
    st.header("Analyse des risques et périodes de crise :")
    st.write("Les indicateurs montrent que le **portefeuille a subi des baisses significatives (drawdown maximal d’environ -25 %)** avant de rebondir grâce à sa composition. La VaR reste élevée, signalant un risque important en cas de nouvelles tensions.")
    
    st.write("Les rendements et les volatilités des différentes catégories d'actifs ont été calculés à partir des prix à long terme (rendements par logarithmes et volatilités comme écart type annualisé).")
    st.write("Pour garantir la comparabilité, les prix ont été convertis en euros via les taux de change correspondants.")
    st.write("La volatilité glissante est calculée avec une moyenne mobile sur 252 jours, tandis que la volatilité EWMA attribue des pondérations décroissantes aux rendements historiques.")
    st.write("La VaR paramétrique normale utilise la volatilité EWMA et la fonction de distribution normale pour estimer le risque de perte maximale potentielle, et le Maximum Drawdown représente la plus grande baisse sur la période d'analyse.")
    
    # Graphique 6 : Drawdown du portefeuille
    st.markdown("### Drawdown du portefeuille")
    st.write("Le graphique suivant présente le drawdown du portefeuille, illustrant les périodes de baisse avant le rebond, avec un drawdown maximal d’environ -25 %.")
    Daily_Drawdown = (indicateur_ptf["Ptf_value"].cummax() - indicateur_ptf["Ptf_value"]) / indicateur_ptf["Ptf_value"].cummax() * 100
    fig_drawdown = px.area(x=indicateur_ptf.index, y=Daily_Drawdown, title="Drawdown du portefeuille", 
                           labels={"y": "Drawdown (%)"})
    st.plotly_chart(fig_drawdown, use_container_width=True)

# Ajout de la section pour le portefeuille arbitré
if page == "Portefeuille Arbitré":
    st.markdown("# 📈 **Analyse du Portefeuille Arbitré**")
    
    st.write("Le portefeuille offensif a été conçu pour maximiser le rendement en captant les tendances technologiques majeures tout en maintenant un équilibre entre diversification et gestion du risque. L’objectif est d’exploiter la dynamique des secteurs à forte croissance tels que l’intelligence artificielle, la blockchain, la robotique et les infrastructures technologiques, tout en intégrant des actifs stratégiques qui assurent une stabilité financière sur le long terme.")
    st.write("L’allocation repose sur une double approche :")
    st.write("• Des fonds d’investissement spécialisés qui garantissent une diversification optimisée et une gestion active des tendances du marché.")
    st.write("• Des actions individuelles qui permettent une exposition directe aux leaders technologiques et industriels.")
    st.write("Afin de limiter l’exposition aux risques spécifiques d’un seul secteur, nous avons inclus des entreprises et des fonds qui, bien qu’étant liés aux nouvelles technologies, bénéficient aussi d’activités diversifiées.")
    
    st.write("Le portefeuille arbitré a été rééquilibré afin d'optimiser le rapport rendement/risque. La proposition d'arbitrage vise à améliorer la performance globale en augmentant le rendement annualisé tout en maîtrisant la volatilité par rapport au portefeuille initial. Les indicateurs recalculés montrent une progression soutenue de la valeur du portefeuille, avec une volatilité maîtrisée et une diminution de la Tracking Error par rapport au portefeuille initial.")
    
    # Graphique 1 : Évolution du portefeuille arbitré
    st.markdown("### Évolution du Portefeuille Arbitré")
    st.write("""
    Le graphique ci-dessous présente l'évolution de la valeur du portefeuille arbitré au fil du temps. 
    On observe une tendance ascendante régulière, avec quelques zones de correction correspondant aux périodes de crise 
    (représentées par les zones grisées). Cela démontre la capacité du portefeuille à absorber les chocs de marché 
    tout en poursuivant sa progression globale.
    """)
    
    value_column = "Valeur_Ptf" if "Valeur_Ptf" in indicateurs_offensifs.columns else "Ptf_value"
    
    fig_arb = go.Figure()
    fig_arb.add_trace(go.Scatter(
        x=indicateurs_offensifs.index,
        y=indicateurs_offensifs[value_column],
        mode='lines',
        name="Valeur du portefeuille",
        line=dict(color='orange')
    ))
    
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
    
    # Graphique 2 : Composition du portefeuille arbitré
    st.markdown("### Composition du Portefeuille Arbitré")
    st.write("""
    Le diagramme ci-dessous illustre la répartition des poids entre les différents actifs du portefeuille arbitré. 
    Cette composition a été optimisée pour se rapprocher d'une volatilité cible de 20 %, avec une répartition équilibrée 
    mais privilégiant les actifs monétaires et obligataires pour limiter les fluctuations, tout en maintenant une exposition aux actions pour stimuler la croissance.
    """)
    
    fig_comp = px.pie(
        df_poids_offensif,
        values='Poids%',
        names='Isin',
        title="Répartition du Portefeuille Arbitré"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Insertion du texte mot pour mot sur l'interprétation de la répartition et des choix d'actifs
    st.markdown("""
    **En observant la répartition du portefeuille arbitré, on remarque qu’une partie importante est allouée à des titres et des fonds fortement exposés aux secteurs technologiques et à l’innovation (intelligence artificielle, robotique, data centers, etc.). Cette orientation se reflète dans plusieurs choix clés :**

    **NVIDIA (NVDA US)** : Pondération la plus élevée. Leader dans les GPU et l’infrastructure de calcul pour l’IA et les supercalculateurs. Son allocation importante (28,5 % environ) souligne la volonté de capter la croissance exceptionnelle du secteur de l’IA. En contrepartie, cela ajoute de la volatilité au portefeuille, car NVIDIA peut connaître des fluctuations marquées en fonction des annonces de résultats et des cycles d’investissement technologique.

    **Amundi US Equity Fund (LU1883854199)** : Deuxième plus gros poids. Ce fonds expose le portefeuille aux grandes capitalisations américaines (Apple, Microsoft, Alphabet, etc.). Il combine croissance et résilience, contribuant à la fois à la performance et à la réduction de la volatilité globale.

    **BGF European Value A2 (LU0072462186)** : Troisième position notable. En investissant dans des entreprises européennes à forte valeur intrinsèque, ce fonds joue un rôle de stabilisateur au sein d’un portefeuille offensif. Il aide à modérer les fluctuations liées aux secteurs plus cycliques ou purement technologiques.

    **D’autres actifs, comme Baidu AI (9888 HK), Pictet Robotics (PFLDCPE LX), ASML (ASML NA) ou Siemens (SIE GY)**, renforcent cette dimension de diversification, permettant de répartir le risque géographiquement (Chine, Europe) et sectoriellement (semi-conducteurs, IA industrielle, robotique, etc.). La présence de BGF European High Yield Bond (LU1191877379), bien qu’à un poids plus faible, témoigne également de la volonté de conserver une poche obligataire pour amortir les chocs de marché et contribuer à la stabilité du portefeuille.
    """, unsafe_allow_html=True)
    
    # Présentation des fonds et actions stratégiques
    st.header("📈 Les Fonds et Actions Stratégiques")
    st.write("""
    La sélection des fonds et actions au sein du portefeuille arbitré repose sur des critères de performance, de stabilité et de complémentarité. Chaque actif a été choisi pour sa capacité à contribuer à la croissance globale tout en atténuant la volatilité du portefeuille.
    """)
    fonds_descriptions = {
        "LU1883854199": "Amundi US Equity Fund : Ce fonds est un pilier du portefeuille grâce à son exposition aux grandes capitalisations américaines comme Apple, Microsoft et Alphabet. Il combine croissance et résilience, en captant la performance des leaders technologiques tout en offrant une certaine stabilité. Son poids important reflète sa capacité à atténuer la volatilité tout en maximisant le rendement.",
        "BEAN SW": "Belimo : Cet acteur de la gestion énergétique est essentiel pour la croissance des infrastructures technologiques, notamment les data centers. Son rôle dans l’efficacité énergétique lui permet de profiter de l’essor de l’IA tout en restant un actif défensif. Sa faible corrélation avec le secteur purement technologique en fait un stabilisateur du portefeuille.",
        "SIE GY": "Siemens : Leader dans l’automatisation et les infrastructures industrielles, Siemens joue un rôle central dans la digitalisation et l’IA industrielle. Son exposition aux transitions énergétiques et à l’industrie 4.0 lui permet de combiner croissance et stabilité, ce qui justifie son allocation significative dans le portefeuille.",
        "ASML NA": "ASML : Cette entreprise est un maillon essentiel de la chaîne d’approvisionnement des semi-conducteurs, qui sont la base des innovations en IA et en informatique. Son monopole sur la lithographie avancée lui confère une position unique sur le marché et justifie son poids stratégique dans le portefeuille.",
        "NVDA US": "NVIDIA : Indispensable pour le développement des modèles d’IA et des supercalculateurs, NVIDIA est un choix stratégique avec un potentiel de croissance immense. Son allocation élevée reflète son rôle clé dans le domaine de l’IA et sa performance historique impressionnante, en faisant un moteur essentiel du portefeuille.",
        "LU1244893696": "EdRF Big Data : Investissement dans les infrastructures de gestion et d’analyse des données. Ce fonds permet d’exploiter la dynamique du Big Data, qui est le socle de nombreuses applications d’IA. Son allocation est justifiée par le besoin croissant en solutions de stockage et d’analyse avancée.",
        "PIRPEUR LX": "Pictet Digital : Ce fonds offre une exposition aux entreprises du Web3 et de la blockchain, qui transforment les modèles économiques numériques. Son inclusion permet une diversification vers des secteurs d’avenir, renforçant la portée offensive du portefeuille.",
        "ALGAATU LX": "Allianz Global Artificial Intelligence : Ce fonds regroupe les principaux acteurs mondiaux du secteur de l’IA, assurant une exposition optimisée aux entreprises les plus innovantes. Son allocation élevée garantit une participation aux avancées majeures du secteur.",
        "FFGLCAE LX": "BGF Next Generation Technology Fund : Un fonds axé sur les technologies émergentes comme l’informatique quantique et la biotechnologie, en complément des valeurs technologiques traditionnelles. Il joue un rôle clé dans la diversification du portefeuille offensif.",
        "9888 HK": "Baidu AI : L’un des leaders chinois de l’intelligence artificielle, ce titre permet une diversification géographique dans un secteur en pleine expansion. Son poids dans le portefeuille reflète son importance stratégique sur le marché asiatique.",
        "LU0154236417": "BGF US Flexible Equity A2 : Un fonds qui permet une allocation flexible sur les marchés américains, optimisant ainsi la répartition du risque. Il complète la stratégie en offrant une couverture aux fluctuations du marché américain.",
        "LU1191877379": "BGF European High Yield Bond : Un fonds obligataire qui permet d’atténuer la volatilité du portefeuille offensif tout en générant du rendement. Son inclusion vise à maintenir un équilibre entre croissance et gestion du risque.",
        "LU1919842267": "ODDO Artificial Intelligence : Ce fonds se concentre sur les applications industrielles et commerciales de l’IA. Il permet une exposition plus large aux entreprises exploitant l’intelligence artificielle dans différents secteurs économiques.",
        "LU0072462186": "BGF European Value A2 : Ce fonds sélectionne des entreprises européennes offrant une valeur intrinsèque forte, ce qui apporte de la stabilité dans un portefeuille offensif. Son allocation vise à diversifier les sources de rendement.",
        "PFLDCPE LX": "Pictet Robotics : En se focalisant sur l’automatisation et la robotique, ce fonds permet de capter les transformations industrielles et les innovations technologiques. Son poids stratégique est justifié par la montée en puissance de l’automatisation mondiale.",
        "LU1893597309": "BSF European Unconstrained Eq : Une allocation flexible sur les marchés européens, permettant de capter les meilleures opportunités du moment tout en optimisant la gestion du risque. Ce fonds garantit une approche plus large pour diversifier le portefeuille offensif.",
    }
    selected_fund = st.selectbox("🔎 Sélectionnez un fonds ou une action pour en savoir plus :", df_poids_offensif['Isin'].tolist())
    
    if selected_fund:
        fond_info = df_poids_offensif[df_poids_offensif['Isin'] == selected_fund].iloc[0]
        st.write(f"{fonds_descriptions.get(selected_fund, selected_fund)}")
        st.write(f"**Part dans le portefeuille :** {fond_info['Poids%']}")
    
    st.markdown(
        """
        **Ce portefeuille allie croissance et résilience, captant les opportunités offertes par les innovations technologiques tout en maintenant une allocation stratégique.**
        """
    )
    
    # Graphique 3 : Volatilité glissante
    st.markdown("### Volatilité glissante")
    st.write("""
    Le graphique suivant présente la volatilité glissante du portefeuille arbitré, calculée sur une fenêtre de 252 jours. 
    On constate que la volatilité reste globalement alignée sur la cible (environ 20%), indiquant une bonne maîtrise du risque 
    malgré quelques fluctuations liées aux chocs de marché.
    """)

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
    

    # Graphique 4 : Volatilité EWMA et Value at Risk (VaR)
    st.markdown("### Volatilité EWMA et Value at Risk (VaR)")
    st.write("""
    Cette section superpose la volatilité EWMA  et la VaR. 
    La volatilité EWMA attribue plus de poids aux données récentes, reflétant la sensibilité du portefeuille aux événements récents. 
    La VaR mesure le risque de perte extrême sur le court terme. Lorsque la VaR augmente, cela indique un risque accru de pertes importantes, 
    bien que l’on observe une retombée progressive de ce risque.
    """)
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
    
    st.markdown("### Volatilité EWMA")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=indicateurs_offensifs.index,
        y=indicateurs_offensifs["vol_ewma"],
        mode='lines',
        name="Volatilité",
        line=dict(color='red')
    ))
    fig_vol.update_layout(
        title="Volatilité EWMA",
        xaxis_title="Date",
        yaxis_title="Volatilité"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    

    # Graphique 5 : Évolution du portefeuille arbitré et Drawdown
    st.markdown("### Drawdown")
    st.write("""
    Ce graphique présente le drawdown , c’est-à-dire la baisse maximale par rapport au dernier sommet. 
    La courbe ascendante de la valeur montre la performance tirée par les secteurs innovants (IA, Big Data, robotique), tandis que le drawdown reste contenu et rapidement résorbé, 
    démontrant la résilience du portefeuille.
    """)
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

    # Conclusion finale sur la performance et la gestion du risque
    st.write("""
    **Conclusion :**  
    Le portefeuille offensif capitalise sur la croissance des secteurs technologiques d’avenir (IA, robotique, Big Data) tout en maintenant une diversification efficace pour limiter les chocs de marché.  
    - La volatilité atteint initialement des niveaux élevés, principalement en raison de la forte exposition à des valeurs technologiques comme NVIDIA et ASML, avant de se stabiliser autour de la cible de 20 %.  
    - La volatilité EWMA et la VaR montrent que le risque de pertes extrêmes est maîtrisé, avec des pics rapidement compensés par des périodes de stabilisation.  
    - Le drawdown, bien que présent lors des périodes de crise, reste contenu et est rapidement résorbé, témoignant de la résilience de l’allocation.  
    En observant la répartition du portefeuille arbitré, on remarque qu’une partie importante est allouée à des titres et des fonds fortement exposés aux secteurs technologiques et à l’innovation (intelligence artificielle, robotique, data centers, etc.). Cette orientation se reflète dans plusieurs choix clés :

    **NVIDIA (NVDA US)** : Pondération la plus élevée. Leader dans les GPU et l’infrastructure de calcul pour l’IA et les supercalculateurs. Son allocation importante (28,5 % environ) souligne la volonté de capter la croissance exceptionnelle du secteur de l’IA. En contrepartie, cela ajoute de la volatilité au portefeuille, car NVIDIA peut connaître des fluctuations marquées en fonction des annonces de résultats et des cycles d’investissement technologique.

    **Amundi US Equity Fund (LU1883854199)** : Deuxième plus gros poids. Ce fonds expose le portefeuille aux grandes capitalisations américaines (Apple, Microsoft, Alphabet, etc.). Il combine croissance et résilience, contribuant à la fois à la performance et à la réduction de la volatilité globale.

    **BGF European Value A2 (LU0072462186)** : Troisième position notable. En investissant dans des entreprises européennes à forte valeur intrinsèque, ce fonds joue un rôle de stabilisateur au sein d’un portefeuille offensif. Il aide à modérer les fluctuations liées aux secteurs plus cycliques ou purement technologiques.

    **D’autres actifs, comme Baidu AI (9888 HK), Pictet Robotics (PFLDCPE LX), ASML (ASML NA) ou Siemens (SIE GY)**, renforcent cette dimension de diversification, permettant de répartir le risque géographiquement (Chine, Europe) et sectoriellement (semi-conducteurs, IA industrielle, robotique, etc.). La présence de BGF European High Yield Bond (LU1191877379), bien qu’à un poids plus faible, témoigne également de la volonté de conserver une poche obligataire pour amortir les chocs de marché et contribuer à la stabilité du portefeuille.
    """)
    
    
elif page == "Comparaison Portefeuilles":
    st.markdown("# 📊 Comparaison des Portefeuilles (2022–2025)")

    st.header("""***Conclusion***""")
    st.write("""L'analyse comparative de nos deux portefeuilles met en lumière des approches d'investissement distinctes et complémentaires. Le portefeuille initial, caractérisé par une allocation plus défensive, a permis de bénéficier d'une performance régulière grâce à une exposition équilibrée aux actions européennes et américaines, tout en maintenant une volatilité modérée. En revanche, le portefeuille arbitré, conçu pour exploiter les secteurs technologiques d'avenir – notamment l'intelligence artificielle, la robotique et le Big Data – affiche un rendement annualisé nettement supérieur, bien que cette stratégie soit associée à une volatilité cible plus élevée. La répartition judicieuse des actifs, combinant des titres à fort potentiel de croissance comme NVIDIA et ASML avec des fonds spécialisés et des investissements défensifs, permet de capter les tendances innovantes du marché tout en limitant l'impact des corrections. En définitive, ces résultats illustrent que le choix entre une approche plus prudente et une stratégie offensive dépend du profil de l'investisseur, chacun présentant ses avantages en termes de performance et de maîtrise du risque, tout en soulignant l'importance d'une diversification et d'un suivi régulier pour s'adapter aux évolutions du marché.""")
    # Définir la période d'analyse
    start_date = "2022-01-01"
    end_date   = "2025-01-01"
    
    # Filtrer les données pour la période souhaitée
    initial_period = indicateur_comparaison.loc[start_date:end_date]
    arbitre_period = indicateurs_offensifs.loc[start_date:end_date]
    
    # Graphique comparatif de l'évolution des valeurs
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=initial_period.index,
        y=initial_period["Ptf_value"],
        mode="lines",
        name="Portefeuille Initial",
        line=dict(color="blue")
    ))
    fig_compare.add_trace(go.Scatter(
        x=arbitre_period.index,
        y=arbitre_period["Valeur_Ptf"],
        mode="lines",
        name="Portefeuille Arbitré",
        line=dict(color="orange")
    ))
    fig_compare.update_layout(
        title="Évolution de la valeur des portefeuilles (2022–2025)",
        xaxis_title="Date",
        yaxis_title="Valeur (€)",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    rendement_init = str(metriques_initial_comparable['Rendement annuel'].iloc[0])
    vol_init       = str(metriques_initial_comparable['Volatilité annuelle'].iloc[0])
    
    rendement_arb  = str(metriques_ptf_arbitre['Rendement annuel'].iloc[0])
    vol_arb        = str(metriques_ptf_arbitre['Volatilité annuelle'].iloc[0])
    
    st.markdown("## Indicateurs Clés")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Portefeuille Initial")
        st.metric("Rendement Annuel", rendement_init)
        st.metric("Volatilité Annuelle", vol_init)
        
    with col2:
        st.markdown("### Portefeuille Arbitré")
        st.metric("Rendement Annuel", rendement_arb)
        st.metric("Volatilité Annuelle", vol_arb)

        
st.success("Analyse terminée !")
