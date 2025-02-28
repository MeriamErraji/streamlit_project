#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:07:45 2025

@author: Meriam
"""
import pandas as pd
import os
from datetime import datetime
from concurrent import futures
import multiprocessing
from os.path import join
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class PortfolioDataReader:
    def __init__(self, directory_base: str = "donnees_brutes", multithreaded: bool = True):
        """
        Initialise le lecteur de données de portfolio avec support multithread
        @param directory_base: chemin relatif vers le dossier contenant les fichiers Excel
        @param multithreaded: active le traitement multithread si True
        """
        self.directory_base = directory_base
        self.multithreaded = multithreaded
        self.excel_files = [f for f in os.listdir(directory_base) 
                            if f.endswith('.xlsx') and not f.startswith('~')]

    def read_asset_info(self, file_path: str) -> dict:
        """Lit les informations descriptives de l'actif"""
        full_path = join(self.directory_base, file_path)
        header_info = pd.read_excel(full_path, nrows=6, header=None)
        return {
            'security': header_info.iloc[0, 1],
            'start_date': header_info.iloc[1, 1],
            'end_date': header_info.iloc[2, 1],
            'currency': header_info.iloc[4, 1]
        }

    def process_single_file(self, file_name: str) -> tuple:
        """Traite un fichier Excel"""
        try:
            full_path = join(self.directory_base, file_name)

            # Lecture des informations de l'actif
            asset_info = self.read_asset_info(file_name)

            # Lecture et nettoyage des données
            prices = pd.read_excel(full_path, skiprows=6)  #ajout2 : Lecture sans usecols d'abord
            prices.columns = prices.columns.str.strip()  #ajout2 : Nettoyage des colonnes pour éviter les espaces
            
            #print(f"Debug : Colonnes trouvées dans {file_name} ->", prices.columns.tolist())  #ajout2 : Debugging affichage colonnes
            
            if 'Date' not in prices.columns or 'PX_LAST' not in prices.columns:  #ajout2 : Vérification de structure
                raise ValueError(f"Erreur : Colonnes manquantes dans {file_name} ! Colonnes trouvées : {prices.columns.tolist()}")  

            # Sélection et renommage des colonnes
            prices = prices[['Date', 'PX_LAST']]  #ajout2 : Sélection après nettoyage
            prices = prices.rename(columns={'PX_LAST': file_name.replace('.xlsx', '')})  #ajout2 : Renommage propre
            
            prices['Date'] = pd.to_datetime(prices['Date'], format='%d/%m/%Y')  #ajout2 : Conversion en datetime

            #ajout2 : Affichage des informations pour vérification
            #print(f"\nFichier traité : {file_name}")
            #print("Colonnes trouvées :", prices.columns.tolist())
            #print("Premières lignes :")
            #print(prices.head())

            return asset_info, prices

        except Exception as e:
            print(f"Erreur lors du traitement de {file_name} : {str(e)}")  #ajout2 : Message d'erreur détaillé
            return None, None

    def process_all_files(self) -> tuple:
        """Traite tous les fichiers Excel avec support multithread"""
        assets_info = {}
        price_dfs = []

        if self.multithreaded:
            cpu_count = max(2, multiprocessing.cpu_count() - 2)
            print(f"Traitement multithread avec {cpu_count} threads.")

            with futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
                future_to_file = {executor.submit(self.process_single_file, file): file 
                                  for file in self.excel_files}

                for future in futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        asset_info, prices = future.result()
                        if asset_info and prices is not None:
                            assets_info[file] = asset_info
                            price_dfs.append(prices)
                    except Exception as e:
                        print(f"Erreur lors du traitement de {file}: {str(e)}")
        else:
            for file in self.excel_files:
                asset_info, prices = self.process_single_file(file)
                if asset_info and prices is not None:
                    assets_info[file] = asset_info
                    price_dfs.append(prices)

        return assets_info, price_dfs

    def create_price_matrix(self, price_dfs: list) -> pd.DataFrame:
        """Crée la matrice de prix finale"""
        if not price_dfs:
            raise ValueError("Aucune donnée de prix n'a pu être lue correctement")

        merged_df = price_dfs[0].set_index('Date')
        for df in price_dfs[1:]:
            merged_df = merged_df.join(df.set_index('Date'), how='outer')

        merged_df = merged_df.sort_index()
        merged_df = merged_df.ffill()
        return merged_df

    def convert_to_eur(self, price_matrix: pd.DataFrame, assets_info: dict, fx_data: pd.DataFrame) -> pd.DataFrame:
        """Convertit tous les prix en EUR"""
        eur_matrix = price_matrix.copy()

        for column in price_matrix.columns:
            if column != 'EURUSD':  # Ne pas convertir le taux de change
                if column + '.xlsx' in assets_info:  #ajout2 : Vérification que l'ISIN existe bien dans assets_info
                    asset_currency = assets_info[column + '.xlsx']['currency']
                    if asset_currency == 'USD':  # Conversion uniquement pour USD
                        eur_matrix[column] = price_matrix[column] / fx_data['EURUSD']
                else:
                    print(f"Attention : {column}.xlsx non trouvé dans assets_info")  #ajout2 : Alerte si la clé est absente

        return eur_matrix

   
    
def calculate_tracking_error(eur_price_matrix: pd.DataFrame, indicateur_ptf: pd.DataFrame, fx_data: pd.DataFrame) -> float:
    """Calcul de la Tracking Error avec les benchmarks"""
    
    # Chargement des benchmarks
    data_folder = "donnees_benchmark" 
    os.makedirs(data_folder, exist_ok=True)

    reader = PortfolioDataReader(directory_base=data_folder, multithreaded=True)
    
    benchmark_info, price_benchmark = reader.process_all_files()
    bench_matrix = reader.create_price_matrix(price_benchmark)
    
    #print(bench_matrix)
    #print(benchmark_info)
    eur_bench_matrix = reader.convert_to_eur(bench_matrix, benchmark_info, fx_data)
    """
    for asset, info in benchmark_info.items():
        
        print(f"\n{asset}:")
        for key, value in info.items():
            print(f"{key}: {value}")
    """ 
    #print("\nMatrice de prix en EUR:")
    #print(eur_bench_matrix.head())    
    
    # Définition des pondérations du benchmark composite
    df_weightbench = pd.DataFrame({
        'Isin': ["LEGATRUU Index", "MXWD Index"],
        'Poids%': [0.6, 0.4]
    })
    
    print("\nComposition du benchmark composite:")
    print(df_weightbench)
    
    # Extraction des prix du benchmark
    extract_bench = bench_matrix[[ele for ele in df_weightbench['Isin'].unique()]]
    
    # Vérification des dates communes avec le portefeuille
    common_dates = eur_price_matrix.index.intersection(extract_bench.index)
    if common_dates.empty:
        raise ValueError("Erreur : Aucune date commune entre le portefeuille et le benchmark.")
    
    extract_bench = extract_bench.loc[common_dates]
    
    # Calcul de rendements et volatilité du ptf initial
    bench_indicateurs = pd.DataFrame({"prix_initial":extract_bench.dropna().iloc[1],'prix_final':extract_bench.dropna().iloc[-1]})
    
    bench_indicateurs['performance_totale']=(bench_indicateurs['prix_final']/bench_indicateurs['prix_initial'])-1
    
    bench_indicateurs['performance_annualisee']=(1+bench_indicateurs['performance_totale'])**(252/len(extract_bench))-1

    log_bench_rdt = np.log(extract_bench/extract_bench.shift(1)).iloc[1:]
    bench_indicateurs['volatilite'] = log_bench_rdt.std()*252**0.5
    
    cov_bench = log_bench_rdt.cov()
    
    wgt_bench = np.transpose(df_weightbench['Poids%'])
    vol_bench_ini =  (252**0.5)*np.dot(np.transpose(df_weightbench['Poids%']), np.dot(cov_bench, df_weightbench['Poids%']))**0.5
    #testvol =  (252**0.5)*np.dot(np.transpose(wgt), np.dot(cov_bench, wgt_bench))**0.5
      
   # print(wgt)
    print('moi', vol_bench_ini)
    #print('test', testvol)

    # Calcul de le rendement annualisé du ptf initial
    er2=bench_indicateurs['performance_annualisee']
    #ptf_er2=lambda x: np.dot(np.transpose(wgt_bench) ,er2)


    ptf_er2=lambda x: np.dot(wgt_bench ,er2)
    
    rdt_test=ptf_er2(df_weightbench)


    import matplotlib.pyplot as plt
    
    # Création du DataFrame des indicateurs du portefeuille
    indicateur_benchmark = pd.DataFrame()
    
    #Rendement du portefeuille (pondéré par les poids
    indicateur_benchmark["Ptf_rdt"] = np.dot(log_bench_rdt, wgt_bench)
    
    #Performance cumulée du portefeuille
    indicateur_benchmark["Ptf_rdt_cum"] = (1 + indicateur_benchmark["Ptf_rdt"]).cumprod()
    
    #Calcul de la valeur du portefeuille avec une valeur initiale
    valeur_initiale = 1499781.611  # Exemple de capital initial
    indicateur_benchmark["Ptf_value"] = indicateur_benchmark["Ptf_rdt_cum"] * valeur_initiale
    
    print("REGARDE ICI \n", indicateur_benchmark)
    
    # Calcul des excès de rendement entre le portefeuille et le benchmark
    ptf_returns =indicateur_ptf["Ptf_rdt"]
    bench_returns=indicateur_benchmark["Ptf_rdt"]
    excess_return = ptf_returns - bench_returns
    
    # Suppression des valeurs NaN
    excess_return = excess_return.dropna()
    
    # Calcul de la Tracking Error (volatilité des excès de rendement annualisée)
    tracking_error = excess_return.std() * np.sqrt(252)
    
    print(f"\nTracking Error : {tracking_error:.2%}")
    
    return tracking_error, excess_return, common_dates



    
if __name__ == "__main__":
    #main()
    
    data_folder = "donnees_brutes" 
    os.makedirs(data_folder, exist_ok=True)

    reader = PortfolioDataReader(
        directory_base=data_folder,
        multithreaded=True
    )

    assets_info, price_dfs = reader.process_all_files()
    price_matrix = reader.create_price_matrix(price_dfs)
    
    # Lecture spécifique du fichier EURUSD
    fx_path = join(data_folder, "EURUSD.xlsx")
    fx_data = pd.read_excel(fx_path, skiprows=6) 
        
    # Vérification et conversion des colonnes pour EURUSD
    if 'PX_LAST' in fx_data.columns:
        fx_data = fx_data.rename(columns={'PX_LAST': 'EURUSD'})
    
    fx_data['Date'] = pd.to_datetime(fx_data['Date'], format='%d/%m/%Y')
    fx_data = fx_data.set_index('Date')
    
    eur_price_matrix = reader.convert_to_eur(price_matrix, assets_info, fx_data)
    
    # Affichage des résultats
    #for asset, info in assets_info.items():
        #print(f"\n{asset}:")
        #for key, value in info.items():
            #print(f"{key}: {value}")
    
    #print("\nMatrice de prix en EUR:")
    print(eur_price_matrix)

    # Création du DataFrame de composition
    valorisations_initiales = {
        'LU1919842267': 44909.55 ,  
        'LU1883854199': 150013.50 ,  
        'LU1893597309': 224915.04,  
        'FR0010983924': 74926.60,  
        'LU0154236417': 149951.66,   
        'LU1103207525': 45195.35 ,   
        'LU0072462186': 224934.50 ,   
        'LU1191877379': 74982.11,   
        'LU1161527038': 179968.80,   
        'LU1160351208': 105079.80,  
        'LU1279613365': 29973.96 ,   
        'LU1244893696': 74883.69 ,   
        'LU1882449801': 29991.99,   
        'FR0011288513': 90055.06     
    }

    df_weight = pd.DataFrame({
        'Isin': valorisations_initiales.keys(),
        'Valo_2020-01-02': valorisations_initiales.values()
    })
    #Partie 1: evaluation du portefeuille
    
    ##Préparation de la base de données df_poids
   
    ### Calcul des poids
    df_weight['Poids%'] = df_weight['Valo_2020-01-02']/sum(df_weight['Valo_2020-01-02'])
    #print(df_weight['Poids%'].sum())
    
    #print("Composition du portefeuille: \n")
    #print(df_weight)
    
    # Extraction des prix pour les composants du ptf initial 
    extract =  eur_price_matrix [[ele for ele in df_weight['Isin'].unique()]]
    extract.index=eur_price_matrix.index
    
    # Calcul de rendements et volatilité du ptf initial
    df_indicateurs = pd.DataFrame({"prix_initial":extract.dropna().iloc[1],'prix_final':extract.dropna().iloc[-1]})
    
    df_indicateurs['performance_totale']=(df_indicateurs['prix_final']/df_indicateurs['prix_initial'])-1
    
    df_indicateurs['performance_annualisee']=(1+df_indicateurs['performance_totale'])**(252/len(extract))-1

    log_rendement = np.log(extract/extract.shift(1)).iloc[1:]
    df_indicateurs['volatilite'] = log_rendement.std()*252**0.5
    
    cov_compo = log_rendement.cov()
    
    #print(df_indicateurs)
    
    # Calcul de la volatilité annualisée du ptf initial
   
    wgt = np.transpose(df_weight['Poids%'])
    vol_ptf_ini =  (252**0.5)*np.dot(np.transpose(df_weight['Poids%']), np.dot(cov_compo, df_weight['Poids%']))**0.5
   
    
    print('Calcul de la volatilité annualisée du ptf initial', vol_ptf_ini)
    

    # Calcul de le rendement annualisé du ptf initial
    er2=df_indicateurs['performance_annualisee']
    ptf_er2=lambda x: np.dot(np.transpose(wgt) ,er2)


    ptf_er2=lambda x: np.dot(wgt ,er2)
    
    rdt_test=ptf_er2(df_weight)
    print("Rendement annualisé du ptf initial",rdt_test)

    
    
    # Création du DataFrame des indicateurs du portefeuille
    indicateur_ptf = pd.DataFrame(index=extract.index)
    indicateur_ptf = indicateur_ptf.iloc[:-1]
    
    #Rendement du portefeuille (pondéré par les poids
    indicateur_ptf["Ptf_rdt"] = np.dot(log_rendement, wgt)
    
    #Performance cumulée du portefeuille
    indicateur_ptf["Ptf_rdt_cum"] = (1 + indicateur_ptf["Ptf_rdt"]).cumprod()
    
    #Calcul de la valeur du portefeuille avec une valeur initiale
    valeur_initiale = 1499781.611  # Exemple de capital initial
    indicateur_ptf["Ptf_value"] = indicateur_ptf["Ptf_rdt_cum"] * valeur_initiale
    
    #Calcul de la volatilité glissante (fenêtre 252 jours, annualisée)
    indicateur_ptf["roll_vol"] = indicateur_ptf["Ptf_rdt"].rolling(252).std() * np.sqrt(252)
    #print(indicateur_ptf)
    
    # Calcul de la volatilité glissante
    roll_vol = indicateur_ptf[["Ptf_rdt"]].rolling(252).std() * np.sqrt(252)
    indicateur_ptf["roll_vol"] = roll_vol
    
    # Statistiques de volatilité
    stats_vol = {
        "last": roll_vol[["Ptf_rdt"]].values[-1][0],
        "mean": roll_vol["Ptf_rdt"].mean(),
        "q_10": roll_vol["Ptf_rdt"].quantile(0.10),
        "q_25": roll_vol["Ptf_rdt"].quantile(0.25),
        "q_75": roll_vol["Ptf_rdt"].quantile(0.75)
    }
    
    # Calcul de la volatilité EWMA
    l = 0.94
    lambdas = [[
        (1-l)*l**i 
        for i in range(251,-1,-1)
    ]]
    
    def vol_ewma(l_ret):
        r_squared = l_ret**2
        return (np.dot(r_squared, np.transpose(lambdas))[0]*252)**0.5
    
    indicateur_ptf["vol_ewma"] = indicateur_ptf[["Ptf_rdt"]].rolling(252).apply(vol_ewma)
    
    # Calcul VaR
    def var_normale(vol, confiance, maturity):
        return -vol * norm.ppf(confiance) * np.sqrt(maturity)
    
    indicateur_ptf["VaR Norm(95%, 1Y)"] = indicateur_ptf.apply(
        lambda row: var_normale(row["vol_ewma"], 0.95, 1),
        axis=1
    )
    
    # Maximum Drawdown
    max_portfolio_value = np.maximum.accumulate(indicateur_ptf['Ptf_value'].dropna())
    initial_portfolio_drawdown = indicateur_ptf['Ptf_value'] / max_portfolio_value - 1.0
    drawdown_end_date = initial_portfolio_drawdown.idxmin()
    
    max_drawdown_stats = {
        "MaxDD": initial_portfolio_drawdown.min(),
        "start": indicateur_ptf['Ptf_value'].loc[:drawdown_end_date].idxmax(),
        "end": drawdown_end_date
    }
    

    # Calcul de la Tracking Errordef calculate_tracking_error(eur_price_matrix: pd.DataFrame, indicateur_ptf: pd.DataFrame, fx_data: pd.DataFrame) -> float:
    """Calcul de la Tracking Error avec les benchmarks"""
    
    # Chargement des benchmarks
    data_folder = "donnees_benchmark" 
    os.makedirs(data_folder, exist_ok=True)

    reader = PortfolioDataReader(directory_base=data_folder, multithreaded=True)
    
    benchmark_info, price_benchmark = reader.process_all_files()
    bench_matrix = reader.create_price_matrix(price_benchmark)
    

    eur_bench_matrix = reader.convert_to_eur(bench_matrix, benchmark_info, fx_data)
    """ 
    for asset, info in benchmark_info.items():
        
        print(f"\n{asset}:")
        for key, value in info.items():
            print(f"{key}: {value}")
    """
   # print("\nMatrice de prix en EUR:")
    #print(eur_bench_matrix.head())    
    
    # Définition des pondérations du benchmark composite
    df_weightbench = pd.DataFrame({
        'Isin': ["LEGATRUU Index", "MXWD Index"],
        'Poids%': [0.6, 0.4]
    })
    
    #print("\nComposition du benchmark composite:")
    #print(df_weightbench)
    
    # Extraction des prix du benchmark
    extract_bench = bench_matrix[[ele for ele in df_weightbench['Isin'].unique()]]
    
    # Vérification des dates communes avec le portefeuille
    common_dates = eur_price_matrix.index.intersection(extract_bench.index)
    if common_dates.empty:
        raise ValueError("Erreur : Aucune date commune entre le portefeuille et le benchmark.")
   # print("COMMON DATES \n", common_dates)
    extract_bench = extract_bench.loc[common_dates]
    extract_bench.index=common_dates
   
    # Calcul de rendements et volatilité du ptf initial
    bench_indicateurs = pd.DataFrame({"prix_initial":extract_bench.dropna().iloc[1],'prix_final':extract_bench.dropna().iloc[-1]})
    
    bench_indicateurs['performance_totale']=(bench_indicateurs['prix_final']/bench_indicateurs['prix_initial'])-1
    
    bench_indicateurs['performance_annualisee']=(1+bench_indicateurs['performance_totale'])**(252/len(extract_bench))-1

    log_bench_rdt = np.log(extract_bench/extract_bench.shift(1)).iloc[1:]
    bench_indicateurs['volatilite'] = log_bench_rdt.std()*252**0.5
    
    cov_bench = log_bench_rdt.cov()
    
    wgt_bench = np.transpose(df_weightbench['Poids%'])
    vol_bench_ini =  (252**0.5)*np.dot(np.transpose(df_weightbench['Poids%']), np.dot(cov_bench, df_weightbench['Poids%']))**0.5
    #testvol =  (252**0.5)*np.dot(np.transpose(wgt), np.dot(cov_bench, wgt_bench))**0.5
      
   # print(wgt)
    print('Calcul de la volatilité annualisée du benchmark ', vol_bench_ini)
    #print('test', testvol)


    
    # Création du DataFrame des indicateurs du portefeuille
    indicateur_benchmark = pd.DataFrame(index=extract_bench.index)
    indicateur_benchmark = indicateur_benchmark.iloc[:-1]
    
    #Rendement du portefeuille (pondéré par les poids
    indicateur_benchmark["Bench_rdt"] = np.dot(log_bench_rdt, wgt_bench)
    
    #Performance cumulée du portefeuille
    indicateur_benchmark["Bench_rdt_cum"] = (1 + indicateur_benchmark["Bench_rdt"]).cumprod()
    
    #Calcul de la valeur du portefeuille avec une valeur initiale
    valeur_initiale = 1499781.611  # Exemple de capital initial
    indicateur_benchmark["Bench_value"] = indicateur_benchmark["Bench_rdt_cum"] * valeur_initiale
    
    #print("BENCHMARCK ICI \n", indicateur_benchmark)
    #print("PTF ICI \n", indicateur_ptf)
    # Calcul des excès de rendement entre le portefeuille et le benchmark
    #ptf_returns =indicateur_ptf["Ptf_rdt"]
    #bench_returns=indicateur_benchmark["Bench_rdt"]

    common_index = indicateur_benchmark.index.intersection(indicateur_ptf.index)
    ptf_returns = indicateur_ptf["Ptf_rdt"].loc[common_index]
    bench_returns = indicateur_benchmark["Bench_rdt"].loc[common_index]
    excess_return = ptf_returns - bench_returns
    """
    print("Premières dates indicateur_benchmark:\n", indicateur_benchmark.index[:5])
    print("Premières dates indicateur_ptf:\n", indicateur_ptf.index[:5])
    
    print("Dernières dates indicateur_benchmark:\n", indicateur_benchmark.index[-5:])
    print("Dernières dates indicateur_ptf:\n", indicateur_ptf.index[-5:])
    
    print("Taille indicateur_benchmark:", len(indicateur_benchmark))
    print("Taille indicateur_ptf:", len(indicateur_ptf))
    
    # Vérifier les dates communes
    common_index = indicateur_benchmark.index.intersection(indicateur_ptf.index)
    print("Nombre de dates communes:", len(common_index))
    print("Premières dates communes:\n", common_index[:5])
    print("Dernières dates communes:\n", common_index[-5:])
    """
    # Suppression des valeurs NaN
    excess_return = excess_return.dropna()
    #print("\n EXCESSSSS RETURN \n", excess_return)
    
    # Calcul de la Tracking Error (volatilité des excès de rendement annualisée)
    tracking_error_initiale = excess_return.std() * np.sqrt(252)
    
    print(f"\nTracking Error : {tracking_error_initiale:.2%}")
    
# Calcul de la Tracking Error sur une fenêtre glissante de 252 jours (1 an)
tracking_error_series = excess_return.rolling(252).std() * np.sqrt(252)
tracking_error_series_smoothed = tracking_error_series.ewm(span=30).mean()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

# Définition des périodes de crise (à ajuster si besoin)
crises = [
    ("2020-02", "2020-05"),  # COVID-19 crash
    ("2022-02", "2022-07"),  # Conflit Ukraine
    ("2023-03", "2023-06"),  # Crise bancaire US
]

# Création de la figure avec 4 sous-graphiques
fig, axs = plt.subplots(4, 1, figsize=(12, 16))

# 1. ÉVOLUTION DE LA VALEUR DU PORTEFEUILLE
axs[0].set_title("Évolution de la valeur du portefeuille")
axs[0].plot(indicateur_ptf.index, indicateur_ptf["Ptf_value"], 'b-', label="Valeur du portefeuille")
axs[0].set_ylabel("Valeur (€)")
axs[0].legend(loc="upper left")
axs[0].grid(True)
axs[0].yaxis.set_major_formatter(ScalarFormatter())  # Suppression notation scientifique

# Ajout des zones de crise
for start, end in crises:
    axs[0].axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# 2. ÉVOLUTION DES FONDS DU PORTEFEUILLE
axs[1].set_title("Évolution des fonds du portefeuille")

# Exclure la colonne EURUSD si elle est présente
fonds_a_afficher = eur_price_matrix.drop(columns=["EURUSD"], errors="ignore")

# Affichage de tous les fonds sans EURUSD
for fund in fonds_a_afficher.columns:
    axs[1].plot(fonds_a_afficher.index, fonds_a_afficher[fund], label=fund, alpha=0.7)

axs[1].set_ylabel("Valeur des fonds (€)")
axs[1].legend(loc="upper left", fontsize="small", ncol=2)  # Légende optimisée
axs[1].grid(True)

# Ajout des zones de crise
for start, end in crises:
    axs[1].axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# 3. VOLATILITÉ ET VAR
axs[2].set_title("Volatilité et Value at Risk (VaR)")
ax2_bis = axs[2].twinx()

axs[2].plot(indicateur_ptf.index, indicateur_ptf["roll_vol"], 'r--', label="Volatilité")
ax2_bis.plot(indicateur_ptf.index, indicateur_ptf["VaR Norm(95%, 1Y)"], 'g-', label="VaR")

axs[2].set_ylabel("Volatilité")
ax2_bis.set_ylabel("VaR")

axs[2].legend(loc="upper left")
ax2_bis.legend(loc="upper right")
axs[2].grid(True)

# Ajustement des échelles pour éviter l’écrasement des courbes
axs[2].set_ylim(indicateur_ptf["roll_vol"].min() * 0.9, indicateur_ptf["roll_vol"].max() * 1.1)
ax2_bis.set_ylim(indicateur_ptf["VaR Norm(95%, 1Y)"].min() * 1.1, 0)

# Ajout des zones de crise
for start, end in crises:
    axs[2].axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# 4. ÉVOLUTION DU PORTEFEUILLE ET DRAWDOWN
axs[3].set_title("Évolution du portefeuille et Drawdown")
ax3_bis = axs[3].twinx()

axs[3].plot(indicateur_ptf.index, indicateur_ptf["Ptf_value"], 'b-', label="Valeur du portefeuille")
ax3_bis.plot(indicateur_ptf.index, initial_portfolio_drawdown, 'm--', label="Drawdown")

axs[3].set_ylabel("Valeur du portefeuille (€)")
ax3_bis.set_ylabel("Drawdown (%)")

axs[3].legend(loc="upper left")
ax3_bis.legend(loc="upper right")
axs[3].grid(True)
axs[3].yaxis.set_major_formatter(ScalarFormatter())  # Suppression notation scientifique

# Ajout des zones de crise
for start, end in crises:
    axs[3].axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# Formatage de l'axe des dates pour plus de lisibilité
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

# Ajustement automatique de la mise en page
plt.tight_layout()
plt.show()












import scipy.optimize as opt

# Fonction d'optimisation des poids
def optimiser_portefeuille(er_fonction, w0, contraintes, bornes):
    """
    Optimise les poids d'un portefeuille en minimisant une fonction d'objectif.

    :param er_fonction: Fonction de rendement à maximiser
    :param w0: Poids initiaux
    :param contraintes: Contraintes de l'optimisation
    :param bornes: Bornes des poids
    :return: Résultat de l'optimisation
    """
    return opt.minimize(
        er_fonction,
        w0,
        method='SLSQP',
        constraints=contraintes,
        bounds=bornes
    )

# Définition du dossier contenant les données des fonds
dossier_offensif = "donnees_offensif"
os.makedirs(dossier_offensif, exist_ok=True)

# Lecture des données du portefeuille offensif
lecteur_offensif = PortfolioDataReader(directory_base=dossier_offensif, multithreaded=True)
infos_fonds_offensifs, prix_fonds_offensifs = lecteur_offensif.process_all_files()
matrice_prix_offensif = lecteur_offensif.create_price_matrix(prix_fonds_offensifs)

# Conversion des prix en EUR
matrice_prix_offensif_eur = lecteur_offensif.convert_to_eur(matrice_prix_offensif, infos_fonds_offensifs, fx_data)

# Suppression des données avant le 31/12/2021
matrice_prix_offensif_eur = matrice_prix_offensif_eur.loc["2022-01-01":]

# Définition des fonds et montants investis
valeurs_offensives = {
    'LU1883854199': 150000.00,  'LU1103207525': 150000.00,  
    'LU1244893696': 150000.00,  'LU1919842267': 150000.00,  
    'LU1861132840': 150000.00,  'PIRPEUR LX': 150000.00,    
    'ALGAATU LX': 150000.00,    'PFLDCPE LX': 150000.00,    
    'FFGLCAE LX': 150000.00,    'NVDA US': 150000.00,  
    '9888 HK': 150000.00,       'BEAN SW': 150000.00,  
    'ASML NA': 150000.00,       'SIE GY': 150000.00   
}

# Création du DataFrame des pondérations
df_poids_offensif = pd.DataFrame({
    'Isin': valeurs_offensives.keys(),
    'Valorisation': valeurs_offensives.values()
})

# Calcul des poids du portefeuille offensif
df_poids_offensif["Poids%"] = df_poids_offensif["Valorisation"] / df_poids_offensif["Valorisation"].sum()

# Extraction des prix des actifs du portefeuille offensif
extraction_offensive = matrice_prix_offensif_eur[df_poids_offensif['Isin'].unique()]
extraction_offensive.index = matrice_prix_offensif_eur.index
extraction_offensive = extraction_offensive.iloc[:-1]

# Vérification des prix bruts avant optimisation
print("\nPrix des fonds offensifs avant optimisation :")
print(extraction_offensive.head())

# Calcul des rendements logarithmiques et matrice de covariance
rendements_log_offensif = np.log(matrice_prix_offensif_eur / matrice_prix_offensif_eur.shift(1)).dropna()
covariance_offensive = rendements_log_offensif.cov()

# Objectif de volatilité
volatilite_cible = 0.20

# Fonction de volatilité du portefeuille
def volatilite_portefeuille(poids):
    return np.sqrt(252 * np.dot(np.transpose(poids), np.dot(covariance_offensive, poids)))

# Définition des poids initiaux
poids_initiaux = np.array([1 / len(df_poids_offensif)] * len(df_poids_offensif))

# Définition des bornes des poids (min 1%, max 100%)
bornes_optimisation = [(0.01, 1.0)] * len(df_poids_offensif)

# Optimisation des poids
resultat_optimisation = optimiser_portefeuille(
    lambda w: -np.dot(np.transpose(w), rendements_log_offensif.mean()),
    poids_initiaux,
    contraintes=[
        {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)},  # Somme des poids = 1
        {'type': 'eq', 'fun': lambda w: volatilite_cible - volatilite_portefeuille(w)}  # Contraintes de volatilité
    ],
    bornes=bornes_optimisation
)

# Mise à jour des poids optimisés
df_poids_offensif["Poids%"] = resultat_optimisation.x

# Vérification des résultats de l'optimisation
print("\nPondérations optimisées :")
print(df_poids_offensif)
print(f"\nSomme des poids : {df_poids_offensif['Poids%'].sum():.6f} (Doit être proche de 1.0)")

# Calcul des performances du portefeuille offensif après optimisation

# Assurer que l'index reste basé sur les dates de la matrice des prix

indicateurs_offensifs = pd.DataFrame({"Ptf_rdt": np.dot(rendements_log_offensif, df_poids_offensif["Poids%"])})
indicateurs_offensifs.index = matrice_prix_offensif_eur.index[:-1]  # On aligne avec la taille de la série

indicateurs_offensifs = indicateurs_offensifs.iloc[:-1]

indicateurs_offensifs["Ptf_rdt_cum"] = (1 + indicateurs_offensifs["Ptf_rdt"]).cumprod()
indicateurs_offensifs["Valeur_Ptf"] = indicateurs_offensifs["Ptf_rdt_cum"] * 1_500_000  # Capital initial

# Vérification des performances après arbitrage
print("\nPerformances du portefeuille offensif après arbitrage :")
print(indicateurs_offensifs)

# Calcul des statistiques du portefeuille offensif
rendement_annuel = indicateurs_offensifs['Ptf_rdt_cum'].iloc[-1] ** (252 / len(indicateurs_offensifs)) - 1
volatilite_annuelle = np.sqrt(252 * np.dot(np.transpose(df_poids_offensif["Poids%"]), 
                                           np.dot(covariance_offensive, df_poids_offensif["Poids%"])))

# Calcul de la volatilité glissante sur 252 jours
indicateurs_offensifs["Volatilite_glissante"] = indicateurs_offensifs["Ptf_rdt"].rolling(252).std() * np.sqrt(252)

# Affichage des résultats
print(f"\nRendement annuel : {rendement_annuel:.2%}")
print(f"Volatilité annuelle : {volatilite_annuelle:.2%}")


    
# Maximum Drawdown pour le portefeuille offensif
p_offensif = np.maximum.accumulate(indicateurs_offensifs['Valeur_Ptf'].dropna())
Daily_Drawdown_offensif = indicateurs_offensifs['Valeur_Ptf'] / p_offensif - 1.0
end_offensif = Daily_Drawdown_offensif.idxmin()

max_drawdown_stats = {
    "MaxDD": Daily_Drawdown_offensif.min(),
    "start": indicateurs_offensifs['Valeur_Ptf'].loc[:end_offensif].idxmax(),
    "end": end_offensif
}
    
indicateurs_offensifs["vol_ewma"] = indicateur_ptf[["Ptf_rdt"]].rolling(252).apply(vol_ewma)

indicateurs_offensifs["VaR Norm(95%, 1Y)"] = indicateurs_offensifs.apply(
    lambda row: var_normale(row["vol_ewma"], 0.95, 1),
    axis=1
)
    

common_index = indicateurs_offensifs.index.intersection(indicateur_ptf.index)
ptf_returns = indicateur_ptf["Ptf_rdt"].loc[common_index]
bench_returns = indicateurs_offensifs["Ptf_rdt"].loc[common_index]
excess_return = ptf_returns - bench_returns


# Suppression des valeurs NaN
excess_return = excess_return.dropna()

# Calcul de la Tracking Error (volatilité des excès de rendement annualisée)
tracking_error_offensif = excess_return.std() * np.sqrt(252)

print(f"\nTracking Error : {tracking_error_offensif:.2%}")

# Calcul de la Tracking Error sur une fenêtre glissante de 252 jours (1 an)
tracking_error_series = excess_return.rolling(252).std() * np.sqrt(252)
tracking_error_series_smoothed = tracking_error_series.ewm(span=30).mean()

# Définition manuelle des limites de l'axe des abscisses
start_date = "2021-12-31"
end_date = "2025-01-02"

fig, axs = plt.subplots(4, 1, figsize=(12, 16))

# 1. Évolution de la valeur du portefeuille arbitré
axs[0].set_title("Évolution de la valeur du portefeuille arbitré")
axs[0].plot(indicateurs_offensifs.index, indicateurs_offensifs["Valeur_Ptf"], 'b-', label="Valeur du portefeuille arbitré")
axs[0].set_ylabel("Valeur (€)")
axs[0].legend(loc="upper left")
axs[0].grid(True)
axs[0].yaxis.set_major_formatter(ScalarFormatter())
axs[0].set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))  # Fixer la plage d'affichage

# 2. Évolution des fonds du portefeuille arbitré
axs[1].set_title("Évolution des fonds du portefeuille arbitré")
fonds_a_afficher = matrice_prix_offensif_eur.drop(columns=["EURUSD"], errors="ignore")

for fund in fonds_a_afficher.columns:
    axs[1].plot(fonds_a_afficher.index, fonds_a_afficher[fund], label=fund, alpha=0.7)

axs[1].set_ylabel("Valeur des fonds (€)")
axs[1].legend(loc="upper left", fontsize="small", ncol=2)
axs[1].grid(True)
axs[1].set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))  # Fixer la plage d'affichage

# 3. Volatilité et Value at Risk (VaR)
axs[2].set_title("Volatilité et Value at Risk (VaR)")
ax2_bis = axs[2].twinx()

axs[2].plot(indicateurs_offensifs.index, indicateurs_offensifs["Volatilite_glissante"], 'r--', label="Volatilité")
ax2_bis.plot(indicateurs_offensifs.index, indicateurs_offensifs["VaR Norm(95%, 1Y)"], 'g-', label="VaR")

axs[2].set_ylabel("Volatilité")
ax2_bis.set_ylabel("VaR")

axs[2].legend(loc="upper left")
ax2_bis.legend(loc="upper right")
axs[2].grid(True)
axs[2].set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))  # Fixer la plage d'affichage

# 4. Évolution du portefeuille arbitré et Drawdown
axs[3].set_title("Évolution du portefeuille arbitré et Drawdown")
ax3_bis = axs[3].twinx()

axs[3].plot(indicateurs_offensifs.index, indicateurs_offensifs["Valeur_Ptf"], 'b-', label="Valeur du portefeuille arbitré")
ax3_bis.plot(Daily_Drawdown_offensif.index, Daily_Drawdown_offensif, 'm--', label="Drawdown")

axs[3].set_ylabel("Valeur du portefeuille (€)")
ax3_bis.set_ylabel("Drawdown (%)")

axs[3].legend(loc="upper left")
ax3_bis.legend(loc="upper right")
axs[3].grid(True)
axs[3].yaxis.set_major_formatter(ScalarFormatter())
axs[3].set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))  # Fixer la plage d'affichage

# Formatage de l'axe des dates
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.show()



"""
# 📌 Dossier de sauvegarde
output_folder = "resultats_exportes"
os.makedirs(output_folder, exist_ok=True)

# 📌 Sauvegarde en CSV
indicateur_ptf.to_csv(os.path.join(output_folder, "indicateur_ptf.csv"))
indicateurs_offensifs.to_csv(os.path.join(output_folder, "indicateurs_offensifs.csv"))
Daily_Drawdown.to_csv(os.path.join(output_folder, "daily_drawdown.csv"))

# 📌 Sauvegarde en JSON
indicateur_ptf.to_json(os.path.join(output_folder, "indicateur_ptf.json"), orient="records", date_format="iso")
indicateurs_offensifs.to_json(os.path.join(output_folder, "indicateurs_offensifs.json"), orient="records", date_format="iso")
Daily_Drawdown.to_json(os.path.join(output_folder, "daily_drawdown.json"), orient="records", date_format="iso")

# 📌 Sauvegarde en Pickle (format rapide pour chargement)
indicateur_ptf.to_pickle(os.path.join(output_folder, "indicateur_ptf.pkl"))
indicateurs_offensifs.to_pickle(os.path.join(output_folder, "indicateurs_offensifs.pkl"))
Daily_Drawdown.to_pickle(os.path.join(output_folder, "daily_drawdown.pkl"))

print("\n📂 Export terminé ! Les fichiers sont enregistrés dans le dossier 'resultats_exportes'.")

"""
# Création des dossiers
output_folder = "resultats_analyses"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}/graphiques", exist_ok=True)
os.makedirs(f"{output_folder}/metriques", exist_ok=True)
os.makedirs(f"{output_folder}/donnees", exist_ok=True)

# Sauvegarde des graphiques individuels - Portefeuille Initial
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(indicateur_ptf.index, indicateur_ptf["Ptf_value"], 'b-')
ax1.set_title("Évolution de la valeur du portefeuille")
plt.savefig(f"{output_folder}/graphiques/ptf_initial_evolution.png", dpi=300, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(figsize=(12, 6))
for fund in fonds_a_afficher.columns:
    ax2.plot(fonds_a_afficher.index, fonds_a_afficher[fund], label=fund, alpha=0.7)
ax2.set_title("Évolution des fonds")
plt.savefig(f"{output_folder}/graphiques/ptf_initial_fonds.png", dpi=300, bbox_inches='tight')
plt.close()

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(indicateur_ptf.index, indicateur_ptf["roll_vol"], 'r--')
ax3.set_title("Volatilité et VaR")
plt.savefig(f"{output_folder}/graphiques/ptf_initial_vol_var.png", dpi=300, bbox_inches='tight')
plt.close()

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(indicateur_ptf.index, initial_portfolio_drawdown, 'm--')
ax4.set_title("Drawdown")
plt.savefig(f"{output_folder}/graphiques/ptf_initial_drawdown.png", dpi=300, bbox_inches='tight')
plt.close()

# Sauvegarde des graphiques individuels - Portefeuille Arbitré
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.plot(indicateurs_offensifs.index, indicateurs_offensifs["Valeur_Ptf"], 'b-')
ax5.set_title("Évolution de la valeur du portefeuille arbitré")
plt.savefig(f"{output_folder}/graphiques/ptf_arbitre_evolution.png", dpi=300, bbox_inches='tight')
plt.close()

fig6, ax6 = plt.subplots(figsize=(12, 6))
for fund in fonds_a_afficher.columns:
    ax6.plot(fonds_a_afficher.index, fonds_a_afficher[fund], label=fund, alpha=0.7)
ax6.set_title("Évolution des fonds du portefeuille arbitré")
plt.savefig(f"{output_folder}/graphiques/ptf_arbitre_fonds.png", dpi=300, bbox_inches='tight')
plt.close()

fig7, ax7 = plt.subplots(figsize=(12, 6))
ax7.plot(indicateurs_offensifs.index, indicateurs_offensifs["Volatilite_glissante"], 'r--')
ax7.set_title("Volatilité et VaR du portefeuille arbitré")
plt.savefig(f"{output_folder}/graphiques/ptf_arbitre_vol_var.png", dpi=300, bbox_inches='tight')
plt.close()

fig8, ax8 = plt.subplots(figsize=(12, 6))
ax8.plot(Daily_Drawdown_offensif.index, Daily_Drawdown_offensif, 'm--')
ax8.set_title("Drawdown du portefeuille arbitré")
plt.savefig(f"{output_folder}/graphiques/ptf_arbitre_drawdown.png", dpi=300, bbox_inches='tight')
plt.close()

# Sauvegarde des métriques
metriques_initial = {
    'Rendement annuel': f"{rdt_test:.2%}",
    'Volatilité annuelle': f"{vol_ptf_ini:.2%}",
    'Tracking Error': f"{tracking_error_initiale:.2%}"
}

metriques_arbitre = {
    'Rendement annuel': f"{rendement_annuel:.2%}",
    'Volatilité annuelle': f"{volatilite_annuelle:.2%}",
    'Tracking Error': f"{tracking_error_offensif:.2%}"
}

pd.DataFrame([metriques_initial]).to_excel(f"{output_folder}/metriques/metriques_ptf_initial.xlsx")
pd.DataFrame([metriques_arbitre]).to_excel(f"{output_folder}/metriques/metriques_ptf_arbitre.xlsx")

# Sauvegarde des DataFrames
dataframes = {
    'prix_matrix': price_matrix,
    'eur_price_matrix': eur_price_matrix,
    'indicateur_ptf': indicateur_ptf,
    'bench_matrix': bench_matrix,
    'indicateur_benchmark': indicateur_benchmark,
    'indicateurs_offensifs': indicateurs_offensifs,
    'df_weight': df_weight,
    'df_poids_offensif': df_poids_offensif
}

# Export en Excel avec plusieurs onglets
with pd.ExcelWriter(f"{output_folder}/donnees/donnees_completes.xlsx") as writer:
    for name, df in dataframes.items():
        df.to_excel(writer, sheet_name=name)

# Export en CSV
for name, df in dataframes.items():
    df.to_csv(f"{output_folder}/donnees/{name}.csv")

# Export en JSON
for name, df in dataframes.items():
    df.to_json(f"{output_folder}/donnees/{name}.json", orient="records", date_format="iso")

# Export en Pickle
for name, df in dataframes.items():
    df.to_pickle(f"{output_folder}/donnees/{name}.pkl")

print(f"Fichiers sauvegardés dans le dossier: {output_folder}")
