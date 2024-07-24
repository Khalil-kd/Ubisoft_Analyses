import pandas as pd
import os
from wordcloud import WordCloud
import plotly.graph_objs as go
from collections import Counter
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_elements import elements, dashboard, media, mui, nivo
import plotly.express as px
from PIL import Image
import base64
import streamlit as st 
from streamlit_option_menu import option_menu 
from textblob import TextBlob
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import wordcloud
from plotly.subplots import make_subplots

df = pd.read_csv("vgsales.csv")
all_comments = pd.read_csv('all_comments.csv')

all_followers = pd.read_csv('all_followers.csv')
all_reviews = pd.read_csv('all_reviews.csv')
all_players_filtered = pd.read_csv('all_players_filtered.csv')
all_prices = pd.read_csv('all_prices.csv')

df_steam_ACV = pd.read_csv('df_steam_ACV.csv')
df_steam_WDL = pd.read_csv('df_steam_WDL.csv')
df_steam_FC = pd.read_csv('df_steam_FC.csv')

def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Fonction pour analyser les sentiments des tokens
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

# Définir une fonction pour rendre les boutons homogènes
def sidebar_button(label, key):
    return st.button(label, key=key, use_container_width=True)

# Initialise la sélection avec une valeur par défaut
if 'selection' not in st.session_state:
    st.session_state.selection = "🗂️ Projet"

# Configurer la page en mode large
st.set_page_config(layout="wide")

# Sidebar menu
with st.sidebar:
    st.header(" Sommaire ")
    if sidebar_button("🗂️ Projet", key="Projet"):
        st.session_state.selection = "🗂️ Projet"
    if sidebar_button("📈 Analyse VG sales", key="Analyse VG sales"):
        st.session_state.selection = "📈 Analyse VG sales"
    if sidebar_button("📉 Analyse SteamDB", key="Analyse SteamDB"):
        st.session_state.selection = "📉 Analyse SteamDB"
    if sidebar_button("🕸️ Scraping Steam", key="Scraping Steam"):
        st.session_state.selection = "🕸️ Scraping Steam"
    if sidebar_button("💬 Analyse de sentiment", key="Analyse de sentiment"):
        st.session_state.selection = "💬 Analyse de sentiment"
    if sidebar_button("🔍 Conclusion", key="Conclusion"):
        st.session_state.selection = "🔍 Conclusion"
    if sidebar_button("🧠 APP NLP", key="APP NLP"):
        st.session_state.selection = "🧠 APP NLP"
        
# Ajouter un espace entre le sommaire et les informations
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Ajouter les informations sous le sommaire
    st.markdown("""
    <div style="background-color: #fcfcfc; padding: 10px; border-radius: 10px;">
        <h4>Auteur </h4>
        <ul>
            <li>Khalil KADRI - <a href="https://www.linkedin.com/in/khalil-kadri-544948195/" target="_blank">LinkedIn</a></li>
        </ul>
        <h4>Bootcamp Data Analyst Mai 2024 </h4>
        <ul>
            <li><a href="https://datascientest.com/formation-data-analyst/" target="_blank">DataScientest - Data Analyst </a></li>
        </ul>
        <h4>Données </h4>
        <ul>
            <li><a href="https://www.kaggle.com/datasets/gregorut/videogamesales/" "target="_blank">Kaggle</a></li>
            <li><a href="https://steamdb.info/"target="_blank">Steamdb </a></li>
            <li><a href="https://steamcommunity.com/" "target="_blank">Steamcommunity </a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
if st.session_state.selection == "🗂️ Projet":
    #st.title("Ubisoft Analyses")
    st.header("Projet")
    
    image_path = os.path.join('images', 'ubisoft.png')
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
  
    st.markdown("""
<div style="background-color: black; display: flex; justify-content: center; align-items: center; height: 60px; text-align: center;">
    <h1 style="color: white; margin: 0;">From Data to Strategy - UBISOFT</h1>
</div>
""", unsafe_allow_html=True)
    
# Contenu en fonction de la sélection de la barre latérale
if st.session_state.selection == "🗂️ Projet":
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.write(""" L'industrie du jeu vidéo est l'une des industries les plus dynamiques et innovantes au monde, caractérisée par une concurrence intense et des attentes élevées de la part des consommateurs. Pour les développeurs et éditeurs de jeux, comme Ubisoft, comprendre les perceptions et les réactions des joueurs est crucial pour améliorer les produits et les stratégies marketing. Les plateformes comme Steam fournissent une mine d'informations précieuses à travers les avis et les commentaires des utilisateurs.""")
    st.write(""" L'objectif principal de cette analyse est de fournir des insights approfondis qui pourront guider la stratégie marketing future. En comprenant mieux les attentes et les réactions des joueurs, nous serons en mesure de concevoir des campagnes marketing plus ciblées et efficace.       Cette approche narrative nous permet également de raconter l'histoire des ventes de jeux vidéo d'une manière plus engageante et informatif.""") 
    
    st.markdown("---")

    st.subheader("Plan du projet")
    st.write("1. Étude du marché des jeux vidéos")
    st.write("2. Analyse des données SteamDB")
    st.write("4. Scraping des commentaires sur Steam")
    st.write("5. Analyse de sentiment")
    st.write("6. Conclusion")
    st.write("7. Application NLP")
    
elif st.session_state.selection == "📈 Analyse VG sales":
    
    st.header("Analyse VG sales")
    
    st.subheader("Contexte")
    
    
    st.markdown("""
Les données utilisées dans cette analyse ont été fournies par DataScientest et proviennent d'un scraping effectué sur le site VGChartz. Ces données sont disponibles sur le site Kaggle.

Dans cette section, nous allons commencer par visualiser les données initiales pour comprendre leur structure et leur contenu. Cette première exploration nous permettra de justifier la nécessité de recueillir nos propres données et pourquoi nous avons choisit UBISOFT comme éditeur.

Vous trouverez ci-dessous les données brutes ainsi que les résultats de notre première exploration.
""")
    st.markdown("---")
    
    st.dataframe(df.head())
    
    st.markdown(""" **Commentaire :**
Wii Sports, la vente numéro 1, se distingue nettement des autres. Cette valeur exceptionnelle s'explique facilement par le fait que ce jeu 
était systématiquement inclus avec chaque Nintendo Wii (sauf au Japon et en Corée du Sud). On ne peut donc pas vraiment parler d'un choix du consommateur,
mais plutôt d'un jeu 'gratuit'. Pour éviter de biaiser notre analyse, nous supprimerons la ligne concernant Wii Sports. """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    df = df[df["Name"] != "Wii Sports"]
   
    # Définir les sections
    sections = {
        "Dimensions": "Dimensions du dataframe : ",
        "Doublons": "Nombre de doublons : ",
        "Valeurs manquantes": "Valeurs manquantes : ",
        "Distribution": "Distribution des années : "
    }

    # Afficher le menu horizontal
    selected = option_menu(
        menu_title=None,  # requis
        options=list(sections.keys()),  # requis
        icons=['table', 'copy', 'exclamation', 'bar-chart-line'],  # optionnel
        menu_icon="cast",  # optionnel
        default_index=0,  # optionnel
        orientation="horizontal"
    )

    # Afficher le contenu basé sur la sélection
    st.write(f" {sections[selected]}")
    
    if selected == "Dimensions":
        st.write(df.shape)
         
    elif selected == "Doublons":
        st.write(df.duplicated().sum())
        
    elif selected == "Valeurs manquantes":
        st.dataframe(df.isna().sum())
        
    elif selected == "Distribution":
        st.write(df['Year'].value_counts())
    
    st.markdown("<br>", unsafe_allow_html=True)

    df = df[~df['Year'].isin([2017, 2018, 2019, 2020])]
    
    st.markdown(""" **Commentaire :**
Il est important de mentionner qu'il y a 271 valeurs manquantes pour les années et 58 pour les éditeurs. 
Cependant, comme nous allons ultérieurement créer nos propres ensembles de données via le scraping,
il n'est pas nécessaire de s'attarder sur ces valeurs manquantes à ce stade.
""")
     
    st.markdown(""" 
Après avoir exécuté `value_counts` sur la colonne "Year" pour analyser la répartition des données, 
il est apparu que les années de 2017 à 2020 ne contiennent pas suffisamment de valeurs pour une analyse statistiquement significative. 
En conséquence, il a été décidé de supprimer ces lignes afin de se concentrer sur les années avec des données plus robustes et représentatives.
Cette approche permet de garantir la qualité et la pertinence des résultats de l'analyse.
""")

   

    st.markdown("---")
    
    st.subheader("Analyse des ventes globales par éditeur")
    
    # Créer deux colonnes
    col1, col2 = st.columns(2)
    
# Calculer les meilleures ventes en CA par éditeur
    best_Publisher_Sales = df[['Publisher', 'Global_Sales']].groupby('Publisher').sum().sort_values(by='Global_Sales', ascending=False).reset_index().head(10)
    
# Créer le graphique avec Plotly

    with col1:
        fig_CA = px.bar(best_Publisher_Sales,
                x='Publisher',
                y='Global_Sales',
                title='Global Sales by Publisher',
                labels={'Publisher': 'Publisher', 'Global_Sales': 'Global Sales'},
                color='Publisher',
                text='Global_Sales',
                width=1200, height=600)

        fig_CA.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_CA.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig_CA.update_xaxes(tickangle=90)
        
        st.plotly_chart(fig_CA, use_container_width=True)
        
# Calculer les meilleures ventes en unité par éditeur
    best_publisher_units = df.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).head(10)
    best_publisher_units = pd.DataFrame(best_publisher_units).reset_index()
    best_publisher_units = best_publisher_units.rename(columns={"Year": "Global_Units"})
        
# Créer le graphique avec Plotly
    with col2:
        fig_UN = px.bar(best_publisher_units, 
                x='Publisher', 
                y='Global_Units', 
                title='Global Units by Publisher', 
                labels={'Publisher': 'Publisher', 'Year': 'Global Units'},
                color='Publisher',
                text='Global_Units', width=1200, height=600)

        # Personnaliser l'apparence du graphique
        fig_UN.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_UN.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig_UN.update_xaxes(tickangle=90)

        st.plotly_chart(fig_UN, use_container_width=True)

    st.write(" **Commentaire :** Nintendo est en tête des ventes mondiales avec 1703.80 millions de ventes, ce qui est significativement supérieur à Electronic Arts (1110.32 millions) et Activision (727.46 millions).")
    st.write(" Bien qu'Electronic Arts ait sorti le plus grand nombre de titres (1339), ils ne se classent qu'au deuxième rang pour les ventes globales avec 1110.32 millions. Cela suggère qu'une grande partie de leurs titres n'atteignent pas le même niveau de succès que ceux de Nintendo.")
    st.write(" Les éditeurs avec un grand nombre de titres comme Ubisoft et Activision ne dominent pas nécessairement en termes de ventes globales, contrairement à Nintendo qui a un nombre plus restreint de titres mais des ventes globales bien supérieures. ")
    st.write(" Cela pourrait indiquer que Nintendo se concentre davantage sur la qualité et l'impact de chaque titre plutôt que sur la quantité.")
    
    st.markdown("---")
    
    st.subheader("Analyse des ventes globales par genre")
    
    publishers_to_keep = ['Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment', 'Ubisoft']

    df_top5_Publisher_sales = df[df['Publisher'].isin(publishers_to_keep)]

    df_Genre_Sales = df_top5_Publisher_sales.groupby(['Genre', 'Publisher']).agg({'Global_Sales': 'sum'}).reset_index()
    df_Genre_Sales = df_Genre_Sales.sort_values(by='Global_Sales', ascending=False).reset_index(drop=True)
    
    fig_GR = px.bar(df_Genre_Sales, x='Genre', y='Global_Sales', color='Publisher',
             labels={"Genre": "Genre", "Global_Sales": "Ventes totales (M$)", "Publisher": "Éditeur"},
             title="Total des ventes par genre et éditeur", width=1200, height=600)

    st.plotly_chart(fig_GR, use_container_width=True)

    st.write(" **Commentaire :** Le graphique montre que certains éditeurs se spécialisent et dominent certains genres. Par exemple, Nintendo est très dominant dans les genres Platform et Role-Playing, tandis qu'Electronic Arts excelle dans le genre Sports. Activision est un leader le genre Shooter. Les genres tels que Adventure, Fighting, Puzzle, et Simulation montrent des ventes plus modestes et une répartition plus équilibrée entre les différents éditeurs. Cette répartition met en évidence les stratégies de spécialisation et de diversification des éditeurs dans différents segments du marché des jeux vidéo.")

    df_best_Genre = df_top5_Publisher_sales.groupby(['Year','Genre']).agg({"Global_Sales":"sum"}).reset_index(level=['Year','Genre'])

    df_best_Genre = df_top5_Publisher_sales.groupby(['Year','Genre']).agg({"Global_Sales":"sum"}).reset_index(level=['Year','Genre'])

    fig_BG = px.bar(df_best_Genre, x="Year", y="Global_Sales", color = "Genre", 
                        labels={"Genre": "Genre",
                                "Global_Sales" : "Ventes total (M$)",
                                "Year": "Année"},
                        title="Evolution des ventes par genre")

    st.plotly_chart(fig_BG, use_container_width=True)

    st.write(" **Commentaire :** Les genres Sports et Platform dominent particulièrement durant cette période. Après 2010, une légère diminution est observée, bien que les ventes restent élevées. La diversité des genres augmente également avec le temps, indiquant une diversification du marché des jeux vidéo. Cette tendance peut être justifiée par l'évolution technologique et l'élargissement des audiences de joueurs, favorisant une variété de genres.")
    
    st.markdown("---")
    
    st.subheader("Analyse des bestsellers par éditeur")
    
    df_best_sellers = df_top5_Publisher_sales.groupby(['Year','Publisher','Name']).agg({"Global_Sales":"sum"}).reset_index(level=['Year','Publisher','Name'])
    df_best_sellers_c = df_best_sellers.groupby('Year')['Global_Sales'].transform('max') == df_best_sellers['Global_Sales']

    fig_BS = px.bar(df_best_sellers[df_best_sellers_c], 
                        x="Year", 
                        y="Global_Sales", 
                        color = "Publisher",    
                        labels={"Publisher": "Editeurs",
                        "Global_Sales" : "Ventes total (M$)",
                        "Year": "Année"},
                        title="Meilleur titre de l'année",
                        hover_name="Name")

    st.plotly_chart(fig_BS, use_container_width=True)
    
    st.write(" **Commentaire :** Le graphique montre que Nintendo domine les ventes de jeux vidéo entre 1980 et 2009, avec des pics significatifs en 1985 et 1990. À partir de 2010, Activision prend le relais avec des ventes fortes jusqu'en 2015. Sony Computer Entertainment et Electronic Arts montrent également des pics notables à partir des années 2000. Globalement, Nintendo reste un acteur historique majeur, tandis qu'Activision et Sony ont gagné en importance ces dernières années, reflétant une concurrence accrue dans l'industrie.")
    
    st.write("En prenant en compte nos trois derniers graphes nous pouvons en déduire qu'il n'ya pas spécialement une relation entre le genre et le chiffre d'affaire généré par le jeux vidéo, en prenant l'exemple d'Activistion qui a dominé le marché avec un seul genre qui est de Shooter et Nintendo qui a dominé le marché en commercialisant plusieurs genres et chaque année son best seller n'est pas forcément le même genre de l'année dernière.")

    st.markdown("---")
    
    st.subheader("Analyse des ventes par plateforme")
    
    df_platform = df_top5_Publisher_sales.groupby(by=['Platform', 'Publisher'])['Global_Sales'].sum()
    df_platform = df_platform.reset_index()
    df_platform = df_platform.sort_values(by=['Global_Sales'], ascending=False)

    fig_PL =  px.bar(df_platform, x="Platform", y="Global_Sales", color = "Publisher", 
                        labels={"Platform": "Platform",
                                "Global_Sales" : "Ventes total (M$)",
                                "Publisher": "Publisher"},
                        title="Total des ventes par genres")

    st.plotly_chart(fig_PL, use_container_width=True)
    
    st.write(" **Commentaire :** Les analyses montrent qu'il n'y a pas de corrélation directe entre la plateforme de publication et le succès d'un jeu en tant que best-seller. Nintendo domine les ventes totales par plateforme avec des consoles comme DS et Wii, mais n'est pas systématiquement en tête des meilleurs titres chaque année. Les éditeurs variés comme Sony Computer Entertainment et Electronic Arts apparaissent souvent dans les meilleurs titres, indépendamment de la plateforme. Le succès d'un jeu dépend de nombreux facteurs tels que la qualité, les campagnes marketing et les préférences des consommateurs, plutôt que simplement de la plateforme utilisée.")

    st.markdown("---")
    
    st.subheader("Pourquoi UBISOFT ? ")
    st.markdown("""
Le cas de chaque éditeur est fascinant à examiner mais nous avons décidé d'étudier le cas d'Ubisoft pour 3 raisons : 

1. **Productivité vs. Performance** : Ubisoft a publié un grand nombre de jeux (plus de 900), ce qui indique une forte productivité. 
    Cependant, son chiffre d'affaires total est très loin de celui de ses principaux concurrents.

2. **Absence de Best-Sellers** : Contrairement à d'autres grands éditeurs, Ubisoft n'a pas produit de best-sellers au cours des années représentées.
    Analyser pourquoi Ubisoft n'a pas réussi à avoir des titres dominants malgré une forte présence.

3. **Contribution aux Genres Populaires** : Bien qu'Ubisoft ne domine pas, il contribue de manière significative à des genres populaires comme 
    l'aventure et l'action. 

En analysant en profondeur les 3 derniers bestsellers, Nous visons à obtenir des insights plus précis sur les facteurs de succès et 
les stratégies de prix de cet éditeur. Cette démarche nous permettra d’explorer les dynamiques spécifiques d'Ubisoft et de mieux comprendre 
les raisons derrière leurs performances sur le marché. 

""")

    st.markdown("""
### Choix des jeux

Les jeux ont été sélectionnés en fonction de deux critères principaux : 
- **Prix de sortie** : 59,99 euros
- **Genre** : Action, Aventure

Les jeux sélectionnés sont :
""")
    # Créer trois colonnes
    col1, col2, col3 = st.columns(3)

    # Affichage de Far Cry 6 dans la première colonne
    with col1:
        image_path = os.path.join('images', 'FC.jpeg')
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        

    # Affichage de Assassin's Creed Valhalla dans la deuxième colonne
    with col2:
        image_path = os.path.join('images', 'ACV.jpeg')
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        

    # Affichage de Watch Dogs: Legion dans la troisième colonne
    with col3:
        image_path = os.path.join('images', 'WDL.jpeg')
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        

    st.markdown("---")
    
    # Sous-titre pour les problématiques
    st.subheader("Problématiques")

    # Listage des problématiques
    st.write("""
    1. **Le nombre de commentaires et les notes en ligne attirent-ils plus de joueurs ?** 

    2. **Quels sont les thèmes et aspects les plus discutés dans les commentaires des utilisateurs ?**

    3. **Comment ces thèmes varient-ils en fonction du sentiment des avis ?**
    """)

elif st.session_state.selection == "📉 Analyse SteamDB":
      
    
    st.header("Analyse SteamDB")
    
    # Introduction
    st.markdown("""
    ### Introduction
    Dans cette section, nous allons analyser les données que nous avons téléchargé sur SteamDB pour nos trois jeux : Far Cry 6 (FC), Assassin's Creed Valhalla (ACV), et Watch Dogs: Legion (WDL). 
    Nous commencerons par une analyse combinée des trois jeux, suivie d'une analyse détaillée par jeux.
    """)
    st.markdown("---")
    
    # Graphique des followers over time
        
    st.subheader("Analyse de l'évolution des abonnées ")
    
    fig_followers = px.line(all_followers, x='DateTime', y='Followers', color='Source', title='Followers Over Time for Different Sources', height=700, width=1000)
    st.plotly_chart(fig_followers)
    st.markdown(""" **Commentaire :**
    Le graphique montre le nombre de followers au fil du temps pour trois jeux : Assassin's Creed Valhalla (ACV), Watch Dogs: Legion (WDL), et Far Cry (FC). 
    ACV a le plus grand nombre de followers, atteignant plus de 60 000 en juillet 2024, suivi par FC et WDL. Tous les jeux montrent une croissance continue des followers, 
    ACV ayant une croissance particulièrement rapide, surtout au début de 2023. FC connaît également une augmentation significative vers mai 2023.
    La tendance globale indique que l'intérêt des joueurs pour ces jeux reste stable et en croissance sur la période observée.
    """)

    # Graphique des positive reviews over time
    
    st.markdown("---")
    
    st.subheader("Analyse de l'évolution des commentaires positives et négatives ")
    
    fig_positive_reviews = px.line(all_reviews, x='DateTime', y='Positive_reviews', color='Source', title='Positive Reviews Over Time by Source')
    st.plotly_chart(fig_positive_reviews)
    st.markdown(""" **Commentaire :**
    Le graphique montre le nombre d'avis positifs au fil du temps pour Assassin's Creed Valhalla (ACV), Watch Dogs: Legion (WDL), et Far Cry (FC). 
    Les trois jeux présentent des pics significatifs d'avis positifs, particulièrement en début de leur lancement et en fin d'année.
    """)

    # Graphique des negative reviews over time
    
    fig_negative_reviews = px.line(all_reviews, x='DateTime', y='Negative_reviews', color='Source', title='Negative Reviews Over Time by Source')
    st.plotly_chart(fig_negative_reviews)
    st.markdown(""" **Commentaire :**
    Le graphique montre le nombre d'avis négatifs au fil du temps pour Assassin's Creed Valhalla (ACV), Watch Dogs: Legion (WDL), et Far Cry (FC). 
    Les pics d'avis négatifs coïncident souvent avec les premières semaines suivant la sortie ou les mises à jour majeures des jeux.
    Cela peut être dû à des attentes élevées et à une grande visibilité des problèmes techniques ou des aspects du jeux qui ne répondent pas aux attentes des joueurs.
    Après ces périodes initiales, le nombre d'avis négatifs tend à diminuer, probablement en raison de correctifs apportés par les développeurs et de l'ajustement des attentes des joueurs.
    """)

    st.markdown("---")
    
    st.subheader("Analyse de l'évolution du nombre de joueurs ")
    
    # Graphique des number of players over time
    
    fig_players = px.line(all_players_filtered, x='DateTime', y='Players', color='Source', title='Number of Players Over Time by Source')
    st.plotly_chart(fig_players)
    st.markdown(""" **Commentaire :**
    Les trois jeux présentent des pics de joueurs lors de leur lancement, suivis d'une diminution progressive. ACV et FC montrent des augmentations substantielles des joueurs 
    autour de leurs dates de lancement ou de mises à jour majeures. WDL a un pic initial plus modéré et une stabilité relative par la suite. La tendance générale indique que les 
    lancements attirent un grand nombre de joueurs initialement, mais que ce nombre diminue ensuite.
    """)

    st.markdown("---")
    
    st.subheader("Analyse de l'évolution des audiences TWITCH")
    
    # Graphique des Twitch viewers over time
    
    fig_twitch_viewers = px.line(all_players_filtered, x='DateTime', y='Twitch_Viewers', color='Source', title='Twitch Viewers Over Time by Source')
    st.plotly_chart(fig_twitch_viewers)
    st.markdown(""" **Commentaire :**
    Les pics initiaux en janvier 2023 indiquent une forte audience au lancement des jeux, particulièrement pour ACV et FC. Après ces pics initiaux, le nombre de spectateurs Twitch diminue 
    et devient plus stable avec des augmentations occasionnelles.
    """)

    st.markdown("---")
    
    st.subheader("Analyse de l'évolution des prix ")
    
    # Graphique des final price over time
  
    fig_final_price = px.line(all_prices, x='DateTime', y='Final_price', color='Source', title='Final Price Over Time by Source', height=700, width=1000)
    st.plotly_chart(fig_final_price)
    st.markdown(""" **Commentaire :**
    Les prix fluctuent fréquemment, avec des baisses notables suivies de hausses. En regardant plus attentivement le graphique, il est évident que malgré des dates de lancement différentes, 
    les jeux ACV, WDL, et FC semblent suivre un schéma similaire de promotions et de hausses de prix en même temps. Cela pourrait indiquer que Ubisoft utilise des stratégies de tarification 
    coordonnées pour maximiser l'impact de leurs ventes et promotions.
    """)

    st.markdown("---")

    st.subheader("Analyse de l'évolution du nombre de joueurs et des commentaires ")
    st.markdown("<br>", unsafe_allow_html=True)
    # Créer deux colonnes
    col1, col2 = st.columns([2, 1])
    with col1:
        image_path = os.path.join('images', 'comnegpos.png')
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(""" **Commentaire :**
    Il semble y avoir une relation entre les avis positifs et négatifs et le nombre de joueurs pour les trois jeux analysés :
    
    1. **Corrélation Positive** : Lorsque le nombre de joueurs augmente, le nombre d'avis (surtout positifs) augmente également. Cela peut s'expliquer par une plus grande visibilité et interaction avec le jeu lors des pics de joueurs.
    
    2. **Avis Négatifs** : Bien que les avis positifs soient plus nombreux lors des pics de joueurs, les avis négatifs augmentent également. Cela peut refléter une plus grande diversité d'opinions lorsque plus de joueurs interagissent avec le jeu.
    
    3. **Impact des Événements** : Les pics de joueurs, souvent liés à des événements spécifiques (lancements, mises à jour, promotions), génèrent plus d'avis.
    
    Ces observations suggèrent que les périodes de forte activité des joueurs sont cruciales pour recueillir des avis, tant positifs que négatifs. Analyser ces périodes peut offrir des insights sur les réactions des joueurs et aider à comprendre l'impact des événements spécifiques sur l'engagement des joueurs.
    """)
    
    st.markdown("---")
    
    st.subheader("Analyse de l'évolution du nombre de joueurs et des prix ")
    st.markdown("<br>", unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1, 2])
    with col2:
        image_path = os.path.join('images', 'prjr.png')
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        
    
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(""" **Commentaire :**
    Les trois graphes suggèrent une relation inverse entre le prix du jeu et le nombre de joueurs :
    
    **1- Baisse des Prix et Augmentation des Joueurs** : Les périodes de baisse de prix semblent correspondre à des augmentations du nombre de joueurs.
    Cela pourrait indiquer que les promotions et les baisses de prix attirent plus de joueurs.
    
    **2- Pics de Joueurs** : Les pics du nombre de joueurs apparaissent souvent après une réduction de prix, ce qui montre l'impact des stratégies de tarification sur l'engagement des joueurs.
    
    Les éditeurs pourraient bénéficier de stratégies de tarification dynamiques, en ajustant les prix pour attirer plus de joueurs pendant les périodes de faible activité.
    Promotions et Réductions : La mise en place de promotions et de réductions de prix peut être efficace pour stimuler l'engagement des joueurs et augmenter le nombre de joueurs actifs.
    """)

    st.markdown("---")
    
    st.markdown("## Conclusion")
      
    st.markdown("""
    Nous avons exploré deux aspects cruciaux de la performance des jeux Ubisoft à travers nos analyses précédentes. 
    D'une part, nous avons constaté une corrélation entre les variations de prix des jeux et le nombre de joueurs actifs. 
    En général, les baisses de prix semblent attirer davantage de joueurs, soulignant l'importance des stratégies de tarification 
    pour stimuler l'engagement. D'autre part, nous avons examiné l'impact des avis positifs et négatifs sur le nombre de joueurs. 
    Les périodes de forte activité des joueurs génèrent plus d'avis, et ces avis, qu'ils soient positifs ou négatifs, 
    reflètent souvent l'expérience des joueurs avec le jeu.

    Cette dualité entre les prix et les avis des joueurs ouvre la voie à une analyse plus approfondie. Il est essentiel de comprendre non seulement 
    la quantité des avis, mais aussi leur qualité. C'est là que l'analyse des sentiments entre en jeu. En examinant les sentiments des avis, 
    nous pouvons identifier les aspects spécifiques des jeux qui suscitent des réactions positives ou négatives. De plus, en analysant les mots clés 
    récurrents dans les commentaires, nous pouvons détecter les motifs de satisfaction ou de mécontentement parmi les joueurs.

    Ainsi, pour approfondir notre compréhension des dynamiques entre les prix, les avis et le comportement des joueurs, nous allons maintenant 
    effectuer une analyse des sentiments des commentaires. Cette analyse nous permettra de :

    1. Identifier les sentiments prédominants exprimés dans les avis positifs et négatifs.
    2. Déterminer les mots clés associés aux expériences positives et négatives des joueurs.

    Passons donc à l'analyse des sentiments et des mots clés dans les avis des joueurs pour obtenir des insights plus détaillés sur les réactions des utilisateurs aux jeux Ubisoft.
    """)



elif st.session_state.selection == "🕸️ Scraping Steam":
    st.header("Scraping Steam")
    
    image_path = os.path.join('images', 'steam.jpg')
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    
    
    st.subheader("Première étape : Présentation du Code de Scraping")
    st.markdown("""
    Dans cette section, nous allons présenter le code utilisé pour effectuer le scraping des données de Steam. Le scraping nous permet d'extraire des informations utiles sur les jeux directement à partir du site web de Steam.
    """)
    st.code("""
    
# Importer les modules nécessaires
import time  # Importer le module time pour gérer les pauses
import pandas as pd  # Importer pandas pour manipuler les DataFrames
from selenium import webdriver  # Importer webdriver de Selenium pour automatiser le navigateur
from selenium.webdriver.common.by import By  # Importer By pour localiser les éléments
from selenium.webdriver.chrome.service import Service  # Importer Service pour gérer le chromedriver
from selenium.webdriver.chrome.options import Options  # Importer Options pour configurer le navigateur Chrome
from selenium.webdriver.support.ui import WebDriverWait  # Importer WebDriverWait pour gérer les attentes
from selenium.webdriver.support import expected_conditions as EC  # Importer expected_conditions pour définir les conditions d'attente
from bs4 import BeautifulSoup  # Importer BeautifulSoup pour parser le HTML
from tqdm import tqdm  # Importer tqdm pour afficher une barre de progression

# Configurer Selenium WebDriver
chrome_options = Options()  # Initialiser les options de Chrome
chrome_options.add_argument("--no-sandbox")  # Ajouter l'option no-sandbox
chrome_options.add_argument("--disable-dev-shm-usage")  # Désactiver le partage de mémoire

# Initialiser le service pour le chromedriver
service = Service('C:\Program Files (x86)\chromedriver.exe')  
driver = webdriver.Chrome(service=service, options=chrome_options) 

# Initialiser des listes pour stocker les données
dates = []  # Liste pour stocker les dates
comments = []  # Liste pour stocker les commentaires
recommendations = []  # Liste pour stocker les recommandations
comment_set = set()  # Ensemble pour vérifier les doublons

# Fonction pour nettoyer le texte
def clean_text(text):  # Nettoyer le texte en supprimant les caractères indésirables
    return text.replace('\n', '').replace('\r', '').replace('\t', '').strip()

# Scraper les commentaires
max_comments = 10000  # Définir le nombre maximum de commentaires à scraper
page = 1  # Initialiser le numéro de page
comment_count = 0  # Initialiser le compteur de commentaires

# Ouvrir la page web
url = 'https://steamcommunity.com/app/2208920/reviews/?p=1&browsefilter=toprated&filterLanguage=all'  # URL de la page à scraper
driver.get(url)  # Ouvrir la page web avec le navigateur

# Fermer les fenêtres modales si elles existent
try:
    close_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, 'newmodal_close')))  # Attendre que le bouton de fermeture de la fenêtre modale soit cliquable
    close_button.click()  # Cliquer sur le bouton de fermeture
    print("Fenêtre modale fermée.")  # Imprimer un message de confirmation
except:
    print("Aucune fenêtre modale trouvée ou impossible à fermer.")  # Imprimer un message si aucune fenêtre modale n'est trouvée

# Cliquer sur "Voir le hub de la communauté"
try:
    community_hub_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Voir le hub de la communauté")]')))  # Attendre que le bouton "Voir le hub de la communauté" soit cliquable
    community_hub_button.click()  # Cliquer sur le bouton
    print("Bouton 'Voir le hub de la communauté' cliqué.")  # Imprimer un message de confirmation
except:
    print("Impossible de cliquer sur 'Voir le hub de la communauté'.")  # Imprimer un message si le bouton ne peut pas être cliqué

# Initialiser la barre de progression
pbar = tqdm(total=max_comments, desc="Scraping comments", unit="comment")  

while comment_count < max_comments:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Faire défiler la page vers le bas
    time.sleep(5)  # Attendre le chargement

    soup = BeautifulSoup(driver.page_source, 'html.parser')  # Parser le HTML de la page

    comment_divs = soup.find_all('div', class_='apphub_Card')  # Trouver toutes les divs avec la classe 'apphub_Card'

    if not comment_divs:
        print("Aucun commentaire trouvé.")  # Imprimer un message si aucun commentaire n'est trouvé
        break  # Sortir de la boucle si aucun commentaire n'est trouvé

    for comment_div in comment_divs:
        if comment_count >= max_comments:
            break  # Sortir de la boucle si le nombre maximum de commentaires est atteint

        try:
            comment_div_content = comment_div.find('div', class_='apphub_CardTextContent')  # Trouver la div contenant le texte du commentaire
            comment_text = clean_text(comment_div_content.text)  # Nettoyer le texte du commentaire
        except:
            comment_text = 'No comment available'  # Définir un texte par défaut si le commentaire n'est pas disponible

        if comment_text in comment_set:
            continue  # Sauter les doublons
        else:
            comment_set.add(comment_text)  # Ajouter le commentaire à l'ensemble des commentaires
            comments.append(comment_text)  # Ajouter le commentaire à la liste des commentaires

        try:
            recommendation_div = comment_div.find('div', class_='title')  # Trouver la div contenant la recommandation
            recommendation = clean_text(recommendation_div.text)  # Nettoyer le texte de la recommandation
            recommendations.append(recommendation)  # Ajouter la recommandation à la liste des recommandations
        except:
            recommendations.append('No recommendation available')  # Ajouter une recommandation par défaut si elle n'est pas disponible

        try:
            date_div = comment_div.find('div', class_='date_posted')  # Trouver la div contenant la date
            date = clean_text(date_div.text)  # Nettoyer le texte de la date
            dates.append(date)  # Ajouter la date à la liste des dates
        except:
            dates.append('No date available')  # Ajouter une date par défaut si elle n'est pas disponible

        comment_count += 1  # Incrémenter le compteur de commentaires
        pbar.update(1)  # Mettre à jour la barre de progression

    if len(comment_divs) == 0:
        print("Aucun nouveau commentaire chargé.")  # Imprimer un message si aucun nouveau commentaire n'est chargé
        break  # Sortir de la boucle si aucun nouveau commentaire n'est chargé

# Fermer le driver
driver.quit()  # Fermer le navigateur

pbar.close()  # Fermer la barre de progression

# Créer un DataFrame avec les données
df = pd.DataFrame({
    'Date': dates,  # Ajouter la colonne 'Date' au DataFrame
    'Recommendation': recommendations,  # Ajouter la colonne 'Recommendation' au DataFrame
    'Comment': comments  # Ajouter la colonne 'Comment' au DataFrame
})

# Sauvegarder les données dans un fichier CSV
df.to_csv('steam_comments_WDL.csv', index=False)  # Sauvegarder le DataFrame dans un fichier CSV sans l'index


    """, language='python')

    st.subheader("Deuxième étape : Traduction")
    st.markdown("""
    Après avoir extrait les données brutes, nous devons les traduire en un format utilisable pour notre analyse. Cette étape inclut la transformation des textes et la traduction si nécessaire.
    """)
    st.code("""
    # Importer le module Translator de googletrans pour traduire le texte
from googletrans import Translator  
# Importer tqdm pour afficher une barre de progression
from tqdm import tqdm  

# Initialiser le traducteur
translator = Translator()  

# Fonction pour traduire le texte avec la détection de la langue et la barre de progression
def translate_column(column):
    # Initialiser une liste pour stocker les textes traduits
    translated_texts = []  
    # Boucle pour chaque texte dans la colonne avec une barre de progression
    for text in tqdm(column, desc=f'Translating {column.name}', unit="text"):
        try:
            # Détecter la langue du texte
            detected_lang = translator.detect(text).lang  
            # Traduire le texte en anglais
            translated = translator.translate(text, src=detected_lang, dest='en')  
            # Ajouter le texte traduit à la liste
            translated_texts.append(translated.text)  
        except Exception as e:
            # Afficher un message en cas d'erreur de traduction
            print(f"Erreur de traduction: {e}")  
            # Ajouter le texte original en cas d'erreur
            translated_texts.append(text)  
    # Retourner la liste des textes traduits
    return translated_texts  

# Traduire les colonnes du DataFrame avec la barre de progression
# Appliquer la fonction de traduction à la colonne 'Comment' du DataFrame
df['Comment'] = translate_column(df['Comment'])  

    """, language='python')

    st.subheader("Troisième étape : Vérification des Données")
    st.markdown("""
    Une fois les données traduites, il est crucial de vérifier si nos commentaires ont bien été traduits et donc le cas contraires, ils seront supprimés.
    """)
    st.code("""
    # Fonction pour vérifier si une chaîne contient uniquement des caractères latins (utilisés pour l'anglais)
def is_english(text):
    # Vérifie si tous les caractères sont des lettres latines ou des espaces
    return re.match(r'^[\x00-\x7F]+$', text) is not None

# Appliquer la fonction sur la colonne 'Comment' et filtrer les lignes où le commentaire est en anglais
df_cleaned = df[df['Comment'].apply(is_english)]

    """, language='python')

    st.subheader("Quatrième étape : Traitement des Données")

    st.markdown("""
    Passons maintenant à la quatrième partie où nous allons effectuer le traitement du texte : Cela inclut la suppression des caractères spéciaux, la normalisation des textes, la suppression des stop words, tokenisation et lemmatisation.
    """)
    st.code("""
   
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import sys

# Rediriger temporairement stdout et stderr #
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Télécharger les ressources nécessaires pour NLTK #
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Réinitialiser stdout et stderr à leurs valeurs originales #
sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr

# Définir la fonction de nettoyage du texte #
def clean_text(text):
    # Suppression des caractères spéciaux et de la ponctuation #
    processed_text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Conversion en minuscules #
    processed_text = processed_text.lower()
    
    # Tokenisation du texte #
    tokens = word_tokenize(processed_text)
    
    # Suppression des mots vides #
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatisation des tokens #
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Joindre les tokens en une seule chaîne #
    cleaned_text = " ".join(tokens)
    
    return cleaned_text

# Appliquer le nettoyage du texte sur la colonne 'Review' #
all_comments['Cleaned_Review'] = all_comments['Review'].apply(clean_text)


    """, language='python')

    st.write("""Nous allons commencer par afficher les premières lignes du DataFrame, ce qui nous donnera un aperçu des données disponibles. 
Cet aperçu inclut les commentaires, les recommandations des utilisateurs, et les dates auxquelles ces commentaires ont été publiés. 
Ce point de départ est crucial pour comprendre le contexte et la diversité des opinions que nous analyserons en profondeur par la suite.

Voyons maintenant un aperçu des données :
""")
    
    st.dataframe(all_comments.head())
    st.write("Dimension de notre dataframe : ")
    st.write(all_comments.shape)
    
elif st.session_state.selection == "💬 Analyse de sentiment":
    
  
    
    st.header("Analyse de sentiment")
   
    
    st.write("""
L'analyse des sentiments est une technique puissante pour comprendre les opinions exprimées dans les commentaires des utilisateurs. 
Dans cette section, nous examinerons un échantillon des données de commentaires que nous avons recueillies pour évaluer les sentiments 
généraux des utilisateurs concernant les jeux vidéo. 

""")

# Afficher les premières lignes du DataFrame
    
    all_comments['Cleaned_Review'] = all_comments['Cleaned_Review'].astype(str)
    
    # Fonction pour calculer la polarité
    def get_polarity(text):
        return TextBlob(text).sentiment.polarity

    # Fonction pour calculer la subjectivité
    def get_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    # Ajouter les colonnes 'polarity' et 'subjectivity'
    all_comments['polarity'] = all_comments['Cleaned_Review'].apply(get_polarity)
    all_comments['subjectivity'] = all_comments['Cleaned_Review'].apply(get_subjectivity)

    # Ajouter la colonne 'sentiment' en fonction de la polarité
    all_comments['sentiment'] = all_comments['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Afficher les premiers résultats pour vérifier
    
    st.write("Nous analysons les sentiments des commentaires des utilisateurs pour comprendre leurs opinions et réactions envers les jeux en utilisant les fonction get_polarity et get_subjectivity.")
    st.dataframe(all_comments.head())
    
    st.markdown("---")
    
    st.subheader("Analyse de la Polarité et de la Subjectivité des Commentaires")

    # Créer une figure vide
    fig = go.Figure()

    # Ajouter le boxplot pour la polarité
    fig.add_trace(go.Box(
        y=all_comments['polarity'],
        name='Polarity',
        marker_color='blue'  # Couleur pour la polarité
    ))

    # Ajouter le boxplot pour la subjectivité
    fig.add_trace(go.Box(
        y=all_comments['subjectivity'],
        name='Subjectivity',
        marker_color='green'  # Couleur pour la subjectivité
    ))

    # Mettre à jour la mise en page de la figure
    fig.update_layout(
        title='Boxplot de la polarité et de la subjectivité',
        xaxis_title='Variables',
        yaxis_title='Valeur',
        width=800,  # Largeur du graphique
        height=600  # Hauteur du graphique
    )

    # Afficher le boxplot dans Streamlit
    st.plotly_chart(fig)
    
     
    st.write("""
    **Commentaire :** 
    Polarité : La ligne centrale du boxplot (médiane) est proche de 0, indiquant que la majorité des commentaires sont neutres en termes de polarité.
    Il y a plusieurs points en dehors de la boîte (outliers), surtout du côté négatif, ce qui montre qu'il y a des commentaires très négatifs, mais moins fréquents.
    Subjectivité : La médiane est autour de 0.5, indiquant que les commentaires sont en moyenne modérément subjectifs.
    Il y a moins d'outliers pour la subjectivité, ce qui montre une distribution plus uniforme des commentaires en termes de subjectivité.
    """)
    
    st.markdown("---")
    
    
    fig = px.scatter(all_comments, x='polarity', y='subjectivity', width=800, height=600)

    # Ajouter des titres et des étiquettes d'axe
    fig.update_layout(
        title='Scatter Plot de la polarité et de la subjectivité',
        xaxis_title='Polarity',
        yaxis_title='Subjectivity'
    )

    # Afficher le scatter plot dans Streamlit
    st.plotly_chart(fig)
    
    st.write("**Commentaire :** Le scatter plot montre que la majorité des avis ont une polarité proche de zéro, indiquant des sentiments neutres. La subjectivité varie considérablement, ce qui suggère que les avis sont souvent basés sur des opinions personnelles et non sur des faits objectifs.")
    
    st.markdown("---")
    
        # Convertir la colonne 'Date' en type de données 'datetime'
    all_comments['Date'] = pd.to_datetime(all_comments['Date'], infer_datetime_format=True)

    # Trier le DataFrame par ordre croissant de dates
    all_comments = all_comments.sort_values('Date')

    # Créer la figure
    fig = go.Figure()

    # Ajouter la trace de ligne pour l'évolution de la polarité dans le temps
    fig.add_trace(go.Scatter(x=all_comments['Date'], y=all_comments['polarity'], mode='lines', name='Polarité'))

    # Mettre en forme le titre et les étiquettes d'axe
    fig.update_layout(
        title="Évolution de la polarité des commentaires dans le temps",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Polarité')
    )

    # Afficher le graphique dans Streamlit
    #st.plotly_chart(fig)

    #st.write("**Commentaire :** L'évolution de la polarité des commentaires montre des variations significatives au fil du temps. Les pics de polarité positive et négative peuvent correspondre à des événements spécifiques ou à des mises à jour majeures du jeu.")

    # Compter le nombre de chaque sentiment
    sentiment_counts = all_comments['sentiment'].value_counts()

    # Obtenir les étiquettes de sentiment et les valeurs correspondantes
    labels = sentiment_counts.index.tolist()
    values = sentiment_counts.values.tolist()
    
    # Tracer le diagramme circulaire avec Plotly Express
    fig = px.pie(values=values, names=labels, title='Répartition des sentiments', hole=0.3)

    # Afficher le diagramme circulaire dans Streamlit
    st.plotly_chart(fig)
    
    
    st.write("**Commentaire :** La répartition des sentiments montre que les avis positifs représentent 39.9%, les avis négatifs 35.7%, et les avis neutres 24.4%. Cela indique une légère prédominance des avis positifs parmi les commentaires.")   
   
    # Calculer le nombre de commentaires par sentiment et par date
    df_agg = all_comments.groupby(['Date', 'sentiment']).size().reset_index(name='counts')

    # Créer le graphique
    fig = px.line(df_agg, x='Date', y='counts', color='sentiment', title='Analyse de sentiment des commentaires')

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    st.write("**Commentaire :** Les tendances montrent une prédominance de sentiments positifs au fil du temps, avec des pics de commentaires négatifs et neutres à certaines périodes. Les variations peuvent être liées à des événements spécifiques ou des mises à jour.")
   
    st.markdown("---")
     
    # Fonction pour générer le Word Cloud
    def generer_wordcloud(tweets, titre):
        tout_texte = " ".join(tweets)
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords={'assassin', 'creed', 'valhalla', 'ubisoft', 'odyssey', 'game', 'dog', 'far', 'cry', 'like', 'get', 'one', 'russia', 'state', 'terrorist', 'steam', 'even'}).generate(tout_texte)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(titre, fontsize=14)
        ax.axis('off')
        st.pyplot(fig)

    # Filtrer les commentaires positifs, négatifs et neutres
    tweets_positifs = all_comments[all_comments['sentiment'] == 'Positive']['Cleaned_Review']
    tweets_negatifs = all_comments[all_comments['sentiment'] == 'Negative']['Cleaned_Review']
    tweets_neutre = all_comments[all_comments['sentiment'] == 'Neutral']['Cleaned_Review']

    # Générer les Word Clouds
    st.subheader("Analyse Word-Cloud")
    generer_wordcloud(tweets_positifs, 'Word Cloud - Commentaires Positifs')

    st.write("**Commentaire :** Les commentaires positifs contiennent souvent des mots comme 'story', 'good', 'fun', indiquant des expériences agréables.")

    
    generer_wordcloud(tweets_negatifs, 'Word Cloud - Commentaires Négatifs')

    st.write("**Commentaire :** Les commentaires négatifs montrent des mots comme 'bad', 'time', 'story', soulignant les aspects négatifs du jeu.")

    
    generer_wordcloud(tweets_neutre, 'Word Cloud - Commentaires Neutres')

    st.write("**Commentaire :** Les commentaires neutres sont variés et contiennent des mots comme 'viking', 'nan', 'achievement', indiquant des discussions diverses.")
   

        # Liste des mots à supprimer
    mots_a_supprimer = {'assassin', 'creed', 'valhalla', 'ubisoft', 'odyssey', 'game', 'dog', 'watch', 'data', 'far', 'cry', 'steam', 'okay'}

    # Fonction pour tokeniser les textes en supprimant certains mots
    def tokenisation(textes):
        tokens = [token for line in textes for token in line.split() if token.lower() not in mots_a_supprimer]
        return tokens

    # Fonction pour obtenir les mots les plus utilisés
    def get_max_token(textes, num=30):
        tokens = tokenisation(textes)
        word_tokens = Counter(tokens)
        max_common = word_tokens.most_common(num)
        return dict(max_common)

    # Fonction pour visualiser les mots les plus utilisés
    def token_df_vis(x, title):
        df = pd.DataFrame(get_max_token(x).items(), columns=['words', 'count'])
        colors = ['red', 'blue', 'green', 'purple', 'orange']
    
        # Attribuer des couleurs aux 5 premiers mots
        df['color'] = ['red' if i < 5 else 'gray' for i in range(len(df))]
        
        fig = px.bar(df, x='words', y='count', title=title, color='color',color_discrete_map='identity')
        st.plotly_chart(fig)

    # Filtrer les commentaires positifs, négatifs et neutres
    tweets_positifs = all_comments[all_comments['sentiment'] == 'Positive']['Cleaned_Review']
    tweets_negatifs = all_comments[all_comments['sentiment'] == 'Negative']['Cleaned_Review']
    tweets_neutre = all_comments[all_comments['sentiment'] == 'Neutral']['Cleaned_Review']

    # Visualiser les mots les plus utilisés pour chaque catégorie de sentiment
    
    token_df_vis(tweets_positifs, 'Mots les Plus Utilisés - Commentaires Positifs')

    
    token_df_vis(tweets_negatifs, 'Mots les Plus Utilisés - Commentaires Négatifs')

    
    token_df_vis(tweets_neutre, 'Mots les Plus Utilisés - Commentaires Neutres')

    st.write("**Commentaire :** Les barres montrent les mots les plus fréquents dans chaque catégorie de sentiment. Les commentaires positifs contiennent souvent des mots comme 'good', 'like', 'story', tandis que les commentaires négatifs mettent en avant des termes comme 'time', 'bad', 'story'.") 
   
    st.markdown("---")
    
    st.subheader("Catégorisation des commentaires")
    
    st.write("""
    Dans cette analyse, nous avons créé trois listes de mots clés pour catégoriser et filtrer les commentaires en fonction de différents aspects. Ces listes sont cruciales pour comprendre les préoccupations des utilisateurs et les aspects spécifiques des jeux qui influencent leur satisfaction. Voici une explication des trois catégories de mots clés :

    1. **Mots Clés de Gameplay** :
        - **Objectif** : Identifier les aspects du gameplay mentionnés dans les commentaires.
        - **Exemples de Mots Clés** : 'story', 'character', 'mission', 'level', 'multiplayer'.
        - **Utilisation** : Ces mots nous aident à comprendre quelles parties du gameplay sont les plus discutées, qu'il s'agisse de la narration, des personnages, ou des mécanismes de jeu.

    2. **Mots Clés de promotion** :
        - **Objectif** : Isoler les mentions des offres promotionnelles et aux événements.
        - **Exemples de Mots Clés** : 'sale', 'discount', , 'promo', 'expansion', 'event'.
        - **Utilisation** : Ces mots sont utilisés pour filtrer les discussions sur les promotions, les nouvelles versions, les ventes saisonnières, et les événements spéciaux, ce qui nous permet de voir comment ces éléments affectent la perception des utilisateurs.

    3. **Mots Clés de Mise à Jour** :
        - **Objectif** : Détecter les commentaires relatifs aux mises à jour, aux problèmes techniques et les améliorations de performance
        - **Exemples de Mots Clés** : 'bug', 'fix', 'patch', 'improve', 'performance', 'update'.
        - **Utilisation** : En analysant ces mots, nous pouvons identifier les principaux problèmes techniques rencontrés par les utilisateurs et les améliorations souhaitées, ce qui est essentiel pour prioriser les correctifs et les améliorations dans les futures mises à jour.

    Ces listes de mots clés permettent une analyse plus fine et ciblée des commentaires, facilitant ainsi l'identification des tendances et des domaines d'amélioration prioritaires pour les développeurs.
    """)  
    st.code(""" 
           mots_gameplay = [
    'story', 'narrative', 'plot', 'world', 'universe', 'combat', 'fight', 'battle', 
    'character', 'protagonist', 'antagonist', 'npc', 'hero', 'villain', 'play', 
    'feel', 'experience', 'time', 'hours', 'fun', 'enjoy', 'enjoyable', 'entertain', 
    'entertaining', 'graphic', 'visual', 'bug', 'glitch', 'issue', 'problem', 
    'frame rate', 'fps', 'animation', 'ac', 'im', 'difficulty', 'challenge', 
    'level', 'mission', 'quest', 'adventure', 'exploration', 'puzzle', 'skill', 
    'multiplayer', 'coop', 'co-op', 'singleplayer', 'solo', 'online', 'offline', 
    'controller', 'keyboard', 'mouse', 'ai', 'artificial intelligence'
]
mots_promotion = [
    'sale', 'discount', 'offer', 'price', 'deal', 'promotion', 'promo', 'event', 
    'free', 'giveaway', 'launch', 'release', 'early access', 'beta', 'pre-order', 
    'bonus', 'bundle', 'package', 'exclusive', 'limited', 'special', 'anniversary', 
    'holiday', 'black friday', 'cyber monday', 'seasonal', 'christmas', 'easter', 
    'new year', 'valentine', 'halloween', 'festival', 'weekend deal', 'weeklong deal', 
    'flash sale', 'daily deal', 'membership', 'subscription', 'pass', 'season pass', 
    'expansion', 'dlc', 'add-on', 'content pack', 'update sale', 'promotion event', 
    'launch discount', 'release offer', 'early bird', 'vip', 'reward', 'gift', 
    'redeem', 'coupon', 'voucher', 'cashback', 'savings', 'markdown', 'slashed prices', 
    'price cut', 'half off', 'bogo', 'buy one get one', 'loyalty'
]

mots_update = [
    'bug', 'fix', 'patch', 'update', 'upgrade', 'improve', 'improvement', 'better', 
    'balance', 'balancing', 'rework', 'revamp', 'performance', 'optimize', 'optimization', 
    'stability', 'stable', 'crash', 'fixes', 'adjust', 'adjustment', 'hotfix', 
    'enhancement', 'tweak', 'modification', 'refinement', 'reduce', 'increase', 
    'loading time', 'load time', 'lag', 'latency', 'fps boost', 'frame rate improvement', 
    'smooth', 'smoother', 'efficient', 'efficiency', 'response time', 'input lag', 
    'memory usage', 'gpu', 'cpu', 'driver update', 'compatibility'
]
           
           """, language='python')
   
   
   

    # Listes de mots à utiliser pour l'analyse
    mots_gameplay = [
        'story', 'narrative', 'plot', 'world', 'universe', 'combat', 'fight', 'battle', 
        'character', 'protagonist', 'antagonist', 'npc', 'hero', 'villain', 'play', 
        'feel', 'experience', 'time', 'hours', 'fun', 'enjoy', 'enjoyable', 'entertain', 
        'entertaining', 'graphic', 'visual', 'bug', 'glitch', 'issue', 'problem', 
        'frame rate', 'fps', 'animation', 'ac', 'im', 'difficulty', 'challenge', 
        'level', 'mission', 'quest', 'adventure', 'exploration', 'puzzle', 'skill', 
        'multiplayer', 'coop', 'co-op', 'singleplayer', 'solo', 'online', 'offline', 
        'controller', 'keyboard', 'mouse', 'ai', 'artificial intelligence'
    ]

    mots_update = [
        'sale', 'discount', 'offer', 'price', 'deal', 'promotion', 'promo', 'event', 
        'free', 'giveaway', 'launch', 'release', 'early access', 'beta', 'pre-order', 
        'bonus', 'bundle', 'package', 'exclusive', 'limited', 'special', 'anniversary', 
        'holiday', 'black friday', 'cyber monday', 'seasonal', 'christmas', 'easter', 
        'new year', 'valentine', 'halloween', 'festival', 'weekend deal', 'weeklong deal', 
        'flash sale', 'daily deal', 'membership', 'subscription', 'pass', 'season pass', 
        'expansion', 'dlc', 'add-on', 'content pack', 'update sale', 'promotion event', 
        'launch discount', 'release offer', 'early bird', 'vip', 'reward', 'gift', 
        'redeem', 'coupon', 'voucher', 'cashback', 'savings', 'markdown', 'slashed prices', 
        'price cut', 'half off', 'bogo', 'buy one get one', 'loyalty'
    ]

    mots_promotion = [
        'bug', 'fix', 'patch', 'update', 'upgrade', 'improve', 'improvement', 'better', 
        'balance', 'balancing', 'rework', 'revamp', 'performance', 'optimize', 'optimization', 
        'stability', 'stable', 'crash', 'fixes', 'adjust', 'adjustment', 'hotfix', 
        'enhancement', 'tweak', 'modification', 'refinement', 'reduce', 'increase', 
        'loading time', 'load time', 'lag', 'latency', 'fps boost', 'frame rate improvement', 
        'smooth', 'smoother', 'efficient', 'efficiency', 'response time', 'input lag', 
        'memory usage', 'gpu', 'cpu', 'driver update', 'compatibility'
    ]

    # Fonction pour identifier les références dans les commentaires
    def refer(tweet, refs):
        flag = 0
        for ref in refs:
            if ref in tweet:
                flag = 1
                break
        return flag

    # Appliquer la fonction de référence pour chaque catégorie
    all_comments['gameplay'] = all_comments['Cleaned_Review'].apply(lambda x: refer(x, mots_gameplay))
    all_comments['update'] = all_comments['Cleaned_Review'].apply(lambda x: refer(x, mots_update))
    all_comments['promotion'] = all_comments['Cleaned_Review'].apply(lambda x: refer(x, mots_promotion))

    # Liste des catégories
    categories = ['gameplay', 'update', 'promotion']

    # Calculer la distribution de chaque catégorie en nombre
    category_counts = all_comments[categories].sum()

    df = pd.DataFrame({
    'Categorie': categories,
    'Nombre de Commentaires': category_counts
})

    # Créer un graphique à barres pour la distribution de chaque catégorie
    fig = px.bar(df, x='Categorie', y='Nombre de Commentaires', title='Distribution des Catégories dans les Commentaires')
    # Ajuster la disposition du texte à l'intérieur des barres
    fig.update_traces(textposition='inside')
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
    
    

    st.write("**Commentaire :** La distribution montre que les commentaires se concentrent principalement sur le gameplay, suivis des promotions et des mises à jour. Cela indique que les joueurs sont particulièrement attentifs aux aspects de gameplay des jeux.")
   
   
        # Liste des catégories
    categories = ['gameplay', 'update', 'promotion']

    # Création de sous-figures
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{cat.capitalize()} - Polarity" for cat in categories] + [f"{cat.capitalize()} - Subjectivity" for cat in categories])

    # Ajouter les boxplots de polarité
    for i, category in enumerate(categories):
        df_polarity = all_comments[all_comments[category] == 1]
        fig.add_trace(
            go.Box(y=df_polarity['polarity'], name=category.capitalize(), marker_color='blue'),
            row=1, col=i+1
        )

    # Ajouter les boxplots de subjectivité
    for i, category in enumerate(categories):
        df_subjectivity = all_comments[all_comments[category] == 1]
        fig.add_trace(
            go.Box(y=df_subjectivity['subjectivity'], name=category.capitalize(), marker_color='green'),
            row=2, col=i+1
        )

    # Mettre à jour les titres et les axes
    fig.update_layout(height=800, width=1200, title_text="Boxplots de Polarité et de Subjectivité par Catégorie")
    fig.update_yaxes(title_text="Polarité", row=1, col=1)
    fig.update_yaxes(title_text="Polarité", row=1, col=2)
    fig.update_yaxes(title_text="Polarité", row=1, col=3)
    fig.update_yaxes(title_text="Subjectivité", row=2, col=1)
    fig.update_yaxes(title_text="Subjectivité", row=2, col=2)
    fig.update_yaxes(title_text="Subjectivité", row=2, col=3)
    
# Afficher les boxplots dans Streamlit
    st.plotly_chart(fig)

    st.write("**Commentaire :** Les boxplots montrent que les commentaires sur le gameplay ont une polarité plus élevée comparée aux mises à jour et promotions. La subjectivité varie également, indiquant une diversité d'opinions dans chaque catégorie.")
   
    # Initialiser un DataFrame vide pour les résultats
    results_df = pd.DataFrame(columns=[
        'Category', 'Mean Polarity', 'Max Polarity', 'Min Polarity', 'Median Polarity',
        'Mean Subjectivity', 'Max Subjectivity', 'Min Subjectivity', 'Median Subjectivity'
    ])

    # Liste des catégories
    categories = ['gameplay', 'update', 'promotion']

    # Parcourir les catégories
    for category in categories:
        # Filtrer le DataFrame pour la catégorie spécifique
        filtered_df = all_comments[all_comments[category] == 1]
        
        # Calculer les statistiques pour la polarité et la subjectivité
        polarity_stats = filtered_df['polarity'].agg(['mean', 'max', 'min', 'median'])
        subjectivity_stats = filtered_df['subjectivity'].agg(['mean', 'max', 'min', 'median'])
        
        # Ajouter les résultats au DataFrame des résultats
        new_row = pd.DataFrame([{
            'Category': category.capitalize(),
            'Mean Polarity': polarity_stats['mean'],
            'Max Polarity': polarity_stats['max'],
            'Min Polarity': polarity_stats['min'],
            'Median Polarity': polarity_stats['median'],
            'Mean Subjectivity': subjectivity_stats['mean'],
            'Max Subjectivity': subjectivity_stats['max'],
            'Min Subjectivity': subjectivity_stats['min'],
            'Median Subjectivity': subjectivity_stats['median']
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
  
    

    # Convertir la colonne 'Date' en type datetime
    all_comments['Date'] = pd.to_datetime(all_comments['Date'], format='%Y-%m-%d')

    # Liste des catégories
    categories = ['gameplay', 'update', 'promotion']

    # Créer des graphiques séparés pour l'évolution de la moyenne mobile de la polarité par date pour chaque catégorie
    for category in categories:
        filtered_df = all_comments[all_comments[category] == 1]
        filtered_df = filtered_df.set_index('Date')
        filtered_df = filtered_df.sort_index()
        filtered_df['polarity_moving_avg'] = filtered_df['polarity'].rolling(window=7).mean()  # Moyenne mobile sur 7 jours

        fig = px.line(filtered_df, x=filtered_df.index, y='polarity_moving_avg',
                    title=f'Évolution de la Moyenne Mobile de la Polarité ({category.capitalize()})',
                    labels={'polarity_moving_avg': 'Moyenne Mobile de la Polarité', 'Date': 'Date'})
        
        st.plotly_chart(fig)
   
    st.write("**Commentaire :** L'évolution de la moyenne mobile de la polarité montre des fluctuations régulières dans toutes les catégories, avec des pics et des creux distincts. Cela peut indiquer des variations dans la réception des jeux au fil du temps, influencées par des mises à jour, des promotions ou des éléments de gameplay.")

   

elif st.session_state.selection == "🔍 Conclusion":
    
        # Titre de l'application
    st.title("Conclusion")

    # Section 1: Analyse des données Steam DB
    st.header("1. Analyse des données Steam DB")
    st.write("""
    L'analyse des données Steam DB a révélé des tendances significatives concernant les avis positifs et négatifs, ainsi que les prix des jeux au fil du temps pour "Assassin's Creed Valhalla" (ACV), "Watch Dogs: Legion" (WDL) et "Far Cry" (FC).

    Les trois jeux avaient connu des pics d'avis positifs et négatifs autour des dates de lancement et des mises à jour majeures. Les avis positifs avaient tendance à augmenter lors des lancements et des événements spéciaux, tandis que les avis négatifs augmentaient souvent suite aux problèmes techniques ou aux aspects du jeu qui ne répondaient pas aux attentes des joueurs.

    Les fluctuations des prix montraient des baisses notables suivies de hausses, indiquant une stratégie de tarification dynamique. Ces fluctuations coïncidaient souvent avec des périodes de promotions et de mises à jour. La corrélation entre les baisses de prix et les augmentations du nombre de joueurs suggérait que les promotions attiraient plus de joueurs, augmentant ainsi l'engagement des utilisateurs.
    """)

    # Section 2: Analyse des Sentiments des Commentaires Steam
    st.header("2. Analyse des Sentiments des Commentaires Steam")
    st.write("""
    L'analyse des sentiments des commentaires Steam avait permis de catégoriser les avis en positifs, négatifs et neutres, offrant une vue détaillée des perceptions des utilisateurs.

    La majorité des commentaires étaient positifs (39.9%), suivis de négatifs (35.7%) et de neutres (24.4%), montrant une légère prédominance des avis positifs. Les sentiments variaient considérablement au fil du temps, avec des pics de sentiments négatifs et neutres à certaines périodes, souvent liés à des événements spécifiques ou des mises à jour.

    Les avis montraient que la majorité des commentaires avaient une polarité neutre, tandis que la subjectivité était répartie autour de 0.5, indiquant des avis variés et souvent personnels. La majorité des avis avaient une polarité proche de zéro, avec une subjectivité variant considérablement, suggérant que les avis étaient souvent basés sur des opinions personnelles.

    L'évolution de la polarité des commentaires montrait des variations significatives, avec des pics de polarité positive et négative correspondant à des événements spécifiques ou des mises à jour majeures du jeu.

    Les analyses des mots les plus fréquemment utilisés dans les commentaires positifs, négatifs et neutres montraient les aspects les plus discutés par les joueurs.
    """)

    # Conclusion globale
    st.header("Conclusion générale")
    st.write("""
    Les analyses combinées de Steam DB et des sentiments des commentaires ont offert une vue d'ensemble précieuse sur la perception des jeux par les joueurs et les stratégies de tarification efficaces. Les avis des utilisateurs montraient une corrélation avec les événements du jeu, les mises à jour et les stratégies de prix, soulignant l'importance de ces facteurs dans la satisfaction et l'engagement des joueurs. En utilisant des techniques d'analyse de sentiments, il a été possible d'obtenir des insights approfondis pour améliorer les jeux et les stratégies marketing. Ces analyses ont permis de mieux comprendre les préférences des joueurs et d'ajuster les stratégies en conséquence pour maximiser la satisfaction et l'engagement des utilisateurs.
    """)


elif st.session_state.selection == "🧠 APP NLP":
    st.header("Modèle de Traitement Automatique du Langage Naturel (TALN)")
    st.write("L'application d'analyse de sentiment permet aux utilisateurs d'entrer du texte et d'obtenir une évaluation de la polarité et de la subjectivité de leurs propos. Utilisant des bibliothèques Python telles que TextBlob et VADER (SentimentIntensityAnalyzer), cette application analyse le sentiment global et les sentiments individuels des mots dans le texte. Elle présente les résultats sous forme de tableau de données et de graphique à barres interactif, offrant ainsi une vue d'ensemble claire et intuitive des sentiments exprimés.")
    
    # Emplacement pour le modèle
    st.subheader("Entrer un texte pour l'analyse")
    with st.form(key='nlpForm'):
        user_input = st.text_area("Texte à analyser")
        submit_button = st.form_submit_button(label='Analyser')
        
    if submit_button and user_input:
        st.write("Analyse du texte:")

        # Analyse avec TextBlob
        sentiment = TextBlob(user_input).sentiment
        st.write(sentiment)

        # Emoji
        if sentiment.polarity > 0:
            st.markdown("Sentiment:: Positive :smiley:")
        elif sentiment.polarity < 0:
            st.markdown("Sentiment:: Negative :angry:")
        else:
            st.markdown("Sentiment:: Neutral 😐")

        # Dataframe
        result_df = convert_to_df(sentiment)
        st.dataframe(result_df)

        # Visualization
        c = alt.Chart(result_df).mark_bar().encode(
            x='metric',
            y='value',
            color='metric'
        )
        st.altair_chart(c, use_container_width=True)

        # Analyse des tokens avec VADER
        st.info("Token Sentiment")
        token_sentiments = analyze_token_sentiment(user_input)
        st.write(token_sentiments)

        # Compter le nombre de chaque type de sentiment
        positive_count = len(token_sentiments['positives'])
        negative_count = len(token_sentiments['negatives'])
        neutral_count = len(token_sentiments['neutral'])

        # Créer un DataFrame pour les proportions de chaque sentiment
        data = {
            'sentiment': ['positives', 'negatives', 'neutral'],
            'count': [positive_count, negative_count, neutral_count]
        }
        df = pd.DataFrame(data)

        # Créer le graphique en cercle
        fig = px.pie(df, values='count', names='sentiment', 
                    title='Distribution des Sentiments des Tokens',
                    color='sentiment',
                   )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

    
  
# Footer
st.markdown("""
    <style>
        footer {
            visibility: hidden;
        }
        footer:after {
            content:'© 2024 Ubisoft Analyses. Tous droits réservés.';
            visibility: visible;
            display: block;
            position: relative;
            padding: 10px;
            top: 2px;
         }
     </style>
    """, unsafe_allow_html=True)
    


