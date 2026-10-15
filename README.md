# 📈 AI Stock Price Prediction with News Sentiment (LSTM & Streamlit)

Application web interactive de Deep Learning développée en Python qui combine l'analyse des **tendances boursières historiques** (via Yahoo Finance) et **l'analyse de sentiment des actualités financières** (via NLTK/VADER) pour prédire les cours boursiers à l'aide d'un réseau de neurones récurrents (**LSTM**).

## 🚀 Fonctionnalités Clés
* **Double Source de Données (Multimodal) :** Intégration automatique des prix de clôture et du score de sentiment des actualités financières en temps réel pour affiner la modélisation[cite: 5, 10].
* **Architecture LSTM Profonde :** Réseau séquentiel multi-couches avec régularisation (`Dropout`) pour l'apprentissage des séries temporelles financières[cite: 10].
* **Prévision Future (3 Mois) :** Simulation et projection des tendances boursières à moyen terme[cite: 9].
* **Interface Interactive (Streamlit) :** Permet une configuration dynamique (choix de l'action, des dates, des époques d'entraînement et de la taille des batchs)[cite: 6, 7].

## 🛠️ Stack Technique
* **Langage :** Python
* **Interface Web :** Streamlit[cite: 6]
* **Deep Learning :** TensorFlow, Keras[cite: 10]
* **NLP & Sentiment :** NLTK (VADER), Feedparser, Finvizfinance[cite: 5]
* **Data Science & API :** Pandas, NumPy, Scikit-Learn, yfinance[cite: 6, 10]
* **Visualisation :** Matplotlib[cite: 8, 9, 10]

## 📊 Aperçu de l'Application & Parcours Utilisateur

**1. Page d'accueil et visualisation des données boursières historiques :**
![Accueil et Données Boursières](app_screenshot1.png)

**2. Analyse du sentiment des actualités financières en temps réel (multi-sources) :**
![Analyse des Actualités et Sentiments](app_screenshot2.jpg)

**3. Dataset fusionné (Association des cours de clôture et des scores de sentiment journaliers) :**
![Dataset Fusionné](app_screenshot3.png)

**4. Entraînement et évaluation du modèle (Comparaison Prix Réels vs Prédits) :**
![Évaluation du Modèle LSTM](app_screenshot4.png)

**5. Projection et simulation des tendances boursières sur 3 mois :**
![Prévision sur 3 Mois](app_screenshot5.png)

**6. Estimation finale et prédiction du cours pour le jour suivant :**
![Prédiction du Prochain Jour](app_screenshot6.png)

## ⚙️ Comment lancer l'application en local ?

1. Clonez ou téléchargez ce dépôt.
2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
