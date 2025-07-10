## 1\. Intro

Ce projet présente un **agent de Machine Learning adaptatif** conçu pour automatiser les étapes clés de la création d'un modèle de classification à partir de **n'importe quel dataset tabulaire**. L'agent est capable d'effectuer le *feature engineering*, d'entraîner un modèle, d'évaluer ses performances et de tracer l'intégralité de l'expérience à l'aide de **MLflow**.

Le but est de fournir un système flexible où l'utilisateur n'a qu'à fournir un chemin vers son dataset et le nom de la colonne cible, et l'agent gère le reste de manière autonome, s'adaptant à la structure des données.

-----

## 2\. Fonctionnalités

  * **Chargement Adaptatif des Données** : Lit des fichiers CSV génériques.
  * **Analyse et *Feature Engineering* Automatisés** :
      * Détection automatique des caractéristiques numériques et catégorielles.
      * Gestion des valeurs manquantes (imputation par la médiane/mode).
      * Encodage des variables catégorielles (One-Hot Encoding).
      * Mise à l'échelle des caractéristiques numériques (StandardScaler).
      * Identification et suppression des colonnes non pertinentes (trop de valeurs uniques/manquantes).
  * **Entraînement et Évaluation de Modèle** : Utilise un `RandomForestClassifier` pour l'entraînement et fournit des métriques de performance clés (accuracy, precision, recall, F1-score).
  * **Intégration MLflow** : Trace automatiquement les paramètres, les métriques et le modèle entraîné pour chaque exécution, facilitant la reproductibilité et la comparaison des expériences.
  * **Interface API (FastAPI)** : Permet de déclencher le pipeline ML via une simple requête HTTP, facilitant l'intégration dans d'autres systèmes ou des interfaces utilisateur.
  * **Conteneurisation (Docker)** : L'ensemble de l'application est packagé dans un conteneur Docker pour une installation et un déploiement faciles et reproductibles.

-----

## 3\. Architecture du Projet

Le projet suit une architecture modulaire pour une meilleure maintenabilité et évolutivité :

```
.
├── Dockerfile                  # Configuration de l'image Docker
├── docker-compose.yml          # Orchestration des services Docker (application et potentiellement MLflow)
├── requirements.txt            # Liste des dépendances Python
├── .dockerignore               # Fichiers à ignorer lors du build Docker
├── data/                       # Dossier pour stocker les datasets (Titanic, Framingham, etc.)
│   └── titanic_train.csv
│   └── framingham.csv
└── src/                        # Code source de l'application
    ├── __init__.py
    ├── main.py                 # Point d'entrée de l'API FastAPI
    ├── ml_agent/               # Logique principale de l'agent ML
    │   ├── __init__.py
    │   ├── agent.py            # Orchestrateur du pipeline ML
    │   ├── data_processor.py   # Gestion adaptative du chargement et feature engineering
    │   ├── model_trainer.py    # Entraînement et évaluation du modèle
    │   └── mlflow_logger.py    # Fonctions d'interaction avec MLflow
    └── utils/                  # Fonctions utilitaires diverses
        ├── __init__.py
        └── data_loader.py      # Chargement générique de données
```

-----

## 4\. Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

  * **Docker Desktop** (incluant Docker Engine et Docker Compose) : [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

-----

## 5\. Lancement de l'Application

### Cloner le Dépôt

Commencez par cloner ce dépôt GitHub (remplacez `[URL_DU_DEPOT]` par l'URL réelle de votre dépôt) :

```bash
git clone [URL_DU_DEPOT]
cd [nom-du-dossier-du-depot]
```

### Préparer les Datasets

1.  Créez un dossier nommé `data/` à la racine de votre projet (au même niveau que `Dockerfile` et `docker-compose.yml`).
    ```bash
    mkdir data
    ```
2.  **Téléchargez les datasets de test** et placez-les dans ce dossier `data/` :
      * **Titanic Survival Prediction (`titanic_train.csv`)** :
        [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
      * **Framingham Heart Study Dataset (`framingham.csv`)** :
        [https://www.kaggle.com/datasets/mirichoi0218/framingham-heart-study-dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/mirichoi0218/framingham-heart-study-dataset)

### Démarrer les Services Docker

Naviguez vers le répertoire racine de votre projet dans votre terminal et exécutez les commandes suivantes :

1.  **Construire l'image Docker :**
    ```bash
    docker-compose build
    ```
2.  **Démarrer le conteneur de l'application :**
    ```bash
    docker-compose up
    ```
    Laissez ce terminal ouvert pour voir les logs de l'application.

-----

## 6\. Utilisation de l'API

L'API est maintenant accessible à l'adresse `http://localhost:80`.

### Endpoint Principal

  * **GET `/`** : Vérifie si l'API est opérationnelle.
    ```bash
    curl http://localhost:80
    # Expected output: {"message": "Agent ML API est opérationnel !"}
    ```

### Lancement du Pipeline ML

  * **POST `/run_ml_pipeline/`** : Déclenche l'exécution du pipeline ML adaptatif.

    Vous devez fournir le chemin du dataset (relatif au dossier `/app/data/` dans le conteneur) et le nom de la colonne cible.

      * **Exemple avec le dataset Titanic (`titanic_train.csv`, cible: `Survived`) :**

        ```bash
        curl -X POST \
             -H "Content-Type: application/json" \
             -d '{"data_path": "titanic_train.csv", "target_column": "Survived"}' \
             http://localhost:80/run_ml_pipeline/
        ```

      * **Exemple avec le dataset Framingham (`framingham.csv`, cible: `diabetes`) :**

        ```bash
        curl -X POST \
             -H "Content-Type: application/json" \
             -d '{"data_path": "framingham.csv", "target_column": "diabetes"}' \
             http://localhost:80/run_ml_pipeline/
        ```

      * **Réponse attendue (succès) :**

        ```json
        {"status": "success", "message": "Pipeline ML exécuté avec succès. Vérifiez les logs MLflow."}
        ```

      * **En cas d'erreur :**

        ```json
        {"status": "error", "message": "Une erreur est survenue: [description de l'erreur]"}
        ```

-----

## 7\. Accéder à l'Interface Utilisateur MLflow

Pour visualiser les expériences MLflow journalisées par l'agent :

1.  Ouvrez un **nouveau terminal** (sans arrêter le `docker-compose up` en cours).
2.  Naviguez vers la racine de votre projet.
3.  Lancez l'interface utilisateur MLflow :
    ```bash
    mlflow ui
    ```
4.  Ouvrez votre navigateur et allez à l'adresse : `http://localhost:5000`

Vous y trouverez toutes les "runs" de votre agent, avec les paramètres utilisés, les métriques de performance et les modèles sauvegardés.

-----

## 8\. Exemples de Datasets de Test

L'agent est conçu pour s'adapter à divers datasets tabulaires de classification. N'hésitez pas à télécharger d'autres datasets (par exemple depuis Kaggle ou l'UCI Machine Learning Repository) et à les placer dans le dossier `data/` pour les tester.

Assurez-vous simplement que :

  * Le fichier est au format CSV.
  * Vous connaissez le nom exact de la colonne cible à prédire.

-----

## 9\. Améliorations Possibles

Cet agent peut être enrichi de nombreuses façons :

  * **Optimisation des Hyperparamètres** : Intégrer `GridSearchCV` ou `RandomizedSearchCV` dans `ModelTrainer` pour trouver automatiquement les meilleurs hyperparamètres.
  * **Sélection de Modèle Automatisée** : Permettre à l'agent de tester plusieurs types de classificateurs (Logistic Regression, XGBoost, LightGBM, etc.) et de choisir le plus performant.
  * **API d'Inférence** : Ajouter un *endpoint* API pour effectuer des prédictions sur de nouvelles données en utilisant un modèle entraîné et loggué via MLflow.
  * ***Feature Engineering* Avancé** : Intégrer des logiques plus complexes pour la création de nouvelles *features* (par exemple, à partir de dates, de textes courts, ou d'interactions de colonnes).
  * **Gestion des Valeurs Aberrantes (Outliers)** : Implémenter des stratégies pour détecter et traiter les valeurs aberrantes.
  * **Déploiement du Tracking Server MLflow** : Mettre en place un service MLflow Tracking Server persistant via Docker Compose pour une gestion des expériences plus robuste.