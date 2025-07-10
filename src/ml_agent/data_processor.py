import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class DataProcessor:
    """
    Classe responsable du chargement, du nettoyage et du feature engineering des données.
    Capable de s'adapter à différents datasets tabulaires de classification.
    """
    def __init__(self, data_path: str, target_column: str, unique_threshold: float = 0.9, missing_threshold: float = 0.8):
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.preprocessor = None # Pour stocker le ColumnTransformer ou le pipeline de preprocessing
        self.numerical_features = []
        self.categorical_features = []
        self.features_to_drop = []
        self.unique_threshold = unique_threshold # Seuil pour détecter des colonnes quasi-uniques (ex: IDs)
        self.missing_threshold = missing_threshold # Seuil pour détecter des colonnes avec trop de valeurs manquantes

    def load_data(self) -> pd.DataFrame:
        """Charge le dataset."""
        print(f"Chargement du dataset depuis: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print("Dataset chargé.")
        print(f"Forme du dataset: {self.df.shape}")
        return self.df

    def _analyze_and_select_features(self):
        """
        Analyse le DataFrame pour détecter automatiquement les types de colonnes
        et identifier les colonnes à supprimer ou à traiter.
        """
        print("Analyse des caractéristiques du dataset pour adaptation...")
        
        # Exclure la colonne cible de l'analyse des caractéristiques
        X_temp = self.df.drop(columns=[self.target_column])

        # Identification des colonnes à supprimer
        for col in X_temp.columns:
            # Supprimer les colonnes avec trop de valeurs uniques (potentiels IDs)
            if X_temp[col].nunique() / len(X_temp) > self.unique_threshold:
                self.features_to_drop.append(col)
                print(f"  Colonne '{col}' identifiée comme quasi-unique (ID) et sera supprimée.")
            
            # Supprimer les colonnes avec trop de valeurs manquantes
            elif X_temp[col].isnull().sum() / len(X_temp) > self.missing_threshold:
                self.features_to_drop.append(col)
                print(f"  Colonne '{col}' identifiée avec trop de valeurs manquantes et sera supprimée.")

        # Créer un DataFrame temporaire sans les colonnes à supprimer pour l'analyse des types
        X_filtered = X_temp.drop(columns=self.features_to_drop, errors='ignore')
        
        # Détection des types de colonnes restants
        for col in X_filtered.columns:
            if pd.api.types.is_numeric_dtype(X_filtered[col]):
                self.numerical_features.append(col)
            elif pd.api.types.is_object_dtype(X_filtered[col]) or pd.api.types.is_categorical_dtype(X_filtered[col]):
                # Pour les colonnes object, vérifier si elles ont un nombre raisonnable de catégories
                # Un seuil de 50 est un point de départ, peut être ajusté.
                if X_filtered[col].nunique() < 50: 
                    self.categorical_features.append(col)
                else:
                    self.features_to_drop.append(col) # Sinon, la traiter comme non utilisable (ex: texte libre)
                    print(f"  Colonne '{col}' identifiée comme catégorielle avec trop de catégories uniques et sera supprimée ou non traitée.")
            else: # Autres types (datetime, etc.) sont ignorés pour l'instant
                self.features_to_drop.append(col)
                print(f"  Colonne '{col}' identifiée comme un type non supporté actuellement et sera supprimée.")
        
        print(f"Caractéristiques numériques détectées: {self.numerical_features}")
        print(f"Caractéristiques catégorielles détectées: {self.categorical_features}")
        print(f"Caractéristiques à supprimer (finale): {self.features_to_drop}")

    def _define_preprocessing_pipeline(self):
        """
        Définit le pipeline de preprocessing pour les caractéristiques numériques et catégorielles
        en utilisant les listes de features auto-détectées.
        """
        transformers = []

        if self.numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, self.numerical_features))
            print(f"  Pipeline numérique créé pour: {self.numerical_features}")

        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_features))
            print(f"  Pipeline catégoriel créé pour: {self.categorical_features}")

        if not transformers:
            # Ceci pourrait arriver si toutes les colonnes sont supprimées ou non supportées.
            raise ValueError("Aucune caractéristique numérique ou catégorielle valide n'a été détectée après l'analyse. Impossible de créer le pipeline de preprocessing.")

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop' # Supprime les colonnes non spécifiées (celles qui ont été ignorées ou à supprimer)
        )
        print("Pipeline de preprocessing dynamique défini.")

    def process_data(self):
        """
        Exécute le pipeline complet de traitement des données de manière adaptative.
        Retourne les ensembles d'entraînement/test prétraités et le preprocessor.
        """
        df = self.load_data()

        # Effectuer l'analyse et la sélection des caractéristiques
        self._analyze_and_select_features()

        # Séparer les caractéristiques (X) de la cible (y)
        # S'assurer que la colonne cible est présente
        if self.target_column not in df.columns:
            raise ValueError(f"La colonne cible '{self.target_column}' n'a pas été trouvée dans le dataset.")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Supprimer les colonnes identifiées comme non pertinentes ou problématiques de X
        X = X.drop(columns=self.features_to_drop, errors='ignore')

        # Double vérification pour s'assurer que les listes de features ne contiennent que les colonnes restantes dans X
        # Ceci est important car _analyze_and_select_features peut avoir ajouté des colonnes à self.features_to_drop
        # après la détection initiale des types si elles avaient trop de catégories par exemple.
        self.numerical_features = [col for col in self.numerical_features if col in X.columns]
        self.categorical_features = [col for col in self.categorical_features if col in X.columns]


        # Diviser les données en ensembles d'entraînement et de test
        print("Division des données en ensembles d'entraînement et de test...")
        # Vérifier si y est binaire pour la stratification
        if y.nunique() == 2:
            stratify_y = y
            print("Cible binaire détectée, stratification appliquée.")
        else:
            print("Attention: La colonne cible n'est pas binaire ou n'a pas 2 classes. La stratification ne sera pas appliquée.")
            stratify_y = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_y
        )
        print(f"Forme de X_train: {X_train.shape}, Forme de X_test: {X_test.shape}")

        # Définir et ajuster le pipeline de preprocessing sur les données d'entraînement
        self._define_preprocessing_pipeline()
        print("Application du preprocessing sur les données d'entraînement et de test...")
        # X_train_processed et X_test_processed seront des numpy arrays
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print("Données prétraitées avec succès.")
        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor