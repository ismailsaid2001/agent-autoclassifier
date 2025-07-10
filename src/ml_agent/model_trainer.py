import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold # <-- Nouvelle importation
import warnings

# Supprimer les avertissements de métriques si aucune prédiction positive/négative n'est faite
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')

class ModelTrainer:
    """
    Classe responsable de l'entraînement et de l'évaluation du modèle,
    incluant la recherche d'hyperparamètres avec GridSearchCV et validation croisée stratifiée.
    """
    def __init__(self):
        self.model = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Entraîne un modèle de classification (RandomForestClassifier) et l'évalue
        en utilisant GridSearchCV pour l'optimisation des hyperparamètres
        avec une validation croisée stratifiée.
        """
        print("Initialisation de la recherche d'hyperparamètres avec GridSearchCV pour RandomForestClassifier...")

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        base_model = RandomForestClassifier(random_state=42)

        # Définition de la stratégie de validation croisée stratifiée
        # n_splits : nombre de plis (folds)
        # shuffle : mélanger les données avant de les diviser
        # random_state : pour la reproductibilité du mélange
        cv_strategy = StratifiedKFold(n_splits=4
                                      , shuffle=True, random_state=42) # <-- Ajout de StratifiedKFold

        # Configuration de GridSearchCV avec la stratégie de CV explicite
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_strategy, # <-- Utilisation de l'objet StratifiedKFold
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        print("Lancement de l'ajustement du GridSearchCV sur les données d'entraînement...")
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        
        print(f"Meilleurs hyperparamètres trouvés: {grid_search.best_params_}")
        print(f"Meilleur score de validation (Accuracy): {grid_search.best_score_:.4f}")
        print("Modèle entraîné avec les meilleurs hyperparamètres.")

        print("Évaluation du modèle final sur l'ensemble de test...")
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        
        print("\nMétriques du modèle sur l'ensemble de test:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return self.model, metrics