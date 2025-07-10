from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .mlflow_logger import MLflowLogger
import pandas as pd
from sklearn.pipeline import make_pipeline # <-- Nouvelle importation : make_pipeline

class MLAgent:
    """
    Agent de Machine Learning orchestrant les étapes de preprocessing,
    d'entraînement, d'évaluation et de loggage via MLflow.
    Il s'adapte aux datasets tabulaires pour la classification.
    """
    def __init__(self, data_path: str, target_column: str,
                 unique_threshold: float = 0.9, missing_threshold: float = 0.8):
        self.data_path = data_path
        self.target_column = target_column
        self.unique_threshold = unique_threshold
        self.missing_threshold = missing_threshold
        self.mlflow_logger = MLflowLogger(experiment_name="Adaptive_ML_Agent_Classification")
        
        print(f"MLAgent initialisé pour le dataset: {self.data_path} et la cible: {self.target_column}")

    def run_pipeline(self):
        """
        Exécute le pipeline complet de Machine Learning de manière adaptative.
        """
        run_name = f"Run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}_{self.target_column}"
        self.mlflow_logger.start_run(run_name=run_name)
        try:
            # Log initial parameters
            self.mlflow_logger.log_params({
                "data_path": self.data_path,
                "target_column": self.target_column,
                "unique_threshold": self.unique_threshold,
                "missing_threshold": self.missing_threshold
            })

            # 1. Traitement des données adaptatif
            print("Étape 1: Traitement et Feature Engineering des données adaptatif...")
            processor = DataProcessor(
                data_path=self.data_path,
                target_column=self.target_column,
                unique_threshold=self.unique_threshold,
                missing_threshold=self.missing_threshold
            )
            # 'preprocessor' est le ColumnTransformer entraîné par DataProcessor
            X_train, X_test, y_train, y_test, preprocessor = processor.process_data()
            
            # Log des caractéristiques détectées
            self.mlflow_logger.log_params({
                "detected_numerical_features": processor.numerical_features,
                "detected_categorical_features": processor.categorical_features,
                "dropped_features_by_agent": processor.features_to_drop
            })

            # 2. Entraînement et évaluation du modèle
            print("Étape 2: Entraînement et évaluation du modèle...")
            trainer = ModelTrainer()
            model, metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # 3. Loggage des résultats et du pipeline complet avec MLflow
            print("Étape 3: Journalisation des résultats et du pipeline ML complet avec MLflow...")
            self.mlflow_logger.log_metrics(metrics)
            
            # Création du pipeline complet qui combine le préprocesseur et le modèle
            # make_pipeline crée un Pipeline simplifié où les noms des étapes sont générés automatiquement
            full_pipeline = make_pipeline(preprocessor, model) # <-- Création du pipeline complet
            
            # Enregistrer le pipeline complet avec MLflow
            self.mlflow_logger.log_model(model=full_pipeline, artifact_path="full_ml_pipeline") # <-- Log du pipeline complet
            
            print("Pipeline exécuté avec succès. Vérifiez l'interface utilisateur MLflow pour les détails.")

        except Exception as e:
            print(f"Une erreur est survenue lors de l'exécution du pipeline: {e}")
            self.mlflow_logger.end_run(status="FAILED")
            raise # Renvoyer l'exception pour que l'appelant puisse la gérer
        finally:
            self.mlflow_logger.end_run()