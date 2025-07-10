import mlflow
import mlflow.sklearn
from typing import Dict, Any
import os

class MLflowLogger:
    """
    Classe utilitaire pour interagir avec MLflow afin de logguer les expériences.
    """
    def __init__(self, experiment_name: str = "ML_Agent_Classification_Experiment"):
        self.experiment_name = experiment_name
        
        # Configuration du tracking URI de MLflow
        # Si vous avez un serveur MLflow distant, décommentez et ajustez la ligne ci-dessous :
        # os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000" # Ou l'adresse de votre serveur MLflow

        # S'assurer que le répertoire mlruns est créé dans le répertoire de travail local si aucun serveur n'est spécifié
        if "MLFLOW_TRACKING_URI" not in os.environ and not os.path.exists("mlruns"):
            os.makedirs("mlruns", exist_ok=True)
            print("Création du répertoire 'mlruns' local pour le tracking MLflow.")

        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow initialisé. Expérience actuelle: {self.experiment_name}")
        self.active_run = None

    def start_run(self, run_name: str = None):
        """Démarre une nouvelle run MLflow."""
        if self.active_run:
            print("Une run MLflow est déjà active. La terminer avant d'en démarrer une nouvelle.")
            self.end_run() # Assure que la run précédente est bien fermée
        
        self.active_run = mlflow.start_run(run_name=run_name)
        print(f"MLflow Run ID: {self.active_run.info.run_id}")
        return self.active_run

    def end_run(self, status: str = "FINISHED"):
        """Termine la run MLflow active."""
        if self.active_run:
            mlflow.end_run(status=status)
            print(f"MLflow Run {self.active_run.info.run_id} terminée avec le statut: {status}")
            self.active_run = None
        else:
            print("Aucune run MLflow active à terminer.")

    def log_params(self, params: Dict[str, Any]):
        """Loggue un dictionnaire de paramètres."""
        if self.active_run:
            print(f"Journalisation des paramètres: {params}")
            mlflow.log_params(params)
        else:
            print("Pas de run MLflow active pour logguer les paramètres.")

    def log_metrics(self, metrics: Dict[str, float]):
        """Loggue un dictionnaire de métriques."""
        if self.active_run:
            print(f"Journalisation des métriques: {metrics}")
            mlflow.log_metrics(metrics)
        else:
            print("Pas de run MLflow active pour logguer les métriques.")

    def log_model(self, model: Any, artifact_path: str, registered_model_name: str = None):
        """Loggue un modèle Scikit-learn."""
        if self.active_run:
            print(f"Journalisation du modèle vers l'artefact: {artifact_path}")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name # Optionnel: enregistrer dans le Model Registry
            )
            print("Modèle loggué.")
        else:
            print("Pas de run MLflow active pour logguer le modèle.")

    @staticmethod
    def load_model(run_id: str, artifact_path: str):
        """Charge un modèle loggué depuis une run MLflow spécifique."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"Chargement du modèle depuis: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modèle chargé.")
        return model