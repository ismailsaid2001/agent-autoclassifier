import os
from fastapi import FastAPI, HTTPException
from src.ml_agent.agent import MLAgent
from pydantic import BaseModel

# FastAPI application instance
app = FastAPI(
    title="Agent ML FastAPI",
    description="API pour exécuter le pipeline de Machine Learning de manière adaptative."
)

class RunPipelineRequest(BaseModel):
    data_path: str = "data/titanic_train.csv"  # Default for testing
    target_column: str = "Survived"           # Default for testing

@app.get("/")
async def read_root():
    return {"message": "Agent ML API est opérationnel !"}

@app.post("/run_ml_pipeline/")
async def trigger_ml_pipeline(request: RunPipelineRequest):
    """
    Déclenche le pipeline de Machine Learning avec le dataset et la colonne cible fournis.
    L'agent s'adapte au dataset tabulaire pour la classification.
    """
    # Construire le chemin absolu du fichier data_path
    # Le chemin dans le conteneur est /app/data/<filename>
    # Si le client fournit juste "titanic_train.csv", nous le convertissons en "/app/data/titanic_train.csv"
    # Si le chemin est déjà un chemin absolu ou commence par "data/", il sera traité comme tel.
    # Pour une utilisation via Docker Compose, le volume `data` est monté dans `/app/data`.
    # Donc, si le request.data_path est "titanic_train.csv", on doit le faire pointer vers "/app/data/titanic_train.csv"

    # Gérer le chemin du dataset à l'intérieur du conteneur Docker
    # Si request.data_path est juste un nom de fichier, ajoute le préfixe /app/data/
    if not request.data_path.startswith('/'): # If it's a relative path, assume it's in /app/data
        full_data_path_in_container = os.path.join("/app/data", os.path.basename(request.data_path))
    else:
        full_data_path_in_container = request.data_path # If it's already an absolute path

    # Vérifier si le fichier dataset existe à l'emplacement attendu dans le conteneur
    if not os.path.exists(full_data_path_in_container):
        raise HTTPException(
            status_code=404,
            detail=f"Le fichier dataset est introuvable à {full_data_path_in_container}. "
                   f"Veuillez vous assurer qu'il est monté dans le dossier 'data/' du conteneur."
        )

    print(f"Lancement de l'agent ML avec le dataset: {full_data_path_in_container} et la cible: {request.target_column}")
    
    try:
        agent = MLAgent(data_path=full_data_path_in_container, target_column=request.target_column)
        agent.run_pipeline()
        return {"status": "success", "message": "Pipeline ML exécuté avec succès. Vérifiez les logs MLflow."}
    except Exception as e:
        # Log l'erreur complète pour le débogage
        print(f"Erreur lors de l'exécution du pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    # Ceci est utile pour des tests locaux sans Docker/Uvicorn en mode developpement,
    # mais en production ou avec docker-compose, uvicorn appellera directement `app`.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)