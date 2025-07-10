import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV dans un DataFrame pandas.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Fichier '{file_path}' chargé avec succès. Forme: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé.")
        raise
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du fichier: {e}")
        raise