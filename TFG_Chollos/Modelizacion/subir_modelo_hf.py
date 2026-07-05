'''
subir_modelo_hf.py
===================
Sube el Bosque Aleatorio serializado (~1,3 GB, demasiado grande para GitHub) a un repositorio de modelos en Hugging Face Hub, público, para que App_Cloud pueda descargarlo en tiempo de ejecución sin necesidad de token.

Requiere la variable de entorno HF_TOKEN con un token de escritura (https://huggingface.co/settings/tokens). No pases el token como argumento ni lo escribas en este archivo.

Uso (PowerShell):
    $env:HF_TOKEN = "hf_xxx"
    python TFG_Chollos/Modelizacion/subir_modelo_hf.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import os

from huggingface_hub import HfApi

from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
REPO_ID  = "mario878/tfg-chollos-modelos"
ARCHIVO  = "bosque_aleatorio_reg.pkl"

logger = configurar_logger(__name__)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Falta la variable de entorno HF_TOKEN. "
            "Define $env:HF_TOKEN = 'hf_xxx' antes de ejecutar este script."
        )

    BASE = conseguir_ruta_general_TFG()
    ruta_modelo = BASE / 'data' / 'models' / ARCHIVO
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

    logger.info(f'⏳ Subiendo {ruta_modelo} ({ruta_modelo.stat().st_size / 1e9:.2f} GB)...')
    api.upload_file(
        path_or_fileobj=str(ruta_modelo),
        path_in_repo=ARCHIVO,
        repo_id=REPO_ID,
        repo_type="model",
    )
    logger.info(f'✅ Modelo subido: https://huggingface.co/{REPO_ID}')


if __name__ == "__main__":
    main()
