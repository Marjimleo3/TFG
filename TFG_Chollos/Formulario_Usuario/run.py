from TFG_Chollos.utils import conseguir_ruta_general_TFG
import subprocess

BASE = conseguir_ruta_general_TFG()

app = BASE / "Formulario_Usuario/Formulario_Web.py"
subprocess.run(["streamlit", "run", app])