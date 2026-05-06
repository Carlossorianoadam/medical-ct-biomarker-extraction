import json
import os
from pathlib import Path

# ==========================================
# RUTA A TU NUEVO DATASET (Verifica que sea esta)
# ==========================================
DIR_DATASET = "C:/Users/theme/Downloads/Proyecto_IA_Medica/Dataset002_LumbarPsoas"

def generar_json():
    dir_dataset = Path(DIR_DATASET)
    dir_images = dir_dataset / "imagesTr"
    
    if not dir_images.exists():
        print(f"❌ Error: No se encuentra la carpeta {dir_images}. Revisa la ruta.")
        return

    # Contamos cuántos archivos _0000.nii.gz hay
    imagenes = [f for f in os.listdir(dir_images) if f.endswith('_0000.nii.gz')]
    num_casos = len(imagenes)
    
    # Estructura del JSON para nnU-Net v2
    dataset_info = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "T12": 1,
            "L1": 2,
            "L2": 3,
            "L3": 4,
            "L4": 5,
            "psoas_izquierdo": 6,
            "psoas_derecho": 7
        },
        "numTraining": num_casos,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO" # Recomendado para NIfTI
    }
    
    # Guardamos el archivo dataset.json en la raíz del dataset
    ruta_json = dir_dataset / "dataset.json"
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"✅ ¡dataset.json creado con éxito!")
    print(f"🧠 Total de pacientes detectados: {num_casos}")
    print(f"📁 Ubicación: {ruta_json}")

if __name__ == "__main__":
    generar_json()