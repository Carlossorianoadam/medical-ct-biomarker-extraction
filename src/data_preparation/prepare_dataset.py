import os
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path

def limpiar_totalsegmentator_v2(ruta_origen, ruta_destino):
    ruta_origen = Path(ruta_origen)
    ruta_destino = Path(ruta_destino)
    
    # Creamos las carpetas estilo "Machine Learning" (formato MSD)
    dir_images = ruta_destino / "imagesTr"
    dir_labels = ruta_destino / "labelsTr"
    dir_images.mkdir(parents=True, exist_ok=True)
    dir_labels.mkdir(parents=True, exist_ok=True)
    
    pacientes = [d for d in ruta_origen.iterdir() if d.is_dir()]
    print(f"[*] Encontrados {len(pacientes)} pacientes. Iniciando extracción V2.0 (7 clases)...")
    
    # Diccionario con el mapeo exacto (Derecho e Izquierdo separados)
    MAPEO_CLASES = {
        "vertebrae_T12.nii.gz": 1,
        "vertebrae_L1.nii.gz": 2,
        "vertebrae_L2.nii.gz": 3,
        "vertebrae_L3.nii.gz": 4,
        "vertebrae_L4.nii.gz": 5,
        "iliopsoas_left.nii.gz": 6,
        "iliopsoas_right.nii.gz": 7
    }
    
    # Procesar TODOS los pacientes del dataset
    for paciente_dir in pacientes:
        nombre_paciente = paciente_dir.name
        
        ruta_ct = paciente_dir / "ct.nii.gz"
        dir_masks = paciente_dir / "segmentations"
        
        # Si al paciente le falta el TAC base, lo saltamos
        if not ruta_ct.exists():
            print(f"  [!] Saltando {nombre_paciente}: Falta el archivo ct.nii.gz.")
            continue
            
        print(f"  [+] Procesando {nombre_paciente}...")
        
        # 1. Cargamos el TAC para obtener las dimensiones y crear el lienzo vacío
        ct_nifti = nib.load(ruta_ct)
        mascara_unificada = np.zeros(ct_nifti.shape, dtype=np.uint8)
        
        clases_encontradas = 0
        
        # 2. Buscamos y pintamos cada clase que exista en este paciente
        for nombre_archivo, etiqueta in MAPEO_CLASES.items():
            ruta_mascara = dir_masks / nombre_archivo
            
            if ruta_mascara.exists():
                # Cargamos la máscara, la pasamos a booleano y la pintamos con su etiqueta
                mask_data = nib.load(ruta_mascara).get_fdata().astype(bool)
                mascara_unificada[mask_data] = etiqueta
                clases_encontradas += 1
            else:
                print(f"      - Aviso: No se encontró {nombre_archivo}")
                
        # 3. Solo guardamos el paciente si hemos encontrado al menos alguna de las estructuras
        if clases_encontradas > 0:
            # Copiamos el TAC original renombrado para nnU-Net (_0000)
            ruta_destino_ct = dir_images / f"{nombre_paciente}_0000.nii.gz"
            
            # Seguro de vida: Si el TAC ya existe, no lo volvemos a copiar para ahorrar tiempo
            if not ruta_destino_ct.exists(): 
                shutil.copy(ruta_ct, ruta_destino_ct)
            
            # Guardamos la máscara unificada
            nifti_unificado = nib.Nifti1Image(mascara_unificada, ct_nifti.affine)
            nib.save(nifti_unificado, dir_labels / f"{nombre_paciente}.nii.gz")
            print(f"      -> OK: Guardado con {clases_encontradas}/7 estructuras.")
        else:
            print(f"  [!] Saltando {nombre_paciente}: No tenía ninguna de las 7 estructuras diana.")

    print("\n[+] ¡Proceso masivo terminado! Tu dataset 3D Versión 2.0 está listo para entrenar.")

if __name__ == "__main__":
    # Rutas actualizadas para tu usuario 'theme'
    ruta_total_segmentator = "C:/Users/theme/Downloads/Proyecto_IA_Medica/TotalSegmentator"
    
    # IMPORTANTE: He cambiado el nombre de la carpeta destino para no sobrescribir tu trabajo anterior
    ruta_mi_dataset_v2 = "C:/Users/theme/Downloads/Proyecto_IA_Medica/Dataset002_LumbarPsoas"
    
    limpiar_totalsegmentator_v2(ruta_total_segmentator, ruta_mi_dataset_v2)