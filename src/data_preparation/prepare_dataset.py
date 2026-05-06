import os
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path

def limpiar_totalsegmentator_v2(ruta_origen, ruta_destino):
    ruta_origen = Path(ruta_origen)
    ruta_destino = Path(ruta_destino)

    dir_images = ruta_destino / "imagesTr"
    dir_labels = ruta_destino / "labelsTr"
    dir_images.mkdir(parents=True, exist_ok=True)
    dir_labels.mkdir(parents=True, exist_ok=True)

    pacientes = [d for d in ruta_origen.iterdir() if d.is_dir()]
    print(f"[*] Encontrados {len(pacientes)} pacientes. Iniciando extraccin V2.0 (7 clases)...")

    MAPEO_CLASES = {
        "vertebrae_T12.nii.gz": 1,
        "vertebrae_L1.nii.gz": 2,
        "vertebrae_L2.nii.gz": 3,
        "vertebrae_L3.nii.gz": 4,
        "vertebrae_L4.nii.gz": 5,
        "iliopsoas_left.nii.gz": 6,
        "iliopsoas_right.nii.gz": 7
    }

    for paciente_dir in pacientes:
        nombre_paciente = paciente_dir.name

        ruta_ct = paciente_dir / "ct.nii.gz"
        dir_masks = paciente_dir / "segmentations"

        if not ruta_ct.exists():
            print(f"  [!] Saltando {nombre_paciente}: Falta el archivo ct.nii.gz.")
            continue

        print(f"  [+] Procesando {nombre_paciente}...")

        ct_nifti = nib.load(ruta_ct)
        mascara_unificada = np.zeros(ct_nifti.shape, dtype=np.uint8)

        clases_encontradas = 0

        for nombre_archivo, etiqueta in MAPEO_CLASES.items():
            ruta_mascara = dir_masks / nombre_archivo

            if ruta_mascara.exists():
                mask_data = nib.load(ruta_mascara).get_fdata().astype(bool)
                mascara_unificada[mask_data] = etiqueta
                clases_encontradas += 1
            else:
                print(f"      - Aviso: No se encontr {nombre_archivo}")

        if clases_encontradas > 0:
            ruta_destino_ct = dir_images / f"{nombre_paciente}_0000.nii.gz"

            if not ruta_destino_ct.exists():
                shutil.copy(ruta_ct, ruta_destino_ct)

            nifti_unificado = nib.Nifti1Image(mascara_unificada, ct_nifti.affine)
            nib.save(nifti_unificado, dir_labels / f"{nombre_paciente}.nii.gz")
            print(f"      -> OK: Guardado con {clases_encontradas}/7 estructuras.")
        else:
            print(f"  [!] Saltando {nombre_paciente}: No tena ninguna de las 7 estructuras diana.")

    print("\n[+] Proceso masivo terminado! Tu dataset 3D Versin 2.0 est listo para entrenar.")

if __name__ == "__main__":
    ruta_total_segmentator = "C:/Users/theme/Downloads/Proyecto_IA_Medica/TotalSegmentator"

    ruta_mi_dataset_v2 = "C:/Users/theme/Downloads/Proyecto_IA_Medica/Dataset002_LumbarPsoas"

    limpiar_totalsegmentator_v2(ruta_total_segmentator, ruta_mi_dataset_v2)