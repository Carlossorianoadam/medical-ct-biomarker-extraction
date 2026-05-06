# pipeline_3d_multiclase_lote.py

import os
import subprocess
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# =========================================================
# 1. CONFIGURACIÓN DE RUTAS
# =========================================================

ruta_base = "D:/Proyecto_IA_Medica"

os.environ["nnUNet_raw"] = f"{ruta_base}/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = f"{ruta_base}/nnUNet_preprocessed"
os.environ["nnUNet_results"] = f"{ruta_base}/nnUNet_results"

# Carpeta con todos los pacientes en formato NIfTI
# Ejemplo: SMARTLT_CT001_0000.nii.gz, SMARTLT_CT002_0000.nii.gz...
carpeta_entrada = "C:/IA_Test/Entrada_Lote2.2"

# Carpeta donde se guardarán las máscaras y las imágenes de control
carpeta_salida_3d = "C:/IA_Test/Salida_3D_Lote"

os.makedirs(carpeta_salida_3d, exist_ok=True)

NNUNET_EXE = "C:/Users/theme/anaconda3/Scripts/nnUNetv2_predict.exe"

DATASET_ID = "002"
CONFIG = "3d_fullres"
FOLD = "0"
CHECKPOINT = "checkpoint_best.pth"


# =========================================================
# 2. FUNCIÓN PARA GENERAR IMAGEN DE CONTROL
# =========================================================

def generar_mapa_anatomico(ruta_tac, ruta_mask, carpeta_salida, id_paciente):
    tac = nib.load(ruta_tac)
    mask = nib.load(ruta_mask)

    datos_tac = tac.get_fdata()
    datos_mask = mask.get_fdata()

    # Corte con más volumen de L1
    if np.any(datos_mask == 2):
        z_l1 = int(np.argmax(np.sum(datos_mask == 2, axis=(0, 1))))
    else:
        z_l1 = datos_mask.shape[2] // 2

    # Corte con más volumen de L3
    if np.any(datos_mask == 4):
        z_l3 = int(np.argmax(np.sum(datos_mask == 4, axis=(0, 1))))
    else:
        z_l3 = datos_mask.shape[2] // 2

    # Corte coronal centrado en la columna
    coords_columna = np.argwhere((datos_mask >= 1) & (datos_mask <= 5))

    if len(coords_columna) > 0:
        y_coronal = int(np.mean(coords_columna[:, 1]))
    else:
        y_coronal = datos_mask.shape[1] // 2

    colores = [
        "none",    # 0: Fondo
        "purple",  # 1: T12
        "red",     # 2: L1
        "orange",  # 3: L2
        "yellow",  # 4: L3
        "cyan",    # 5: L4
        "lime",    # 6: Psoas izquierdo
        "green"    # 7: Psoas derecho
    ]

    cmap_custom = mcolors.ListedColormap(colores)
    norm_custom = mcolors.BoundaryNorm(np.arange(9), cmap_custom.N)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plt.suptitle(
        f"MAPA ANATÓMICO 3D: {id_paciente}",
        fontsize=16,
        fontweight="bold"
    )

    # Imagen 1: L1
    axs[0].imshow(np.rot90(datos_tac[:, :, z_l1]), cmap="gray")
    overlay_l1 = np.ma.masked_where(
        datos_mask[:, :, z_l1] == 0,
        datos_mask[:, :, z_l1]
    )
    axs[0].imshow(
        np.rot90(overlay_l1),
        cmap=cmap_custom,
        norm=norm_custom,
        alpha=0.6
    )
    axs[0].set_title(f"Nivel L1 / óseo\nCorte Z: {z_l1}", fontsize=12)
    axs[0].axis("off")

    # Imagen 2: L3
    axs[1].imshow(np.rot90(datos_tac[:, :, z_l3]), cmap="gray")
    overlay_l3 = np.ma.masked_where(
        datos_mask[:, :, z_l3] == 0,
        datos_mask[:, :, z_l3]
    )
    axs[1].imshow(
        np.rot90(overlay_l3),
        cmap=cmap_custom,
        norm=norm_custom,
        alpha=0.6
    )
    axs[1].set_title(f"Nivel L3 / muscular\nCorte Z: {z_l3}", fontsize=12)
    axs[1].axis("off")

    # Imagen 3: Coronal
    axs[2].imshow(np.rot90(datos_tac[:, y_coronal, :]), cmap="gray")
    overlay_coronal = np.ma.masked_where(
        datos_mask[:, y_coronal, :] == 0,
        datos_mask[:, y_coronal, :]
    )
    axs[2].imshow(
        np.rot90(overlay_coronal),
        cmap=cmap_custom,
        norm=norm_custom,
        alpha=0.7
    )
    axs[2].set_title("Vista coronal\nColumna completa y psoas", fontsize=12)
    axs[2].axis("off")

    plt.tight_layout()

    ruta_guardado = os.path.join(
        carpeta_salida,
        f"{id_paciente}_anatomia_7clases.png"
    )

    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"📸 Imagen de control guardada: {ruta_guardado}")


# =========================================================
# 3. INFERENCIA CON NNUNET PARA TODOS LOS PACIENTES
# =========================================================

print("==================================================")
print("🧠 INICIANDO RED NEURONAL MULTICLASE EN LOTE")
print("==================================================")

archivos_nifti = [
    archivo for archivo in os.listdir(carpeta_entrada)
    if archivo.endswith("_0000.nii.gz")
]

if len(archivos_nifti) == 0:
    raise RuntimeError("No se encontraron archivos acabados en _0000.nii.gz.")

print(f"Pacientes encontrados: {len(archivos_nifti)}")

# nnU-Net puede procesar todos los pacientes de la carpeta directamente
comando = (
    f'"{NNUNET_EXE}" '
    f'-i "{carpeta_entrada}" '
    f'-o "{carpeta_salida_3d}" '
    f'-d {DATASET_ID} '
    f'-c {CONFIG} '
    f'-f {FOLD} '
    f'-chk {CHECKPOINT}'
)

proceso = subprocess.run(
    comando,
    shell=True,
    capture_output=True,
    text=True
)

if proceso.returncode != 0:
    print("❌ Ha ocurrido un error durante la predicción:")
    print(proceso.stderr)
    raise RuntimeError("La inferencia con nnU-Net ha fallado.")

print("✅ Predicción 3D completada con éxito.")
print("📌 Generando mapas anatómicos por paciente...")


# =========================================================
# 4. GENERACIÓN DE IMÁGENES DE CONTROL POR PACIENTE
# =========================================================

for archivo in archivos_nifti:
    id_paciente = archivo.replace("_0000.nii.gz", "")

    ruta_tac = os.path.join(carpeta_entrada, archivo)

    # nnU-Net elimina el sufijo _0000 al guardar la máscara
    ruta_mask = os.path.join(carpeta_salida_3d, f"{id_paciente}.nii.gz")

    if not os.path.exists(ruta_mask):
        print(f"⚠️ No se encontró máscara para {id_paciente}. Se omite.")
        continue

    print("\n==================================================")
    print(f"🖼️ Procesando visualización: {id_paciente}")
    print("==================================================")

    try:
        generar_mapa_anatomico(
            ruta_tac=ruta_tac,
            ruta_mask=ruta_mask,
            carpeta_salida=carpeta_salida_3d,
            id_paciente=id_paciente
        )

    except Exception as e:
        print(f"❌ Error generando imagen para {id_paciente}:")
        print(e)


print("\n==================================================")
print("✅ PROCESO COMPLETADO")
print(f"Máscaras e imágenes guardadas en:\n{carpeta_salida_3d}")
print("==================================================")