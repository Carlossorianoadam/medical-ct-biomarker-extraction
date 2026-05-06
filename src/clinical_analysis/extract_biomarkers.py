
import os
import csv
import json
import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.ndimage import binary_erosion, center_of_mass
from skimage import draw



CARPETA_TAC = "C:/IA_Test/Entrada_Lote"
CARPETA_MASK = "C:/IA_Test/Salida_3D_Lote"
CARPETA_SALIDA = "C:/IA_Test/Resultados_IA_Lote"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

DATASET_SPLIT = "test"

CLASES = {
    1: "T12",
    2: "L1",
    3: "L2",
    4: "L3",
    5: "L4",
    6: "Psoas izquierdo",
    7: "Psoas derecho",
}

FALLBACK_OSEO = [2, 1, 3]
FALLBACK_MUSCULAR = [4, 5]

AREA_ROI_MM2 = 200
EROSION_TRABECULAR_2D = 2
EROSION_TRABECULAR_3D = 2



def convertir_id(id_original):
    numero = id_original.split("_")[-1]
    return f"SMARTLT_CT{int(numero):03d}"

def cargar_nifti(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontr el archivo: {ruta}")
    return nib.load(ruta)


def obtener_voxel_info(nifti):
    dx, dy, dz = nifti.header.get_zooms()[:3]
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "area_pixel_cm2": (dx * dy) / 100.0,
        "volumen_voxel_cm3": (dx * dy * dz) / 1000.0,
    }


def calcular_mid_slice_desde_mascara(mask_3d):
    coords = np.argwhere(mask_3d)
    if len(coords) == 0:
        return None, None, None

    z_min = int(coords[:, 2].min())
    z_max = int(coords[:, 2].max())
    z_mid = int(math.floor((z_min + z_max) / 2.0))

    return z_mid, z_min, z_max


def dividir_vertebra_anterior_posterior(mask_vertebra_3d, tac_3d):
    coords = np.argwhere(mask_vertebra_3d)
    if len(coords) == 0:
        return np.zeros_like(mask_vertebra_3d, dtype=bool)

    y_min = int(coords[:, 1].min())
    y_max = int(coords[:, 1].max())
    y_mid = int((y_min + y_max) / 2)

    parte_a = np.copy(mask_vertebra_3d)
    parte_b = np.copy(mask_vertebra_3d)

    parte_a[:, y_mid:, :] = False
    parte_b[:, :y_mid, :] = False

    hu_a = np.mean(tac_3d[parte_a]) if np.any(parte_a) else np.inf
    hu_b = np.mean(tac_3d[parte_b]) if np.any(parte_b) else np.inf

    return parte_a if hu_a < hu_b else parte_b


def aislar_trabecular_3d(mask_vertebra_3d, tac_3d):
    cuerpo = dividir_vertebra_anterior_posterior(mask_vertebra_3d, tac_3d)
    if not np.any(cuerpo):
        return cuerpo
    return binary_erosion(cuerpo, iterations=EROSION_TRABECULAR_3D)


def aislar_trabecular_2d(mask_vertebra_2d, tac_2d):
    coords = np.argwhere(mask_vertebra_2d)

    if len(coords) == 0:
        return np.zeros_like(mask_vertebra_2d, dtype=bool)

    y_min = int(coords[:, 1].min())
    y_max = int(coords[:, 1].max())
    y_mid = int((y_min + y_max) / 2)

    parte_a = np.copy(mask_vertebra_2d)
    parte_b = np.copy(mask_vertebra_2d)

    parte_a[:, y_mid:] = False
    parte_b[:, :y_mid] = False

    hu_a = np.mean(tac_2d[parte_a]) if np.any(parte_a) else np.inf
    hu_b = np.mean(tac_2d[parte_b]) if np.any(parte_b) else np.inf

    cuerpo = parte_a if hu_a < hu_b else parte_b

    if not np.any(cuerpo):
        return cuerpo

    return binary_erosion(cuerpo, iterations=EROSION_TRABECULAR_2D)


def seleccionar_nivel_valido(datos_mask, datos_tac, lista_fallbacks):
    for i, clase in enumerate(lista_fallbacks):
        mask_vertebra = datos_mask == clase

        if not np.any(mask_vertebra):
            continue

        cuerpo = dividir_vertebra_anterior_posterior(mask_vertebra, datos_tac)
        z_mid, z_min, z_max = calcular_mid_slice_desde_mascara(cuerpo)

        if z_mid is None:
            continue

        return {
            "clase": clase,
            "nombre": CLASES[clase],
            "fallback_usado": i > 0,
            "z_mid": z_mid,
            "z_min": z_min,
            "z_max": z_max,
            "mask_3d": mask_vertebra,
            "cuerpo_3d": cuerpo,
        }

    return None


def crear_roi_circular_200mm(mask_trabecular_2d, tac_2d, dx, dy):
    if not np.any(mask_trabecular_2d):
        return np.zeros_like(tac_2d, dtype=bool), {
            "roi_valida": False,
            "motivo": "No se detect regin trabecular.",
            "area_roi_real_mm2": 0.0,
        }

    centro_y, centro_x = center_of_mass(mask_trabecular_2d)

    radio_mm = math.sqrt(AREA_ROI_MM2 / math.pi)
    radio_px = radio_mm / np.mean([dx, dy])

    roi_circular = np.zeros_like(tac_2d, dtype=bool)

    rr, cc = draw.disk(
        (int(round(centro_y)), int(round(centro_x))),
        radius=radio_px,
        shape=tac_2d.shape
    )

    roi_circular[rr, cc] = True
    roi_final = roi_circular & mask_trabecular_2d

    area_pixel_mm2 = dx * dy
    area_roi_real_mm2 = float(np.sum(roi_final) * area_pixel_mm2)

    roi_valida = area_roi_real_mm2 >= AREA_ROI_MM2 * 0.70

    info = {
        "roi_valida": roi_valida,
        "motivo": "" if roi_valida else "La ROI circular queda parcialmente fuera del trabecular.",
        "area_roi_real_mm2": area_roi_real_mm2,
    }

    return roi_final, info


def estadisticas_hu(valores):
    if valores is None or len(valores) == 0:
        return {
            "hu_medio": None,
            "hu_mediano": None,
            "hu_std": None,
        }

    return {
        "hu_medio": float(np.mean(valores)),
        "hu_mediano": float(np.median(valores)),
        "hu_std": float(np.std(valores)),
    }


def r1(x):
    return round(float(x), 1) if x is not None else None


def r2(x):
    return round(float(x), 2) if x is not None else None


def convertir_a_json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convertir_a_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convertir_a_json_serializable(v) for v in obj]
    return obj


def guardar_resultados_csv(resultados, ruta_csv):
    existe = os.path.exists(ruta_csv)

    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(resultados.keys()),
            delimiter=";"
        )

        if not existe:
            writer.writeheader()

        writer.writerow(resultados)



def procesar_paciente(id_paciente, ruta_tac, ruta_mask):
    id_smart = convertir_id(id_paciente)
    print("\n==================================================")
    print(f" PROCESANDO PACIENTE: {id_paciente} -> {id_smart}")
    print("==================================================")

    carpeta_paciente = os.path.join(CARPETA_SALIDA, id_smart)
    os.makedirs(carpeta_paciente, exist_ok=True)

    tac_nifti = cargar_nifti(ruta_tac)
    mask_nifti = cargar_nifti(ruta_mask)

    datos_tac = tac_nifti.get_fdata()
    datos_mask = mask_nifti.get_fdata().astype(np.uint8)

    voxel = obtener_voxel_info(tac_nifti)

    dx = voxel["dx"]
    dy = voxel["dy"]
    area_pixel_cm2 = voxel["area_pixel_cm2"]
    volumen_voxel_cm3 = voxel["volumen_voxel_cm3"]

    nivel_oseo = seleccionar_nivel_valido(datos_mask, datos_tac, FALLBACK_OSEO)
    nivel_muscular = seleccionar_nivel_valido(datos_mask, datos_tac, FALLBACK_MUSCULAR)

    if nivel_oseo is None:
        raise RuntimeError("No se encontr ningn nivel seo vlido: L1, T12 o L2.")

    if nivel_muscular is None:
        raise RuntimeError("No se encontr ningn nivel muscular vlido: L3 o L4.")

    print(f" Nivel seo usado: {nivel_oseo['nombre']} | Fallback: {nivel_oseo['fallback_usado']}")
    print(f" Nivel muscular usado: {nivel_muscular['nombre']} | Fallback: {nivel_muscular['fallback_usado']}")


    z_oseo = nivel_oseo["z_mid"]
    clase_osea = nivel_oseo["clase"]

    tac_oseo_2d = datos_tac[:, :, z_oseo]
    mask_osea_2d = datos_mask[:, :, z_oseo] == clase_osea

    trabecular_2d = aislar_trabecular_2d(mask_osea_2d, tac_oseo_2d)
    roi_osea_2d, info_roi = crear_roi_circular_200mm(
        trabecular_2d,
        tac_oseo_2d,
        dx,
        dy
    )

    valores_hu_oseo_std = tac_oseo_2d[roi_osea_2d]
    stats_oseo_std = estadisticas_hu(valores_hu_oseo_std)

    trabecular_3d = aislar_trabecular_3d(nivel_oseo["mask_3d"], datos_tac)
    volumen_oseo_3d_cm3 = float(np.sum(trabecular_3d) * volumen_voxel_cm3)


    z_muscular = nivel_muscular["z_mid"]
    z_min_musc = nivel_muscular["z_min"]
    z_max_musc = nivel_muscular["z_max"]

    comentario_IA_extra = ""

    psoas_izq_2d = datos_mask[:, :, z_muscular] == 6
    psoas_der_2d = datos_mask[:, :, z_muscular] == 7

    area_psoas_mid = np.sum(psoas_izq_2d) + np.sum(psoas_der_2d)

    if area_psoas_mid == 0:
        mejor_z = None
        mejor_area = 0

        for z in range(z_min_musc, z_max_musc + 1):
            izq = datos_mask[:, :, z] == 6
            der = datos_mask[:, :, z] == 7
            area = np.sum(izq) + np.sum(der)

            if area > mejor_area:
                mejor_area = area
                mejor_z = z

        if mejor_z is not None and mejor_area > 0:
            z_muscular = mejor_z
            psoas_izq_2d = datos_mask[:, :, z_muscular] == 6
            psoas_der_2d = datos_mask[:, :, z_muscular] == 7
            comentario_IA_extra += "Fallback de corte muscular dentro del mismo nivel. "
        else:
            comentario_IA_extra += "No se detect psoas en el nivel muscular. "

    tac_musc_2d = datos_tac[:, :, z_muscular]

    area_izq_cm2 = float(np.sum(psoas_izq_2d) * area_pixel_cm2)
    area_der_cm2 = float(np.sum(psoas_der_2d) * area_pixel_cm2)

    hu_izq_2d = float(np.mean(tac_musc_2d[psoas_izq_2d])) if area_izq_cm2 > 0 else None
    hu_der_2d = float(np.mean(tac_musc_2d[psoas_der_2d])) if area_der_cm2 > 0 else None

    slab_psoas_izq = datos_mask[:, :, z_min_musc:z_max_musc + 1] == 6
    slab_psoas_der = datos_mask[:, :, z_min_musc:z_max_musc + 1] == 7

    vol_izq_cm3 = float(np.sum(slab_psoas_izq) * volumen_voxel_cm3)
    vol_der_cm3 = float(np.sum(slab_psoas_der) * volumen_voxel_cm3)


    procesamiento_OK = True
    comentario_IA = comentario_IA_extra if comentario_IA_extra is not None else ""

    if nivel_oseo["fallback_usado"]:
        comentario_IA += f"Fallback seo usado: {nivel_oseo['nombre']}. "

    if nivel_muscular["fallback_usado"]:
        comentario_IA += f"Fallback muscular usado: {nivel_muscular['nombre']}. "

    if not info_roi["roi_valida"]:
        procesamiento_OK = False
        comentario_IA += "ROI vertebral no vlida. "

    if stats_oseo_std["hu_medio"] is None:
        procesamiento_OK = False
        comentario_IA += "HU vertebral no calculable. "
    elif stats_oseo_std["hu_medio"] < 90:
        comentario_IA += "HU vertebral bajo (posible osteopenia). "

    if volumen_oseo_3d_cm3 is None or volumen_oseo_3d_cm3 <= 0:
        procesamiento_OK = False
        comentario_IA += "Volumen vertebral no calculable. "
    elif volumen_oseo_3d_cm3 < 8:
        procesamiento_OK = False
        comentario_IA += "Volumen vertebral muy bajo; segmentacin incompleta. "
    elif volumen_oseo_3d_cm3 < 15:
        comentario_IA += "Volumen vertebral bajo. "

    if area_der_cm2 is None or area_der_cm2 <= 0:
        procesamiento_OK = False
        comentario_IA += "rea psoas derecho nula o no detectada. "
    elif area_der_cm2 < 2:
        procesamiento_OK = False
        comentario_IA += "rea psoas derecho extremadamente baja. "
    elif area_der_cm2 < 4:
        comentario_IA += "rea psoas derecho baja; posible sarcopenia. "

    if area_izq_cm2 is None or area_izq_cm2 <= 0:
        procesamiento_OK = False
        comentario_IA += "rea psoas izquierdo nula o no detectada. "
    elif area_izq_cm2 < 2:
        procesamiento_OK = False
        comentario_IA += "rea psoas izquierdo extremadamente baja. "
    elif area_izq_cm2 < 4:
        comentario_IA += "rea psoas izquierdo baja; posible sarcopenia. "

    if hu_der_2d is None:
        procesamiento_OK = False
        comentario_IA += "HU psoas derecho no calculable. "
    elif hu_der_2d < 30:
        comentario_IA += "HU psoas derecho bajo. "

    if hu_izq_2d is None:
        procesamiento_OK = False
        comentario_IA += "HU psoas izquierdo no calculable. "
    elif hu_izq_2d < 30:
        comentario_IA += "HU psoas izquierdo bajo. "

    if vol_der_cm3 is None or vol_der_cm3 <= 0:
        procesamiento_OK = False
        comentario_IA += "Volumen psoas derecho no calculable. "
    elif vol_der_cm3 < 3:
        procesamiento_OK = False
        comentario_IA += "Volumen psoas derecho extremadamente bajo. "
    elif vol_der_cm3 < 10:
        comentario_IA += "Volumen psoas derecho bajo. "

    if vol_izq_cm3 is None or vol_izq_cm3 <= 0:
        procesamiento_OK = False
        comentario_IA += "Volumen psoas izquierdo no calculable. "
    elif vol_izq_cm3 < 3:
        procesamiento_OK = False
        comentario_IA += "Volumen psoas izquierdo extremadamente bajo. "
    elif vol_izq_cm3 < 10:
        comentario_IA += "Volumen psoas izquierdo bajo. "

    if area_der_cm2 is not None and vol_der_cm3 is not None and area_der_cm2 > 0:
        if vol_der_cm3 < area_der_cm2 * 1.5:
            procesamiento_OK = False
            comentario_IA += "Inconsistencia rea-volumen psoas derecho. "

    if area_izq_cm2 is not None and vol_izq_cm3 is not None and area_izq_cm2 > 0:
        if vol_izq_cm3 < area_izq_cm2 * 1.5:
            procesamiento_OK = False
            comentario_IA += "Inconsistencia rea-volumen psoas izquierdo. "

    if area_der_cm2 is not None and area_izq_cm2 is not None:
        mayor = max(area_der_cm2, area_izq_cm2)
        menor = min(area_der_cm2, area_izq_cm2)
        if menor > 0 and mayor / menor > 2.5:
            procesamiento_OK = False
            comentario_IA += "Asimetra extrema entre psoas. "

    comentario_IA = comentario_IA.strip()

    resultados_ia = {
        "ID": id_smart,
        "dataset_split": DATASET_SPLIT,

        "nivel_vertebra_IA": nivel_oseo["nombre"],
        "HU_vertebra_IA": r1(stats_oseo_std["hu_medio"]),
        "vol_vertebra_IA_cm3": r2(volumen_oseo_3d_cm3),

        "nivel_psoas_IA": nivel_muscular["nombre"],
        "area_psoas_der_IA_cm2": r2(area_der_cm2),
        "area_psoas_izq_IA_cm2": r2(area_izq_cm2),
        "HU_psoas_der_IA": r1(hu_der_2d),
        "HU_psoas_izq_IA": r1(hu_izq_2d),
        "vol_psoas_der_IA_cm3": r2(vol_der_cm3),
        "vol_psoas_izq_IA_cm3": r2(vol_izq_cm3),

        "procesamiento_OK": procesamiento_OK,
        "comentario_IA": comentario_IA.strip(),
    }

    ruta_csv = os.path.join(CARPETA_SALIDA, "02_IA_values_Carlos.csv")
    guardar_resultados_csv(resultados_ia, ruta_csv)


    resultados_debug = {
        "ID": id_smart,
        "archivo_tac": os.path.basename(ruta_tac),
        "archivo_mascara": os.path.basename(ruta_mask),
        "dataset_split": DATASET_SPLIT,

        "nivel_oseo_usado": nivel_oseo["nombre"],
        "fallback_oseo": nivel_oseo["fallback_usado"],
        "corte_oseo_mid": z_oseo,
        "roi_osea_area_mm2": r2(info_roi["area_roi_real_mm2"]),
        "roi_osea_valida": info_roi["roi_valida"],
        "HU_vertebra_IA": r1(stats_oseo_std["hu_medio"]),
        "HU_vertebra_mediana_debug": r1(stats_oseo_std["hu_mediano"]),
        "HU_vertebra_std_debug": r1(stats_oseo_std["hu_std"]),
        "vol_vertebra_IA_cm3": r2(volumen_oseo_3d_cm3),

        "nivel_muscular_usado": nivel_muscular["nombre"],
        "fallback_muscular": nivel_muscular["fallback_usado"],
        "corte_muscular_mid": z_muscular,
        "area_psoas_der_IA_cm2": r2(area_der_cm2),
        "area_psoas_izq_IA_cm2": r2(area_izq_cm2),
        "HU_psoas_der_IA": r1(hu_der_2d),
        "HU_psoas_izq_IA": r1(hu_izq_2d),
        "vol_psoas_der_IA_cm3": r2(vol_der_cm3),
        "vol_psoas_izq_IA_cm3": r2(vol_izq_cm3),
    }

    ruta_json = os.path.join(carpeta_paciente, f"{id_smart}_resultados_biomarcadores_debug.json")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(
            convertir_a_json_serializable(resultados_debug),
            f,
            indent=4,
            ensure_ascii=False
        )


    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle(f"IA-TAC SMART-LT: {id_smart}", fontsize=16, fontweight="bold")

    axs[0].imshow(np.rot90(tac_oseo_2d), cmap="gray")
    overlay_roi = np.ma.masked_where(~roi_osea_2d, roi_osea_2d)
    axs[0].imshow(np.rot90(overlay_roi), cmap="autumn", alpha=0.7)
    axs[0].set_title(
        f"Vertebral: {nivel_oseo['nombre']} mid-level\n"
        f"HU={resultados_ia['HU_vertebra_IA']} | Vol={resultados_ia['vol_vertebra_IA_cm3']} cm",
        fontsize=11
    )
    axs[0].axis("off")

    axs[1].imshow(np.rot90(tac_musc_2d), cmap="gray")

    mask_psoas_colores = np.zeros_like(tac_musc_2d)
    mask_psoas_colores[psoas_izq_2d] = 1
    mask_psoas_colores[psoas_der_2d] = 2

    cmap_psoas = mcolors.ListedColormap(["none", "lime", "cyan"])
    norm_psoas = mcolors.BoundaryNorm([0, 1, 2, 3], cmap_psoas.N)
    overlay_psoas = np.ma.masked_where(mask_psoas_colores == 0, mask_psoas_colores)

    axs[1].imshow(np.rot90(overlay_psoas), cmap=cmap_psoas, norm=norm_psoas, alpha=0.6)
    axs[1].set_title(
        f"Psoas: {nivel_muscular['nombre']} mid-level\n"
        f"Der: {resultados_ia['area_psoas_der_IA_cm2']} cm / {resultados_ia['HU_psoas_der_IA']} HU\n"
        f"Izq: {resultados_ia['area_psoas_izq_IA_cm2']} cm / {resultados_ia['HU_psoas_izq_IA']} HU",
        fontsize=11
    )
    axs[1].axis("off")

    coords_columna = np.argwhere((datos_mask >= 1) & (datos_mask <= 5))
    x_sagital = int(np.mean(coords_columna[:, 0])) if len(coords_columna) > 0 else datos_tac.shape[0] // 2

    axs[2].imshow(np.rot90(datos_tac[x_sagital, :, :]), cmap="gray")
    axs[2].axhline(
        y=datos_tac.shape[2] - z_oseo,
        color="red",
        linewidth=2,
        label=f"Mid {nivel_oseo['nombre']}"
    )
    axs[2].axhline(
        y=datos_tac.shape[2] - z_muscular,
        color="yellow",
        linewidth=2,
        label=f"Mid {nivel_muscular['nombre']}"
    )
    axs[2].axhspan(
        datos_tac.shape[2] - z_max_musc,
        datos_tac.shape[2] - z_min_musc,
        color="lime",
        alpha=0.3,
        label=f"Slab {nivel_muscular['nombre']}"
    )
    axs[2].set_title("Control anatmico", fontsize=11)
    axs[2].legend(loc="upper right")
    axs[2].axis("off")

    plt.tight_layout()

    ruta_standard_scores = os.path.join(carpeta_paciente, f"{id_smart}_standard_scores.png")
    plt.savefig(ruta_standard_scores, dpi=300, bbox_inches="tight")
    plt.close(fig)


    coords_columna = np.argwhere((datos_mask >= 1) & (datos_mask <= 7))
    y_coronal = int(np.mean(coords_columna[:, 1])) if len(coords_columna) > 0 else datos_tac.shape[1] // 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    colores = [
        "none",
        "purple",
        "red",
        "orange",
        "yellow",
        "cyan",
        "lime",
        "green",
    ]

    cmap_custom = mcolors.ListedColormap(colores)
    norm_custom = mcolors.BoundaryNorm(np.arange(9), cmap_custom.N)

    ax.imshow(np.rot90(datos_tac[:, y_coronal, :]), cmap="gray")

    overlay_coronal = np.ma.masked_where(
        datos_mask[:, y_coronal, :] == 0,
        datos_mask[:, y_coronal, :]
    )

    ax.imshow(
        np.rot90(overlay_coronal),
        cmap=cmap_custom,
        norm=norm_custom,
        alpha=0.65
    )

    ax.set_title(f"{id_smart} - Volume overlay", fontsize=13)
    ax.axis("off")

    ruta_volume_overlay = os.path.join(carpeta_paciente, f"{id_smart}_volume_overlay.png")
    plt.savefig(ruta_volume_overlay, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f" {id_smart} completado.")
    print(f"   CSV actualizado: {ruta_csv}")
    print(f"   JSON: {ruta_json}")
    print(f"   standard_scores: {ruta_standard_scores}")
    print(f"   volume_overlay: {ruta_volume_overlay}")



print("==================================================")
print(" INICIANDO EXTRACCIN DE VALORES IA EN LOTE")
print("==================================================")

ruta_csv_global = os.path.join(CARPETA_SALIDA, "02_IA_values_Carlos.csv")
if os.path.exists(ruta_csv_global):
    os.remove(ruta_csv_global)
    print(" CSV anterior eliminado para evitar duplicados.")

archivos_tac = [
    a for a in os.listdir(CARPETA_TAC)
    if a.endswith("_0000.nii.gz")
]

if len(archivos_tac) == 0:
    raise RuntimeError("No se encontraron TACs con formato *_0000.nii.gz.")

print(f"Pacientes encontrados: {len(archivos_tac)}")

for archivo_tac in archivos_tac:
    id_paciente = archivo_tac.replace("_0000.nii.gz", "")

    ruta_tac = os.path.join(CARPETA_TAC, archivo_tac)
    ruta_mask = os.path.join(CARPETA_MASK, f"{id_paciente}.nii.gz")

    if not os.path.exists(ruta_mask):
        print(f" No se encontr mscara para {id_paciente}. Se omite.")
        continue

    try:
        procesar_paciente(
            id_paciente=id_paciente,
            ruta_tac=ruta_tac,
            ruta_mask=ruta_mask
        )

    except Exception as e:
        print(f" Error procesando {id_paciente}:")
        print(e)

print("\n==================================================")
print(" PROCESO COMPLETADO")
print(f"Resultados guardados en:\n{CARPETA_SALIDA}")
print("==================================================")