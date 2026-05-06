
import os
import shutil
import tempfile
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
import numpy as np


carpeta_pacientes = "C:/Users/theme/Downloads/CTS2/Nueva carpeta"
carpeta_entrada = "C:/IA_Test/Entrada_Lote2"

os.makedirs(carpeta_entrada, exist_ok=True)

settings.disable_validate_slice_increment()
settings.disable_validate_slicecount()
settings.disable_validate_orientation()
settings.disable_validate_orthogonal()

MIN_CORTES_VALIDOS = 100
MIN_COBERTURA_MM = 250



def es_dicom(ruta):
    try:
        pydicom.dcmread(ruta, stop_before_pixels=True)
        return True
    except Exception:
        return False


def buscar_dicoms(carpeta):
    archivos = []
    for raiz, _, nombres in os.walk(carpeta):
        for nombre in nombres:
            ruta = os.path.join(raiz, nombre)
            if es_dicom(ruta):
                archivos.append(ruta)
    return archivos


def leer_header(ruta):
    return pydicom.dcmread(ruta, stop_before_pixels=True)


def normal_de_orientacion(ds):
    if not hasattr(ds, "ImageOrientationPatient"):
        return None

    ori = np.array([float(x) for x in ds.ImageOrientationPatient])
    row = ori[:3]
    col = ori[3:]
    normal = np.cross(row, col)

    norma = np.linalg.norm(normal)
    if norma == 0:
        return None

    return normal / norma


def posicion_corte(ds, normal):
    if not hasattr(ds, "ImagePositionPatient") or normal is None:
        return None

    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    return float(np.dot(ipp, normal))


def clave_geometria(ds):
    orientation = tuple(np.round([float(x) for x in ds.ImageOrientationPatient], 4)) \
        if hasattr(ds, "ImageOrientationPatient") else None

    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))

    pixel_spacing = tuple(np.round([float(x) for x in ds.PixelSpacing], 4)) \
        if hasattr(ds, "PixelSpacing") else None

    slice_thickness = round(float(getattr(ds, "SliceThickness", 0)), 4) \
        if hasattr(ds, "SliceThickness") else None

    return (rows, cols, pixel_spacing, slice_thickness, orientation)


def agrupar_series_y_geometria(archivos_dicom):
    grupos = {}

    for ruta in archivos_dicom:
        try:
            ds = leer_header(ruta)

            if getattr(ds, "Modality", "") != "CT":
                continue

            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if series_uid is None:
                continue

            clave = (series_uid, clave_geometria(ds))

            if clave not in grupos:
                grupos[clave] = {
                    "uid": series_uid,
                    "descripcion": str(getattr(ds, "SeriesDescription", "")),
                    "image_type": str(getattr(ds, "ImageType", "")),
                    "archivos": [],
                }

            grupos[clave]["archivos"].append(ruta)

        except Exception:
            continue

    return list(grupos.values())


def calcular_info_candidato(candidato):
    posiciones = []
    normales = []

    for ruta in candidato["archivos"]:
        try:
            ds = leer_header(ruta)
            normal = normal_de_orientacion(ds)
            pos = posicion_corte(ds, normal)

            if pos is not None:
                posiciones.append(pos)

            if normal is not None:
                normales.append(normal)

        except Exception:
            continue

    n_cortes = len(candidato["archivos"])

    if len(posiciones) >= 2:
        cobertura_mm = float(max(posiciones) - min(posiciones))
    else:
        cobertura_mm = 0.0

    if len(normales) > 0:
        normal_media = np.mean(normales, axis=0)
        axial_score = abs(normal_media[2])
    else:
        axial_score = 0.0

    candidato["n_cortes"] = n_cortes
    candidato["cobertura_mm"] = cobertura_mm
    candidato["axial_score"] = axial_score

    return candidato


def seleccionar_mejor_serie(candidatos):
    if not candidatos:
        return None

    candidatos = [calcular_info_candidato(c) for c in candidatos]

    print("\nCandidatos detectados:")
    for c in candidatos:
        print(
            f" - Cortes: {c['n_cortes']:>4} | "
            f"Cobertura: {c['cobertura_mm']:>7.1f} mm | "
            f"Axial: {c['axial_score']:.2f} | "
            f"Desc: {c['descripcion']}"
        )

    candidatos_validos = [
        c for c in candidatos
        if c["n_cortes"] >= MIN_CORTES_VALIDOS
        and c["cobertura_mm"] >= MIN_COBERTURA_MM
        and c["axial_score"] > 0.7
    ]

    if not candidatos_validos:
        print("\n No hay ninguna serie que cumpla criterios mnimos.")
        print("   Se seleccionar la mejor disponible, pero puede estar incompleta.")
        candidatos_validos = candidatos

    mejor = max(
        candidatos_validos,
        key=lambda c: (c["cobertura_mm"], c["n_cortes"], c["axial_score"])
    )

    return mejor


def copiar_serie_a_temporal(archivos, carpeta_temp):
    os.makedirs(carpeta_temp, exist_ok=True)

    for i, ruta in enumerate(archivos):
        destino = os.path.join(carpeta_temp, f"slice_{i:05d}.dcm")
        shutil.copy2(ruta, destino)



print("==================================================")
print(" INICIANDO CONVERSIN EN LOTE: DICOM -> NIfTI")
print("==================================================")

pacientes = [
    p for p in os.listdir(carpeta_pacientes)
    if os.path.isdir(os.path.join(carpeta_pacientes, p))
]

print(f"Pacientes encontrados: {len(pacientes)}")

for id_paciente in pacientes:
    carpeta_dicom_paciente = os.path.join(carpeta_pacientes, id_paciente)
    archivo_salida = os.path.join(carpeta_entrada, f"{id_paciente}_0000.nii.gz")

    print("\n==================================================")
    print(f" PROCESANDO PACIENTE: {id_paciente}")
    print("==================================================")

    if os.path.exists(archivo_salida):
        os.remove(archivo_salida)
        print(" NIfTI anterior eliminado.")

    try:
        archivos_dicom = buscar_dicoms(carpeta_dicom_paciente)

        if len(archivos_dicom) == 0:
            print(" No se encontraron archivos DICOM.")
            continue

        candidatos = agrupar_series_y_geometria(archivos_dicom)

        if len(candidatos) == 0:
            print(" No se encontraron series CT vlidas.")
            continue

        mejor_serie = seleccionar_mejor_serie(candidatos)

        if mejor_serie is None:
            print(" No se pudo seleccionar ninguna serie.")
            continue

        print("\n Serie seleccionada:")
        print(f"UID: {mejor_serie['uid']}")
        print(f"Descripcin: {mejor_serie['descripcion']}")
        print(f"N de cortes: {mejor_serie['n_cortes']}")
        print(f"Cobertura anatmica: {mejor_serie['cobertura_mm']:.1f} mm")
        print(f"Axial score: {mejor_serie['axial_score']:.2f}")

        if mejor_serie["n_cortes"] < MIN_CORTES_VALIDOS or mejor_serie["cobertura_mm"] < MIN_COBERTURA_MM:
            print(" AVISO: serie probablemente incompleta. Se convierte, pero debe revisarse.")

        with tempfile.TemporaryDirectory() as carpeta_temp:
            copiar_serie_a_temporal(mejor_serie["archivos"], carpeta_temp)

            dicom2nifti.dicom_series_to_nifti(
                carpeta_temp,
                archivo_salida,
                reorient_nifti=True
            )

        print(f" Conversin completada: {archivo_salida}")

    except Exception as e:
        print(f" Error al convertir el paciente {id_paciente}:")
        print(e)


print("\n==================================================")
print(" PROCESO COMPLETADO")
print(f"Archivos NIfTI guardados en:\n{carpeta_entrada}")
print("==================================================")