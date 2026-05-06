#preparar_split_nnunet.py

import os
import shutil
import csv
import random
import glob

# Fijamos la "semilla" para que si lo vuelves a ejecutar, el azar sea el mismo 
# y no mezcles a los pacientes de testeo con los de entrenamiento por error.
random.seed(42)

def preparar_split_nnunet(dir_origen, dir_destino, archivo_csv):
    print("[*] Leyendo el registro clínico...")
    
    # 1. Leer los pacientes positivos del Excel
    pacientes_positivos = []
    with open(archivo_csv, mode='r') as f:
        lector = csv.DictReader(f, delimiter=';')
        for fila in lector:
            pacientes_positivos.append(fila["ID_Paciente"])
            
    # 2. Identificar TODOS los pacientes en la carpeta origen
    todos_los_tacs = glob.glob(os.path.join(dir_origen, "imagesTr", "*_0000.nii.gz"))
    todos_los_pacientes = [os.path.basename(t).replace("_0000.nii.gz", "") for t in todos_los_tacs]
    
    # 3. Separar los negativos (los que no están en el Excel)
    pacientes_negativos = [p for p in todos_los_pacientes if p not in pacientes_positivos]
    
    print(f"  -> Total pacientes extraídos: {len(todos_los_pacientes)}")
    print(f"  -> Pacientes CON abdomen: {len(pacientes_positivos)}")
    print(f"  -> Pacientes SIN abdomen: {len(pacientes_negativos)}")
    
    # 4. Mezclar aleatoriamente
    random.shuffle(pacientes_positivos)
    random.shuffle(pacientes_negativos)
    
    # Calcular 80% para entrenar, 20% para la caja fuerte
    num_pos_train = int(len(pacientes_positivos) * 0.8)
    train_pos = pacientes_positivos[:num_pos_train]
    test_pos = pacientes_positivos[num_pos_train:]
    
    # De los negativos, cogemos solo 50 para entrenar (para no sobrecargar) y 10 para probar
    num_neg_train = min(50, len(pacientes_negativos))
    num_neg_test = min(10, max(0, len(pacientes_negativos) - num_neg_train))
    
    train_neg = pacientes_negativos[:num_neg_train]
    test_neg = pacientes_negativos[num_neg_train:num_neg_train+num_neg_test]
    
    lista_train = train_pos + train_neg
    lista_test = test_pos + test_neg
    
    print(f"\n[*] Seleccionados para ENTRENAMIENTO: {len(lista_train)} archivos")
    print(f"[*] Seleccionados para LA CAJA FUERTE (Test): {len(lista_test)} archivos")
    
    # 5. Crear la estructura de carpetas definitiva
    dir_train_img = os.path.join(dir_destino, "imagesTr")
    dir_train_lbl = os.path.join(dir_destino, "labelsTr")
    dir_test_img = os.path.join(dir_destino, "imagesTs")
    dir_test_lbl = os.path.join(dir_destino, "labelsTs")
    
    for carpeta in [dir_train_img, dir_train_lbl, dir_test_img, dir_test_lbl]:
        os.makedirs(carpeta, exist_ok=True)
        
    # 6. Función para copiar de forma segura
    def copiar_archivos(lista_ids, carpeta_img_dest, carpeta_lbl_dest):
        for pid in lista_ids:
            # Copiar TAC
            origen_img = os.path.join(dir_origen, "imagesTr", f"{pid}_0000.nii.gz")
            destino_img = os.path.join(carpeta_img_dest, f"{pid}_0000.nii.gz")
            if os.path.exists(origen_img) and not os.path.exists(destino_img):
                shutil.copy(origen_img, destino_img)
            
            # Copiar Máscara
            origen_lbl = os.path.join(dir_origen, "labelsTr", f"{pid}.nii.gz")
            destino_lbl = os.path.join(carpeta_lbl_dest, f"{pid}.nii.gz")
            if os.path.exists(origen_lbl) and not os.path.exists(destino_lbl):
                shutil.copy(origen_lbl, destino_lbl)

    # 7. Ejecutar copias (esto puede tardar unos minutos)
    print("\n[+] Construyendo carpeta de Entrenamiento (Train)...")
    copiar_archivos(lista_train, dir_train_img, dir_train_lbl)
    
    print("[+] Construyendo carpeta de Prueba (Test)...")
    copiar_archivos(lista_test, dir_test_img, dir_test_lbl)
    
    print(f"\n[+] ¡Éxito absoluto! Tu base de datos oficial está en: {dir_destino}")

if __name__ == "__main__":
    # La carpeta antigua donde están mezclados
    dir_origen = "C:/Users/theme/Downloads/Proyecto_IA_Medica/Mi_Dataset_3D"
    
    # El Excel que generamos antes
    archivo_csv = "C:/Users/theme/Downloads/Proyecto_IA_Medica/resultados_clinicos.csv"
    
    # LA NUEVA CARPETA DEFINITIVA (Formato nnU-Net v2)
    dir_destino = "C:/Users/theme/Downloads/Proyecto_IA_Medica/Dataset001_PsoasL1"
    
    preparar_split_nnunet(dir_origen, dir_destino, archivo_csv)