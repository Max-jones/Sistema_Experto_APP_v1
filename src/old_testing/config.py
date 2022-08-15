# config.py

import os

MAIN_FOLDER = os.path.dirname(os.path.abspath(__file__))[:-4] #without notebooks
print(f'reading files from: {MAIN_FOLDER}')
TRAINING_FILE = MAIN_FOLDER+"/data/Horcon-etiquetado_con_1_etiqueta.csv"
# local_path_WINDOWS = 'C:\\Users\\elmha\\OneDrive - Universidad de Chile\\GitHub\\Sistema_Experto_APP\\data\Horcon-etiquetado_con_1_etiqueta.csv'

