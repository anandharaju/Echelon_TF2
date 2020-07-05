import fasttext
import fasttext.util
import pandas as pd
import numpy as np
from config import settings as cnst
import os


def embed_section_names():
    ft = fasttext.load_model(cnst.FASTTEXT_PATH)
    print("Fasttext loaded from:", cnst.FASTTEXT_PATH)

    sections = pd.read_csv(cnst.PKL_SOURCE_PATH + cnst.ESC + 'available_sections.csv', header=None)
    sections.fillna(value='', inplace=True)
    sections = list(sections.iloc[0])

    print("# of sections to embed:", len(sections))

    sect_names = []
    sect_emb = []

    if os.path.exists(cnst.PKL_SOURCE_PATH + cnst.ESC + 'section_embeddings.csv'):
        os.remove(cnst.PKL_SOURCE_PATH + cnst.ESC + 'section_embeddings.csv')

    for i, section in enumerate(sections):
        if section:
            try:
                sect_names.append(section)
                sect_emb.append(ft.get_word_vector(str(section))[0])
            except Exception as e:
                print(i, "Error Occurred while embedding PE section:", section, str(e))
        else:
            print("\t\tSkipped section name at index - " + str(i) + ", as it is empty")
    print("# of sections successfully embedded:", len(sect_emb))
    pd.DataFrame([sect_names, sect_emb]).to_csv(cnst.PKL_SOURCE_PATH + cnst.ESC + 'section_embeddings.csv', header=None, index=False)
