import pandas as pd
import numpy as np


class RandomizedDuplication:

    '''
            Randomized Duplication - For Class Imbalance
    '''
    def balance_dataset(self, mdf, bdf):
        m_input_length = len(mdf[0])
        b_input_length = len(bdf[0])

        ratio = 0

        if m_input_length < b_input_length:
            ratio = int(b_input_length / m_input_length)
        elif m_input_length > b_input_length:
            ratio = int(m_input_length / b_input_length)
        else:
            return mdf, bdf

        while ratio != 1:
            duplicates = None
            if m_input_length < b_input_length:
                # Malware duplicates
                duplicates = mdf.sample(n=m_input_length, replace=False)
                mdf = mdf.append(duplicates)
            elif m_input_length > b_input_length:
                # Benign duplicates
                duplicates = bdf.sample(n=b_input_length, replace=False)
                bdf = bdf.append(duplicates)
            ratio -= 1

        m_input_length = len(mdf[0])
        b_input_length = len(bdf[0])
        difference = np.abs(b_input_length - m_input_length)
        if m_input_length < b_input_length:
            # Malware duplicates
            duplicates = mdf.sample(n=difference, replace=False)
            mdf = mdf.append(duplicates)
        elif m_input_length > b_input_length:
            # Benign duplicates
            duplicates = bdf.sample(n=difference, replace=False)
            bdf = bdf.append(duplicates)

        return mdf.reset_index(drop=True), bdf.reset_index(drop=True)
