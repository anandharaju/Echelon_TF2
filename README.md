# Echelon
A multi tiered neural network framework for augmenting malware detection performance by locking false positive rate and maximizing true positive rate.

# Environment Details: [PYTHON 3.5.6]
pefile
Keras 2.2.4
tensorflow-gpu 1.12.0
scikit-learn 0.21.2
seaborn 0.9.1
pandas
numpy 1.16.3

numba
nvidia-ml-py3

# Issues In-progress:
Activation Trend Identification:
* Section is found - but data is empty (Section_size_byte = 0) and end_offset = -1
* Offset overlapping across sections - Impact is minimal. Impact is further reduced by using smaller conv window.

# Challenges:
* Orphan samples - due to the problem of Coverage: Does the highly qualified sections cover all samples in training and testing data?
i.e., the number of samples intersected by the selected sections should be equal to 'U'
* Sections with size smaller than convolution window size [500 bytes] - Reset window size to a size less than the minimum section size.
[Caveat: Blows up memory requirement]

# Untracked Changes
* Bi-directional activation trend identification

# Binary Parsing Module:
http://localhost:8888/notebooks/PE_Section_Data_Extractor.ipynb#
