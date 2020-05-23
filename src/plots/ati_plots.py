import matplotlib.pyplot as plt
from config import constants as cnst


def feature_map_histogram(feature_map, prediction):
    feature_map.shape = (20, 20)
    feature_map = feature_map // 1
    ax = plt.subplot(5, 4, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_map, cmap='gray')
    print(feature_map, prediction.shape)