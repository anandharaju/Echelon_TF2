import matplotlib.pyplot as plt
import numpy as np

dpi = 300
figsize = (4, 2.7)

fpr = "C:\\Users\\anand\\Dropbox\\Anand\\Research\\Malware_Analysis\\Echelon\\AAAI\\images\\fprds1.png"
fnr = "C:\\Users\\anand\\Dropbox\\Anand\\Research\\Malware_Analysis\\Echelon\\AAAI\\images\\fnrds1.png"

# DS1
xticks = 50 ,55 ,60,65,70,75,80 ,85,90,91,92,93,94,95,96 ,97,98,99

fpr1 = [0.18,0.16,0.16,0.14,0.13,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.096,0.078,0.078]
fpr2 = [0.18,0.16,0.16,0.14 ,0.13,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.096,0.078,0.078]

fnr1 = [0.93,0.99,1.06,1.18,1.32,1.39,1.57,1.73,2.24,2.37,2.47,2.74,2.95,3.27,3.78,4.56,5.68,8.23]
fnr2 = [0.53,0.54,0.58,0.63,0.70,0.70,0.76,0.81,0.95,0.95,1.06,1.00,1.22,1.13,1.25,1.41,1.55,1.89]

plt.figure(num=None, figsize=figsize, dpi=dpi)
plt.ylabel('FP Rate')
plt.xlabel('Threshold (THD) %')
plt.plot(fpr1, 'black')
#plt.plot(fpr2, 'black', linestyle='--', linewidth=3)
plt.plot(fpr2, 'black', marker='x', markersize=3.5)
x1, y1 = [16, 16], [0.07, 0.117]
plt.plot(x1, y1, 'black', linestyle=':')
plt.legend(['Malconv (Base TIER-1)', 'Echelon (TIER-2)'], fontsize=9)
plt.text(16.2, 0.113, "TIER-1 THD", fontsize=7, rotation=90)
plt.xticks(np.arange(0, len(xticks)+1, 1), xticks, fontsize=7)
plt.yticks(fontsize=7)
#plt.savefig(fpr, bbox_inches='tight')
#plt.show()

plt.figure(num=None, figsize=figsize, dpi=dpi)
plt.ylabel('FN Rate')
plt.xlabel('Threshold (THD) %')
plt.plot(fnr1, 'black')
plt.plot(fnr2, 'black', linestyle='--')
x1, y1 = [16, 16], [0.2, 7]
plt.plot(x1, y1, 'black', linestyle=':')
plt.legend(['Malconv (Base TIER-1)', 'Echelon (TIER-2)'], fontsize=9)
plt.text(16.2, 4.5, "TIER-1 THD", fontsize=7, rotation=90)
plt.xticks(np.arange(0, len(xticks)+1, 1), xticks, fontsize=7)
plt.yticks(fontsize=7)
#plt.savefig(fnr, bbox_inches='tight')
plt.show()

fpr = "C:\\Users\\anand\\Dropbox\\Anand\\Research\\Malware_Analysis\\Echelon\\AAAI\\images\\fprds2.png"
fnr = "C:\\Users\\anand\\Dropbox\\Anand\\Research\\Malware_Analysis\\Echelon\\AAAI\\images\\fnrds2.png"

# DS2
fpr1 = [0.42,  0.3, 0.28, 0.28, 0.27, 0.15, 0.15, 0.15, 0.04]
fpr2 = [0.33, 0.25, 0.25, 0.19, 0.17, 0.10, 0.10, 0.07, 0.04]
fnr1 = [0.51, 0.54, 0.56, 0.58, 0.63, 0.66, 0.71,    1, 1.42]
fnr2 = [0.23, 0.27, 0.31, 0.35, 0.35, 0.37, 0.40, 0.40, 0.41]

plt.figure(num=None, figsize=figsize, dpi=dpi)
plt.ylabel('FP Rate')
plt.xlabel('Threshold (THD) %')
plt.plot(fpr1, 'black')
plt.plot(fpr2, 'black', linestyle='--')
x1, y1 = [8, 8], [0.02, 0.15]
plt.plot(x1, y1, 'black', linestyle=':')
plt.legend(['Malconv (Base TIER-1)', 'Echelon (TIER-2)'], fontsize=9)
plt.text(8.1, 0.12, "TIER-1 THD", fontsize=7, rotation=90)
plt.xticks(np.arange(0, 10, 1), [91, 92, 93, 94, 95, 96, 97, 98, 99])
plt.savefig(fpr, bbox_inches='tight')
plt.show()

plt.figure(num=None, figsize=figsize, dpi=dpi)
plt.ylabel('FN Rate')
plt.xlabel('Threshold (THD) %')
plt.plot(fnr1, 'black')
plt.plot(fnr2, 'black', linestyle='--')
x1, y1 = [8, 8], [0.3, 1.45]
plt.plot(x1, y1, 'black', linestyle=':')
plt.legend(['Malconv (Base TIER-1)', 'Echelon (TIER-2)'], fontsize=9)
plt.text(8.1, 0.6, "TIER-1 THD", fontsize=7, rotation=90)
plt.xticks(np.arange(0, 10, 1), [91, 92, 93, 4, 95, 96, 97, 98, 99])
plt.savefig(fnr, bbox_inches='tight')
plt.show()
