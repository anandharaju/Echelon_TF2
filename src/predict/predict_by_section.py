import time
from utils import utils
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.models import load_model
from config.echelon_meta import EchelonMeta
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ECHELON TIER 1 Neural Network')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--limit', type=float, default=0.)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--model_names', type=str, default=['echelon_section']) #'echelon.text', 'echelon.rdata', 'echelon.rsrc', 'echelon.data', 'echelon.pdata', 'echelon.header'])
parser.add_argument('--model_ext', type=str, default='.h5')
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--confidence_levels', type=int, default=[99]) #[40, 50, 60, 70]) # Percentage
parser.add_argument('--target_confidence', type=int, default=70)
parser.add_argument('--csv', type=str, default=None)


'''
def predict_by_section(wpartition, spartition, sections, model, fn_list, label, batch_size, verbose):
    max_len = 0
    if len(sections) == 1:
        max_len = model.input.shape[1]
    else:
        max_len = model.input[0].shape[1]
    pred = model.predict_generator(
        utils.data_generator_by_section(wpartition, spartition, sections, fn_list, label, max_len, batch_size, shuffle=False),
        steps=len(fn_list) // batch_size + 1,
        verbose=verbose
    )
    return pred'''


def trigger_predict_by_section():
    metaObj = EchelonMeta()
    metaObj.project_details()
    args = parser.parse_args()

    st = time.time()
    all_sections_pred = []
    # read data
    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    Y = df[1]
    label = np.zeros((fn_list.shape))

    for model_name in args.model_names:
        sections = ['.header', '.rsrc', '.data']#['.rsrc', '.text', '.rdata']  #model_name.split('.')[1]
        #sections = ['.header', '.debug', '.idata']
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [PROCESSING SECTION -> ' + str(sections)
              + '] [MODEL NAME -> ' + model_name + args.model_ext + ']')

        # load model
        model = load_model(args.model_path + model_name + args.model_ext)
        #model.summary()

        pred = predict_by_section(sections, model, fn_list, label, args.batch_size, args.verbose)

        for confidence in args.confidence_levels:
            df['predicted score - rounded ' + str(confidence) + '% confidence'] = pred // (confidence/100)
            acc = metrics.accuracy_score(Y, pred > (confidence/100))
            bacc = metrics.balanced_accuracy_score(Y, pred > (confidence/100))
            cm = metrics.confusion_matrix(Y, pred > (confidence/100))
            #print('Confusion Matrix:\t\t[Confidence: ' + str(confidence) + '%] [Acc: ' + str(acc) + "] [Balanced Acc: " + str(bacc) + ']\n   tn   fp   fn   tp')
            #print("%5s%5s%5s%5s" % (str(cm[0][0]), str(cm[0][1]), str(cm[1][0]), str(cm[1][1])))
            #print("Checking results",acc,bacc,cm)
            print("%3s" % str(confidence), " & ", str(acc)[:6], " & %5s & %5s & %5s & %5s \\\\\\hline" % (str(cm[0][0]), str(cm[0][1]), str(cm[1][0]), str(cm[1][1]))   )#, " \\\\\n \\hline")
            #roc = metrics.roc_curve(Y, pred > (confidence/100))
            # print("ROC Curve         : ", roc)
            #print("\n")

        #auc = metrics.roc_auc_score(Y, pred)
        #print("Overall ROC AUC Score     : ", auc)
        fpr, tpr, thds = metrics.roc_curve(Y, pred)
        #auc = metrics.roc_auc_score(Y, pred)
        #print("Overall ROC AUC Score     : ", auc)  # , fpr, tpr)

        #Malconv roc auc
        #maldf = pd.read_csv('aucmalconv.csv', header=None)
        #malfpr = maldf[0]
        #maltpr = maldf[1]
        #plt.plot(malfpr, maltpr, label="auc=0.9983861824925196")

        '''plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend(loc=4)
        plt.show()
        aucdf = pd.DataFrame()
        aucdf['fpr'] = fpr
        aucdf['tpr'] = tpr
        aucdf['thds'] = thds
        aucdf.to_csv('aucechelon.csv', header=None, index=False)'''


        df['predict score'] = pred
        df[0] = [i.split('/')[-1] for i in fn_list]  # os.path.basename
        df.to_csv(args.result_path + model_name + ".result.csv", header=None, index=False)
        print('Results writen in', args.result_path + model_name + ".result.csv")

        malware_list = []
        combined_pred = []
        all_sections_pred.append(pred // (args.target_confidence/100))
    for i in range(0, len(fn_list)):
        if (all_sections_pred[0][i]) >= 1:# + all_sections_pred[1][i] + all_sections_pred[2][i]) >= 3:
            combined_pred.append(1)
            malware_list.append(fn_list[i])
        else:
            combined_pred.append(0)
    print("\n\nECHELON TIER-2 RESULTS:\n-----------------------\nNumber of files processed: ", len(fn_list))

    '''
    acc = metrics.accuracy_score(Y, combined_pred)
    bacc = metrics.balanced_accuracy_score(Y, combined_pred)
    cm = metrics.confusion_matrix(Y, combined_pred)
    print('Confusion Matrix:\t\t[Confidence: ' + str(args.target_confidence) + '%] [Acc: ' + str(acc) + "] [Balanced Acc: "
          + str(bacc) + ']\n   tn   fp   fn   tp')

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # roc = metrics.roc_curve(Y, pred > (confidence / 100))
    print("%5s%5s%5s%5s" % (str(tn), str(fp), str(fn), str(tp)), "FPR: ", fp / (fp + tn), "FNR:", fn / (fn + tp))'''
    print("\n TOTAL TIME ELAPSED FOR SECTION-WISE PREDICTION - ", str(int(time.time() - st) / 60), " minutes\n")

    reconcile(True)


def reconcile(flag):
    print("\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   RECONCILING DATA  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n\n")
    t1 = pd.read_csv('echelon.result.csv', header=None)
    y1 = t1[1]
    p1 = t1[2]
    pv1 = t1[3]

    t2 = pd.read_csv('echelon_section.result.csv', header=None)
    t2.to_csv("echelon_reconciled.csv", header=None, index=False, mode='w')
    mfp = pd.read_csv('malware_fp.csv', header=None)
    mfp.to_csv("echelon_reconciled.csv", header=None, index=False, mode='a')

    es = pd.read_csv('echelon_reconciled.csv', header=None)
    y2 = es[1]
    p2 = es[2]
    pv2 = es[3]

    # Malconv roc auc
    fpr, tpr, thds = metrics.roc_curve(y1, pv1)
    auc = metrics.roc_auc_score(y1, pv1)
    plt.plot(fpr, tpr, 'black', label="Malconv AUC = " + str(auc)[:9])

    # Reconcile roc auc
    fpr, tpr, thds = metrics.roc_curve(y2, pv2)
    auc2 = metrics.roc_auc_score(y2, pv2)
    print("Reconciled ROC AUC Score     : ", auc2, fpr, tpr)
    plt.plot(fpr, tpr, 'black', linestyle='--', linewidth=2, label="Echelon AUC = " + str(auc2)[:9])

    plt.plot([0, 1], [0, 1], 'black', linestyle=':')
    plt.legend(loc=4)
    plt.xlabel("FPR", fontsize=12)
    plt.ylabel("TPR", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.text(0.8, 0.13, '$\it{Standard View}$', fontsize=10)

    aucdf = pd.DataFrame()
    aucdf['fpr'] = fpr
    aucdf['tpr'] = tpr
    aucdf['thds'] = thds
    aucdf.to_csv('aucechelon.csv', header=None, index=False)

    if flag:
        plt.show()

    acc = metrics.accuracy_score(y2, p2)
    bacc = metrics.balanced_accuracy_score(y2, p2)
    cm = metrics.confusion_matrix(y2, p2)
    print('Confusion Matrix:\t\t[Confidence: ] [Acc: ' + str(acc) + "] [Balanced Acc: " + str(bacc) + ']\n   tn   fp   fn   tp')
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    # roc = metrics.roc_curve(Y, pred > (confidence / 100))
    print("%5s %5s %5s %5s" % (str(tn), str(fp), str(fn), str(tp)), "FPR: ", fp / (fp + tn), "FNR:", fn / (fn + tp))


if __name__ == '__main__':
    trigger_predict_by_section()
    show_auc_plot = True
    reconcile(show_auc_plot)

    #a = [1,0.11387,0.04352,0.01817,0.01451,0.01434,0.01080,0.00854,0.00606,0.00546,0.00476,0.00433,0.00217,0.00217,0.00173,0.00173,0.00173,0.00173,0.00144,0.00115,0.00086,0.00081,0.00075,0.00057,0.00043,0.00043,0.00043,0.00043,0.00028,0.00028,0.00028,0.00025,0.00018,0.00014,0.00010,0.00008,0.00007,0.00006,0.00006,0.00002,0.00002,0.00002,0,0,0,0,0,0,0]
    #plt.plot(np.shape(a*100))
    #plt.show()

