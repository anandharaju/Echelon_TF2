import time
import config.constants as cnst


class EchelonMeta:
    @classmethod
    def project_details(cls):

        print(
            "######################################################################################################################################")
        print(
            "######################################################################################################################################")
        print(
            "######################################################################################################################################")
        print(
            "#####                                                                                                                            #####")
        print(
            "#####                                                                                                                            #####")
        print(
            "#####         ########       #######      ###     ###        #######      ###             #########       #####       ###        #####")
        print(
            "#####        #####          #####         ###     ###      #####          ###            ####   ####      ######      ###        #####")
        print(
            "#####        ####           ####          ###     ###      ####           ###            ###     ###      ### ###     ###        #####")
        print(
            "#####        ####           ####          ###     ###      ####           ###            ###     ###      ###  ###    ###        #####")
        print(
            "#####        ########       ####          ###########      ########       ###            ###     ###      ###   ###   ###        #####")
        print(
            "#####        ####           ####          ###     ###      ####           ###            ###     ###      ###    ###  ###        #####")
        print(
            "#####        ####           ####          ###     ###      ####           ###            ###     ###      ###     ### ###        #####")
        print(
            "#####        #####          #####         ###     ###      #####          #########      ####   ####      ###      ######        #####")
        print(
            "#####         ########       #######      ###     ###        #######      #########       #########       ###       #####        #####")
        print(
            "#####                                                                                                                            #####")
        print(
            "#####                                                                                                                            #####")
        print(
            "######################################################################################################################################")
        print(
            "######################################################################################################################################")
        print(
            "######################################################################################################################################")

        time.sleep(5)


    @classmethod
    def run_setup(cls):
        print("\n\n######################################################################################################################################")
        print("\t\t\t\t\t\t\t RUN SETUP")
        print("######################################################################################################################################\n")
        print("Project Base                              :", cnst.PROJECT_BASE_PATH)
        print("Raw & Pickle Data source                  :", cnst.DATA_SOURCE_PATH)
        print("CSV path for the dataset                  :", cnst.ALL_FILE)
        print("GPU Enabled                               :", cnst.USE_GPU)
        print("Folds                                     :", cnst.CV_FOLDS)

        print("Batch Size [Train T1 : Train T2 : Predict]:", "[ "+str(cnst.T1_TRAIN_BATCH_SIZE)+" : "+str(cnst.T2_TRAIN_BATCH_SIZE)+" : "+str(cnst.PREDICT_BATCH_SIZE)+" ]")
        print("Epochs [Tier1 : Tier2]                    :", "[ "+str(cnst.TIER1_EPOCHS)+" : "+str(cnst.TIER2_EPOCHS)+" ]")
        print("\n")
        print("Model used for Tier-1 Training            :", "Pre-trained Malconv - No exposure to Echelon training data" if cnst.USE_PRETRAINED_FOR_TIER1 else "Pre-trained Malconv exposed to Echelon training data")
        print("Model used for Tier-2 Training            :", "Pre-trained Malconv - No exposure to Echelon section-wise training data" if cnst.USE_PRETRAINED_FOR_TIER2 else "Pre-trained Malconv exposed to Echelon section-wise training data")
        print("Target FPR [ Overall : Tier1 : Tier2 ]    :", "[ "+str(cnst.OVERALL_TARGET_FPR)+" : "+str(cnst.TIER1_TARGET_FPR)+" : "+str(cnst.TIER2_TARGET_FPR)+" ]")
        print("Skip Tier-1 Training                      :", cnst.SKIP_TIER1_TRAINING)
        print("Skip Tier-2 Training                      :", cnst.SKIP_TIER2_TRAINING)
        print("Skip ATI Processing                       :", cnst.SKIP_ATI_PROCESSING)
        print("Boost benign files with a lower bound     :", cnst.PERFORM_B2_BOOSTING)
        print("Percentiles for Q_Criteria selection      :", cnst.PERCENTILES)

        print("\n")
        print("Maximum file size set for the model input :", cnst.MAX_FILE_SIZE_LIMIT, "bytes")
        print("Maximum size set for Section Byte Map     :", cnst.MAX_SECT_BYTE_MAP_SIZE)
        print("Layer Number to stunt plugin model for ATI:", cnst.LAYER_NUM_TO_STUNT)
        print("CNN Window size | Stride size             :", str(cnst.CONV_WINDOW_SIZE) + "|" + str(cnst.CONV_STRIDE_SIZE))


if __name__ == '__main__':
    EchelonMeta.project_details()

