import logging
import time
import config.settings as cnst


class EchelonMeta:
    @classmethod
    def project_details(cls):

        logging.info("######################################################################################################################################")
        logging.info("######################################################################################################################################")
        logging.info("######################################################################################################################################")
        logging.info("#####                                                                                                                            #####")
        logging.info("#####                                                                                                                            #####")
        logging.info("#####         ########       #######      ###     ###        #######      ###             #########       #####       ###        #####")
        logging.info("#####        #####          #####         ###     ###      #####          ###            ####   ####      ######      ###        #####")
        logging.info("#####        ####           ####          ###     ###      ####           ###            ###     ###      ### ###     ###        #####")
        logging.info("#####        ####           ####          ###     ###      ####           ###            ###     ###      ###  ###    ###        #####")
        logging.info("#####        ########       ####          ###########      ########       ###            ###     ###      ###   ###   ###        #####")
        logging.info("#####        ####           ####          ###     ###      ####           ###            ###     ###      ###    ###  ###        #####")
        logging.info("#####        ####           ####          ###     ###      ####           ###            ###     ###      ###     ### ###        #####")
        logging.info("#####        #####          #####         ###     ###      #####          #########      ####   ####      ###      ######        #####")
        logging.info("#####         ########       #######      ###     ###        #######      #########       #########       ###       #####        #####")
        logging.info("#####                                                                                                                            #####")
        logging.info("#####                                                                                                                            #####")
        logging.info("######################################################################################################################################")
        logging.info("######################################################################################################################################")
        logging.info("######################################################################################################################################")

        time.sleep(5)


    @classmethod
    def run_setup(cls):
        logging.info("#############################################################################################################################")
        logging.info("\t\t\t\t\t\t\t\t\t RUN SETUP")
        logging.info("#############################################################################################################################\n")
        logging.info("Project Base                              : %s", cnst.PROJECT_BASE_PATH)
        logging.info("Raw & Pickle Data source                  : %s", cnst.DATA_SOURCE_PATH)
        logging.info("CSV path for the dataset                  : %s", cnst.ALL_FILE)
        logging.info("GPU Enabled                               : %s", cnst.USE_GPU)
        logging.info("Folds                                     : %s", cnst.CV_FOLDS)

        logging.info("Batch Size [Train T1 : Train T2 : Predict]: [ "+str(cnst.T1_TRAIN_BATCH_SIZE)+" : "+str(cnst.T2_TRAIN_BATCH_SIZE)+" : "+str(cnst.PREDICT_BATCH_SIZE)+" ]")
        logging.info("Epochs [Tier1 : Tier2]                    : [ "+str(cnst.TIER1_EPOCHS)+" : "+str(cnst.TIER2_EPOCHS)+" ]")
        logging.info("\n")
        logging.info("Model used for Tier-1 Training            : Pre-trained Malconv - No exposure to Echelon training data" if cnst.USE_PRETRAINED_FOR_TIER1 else "Pre-trained Malconv exposed to Echelon training data")
        logging.info("Model used for Tier-2 Training            : Pre-trained Malconv - No exposure to Echelon section-wise training data" if cnst.USE_PRETRAINED_FOR_TIER2 else "Pre-trained Malconv exposed to Echelon section-wise training data")
        logging.info("Target FPR [ Overall : Tier1 : Tier2 ]    : [ "+str(cnst.OVERALL_TARGET_FPR)+" : "+str(cnst.TIER1_TARGET_FPR)+" : "+str(cnst.TIER2_TARGET_FPR)+" ]")
        logging.info("Skip Tier-1 Training                      : %s", cnst.SKIP_TIER1_TRAINING)
        logging.info("Skip Tier-2 Training                      : %s", cnst.SKIP_TIER2_TRAINING)
        logging.info("Skip ATI Processing                       : %s", cnst.SKIP_ATI_PROCESSING)
        logging.info("Boost benign files with a lower bound     : %s", cnst.PERFORM_B2_BOOSTING)
        logging.info("Percentiles for Q_Criteria selection      : %s", cnst.PERCENTILES)

        logging.info("Maximum file size set for the model input : %s", str(cnst.MAX_FILE_SIZE_LIMIT) + "bytes")
        logging.info("Maximum size set for Section Byte Map     : %s", cnst.MAX_SECT_BYTE_MAP_SIZE)
        logging.info("Layer Number to stunt plugin model for ATI: %s", cnst.LAYER_NUM_TO_STUNT)
        logging.info("CNN Window size | Stride size             : %s", str(cnst.CONV_WINDOW_SIZE) + "|" + str(cnst.CONV_STRIDE_SIZE))


if __name__ == '__main__':
    EchelonMeta.project_details()

