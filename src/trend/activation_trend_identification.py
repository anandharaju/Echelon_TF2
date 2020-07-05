import numpy as np
from keras.models import Model
from keras.models import load_model
from os.path import join
import config.settings as cnst
import plots.plots as plots
from predict.predict import predict_byte, predict_byte_by_section
from predict.predict_args import DefaultPredictArguments, Predict as pObj
from .ati_args import SectionActivationDistribution
import pandas as pd
from analyzers.collect_exe_files import get_partition_data, store_partition_data
import gc
import logging


def find_qualified_sections(sd, trend, common_trend, support, fold_index):
    """ Function for training Tier-1 model with whole byte sequence data
        Args:
            sd: object to hold activation distribution of PE sections
            trend: plain activation trend found by core ATI process
            common_trend: not used here
            support: not used here
            fold_index: current fold index of cross validation
        Returns:
            q_sections_by_q_criteria: a dict with q_criterion found for each percentile supplied and
                                      their respective list of sections qualified.
    """
    btrend = trend.loc["BENIGN_ACTIVATION_MAGNITUDE"]
    mtrend = trend.loc["MALWARE_ACTIVATION_MAGNITUDE"]

    # Averaging based on respective benign and malware population
    btrend = btrend / sd.b1_b_truth_count
    mtrend = mtrend / sd.b1_m_truth_count

    btrend[btrend == 0] = 1
    mtrend[mtrend == 0] = 1

    malfluence = mtrend / btrend
    benfluence = btrend / mtrend

    mal_q_criteria_by_percentiles = np.percentile(malfluence, q=cnst.PERCENTILES)
    ben_q_criteria_by_percentiles = np.percentile(benfluence, q=cnst.PERCENTILES)

    q_sections_by_q_criteria = {}
    for i, _ in enumerate(cnst.PERCENTILES):
        q_sections_by_q_criteria[mal_q_criteria_by_percentiles[i]] = np.unique(np.concatenate([trend.columns[malfluence > mal_q_criteria_by_percentiles[i]], trend.columns[benfluence > ben_q_criteria_by_percentiles[i]]]))
        if i == 0:  # Do once for lowest percentile
            list_qsec = np.concatenate([trend.columns[malfluence > mal_q_criteria_by_percentiles[i]], trend.columns[benfluence > ben_q_criteria_by_percentiles[i]]])
            list_avg_act_mag_signed = np.concatenate([malfluence[malfluence > mal_q_criteria_by_percentiles[i]] * -1, benfluence[benfluence > ben_q_criteria_by_percentiles[i]]])

            available_sec = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + 'data' + cnst.ESC + 'available_sections.csv', header=None)
            available_sec = list(available_sec.iloc[0])
            sec_emb = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + 'data' + cnst.ESC + 'section_embeddings.csv')

            list_qsec_id = []
            list_qsec_emb = []
            for q in list_qsec:
                try:
                    list_qsec_emb.append(sec_emb[q][0])
                    list_qsec_id.append(available_sec.index(q) + 1)
                except Exception as e:
                    if not (cnst.LEAK in str(e) or cnst.PADDING in str(e)):
                        logging.debug("The section ["+str(q)+"] is not present in available_sections.csv/section_embeddings.csv")

            qdf = pd.DataFrame([list_qsec, list_qsec_id, list_qsec_emb, list_avg_act_mag_signed], columns=list_qsec)
            qdf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + 'data' + cnst.ESC + 'qsections_meta_'+str(fold_index)+'.csv', header=None, index=False)
            # print("Mal Sections:", trend.columns[malfluence > mal_q_criteria_by_percentiles[i]])
            # print("Ben Sections:", trend.columns[benfluence > ben_q_criteria_by_percentiles[i]])
    logging.info("Qsections found - " + str(len(q_sections_by_q_criteria.keys())))
    logging.info(q_sections_by_q_criteria.keys())
    return q_sections_by_q_criteria


def parse_pe_pkl(file_index, file_id, fjson, unprocessed):
    """ Function to parse pickle file to find the boundaries of PE sections in a sample's pickle representation
    Args:
        file_index: PE sample index
        file_id: PE name
        fjson: pickle data representation of PE sample
        unprocessed: keeps track of count of samples not processed properly
    Returns:
         section_bounds: PE section boundaries
         unprocessed: keeps track of count of samples not processed properly
         file_byte_size: size of full sample
    """
    section_bounds = []
    file_byte_size = None
    max_section_end_offset = 0
    try:
        file_byte_size = fjson['size_byte']
        pkl_sections = fjson["section_info"].keys()
        for pkl_section in pkl_sections:
            section_bounds.append(
                (pkl_section,
                 fjson["section_info"][pkl_section]["section_bounds"]["start_offset"],
                 fjson["section_info"][pkl_section]["section_bounds"]["end_offset"]))
            if fjson["section_info"][pkl_section]["section_bounds"]["end_offset"] > max_section_end_offset:
                max_section_end_offset = fjson["section_info"][pkl_section]["section_bounds"]["end_offset"]

        # Placeholder section "padding" - for activations in padding region
        if max_section_end_offset < fjson["size_byte"]:
            section_bounds.append((cnst.TAIL, max_section_end_offset + 1, fjson["size_byte"]))
        section_bounds.append((cnst.PADDING, fjson["size_byte"] + 1, cnst.MAX_FILE_SIZE_LIMIT))
    except Exception as e:
        logging.Exception("parse failed . . . [FILE INDEX - " + str(file_index) + "]  [" + str(file_id) + "] ")
        unprocessed += 1
    return section_bounds, unprocessed, file_byte_size


def map_act_to_sec(ftype, fmap, sbounds, sd):
    """
        Function to map each hidden layer activation found to corresponding PE section
        Params:
            ftype: Benign or Malware
            fmap: Hidden layer activation map
            sbounds: Dict of PE sections and their boundaries
        Return:
            sd: Object to hold computed activation distribution of PE sections

        Description of other variables/objects used:
            section_support:      Information about how many samples in a given category has a section <Influence by presence>
            activation_histogram: Information about total count of activations occurred in a given section for all samples
                                  of given category <Influence by activation count>
            activation_magnitude: Information about total sum of magnitude of activations occurred in a given section
                                  for all samples of given category <Influence by activation strength>
    """
    # fmap = fmap // 1  # print("FEATURE MAP ", len(feature_map), " : \n", feature_map)
    idx = np.argsort(fmap)[::-1][:len(fmap)]  # Sort activations in descending order -- Helpful to find top activations
    if sbounds is not None:

        for j in range(0, len(sbounds)):
            section = sbounds[j][0]
            sd.a_section_support[section] = (
                        sd.a_section_support[section] + 1) if section in sd.a_section_support.keys() else 1
            if ftype == cnst.BENIGN:
                sd.b_section_support[section] = (
                            sd.b_section_support[section] + 1) if section in sd.b_section_support.keys() else 1
                if section not in sd.m_section_support.keys():
                    sd.m_section_support[section] = 0
            else:
                if section not in sd.b_section_support.keys():
                    sd.b_section_support[section] = 0
                sd.m_section_support[section] = (
                            sd.m_section_support[section] + 1) if section in sd.m_section_support.keys() else 1

        for current_activation_window in range(0, len(fmap)):  # range(0, int(cnst.MAX_FILE_SIZE_LIMIT / cnst.CONV_STRIDE_SIZE)):
            section = None
            offset = idx[current_activation_window] * 500
            act_val = fmap[idx[current_activation_window]]

            ######################################################################################
            # Change for Pooling layer based Activation trend - Only Max activation is traced back
            if act_val == 0:
                continue
            ######################################################################################
            for j in range(0, len(sbounds)):
                cur_section = sbounds[j]
                if cur_section[1] <= offset <= cur_section[2]:
                    section = cur_section[0]
                    break

            if section is not None:
                # if "." not in section: section = "." + section #Same section's name with and without dot are different
                # Sum of Magnitude of Activations
                if section in sd.a_activation_magnitude.keys():
                    sd.a_activation_magnitude[section] += act_val
                    sd.a_activation_histogram[section] += 1
                    if ftype == cnst.BENIGN:
                        if sd.b_activation_magnitude[section] is None:
                            sd.b_activation_magnitude[section] = act_val
                            sd.b_activation_histogram[section] = 1
                        else:
                            sd.b_activation_magnitude[section] += act_val
                            sd.b_activation_histogram[section] += 1
                    else:
                        if sd.m_activation_magnitude[section] is None:
                            sd.m_activation_magnitude[section] = act_val
                            sd.m_activation_histogram[section] = 1
                        else:
                            sd.m_activation_magnitude[section] += act_val
                            sd.m_activation_histogram[section] += 1
                else:
                    sd.a_activation_magnitude[section] = act_val
                    sd.a_activation_histogram[section] = 1
                    if ftype == cnst.BENIGN:
                        sd.b_activation_magnitude[section] = act_val
                        sd.b_activation_histogram[section] = 1
                        sd.m_activation_magnitude[section] = None
                        sd.m_activation_histogram[section] = None
                    else:
                        sd.b_activation_magnitude[section] = None
                        sd.b_activation_histogram[section] = None
                        sd.m_activation_magnitude[section] = act_val
                        sd.m_activation_histogram[section] = 1
            else:
                # !!! VERIFY ALL OFFSET IS MATCHED AND CHECK FOR LEAKAGE !!!
                # print("No matching section found for OFFSET:", offset)
                sd.a_activation_magnitude[cnst.LEAK] += act_val
                sd.a_activation_histogram[cnst.LEAK] += 1
                if ftype == cnst.BENIGN:
                    sd.b_activation_magnitude[cnst.LEAK] += act_val
                    sd.b_activation_histogram[cnst.LEAK] += 1
                else:
                    sd.m_activation_magnitude[cnst.LEAK] += act_val
                    sd.m_activation_histogram[cnst.LEAK] += 1
    return sd


def get_feature_maps(smodel, partition, files):
    """
        Function to obtain hidden layer activation (feature) maps using given stunted model
    Params:
        smodel: stunted model to use
        partition: partition for current set of B1 samples under process
        files: IDs of the samples to be processed from the partition
    Returns:
        raw_feature_maps: hidden layer activation (feature) maps
    """
    predict_args = DefaultPredictArguments()
    predict_args.verbose = cnst.ATI_PREDICT_VERBOSE
    raw_feature_maps = predict_byte(smodel, partition, files, predict_args)
    return raw_feature_maps


def process_files(stunted_model, args, sd):
    """
    Function to process the B1 samples to obtain hidden layer activation maps and trace back their PE sections
    Params:
         stunted_model: Tier-1 model that is stunted up to required hidden layer where activation maps are collected.
         args: contains various config data
    Returns:
         sd: Object to hold computed activation distribution of PE sections
    """
    unprocessed = 0
    samplewise_feature_maps = []
    files = args.t2_x_train
    files_type = args.t2_y_train

    logging.info("FMAP MODULE Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(args.t2_y_train), len(np.where(args.t2_y_train == cnst.BENIGN)[0]), len(np.where(args.t2_y_train == cnst.MALWARE)[0])))

    # file_type = pObj_fmap.ytrue[i]  # Using Ground Truth to get trend of actual benign and malware files
    # file_whole_bytes = {file[:-4]: args.whole_b1_train_partition[file[:-4]]}
    raw_feature_maps = get_feature_maps(stunted_model, args.whole_b1_train_partition, files)
    del args.whole_b1_train_partition
    gc.collect()

    for i in range(0, len(files)):
        section_bounds, unprocessed, fsize = parse_pe_pkl(i, files[i][:-4], args.section_b1_train_partition[files[i][:-4]], unprocessed)
        if cnst.USE_POOLING_LAYER:
            try:
                pooled_max_1D_map = np.sum(raw_feature_maps[i] == np.amax(raw_feature_maps[i], axis=0), axis=1)[:np.min([cnst.MAX_FILE_CONVOLUTED_SIZE,int(fsize/cnst.CONV_STRIDE_SIZE)+2])]
                sd = map_act_to_sec(files_type[i], pooled_max_1D_map, section_bounds, sd)
            except Exception as e:
                logging.exception("$$$$$$$$  " + str(np.shape(raw_feature_maps[i])))  # .size, files[i], args.whole_b1_train_partition[files[i][:-4]])
        else:
            feature_map = raw_feature_maps[i].sum(axis=1).ravel()
            # feature_map_histogram(feature_map, prediction)
            samplewise_feature_maps.append(feature_map)
            sd = map_act_to_sec(files_type[i], feature_map, section_bounds, sd)

    del args.section_b1_train_partition
    gc.collect()
    return sd

    # print(section_stat)
    # print("Unprocessed file count: ", unprocessed)

    # Find activation distribution
    # raw_arr = np.array(np.squeeze(temp_feature_map_list))
    # print(len(raw_arr), raw_arr.max())
    # raw_arr = raw_arr[raw_arr > 0.3]
    # print(len(raw_arr))
    # plt.hist(raw_arr, 10)#range(0, len(raw_arr)))
    # plt.show()

    '''for key in act.keys():
        # key = "."+key if "." not in key else key
        if key is not None and key != '' and key != '.padding':
            with open("BENIGN" if "benign" in section_stat_file else "MALWARE" + "_activation_" + key[1:] + ".csv", mode='a+') as f:
                f.write(str(act[key]))
    '''
    '''
    #overall_stat.append(section_stat)
    for x in pcs_keys:
        overall_stat_str += str(section_stat[x]) + ","
    overall_stat_str = overall_stat_str[:-1] + "\n"

    print("\n[Unprocessed Files : ", unprocessed, "]      Overall Stats: ", overall_stat_str)

    processed_file_count = len(fn_list) - unprocessed
    normalized_stats_str = str(section_stat["header"]/processed_file_count) + "," \
                           + str(section_stat["text"]/processed_file_count) + "," \
                           + str(section_stat["data"]/processed_file_count) + "," \
                           + str(section_stat["rsrc"]/processed_file_count) + "," \
                           + str(section_stat["pdata"]/processed_file_count) + "," \
                           + str(section_stat["rdata"]/processed_file_count) + "\n"
                           #+ str(section_stat["padding"]/processed_file_count) \

    print("Normalized Stats: ", normalized_stats_str)
    #plt.show()

    with open(section_stat_file, 'w+') as f:
        f.write(overall_stat_str)
        f.write("\n")
        f.write(normalized_stats_str)
    '''


def get_stunted_model(args, tier):
    """ Function to stunt the given model up to the required hidden layer
        based on the supplied hidden layer number
    """
    complete_model = load_model(join(args.save_path, args.t1_model_name if tier == 1 else args.t2_model_name))
    # model.summary()
    # redefine model to output right after the sixth hidden layer
    # (ReLU activation layer after convolution - before max pooling)

    stunted_outputs = [complete_model.layers[x].output for x in [args.layer_num]]
    # stunted_outputs = complete_model.get_layer('multiply_1').output
    stunted_model = Model(inputs=complete_model.inputs, outputs=stunted_outputs)
    # stunted_model.summary()
    logging.debug("Model stunted upto " + str(stunted_outputs[0]) + "   Layer number passed to stunt:" + str(args.layer_num))
    return stunted_model


def save_activation_trend(sd):
    """
        Function to save the various activation trends identified in CSV format files.
        Params:
            sd: Object containing computed activation distribution of PE sections
        Returns:
            fmaps_trend: used to identify the qualified sections in subsequent steps
            others: Not in use currently
    """
    fmaps_trend = pd.DataFrame()
    fmaps_common_trend = pd.DataFrame()
    fmaps_section_support = pd.DataFrame()

    fmaps_trend["ACTIVATION / HISTOGRAM"] = ["ALL_ACTIVATION_MAGNITUDE", "BENIGN_ACTIVATION_MAGNITUDE",
                                             "MALWARE_ACTIVATION_MAGNITUDE", "HISTOGRAM_ALL", "HISTOGRAM_BENIGN",
                                             "HISTOGRAM_MALWARE"]
    fmaps_common_trend["COMMON"] = ["ALL_ACTIVATION_MAGNITUDE", "BENIGN_ACTIVATION_MAGNITUDE",
                                    "MALWARE_ACTIVATION_MAGNITUDE", "HISTOGRAM_ALL", "HISTOGRAM_BENIGN",
                                    "HISTOGRAM_MALWARE"]
    fmaps_section_support["SUPPORT"] = ["PRESENCE_IN_ALL", "PRESENCE_IN_BENIGN", "PRESENCE_IN_MALWARE",
                                        "SUPPORT_IN_ALL", "SUPPORT_IN_BENIGN", "SUPPORT_IN_MALWARE"]

    for key in sd.a_activation_histogram.keys():
        fmaps_trend[key] = [int(sd.a_activation_magnitude[key]) if sd.a_activation_magnitude[key] is not None else
                            sd.a_activation_magnitude[key],
                            int(sd.b_activation_magnitude[key]) if sd.b_activation_magnitude[key] is not None else
                            sd.b_activation_magnitude[key],
                            int(sd.m_activation_magnitude[key]) if sd.m_activation_magnitude[key] is not None else
                            sd.m_activation_magnitude[key],
                            int(sd.a_activation_histogram[key]) if sd.a_activation_histogram[key] is not None else
                            sd.a_activation_histogram[key],
                            int(sd.b_activation_histogram[key]) if sd.b_activation_histogram[key] is not None else
                            sd.b_activation_histogram[key],
                            int(sd.m_activation_histogram[key]) if sd.m_activation_histogram[key] is not None else
                            sd.m_activation_histogram[key]]

        if sd.b_activation_histogram[key] is not None and sd.m_activation_histogram[key] is not None:
            fmaps_common_trend[key] = [
                int(sd.a_activation_magnitude[key]) if sd.a_activation_magnitude[key] is not None else
                sd.a_activation_magnitude[key],
                int(sd.b_activation_magnitude[key]) if sd.b_activation_magnitude[key] is not None else
                sd.b_activation_magnitude[key],
                int(sd.m_activation_magnitude[key]) if sd.m_activation_magnitude[key] is not None else
                sd.m_activation_magnitude[key],
                int(sd.a_activation_histogram[key]) if sd.a_activation_histogram[key] is not None else
                sd.a_activation_histogram[key],
                int(sd.b_activation_histogram[key]) if sd.b_activation_histogram[key] is not None else
                sd.b_activation_histogram[key],
                int(sd.m_activation_histogram[key]) if sd.m_activation_histogram[key] is not None else
                sd.m_activation_histogram[key]]

    if sd.b1_count > 0 and sd.b1_b_truth_count > 0 and sd.b1_m_truth_count > 0:
        for key in sd.a_section_support.keys():
            fmaps_section_support[key] = [sd.a_section_support[key], sd.b_section_support[key],
                                          sd.m_section_support[key],
                                          "{:0.1f}%".format(sd.a_section_support[key] / sd.b1_count * 100),
                                          "{:0.1f}%".format(sd.b_section_support[key] / sd.b1_b_truth_count * 100),
                                          "{:0.1f}%".format(sd.m_section_support[key] / sd.b1_m_truth_count * 100)]

    fmaps_trend.fillna(-1, inplace=True)

    fmaps_trend.set_index('ACTIVATION / HISTOGRAM', inplace=True)
    fmaps_common_trend.set_index('COMMON', inplace=True)
    fmaps_section_support.set_index('SUPPORT', inplace=True)

    # Store activation trend identified
    fmaps_trend.to_csv(cnst.COMBINED_FEATURE_MAP_STATS_FILE, index=True)
    fmaps_common_trend.to_csv(cnst.COMMON_COMBINED_FEATURE_MAP_STATS_FILE, index=True)
    fmaps_section_support.to_csv(cnst.SECTION_SUPPORT, index=True)

    # Drop padding and leak information after saving - not useful for further processing
    try:
        fmaps_trend.drop([cnst.PADDING], axis=1, inplace=True)
        fmaps_common_trend.drop([cnst.PADDING], axis=1, inplace=True)
        fmaps_section_support.drop([cnst.PADDING], axis=1, inplace=True)
        fmaps_trend.drop([cnst.LEAK], axis=1, inplace=True)
        fmaps_common_trend.drop([cnst.LEAK], axis=1, inplace=True)
    except:
        logging.info("Proceeding after trying to clean fmap data.")
    return fmaps_trend, fmaps_common_trend, fmaps_section_support


def start_ati_process(args, fold_index, partition_count, sd):
    """
        Function to perform the ATI process over all partitions of B1 training set
        Params:
            args: contains various config data
            fold_index: current fold index of cross validation
            partition_count: count of train B1 partitions
        Returns:
            sd: Object containing computed activation distribution of PE sections
    """
    args.layer_num = cnst.LAYER_NUM_TO_STUNT
    stunted_model = get_stunted_model(args, tier=1)
    for pcount in range(0, partition_count):
        logging.info("ATI for partition: %s", pcount)
        b1datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_train_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
        args.t2_x_train, args.t2_y_train = b1datadf.iloc[:, 0], b1datadf.iloc[:, 1]
        args.whole_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t1")
        args.section_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t2")

        sd = process_files(stunted_model, args, sd)
    del stunted_model
    gc.collect()
    return sd


def get_top_act_blocks(top_acts_idx, sbounds, q_sections, whole_bytes):
    """
        Function to map the top activation back to Qualified section's byte blocks and collating them to form block dataset
        Params:
            top_acts_idx: act as offsets of top activations in the hidden layer activation (feature) map
            sbounds: Pe section boundaries
            q_sections: qualified sections
            whole_bytes: Entire byte content of a PE sample
        Returns:
             top_blocks: single sequence of all top blocks found
    """
    top_blocks = []
    top_acts_idx.sort()
    if sbounds is not None:
        for idx, offset in enumerate(top_acts_idx * cnst.CONV_STRIDE_SIZE):
            for sname, low, upp in sbounds:
                if low <= offset <= upp:
                    if sname in q_sections:
                        try:
                            top_blocks.extend(whole_bytes[offset:offset+cnst.CONV_STRIDE_SIZE])
                            break
                        except Exception as e:
                            logging.exception("[MODULE: get_section_id_vector()] Error occurred while mapping section id: %s  %s  %s  %s  %s  %s",
                                  idx, low, offset, upp, sname, sname in q_sections)
                    # else:
                    #    print(sname, sname in q_sections, sname in available_sections)
    else:
        logging.info("Sections bounds not available. Returning a vector of Zeroes for section id vector.")
    return top_blocks


def collect_b1_block_dataset(args, fold_index, partition_count, mode, qcnt='X'):
    """
        Function to generate the top ativation blocks based dataset from B1 sample set
    Params:
        args: an object containing various config data
        fold_index: current fold index of cross validation
        partition_count: count of B1 train partitions
        mode: phase of data collection - Train / Val / Test
        qcnt: index of the current q_criterion. 'X' for Testing phase
    Returns:
         None (collected data is persisted directly to disk storage)
    """
    args.layer_num = cnst.LAYER_NUM_TO_COLLECT_NN_DATASET
    stunted_model = get_stunted_model(args, tier=cnst.TIER_TO_COLLECT_BLOCK_DATA)
    for pcount in range(0, partition_count):
        logging.info("Collecting Block data for partition: %s", pcount)
        b1datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_"+mode+"_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
        files, files_type = b1datadf.iloc[:, 0], b1datadf.iloc[:, 1]
        args.whole_b1_partition = get_partition_data("b1_"+mode, fold_index, pcount, "t1")
        args.section_b1_partition = get_partition_data("b1_"+mode, fold_index, pcount, "t2")
        unprocessed = 0
        logging.info("Block Module Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(files_type), len(np.where(files_type == cnst.BENIGN)[0]), len(np.where(files_type == cnst.MALWARE)[0])))
        nn_predict_args = DefaultPredictArguments()
        nn_predict_args.verbose = cnst.ATI_PREDICT_VERBOSE
        raw_feature_maps = predict_byte_by_section(stunted_model, args.section_b1_partition, files, args.q_sections, None, nn_predict_args)
        logging.info("Raw feature maps found.")
        for i in range(0, len(files)):
            section_bounds, unprocessed, fsize = parse_pe_pkl(i, files[i][:-4], args.section_b1_partition[files[i][:-4]], unprocessed)
            if cnst.USE_POOLING_LAYER:
                try:
                    cur_fmap = raw_feature_maps[i]
                    top_acts_idx = np.argmax(cur_fmap, axis=0)
                    top_blocks = get_top_act_blocks(top_acts_idx, section_bounds, args.q_sections, args.whole_b1_partition[files[i][:-4]]["whole_bytes"])
                    if sum(top_blocks) == 0:
                        logging.debug("No useful top block data added for sample " + files[i])
                except Exception as e:
                    logging.exception("$$$$ Error occurred in Top Activation Block Module. $$$$")
            args.whole_b1_partition[files[i][:-4]]["whole_bytes"] = top_blocks
        store_partition_data("block_b1_"+mode, fold_index, pcount, "t1", args.whole_b1_partition)
        del args.section_b1_partition
        del args.whole_b1_partition
        gc.collect()
    del stunted_model
    gc.collect()


def init(args, fold_index, partition_count, b1_all_file_cnt, b1b_all_truth_cnt, b1m_all_truth_cnt):
    """ Activation Trend Identification (ATI) Module
        Args:
            args: various data required for ATI
            fold_index: current fold of cross-validation
            partition_count: number of partitions created for b1 training set
            b1_all_file_cnt: count of samples in b1 set
            b1b_all_truth_cnt: count of benign samples in b1 training set
            b1m_all_truth_cnt: count of malware samples in b1 training set
        Returns:
            None (Resultant data are stored in CSV for further use)
    """
    sd = SectionActivationDistribution()
    sd.b1_count = b1_all_file_cnt
    sd.b1_b_truth_count = b1b_all_truth_cnt
    sd.b1_m_truth_count = b1m_all_truth_cnt

    sd = start_ati_process(args, fold_index, partition_count, sd)
    trend, common_trend, support = save_activation_trend(sd)
    # select sections for Tier-2 based on identified activation trend
    q_sections_by_q_criteria = find_qualified_sections(sd, trend, common_trend, support, fold_index)

    # select, drop = plots.save_stats_as_plot(fmaps, qualification_criteria)

    # Save qualified sections by Q_criteria
    qdata = [np.concatenate([[str(q_criterion)], q_sections_by_q_criteria[q_criterion]]) for q_criterion in q_sections_by_q_criteria]
    pd.DataFrame(qdata).to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "out" + cnst.ESC + "result" + cnst.ESC + "qsections_by_qcriteria_" + str(fold_index) + ".csv", index=False, header=None)
    return  # select, drop


if __name__ == '__main__':
    # start_visualization_process(args)
    plots.save_stats_as_plot()

    # pe = pefile.PE("D:\\08_Dataset\\benign\\git-gui.exe")
    # parse_pe(0, "D:\\08_Dataset\\benign\\git-gui.exe", 204800, 0)
    # for section in pe.sections:
    #    print(section)
    # print(pe.OPTIONAL_HEADER, "\n", pe.NT_HEADERS, "\n", pe.FILE_HEADER, "\n", pe.RICH_HEADER, "\n", pe.DOS_HEADER,
    # \"\n", pe.__IMAGE_DOS_HEADER_format__, "\n", pe.header, "\n", "LENGTH", len(pe.header))

'''def display_edit_distance():
    sections = []
    top_sections = []
    malware_edit_distance = []
    print("\n   SECTION [EDIT DISTANCE SCORE]")
    df = pd.read_csv(combined_stat_file)
    df.set_index("type", inplace=True)
    for i in range(0, len(keys)):
        a = df.loc['FN'].values[i]
        b = df.loc['BENIGN'].values[i]
        c = df.loc['MALWARE'].values[i]
        dist1 = norm(a-b) // 1
        dist2 = norm(a-c) // 1
        print(keys[i], dist1, dist2, "[MALWARE]" if dist2 < dist1 else "[BENIGN]", dist1 - dist2)
        if dist2 < dist1:
            malware_edit_distance.append(dist1 - dist2)
            sections.append(keys[i])
    idx = np.argsort(malware_edit_distance)[::-1]
    for t in idx:
        print("%10s" % sections[t], "%20s" % str(malware_edit_distance[t]))
        top_sections.append(sections[t])
    return top_sections[:3]

    def ks(cutoff):
    from scipy import stats
    keys = ['header', 'text', 'data', 'rsrc', 'pdata', 'rdata']
    for key in keys:
        b = pd.read_csv('D:\\03_GitWorks\\echelon\\out\\result_multi\\benign.csv' + ".activation_" + key + ".csv", header=None)
        m = pd.read_csv('D:\\03_GitWorks\\echelon\\out\\result_multi\\malware.csv' + ".activation_" + key + ".csv", header=None)
        b = np.squeeze((b.get_values()))
        m = np.squeeze((m.get_values()))
        b = (b - b.min()) / (b.max() - b.min())
        m = (m - m.min()) / (m.max() - m.min())
        print(key, b.max(), len(b), len(b[b > cutoff]))
        print(key, m.max(), len(m), len(m[m > cutoff]))
        print("Section: ", key[:4], "\t\t", stats.ks_2samp(np.array(b), np.array(m)))
        plt.hist(b[b > cutoff], 100)
        plt.hist(m[m > cutoff], 100)
        plt.legend(['benign', 'malware'])
        plt.show()
        # break
    '''
