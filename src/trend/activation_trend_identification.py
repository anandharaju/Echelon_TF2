import os
from utils import utils
import numpy as np
from keras.models import Model
from keras.models import load_model
import pickle
from os.path import join
import config.constants as cnst
import plots.plots as plots
from plots import ati_plots
from predict.predict import predict_byte
from predict.predict_args import DefaultPredictArguments, Predict as pObj
from plots.ati_plots import feature_map_histogram
from .ati_args import SectionActivationDistribution
import pandas as pd
from analyzers.collect_exe_files import get_partition_data
import gc


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def find_qualified_sections(sd, trend, common_trend, support):
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
    for i, val in enumerate(cnst.PERCENTILES):
        q_sections_by_q_criteria[mal_q_criteria_by_percentiles[i]] = np.unique(np.concatenate([trend.columns[malfluence > mal_q_criteria_by_percentiles[i]], trend.columns[benfluence > ben_q_criteria_by_percentiles[i]]]))
        # print("Mal Sections:", trend.columns[malfluence > mal_q_criteria_by_percentiles[i]])
        # print("Ben Sections:", trend.columns[benfluence > ben_q_criteria_by_percentiles[i]])
    return q_sections_by_q_criteria


def parse_pe_pkl(file_index, file_id, fjson, unprocessed):
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
        print("parse failed . . . [FILE INDEX - ", file_index, "]  [", file_id, "] ", e)
        unprocessed += 1
    return section_bounds, unprocessed, file_byte_size


def map_act_to_sec(ftype, fmap, sbounds, sd):
    # section_support:      Information about how many samples in a given category has a section <Influence by presence>
    # activation_histogram: Information about total count of activations occurred in a given section for all samples
    #                       of given category <Influence by activation count>
    # activation_magnitude: Information about total sum of magnitude of activations occurred in a given section
    #                       for all samples of given category <Influence by activation strength>

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


def get_feature_map(smodel, partition, file):
    predict_args = DefaultPredictArguments()
    predict_args.verbose = cnst.ATI_PREDICT_VERBOSE
    prediction = predict_byte(smodel, partition, np.array([file]), predict_args)
    raw_feature_map = prediction[0]
    return raw_feature_map


def get_feature_maps(smodel, partition, files):
    predict_args = DefaultPredictArguments()
    predict_args.verbose = cnst.ATI_PREDICT_VERBOSE
    raw_feature_maps = predict_byte(smodel, partition, files, predict_args)
    return raw_feature_maps


def process_files(args, sd):
    unprocessed = 0
    samplewise_feature_maps = []
    stunted_model = get_stunted_model(args)
    print("Memory Required:", get_model_memory_usage(cnst.T1_TRAIN_BATCH_SIZE, stunted_model))
    files = args.t2_x_train
    files_type = args.t2_y_train

    print("FMAP MODULE Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(args.t2_y_train), len(np.where(args.t2_y_train == cnst.BENIGN)[0]), len(np.where(args.t2_y_train == cnst.MALWARE)[0])))

    # file_type = pObj_fmap.ytrue[i]  # Using Ground Truth to get trend of actual benign and malware files
    # file_whole_bytes = {file[:-4]: args.whole_b1_train_partition[file[:-4]]}
    raw_feature_maps = get_feature_maps(stunted_model, args.whole_b1_train_partition, files)
    del stunted_model
    del args.whole_b1_train_partition
    gc.collect()

    for i in range(0, len(files)):
        section_bounds, unprocessed, fsize = parse_pe_pkl(i, files[i][:-4], args.section_b1_train_partition[files[i][:-4]], unprocessed)
        if cnst.USE_POOLING_LAYER:
            try:
                pooled_max_1D_map = np.sum(raw_feature_maps[i] == np.amax(raw_feature_maps[i], axis=0), axis=1)[:np.min([cnst.MAX_FILE_CONVOLUTED_SIZE,int(fsize/cnst.CONV_STRIDE_SIZE)+2])]
                sd = map_act_to_sec(files_type[i], pooled_max_1D_map, section_bounds, sd)
            except Exception as e:
                print("$$$$$$$$", str(e), np.shape(raw_feature_maps[i]))  # .size, files[i], args.whole_b1_train_partition[files[i][:-4]])
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


def get_stunted_model(args):
    complete_model = load_model(join(args.save_path, args.t1_model_name))
    # model.summary()
    # redefine model to output right after the sixth hidden layer
    # (ReLU activation layer after convolution - before max pooling)

    #stunted_outputs = [complete_model.layers[x].output for x in [cnst.LAYER_NUM_TO_STUNT]]
    stunted_outputs = complete_model.get_layer('multiply_1').output
    stunted_model = Model(inputs=complete_model.inputs, outputs=stunted_outputs)
    # stunted_model.summary()
    print("Model stunted upto ", stunted_outputs[0], "Layer number passed to stunt:", cnst.LAYER_NUM_TO_STUNT)
    return stunted_model


def save_activation_trend(sd):
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
    fmaps_trend.drop([cnst.PADDING, cnst.LEAK], axis=1, inplace=True)
    fmaps_common_trend.drop([cnst.PADDING, cnst.LEAK], axis=1, inplace=True)
    fmaps_section_support.drop([cnst.PADDING], axis=1, inplace=True)
    return fmaps_trend, fmaps_common_trend, fmaps_section_support


def start_ati_process(args, fold_index, partition_count, sd):
    for pcount in range(0, partition_count):
        print("ATI for partition:", pcount)
        b1datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "b1_train_" + str(fold_index) + "_p" + str(pcount) + ".csv", header=None)
        args.t2_x_train, args.t2_y_train = b1datadf.iloc[:, 0], b1datadf.iloc[:, 1]
        args.whole_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t1")
        args.section_b1_train_partition = get_partition_data("b1_train", fold_index, pcount, "t2")

        sd = process_files(args, sd)
    return sd


def init(args, fold_index, partition_count, b1_all_file_cnt, b1b_all_truth_cnt, b1m_all_truth_cnt):
    sd = SectionActivationDistribution()
    sd.b1_count = b1_all_file_cnt
    sd.b1_b_truth_count = b1b_all_truth_cnt
    sd.b1_m_truth_count = b1m_all_truth_cnt

    sd = start_ati_process(args, fold_index, partition_count, sd)
    trend, common_trend, support = save_activation_trend(sd)
    # select sections for Tier-2 based on identified activation trend
    q_sections_by_q_criteria = find_qualified_sections(sd, trend, common_trend, support)

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
