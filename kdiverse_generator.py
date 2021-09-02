import csv
import os
import time

import metric
import numpy as np
import data_generator
import graph_embedding
import lstm_model
import gibbs_sample
import args_kdiverse

np.random.seed(1234567890)


def generate_result(load_from_file, K, N_min, N_max):
    graph_embedding.get_POI_embeddings(load_from_file=load_from_file)
    lstm_model.get_forward_lstm_model(load_from_file=load_from_file)
    lstm_model.get_backward_lstm_model(load_from_file=load_from_file)

    total_score_curr_f1 = 0
    total_score_curr_pf1 = 0
    total_score_likability = 0
    total_score_intra_div_f1 = 0

    total_traj_curr = 0
    count = 1

    all_gtset = dict()
    all_gtfreqset = dict()
    all_recset = dict()

    print(data_generator.int_to_vocab)
    st = time.time()
    for k, v in data_generator.query_dict_trajectory_test.items():

        # if count>= 3:
        #     break

        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])

        _, lstm_order = gibbs_sample.get_prob_in_idx([poi_start, 0, poi_end], 1)
        lstm_rank = np.argsort(lstm_order)

        def get_next_poi(use_freq_, rank):
            proposed_poi = 0
            for i_ in range(len(rank)):
                if i_ == poi_start or i_ == poi_end:
                    continue
                if (proposed_poi == poi_start or proposed_poi == poi_end) and (i_ != poi_start and i_ != poi_end):
                    proposed_poi = i_
                    continue

                if use_freq_[i_] < use_freq_[proposed_poi]:
                    proposed_poi = i_
                elif use_freq_[i_] == use_freq_[proposed_poi] and rank[i_] < rank[proposed_poi]:
                    proposed_poi = i_

            use_freq_[proposed_poi] += 1
            return use_freq_, proposed_poi

        use_freq = np.zeros([len(lstm_rank)])
        all_traj = []
        for i in range(K):
            use_freq, next_poi = get_next_poi(use_freq, lstm_rank)
            new_traj = gibbs_sample.sampling_algo_2([poi_start, next_poi, poi_end],
                                                    N_max=N_max)
            for j in range(len(new_traj)):
                use_freq[new_traj[j]] += 1
            all_traj.append(new_traj)

        print("{}/{}".format(count, len(data_generator.query_dict_trajectory_test)))
        count += 1
        print([data_generator.int_to_vocab[poi_start], data_generator.int_to_vocab[poi_end]])

        k_converted = str(data_generator.int_to_vocab[poi_start]) + '-' + str(data_generator.int_to_vocab[poi_end])

        dict_temp = dict()
        dict_temp[k] = v
        all_gtset[k_converted] = list(data_generator.convert_int_to_vocab(dict_temp).values())[0]

        all_gtfreqset[k_converted] = [data_generator.query_dict_freq_test[k]]

        dict_temp = dict()
        dict_temp[k] = all_traj
        all_recset[k_converted] = list(data_generator.convert_int_to_vocab(dict_temp).values())[0]

        print(all_gtset[k_converted])
        print(all_recset[k_converted])

        total_score_likability += metric.likability_score_3(v, data_generator.query_dict_freq_test[k], all_traj)
        total_score_curr_f1 += metric.tot_f1_evaluation(v, data_generator.query_dict_freq_test[k], all_traj)
        total_score_curr_pf1 += metric.tot_pf1_evaluation(v, data_generator.query_dict_freq_test[k], all_traj)
        if K > 1:
            total_score_intra_div_f1 += metric.intra_div_F1(all_traj)

        total_traj_curr += np.sum(data_generator.query_dict_freq_test[k]) * len(all_traj)

        avg_likability = total_score_likability / (count - 1)
        if K > 1:
            avg_div = total_score_intra_div_f1 / (count - 1)
        else:
            avg_div = 0
        avg_f1 = total_score_curr_f1 / total_traj_curr
        avg_pf1 = total_score_curr_pf1 / total_traj_curr

        print("Avg. upto now: Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(avg_pf1)
              + " Div: " + str(avg_div))
        print("\n")

    end = time.time()
    print(N_max)
    print(count)
    print("Time: {}".format((end - st) / count))
    print("\n")
    print("Final Score - With K = {}".format(K))
    avg_likability = total_score_likability / (count - 1)
    if K > 1:
        avg_div = total_score_intra_div_f1 / (count - 1)
    else:
        avg_div = 0
    avg_f1 = total_score_curr_f1 / total_traj_curr
    avg_pf1 = total_score_curr_pf1 / total_traj_curr

    print("Likability: " + str(avg_likability) + " F1: " + str(avg_f1) + " PF1: " + str(avg_pf1)
          + " Div: " + str(avg_div))

    write_to_file(all_recset, 'recset_myalgo', N_min=N_min, N_max=N_max)

    return


def write_to_file(dictionary, directory, N_min, N_max, isFreq=False):
    if not isFreq:
        file_path = os.path.join(directory, str(data_generator.embedding_name)) \
                    + "_index_" + str(args_kdiverse.test_index) \
                    + "_min_" + str(N_min) \
                    + "_max_" + str(N_max) \
                    + "_copy_" + str(args_kdiverse.copy_no) + '.csv'
    else:
        file_path = os.path.join(directory, str(data_generator.embedding_name)) + "_" + str(
            args_kdiverse.test_index) + '_freq.csv'

    write_lines = []

    for k, v in dictionary.items():
        for i in range(len(v)):
            write_lines.append(v[i])
        write_lines.append([-1])

    with open(file_path, mode='w+', newline="") as to_csv_file:
        csv_file_writer = csv.writer(to_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in write_lines:
            csv_file_writer.writerow(row)

    return
