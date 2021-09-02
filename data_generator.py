import pprint
import random
import numpy as np
import csv
import args_kdiverse

pp = pprint.PrettyPrinter(indent=4, width=180)
dat_ix = args_kdiverse.dat_ix

random.seed(1234567890)
np.random.seed(1234567890)


def isSame(traj_a, traj_b):
    if len(traj_a) != len(traj_b):
        return False
    for i in range(len(traj_a)):
        if traj_a[i] != traj_b[i]:
            return False

    return True


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


def get_poi_info(dat_ix):
    dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb', 'caliAdv', 'disHolly', 'disland', 'epcot', 'MagicK']
    embedding_name = dat_suffix[dat_ix]
    poi_name = "poi-" + dat_suffix[dat_ix] + ".csv"
    op_tdata = open('origin_data/' + poi_name, 'r')

    POIs = []

    for line in op_tdata.readlines():
        lineArr = line.split(',')
        temp_line = list()
        for item in lineArr:
            temp_line.append(item.strip('\n'))
        POIs.append(temp_line)
    POIs = POIs[1:]

    ALL_POI_IDS = []

    for i in range(len(POIs)):
        ALL_POI_IDS.append(int(POIs[i][0]))

    return embedding_name, POIs, ALL_POI_IDS


def extract_train_test_info(dat_ix):
    embedding_name, POIs, ALL_POI_IDS = get_poi_info(dat_ix)

    query_dict_trajectory_train = dict()
    query_dict_users_train = dict()
    query_dict_traj_ids_train = dict()
    query_dict_traj_time_train = dict()
    query_dict_freq_train = dict()

    with open("processed_data/" + embedding_name + '_train_set.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_no = 0

        row0 = []
        row1 = []
        row2 = []

        for row in csv_reader:

            if line_no == 0:
                row0 = row.copy()
            elif line_no == 1:
                row1 = row.copy()
            elif line_no == 2:
                row2 = row.copy()

                curr_traj = [int(poi) for poi in row0]
                st_poi = curr_traj[0]
                ed_poi = curr_traj[-1]
                qu = str(st_poi) + "-" + str(ed_poi)

                curr_traj_time = [int(poi_time) for poi_time in row1]
                curr_user = row2[0]
                curr_traj_id = int(row2[1])

                gotBefore = False
                all_traj_pos = -1

                if qu in query_dict_trajectory_train.keys():
                    all_traj = query_dict_trajectory_train[qu]
                    for prev_traj_itr in range(len(all_traj)):
                        if isSame(all_traj[prev_traj_itr], curr_traj):
                            gotBefore = True
                            all_traj_pos = prev_traj_itr
                            break

                    if not gotBefore:

                        all_traj.append(curr_traj)
                        query_dict_trajectory_train[qu] = all_traj

                        all_u = query_dict_users_train[qu]
                        all_u.append([curr_user])
                        query_dict_users_train[qu] = all_u

                        all_traj_id = query_dict_traj_ids_train[qu]
                        all_traj_id.append([curr_traj_id])
                        query_dict_traj_ids_train[qu] = all_traj_id

                        all_traj_time = query_dict_traj_time_train[qu]
                        all_traj_time.append([curr_traj_time])
                        query_dict_traj_time_train[qu] = all_traj_time

                        all_freq = query_dict_freq_train[qu]
                        all_freq.append(1)
                        query_dict_freq_train[qu] = all_freq

                    else:

                        all_u = query_dict_users_train[qu]
                        all_u[all_traj_pos].append(curr_user)
                        query_dict_users_train[qu] = all_u

                        all_traj_id = query_dict_traj_ids_train[qu]
                        all_traj_id[all_traj_pos].append(curr_traj_id)
                        query_dict_traj_ids_train[qu] = all_traj_id

                        all_traj_time = query_dict_traj_time_train[qu]
                        all_traj_time[all_traj_pos].append(curr_traj_time)
                        query_dict_traj_time_train[qu] = all_traj_time

                        all_freq = query_dict_freq_train[qu]
                        all_freq[all_traj_pos] += 1
                        query_dict_freq_train[qu] = all_freq

                else:

                    query_dict_trajectory_train.setdefault(qu, []).append(curr_traj)
                    query_dict_users_train.setdefault(qu, []).append([curr_user])
                    query_dict_traj_ids_train.setdefault(qu, []).append([curr_traj_id])
                    query_dict_traj_time_train.setdefault(qu, []).append([curr_traj_time])
                    query_dict_freq_train.setdefault(qu, []).append(1)

            line_no = (line_no + 1) % 3

    query_dict_trajectory_test = dict()
    query_dict_users_test = dict()
    query_dict_traj_ids_test = dict()
    query_dict_traj_time_test = dict()
    query_dict_freq_test = dict()

    with open("processed_data/" + embedding_name + '_test_set.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_no = 0

        row0 = []
        row1 = []
        row2 = []

        for row in csv_reader:

            if (line_no == 0):
                row0 = row.copy()
            elif (line_no == 1):
                row1 = row.copy()
            elif (line_no == 2):
                row2 = row.copy()

                curr_traj = [int(poi) for poi in row0]
                st_poi = curr_traj[0]
                ed_poi = curr_traj[-1]
                qu = str(st_poi) + "-" + str(ed_poi)

                curr_traj_time = [int(poi_time) for poi_time in row1]
                curr_user = row2[0]
                curr_traj_id = int(row2[1])

                gotBefore = False
                all_traj_pos = -1

                if qu in query_dict_trajectory_test.keys():
                    all_traj = query_dict_trajectory_test[qu]
                    for prev_traj_itr in range(len(all_traj)):
                        if isSame(all_traj[prev_traj_itr], curr_traj):
                            gotBefore = True
                            all_traj_pos = prev_traj_itr
                            break

                    if not gotBefore:

                        all_traj.append(curr_traj)
                        query_dict_trajectory_test[qu] = all_traj

                        all_u = query_dict_users_test[qu]
                        all_u.append([curr_user])
                        query_dict_users_test[qu] = all_u

                        all_traj_id = query_dict_traj_ids_test[qu]
                        all_traj_id.append([curr_traj_id])
                        query_dict_traj_ids_test[qu] = all_traj_id

                        all_traj_time = query_dict_traj_time_test[qu]
                        all_traj_time.append([curr_traj_time])
                        query_dict_traj_time_test[qu] = all_traj_time

                        all_freq = query_dict_freq_test[qu]
                        all_freq.append(1)
                        query_dict_freq_test[qu] = all_freq

                    else:

                        all_u = query_dict_users_test[qu]
                        all_u[all_traj_pos].append(curr_user)
                        query_dict_users_test[qu] = all_u

                        all_traj_id = query_dict_traj_ids_test[qu]
                        all_traj_id[all_traj_pos].append(curr_traj_id)
                        query_dict_traj_ids_test[qu] = all_traj_id

                        all_traj_time = query_dict_traj_time_test[qu]
                        all_traj_time[all_traj_pos].append(curr_traj_time)
                        query_dict_traj_time_test[qu] = all_traj_time

                        all_freq = query_dict_freq_test[qu]
                        all_freq[all_traj_pos] += 1
                        query_dict_freq_test[qu] = all_freq

                else:

                    query_dict_trajectory_test.setdefault(qu, []).append(curr_traj)
                    query_dict_users_test.setdefault(qu, []).append([curr_user])
                    query_dict_traj_ids_test.setdefault(qu, []).append([curr_traj_id])
                    query_dict_traj_time_test.setdefault(qu, []).append([curr_traj_time])
                    query_dict_freq_test.setdefault(qu, []).append(1)

            line_no = (line_no + 1) % 3

    return embedding_name, POIs, ALL_POI_IDS, query_dict_trajectory_train, query_dict_freq_train, \
           query_dict_trajectory_test, query_dict_freq_test


embedding_name, POIs, ALL_POI_IDS, query_dict_trajectory_train, query_dict_freq_train, query_dict_trajectory_test, \
query_dict_freq_test = extract_train_test_info(dat_ix)


def get_vocab_to_int(query_dict_trajectory_train, query_dict_freq_train):
    ALL_POI_IDS_FREQ = dict()
    for k, v in query_dict_trajectory_train.items():
        GT_freq = query_dict_freq_train[k]
        for i in range(len(v)):
            for j in range(GT_freq[i]):
                for k in range(len(v[i])):
                    ALL_POI_IDS_FREQ.setdefault(v[i][k], []).append(1)

    for i in range(len(ALL_POI_IDS)):
        ALL_POI_IDS_FREQ.setdefault(ALL_POI_IDS[i], []).append(1)

    for k, v in ALL_POI_IDS_FREQ.items():
        ALL_POI_IDS_FREQ[k] = len(v)

    ALL_POI_IDS_FREQ_SORTED = sorted(ALL_POI_IDS_FREQ.items(), key=lambda item: item[1], reverse=True)
    vocab_to_int = dict()
    for i in range(len(ALL_POI_IDS_FREQ_SORTED)):
        vocab_to_int[ALL_POI_IDS_FREQ_SORTED[i][0]] = i
    vocab_to_int['GO'] = len(ALL_POI_IDS_FREQ_SORTED)
    vocab_to_int['PAD'] = len(ALL_POI_IDS_FREQ_SORTED) + 1
    vocab_to_int['END'] = len(ALL_POI_IDS_FREQ_SORTED) + 2
    int_to_vocab = dict()
    for k, v in vocab_to_int.items():
        int_to_vocab[v] = k

    return vocab_to_int, int_to_vocab


vocab_to_int, int_to_vocab = get_vocab_to_int(query_dict_trajectory_train, query_dict_freq_train)


def convert_vocab_to_int(traj_dict, convert_values=True):
    traj_dict_new = dict()

    if not convert_values:

        for k, v in traj_dict.items():
            str_k = str(k).split("-")
            st_p = int(str_k[0])
            en_p = int(str_k[1])
            new_k = str(vocab_to_int[st_p]) + "-" + str(vocab_to_int[en_p])

            traj_dict_new[new_k] = v

        return traj_dict_new

    for k, v in traj_dict.items():
        st_p = v[0][0]
        en_p = v[0][-1]
        new_k = str(vocab_to_int[st_p]) + "-" + str(vocab_to_int[en_p])
        new_v = []
        for i in range(len(v)):
            new_v_i = [vocab_to_int[poi] for poi in v[i]]
            new_v.append(new_v_i)
        traj_dict_new[new_k] = new_v

    return traj_dict_new


def convert_int_to_vocab(traj_dict, convert_values=True):
    traj_dict_new = dict()

    if not convert_values:

        for k, v in traj_dict.items():
            str_k = str(k).split("-")
            st_p = int(str_k[0])
            en_p = int(str_k[1])
            new_k = str(int_to_vocab[st_p]) + "-" + str(int_to_vocab[en_p])

            traj_dict_new[new_k] = v

        return traj_dict_new

    for k, v in traj_dict.items():
        st_p = v[0][0]
        en_p = v[0][-1]
        new_k = str(int_to_vocab[st_p]) + "-" + str(int_to_vocab[en_p])
        new_v = []
        for i in range(len(v)):
            new_v_i = [int_to_vocab[poi] for poi in v[i]]
            new_v.append(new_v_i)
        traj_dict_new[new_k] = new_v

    return traj_dict_new


query_dict_trajectory_train = convert_vocab_to_int(query_dict_trajectory_train)
query_dict_trajectory_test = convert_vocab_to_int(query_dict_trajectory_test)

query_dict_freq_train = convert_vocab_to_int(query_dict_freq_train, convert_values=False)
query_dict_freq_test = convert_vocab_to_int(query_dict_freq_test, convert_values=False)


def get_all_raw_routes(query_dict_traj, query_dict_freq):
    all_routes = []

    for k, v in query_dict_traj.items():
        for i in range(len(v)):
            for j in range(query_dict_freq[k][i]):
                all_routes.append(v[i])

    return all_routes


def get_poi_transition_matrix(query_dict_trajectory, query_dict_freq):
    transition_mat = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
    all_routes = get_all_raw_routes(query_dict_trajectory, query_dict_freq)

    for traj in all_routes:
        for j in range(len(traj) - 1):
            first_poi = traj[j]
            second_poi = traj[j + 1]
            transition_mat[first_poi][second_poi] += 1

    return transition_mat


poi_transition_matrix = get_poi_transition_matrix(query_dict_trajectory_train, query_dict_freq_train)


def get_poi_transition_matrix_window(query_dict_trajectory, query_dict_freq):
    transition_mat = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
    all_routes = get_all_raw_routes(query_dict_trajectory, query_dict_freq)

    for traj in all_routes:
        for j in range(len(traj)):
            first_poi = traj[j]
            for k in ([-2, -1, 1, 2]):
                if 0 <= j + k < len(traj):
                    second_poi = traj[j + k]
                    transition_mat[first_poi][second_poi] += 1

    return transition_mat


poi_transition_matrix_window = get_poi_transition_matrix_window(query_dict_trajectory_train, query_dict_freq_train)


def normalise_transmat(transmat_cnt):
    transmat = np.array(transmat_cnt.copy())
    t_min = np.min(transmat)
    t_max = np.max(transmat)
    if t_max - t_min != 0:
        transmat = (transmat - t_min) / (t_max - t_min)
    else:
        transmat = (transmat - (t_min - 1))
    return transmat


poi_transition_matrix_normalized = normalise_transmat(poi_transition_matrix_window)


# plt.imshow(poi_transition_matrix_window, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
#
# plt.imshow(poi_transition_matrix_normalized, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()

def get_POI_category_matrix():
    poi_line_dict = dict()

    for i in range(len(POIs)):
        c_poi = vocab_to_int[int(POIs[i][0])]
        poi_line_dict[c_poi] = i

    poi_categories_dict = dict()
    for i in range(len(vocab_to_int) - 3):
        if i in poi_line_dict.keys():
            poi_categories_dict[i] = POIs[poi_line_dict[i]][1]

    transmat = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
    for i in range(len(transmat)):
        for j in range(len(transmat[i])):
            if poi_categories_dict[i] == poi_categories_dict[j]:
                transmat[i, j] = 1

    return transmat


poi_category_matrix = get_POI_category_matrix()


def get_POI_distance_matrix():
    poi_line_dict = dict()

    for i in range(len(POIs)):
        c_poi = vocab_to_int[int(POIs[i][0])]
        poi_line_dict[c_poi] = i

    poi_poi_distance_matrix = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
    for i in range(len(vocab_to_int) - 3):
        poi_1_lon = float(POIs[poi_line_dict[i]][2])
        poi_1_lat = float(POIs[poi_line_dict[i]][3])
        for j in range(len(vocab_to_int) - 3):
            poi_2_lon = float(POIs[poi_line_dict[j]][2])
            poi_2_lat = float(POIs[poi_line_dict[j]][3])

            dist = calc_dist_vec(poi_1_lon, poi_1_lat, poi_2_lon, poi_2_lat)
            poi_poi_distance_matrix[i][j] = np.round(dist, 2)

    poi_poi_distance_matrix_train_gae = np.exp(1 - poi_poi_distance_matrix)
    n_min = np.min(poi_poi_distance_matrix_train_gae)
    n_max = np.max(poi_poi_distance_matrix_train_gae)
    if n_max - n_min != 0:
        poi_poi_distance_matrix = ((poi_poi_distance_matrix_train_gae - n_min) / (n_max - n_min)) - np.eye(
            len(vocab_to_int) - 3)

    return poi_poi_distance_matrix


poi_distance_matrix = get_POI_distance_matrix()


# plt.imshow(poi_distance_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()


class dataset_trajectory(object):
    def __init__(self, input_seqs, input_seq_lengths, backward=False):
        self.input_seqs = input_seqs
        if backward:
            self.input_seqs = [list(reversed(traj)) for traj in self.input_seqs]
        self.input_seq_lengths = input_seq_lengths

        self.ds_length = len(input_seqs)
        self.max_seq_length = np.max(input_seq_lengths)
        self.pad_id = 0

    def process_batch(self, inp_batch, inp_seq_batch):

        sd_batch_final = [[ls[0], ls[-1]] for ls in inp_batch]

        inp_seq_batch_final = [ln - 1 for ln in inp_seq_batch]
        max_length_batch = np.max(inp_seq_batch)
        inp_batch_final = [ls[:-1] + [self.pad_id] * (max_length_batch - len(ls)) for ls in inp_batch]
        tgt_batch_final = [ls[1:] + [self.pad_id] * (max_length_batch - len(ls)) for ls in inp_batch]

        zp = zip(inp_batch_final, inp_seq_batch_final, tgt_batch_final, sd_batch_final)
        zp_l = list(zp)
        zp_l = sorted(zp_l, key=lambda tuple: tuple[1], reverse=True)

        ib, sb, tb, sdb = zip(*zp_l)
        ib = list(ib)
        sb = list(sb)
        tb = list(tb)
        sdb = list(sdb)

        ib = np.array(ib)
        sb = np.array(sb)
        tb = np.array(tb)
        sdb = np.array(sdb)

        return ib, sb, tb, sdb

    def no_training_batches(self, batch_size):

        no_seqs = self.ds_length
        no_batches = int(np.ceil(no_seqs / batch_size))

        return no_batches

    def __call__(self, step, batch_size):

        no_seqs = self.ds_length
        no_batches = int(np.ceil(no_seqs / batch_size))

        step_no = step % no_batches

        if (step_no + 1) * batch_size > no_seqs:
            ts_pad = [self.input_seqs[itr % no_seqs] for itr in range((step_no + 1) * batch_size - no_seqs)]
            tsl_pad = [self.input_seq_lengths[itr % no_seqs] for itr in range((step_no + 1) * batch_size - no_seqs)]

            inp_batch = self.input_seqs[step_no * batch_size:] + ts_pad
            inp_seq_batch = self.input_seq_lengths[step_no * batch_size:] + tsl_pad

        else:
            inp_batch = self.input_seqs[step_no * batch_size:(step_no + 1) * batch_size]
            inp_seq_batch = self.input_seq_lengths[step_no * batch_size:(step_no + 1) * batch_size]

        return self.process_batch(inp_batch, inp_seq_batch)


def get_trajectory_dataset():
    all_traj_data_train = get_all_raw_routes(query_dict_trajectory_train, query_dict_freq_train)
    np.random.shuffle(all_traj_data_train)
    all_traj_data_train_seq = []

    for j in range(len(all_traj_data_train)):
        all_traj_data_train_seq.append(len(all_traj_data_train[j]))

    return dataset_trajectory(all_traj_data_train, all_traj_data_train_seq), dataset_trajectory(all_traj_data_train,
                                                                                                all_traj_data_train_seq,
                                                                                                backward=True)


# dt, dt_b = get_trajectory_dataset()
#
# for i in range(8):
#     ib, sb, tb, sdb = dt(i, 8)
#     print(ib)
#     print(tb)
#     print(sb)
#     print(sdb)
#     print("\n\n\n")
