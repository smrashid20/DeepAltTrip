import time
from pprint import PrettyPrinter

import numpy as np
import random
import metric
import csv
import os
import glob

random.seed(1234567890)
np.random.seed(1234567890)

pp = PrettyPrinter(indent=4, width=250)


def trajectory_extractor(dat_ix):
    dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb', 'caliAdv', 'disHolly', 'disland', 'epcot', 'MagicK']
    poi_name = "poi-" + dat_suffix[dat_ix] + ".csv"  # Edin
    tra_name = "traj-" + dat_suffix[dat_ix] + ".csv"
    print('To Train', dat_suffix[dat_ix])

    embedding_name = dat_suffix[dat_ix]

    op_tdata = open('origin_data/' + poi_name, 'r')
    ot_tdata = open('origin_data/' + tra_name, 'r')

    Trajectory = []
    poiIDs = []

    for line in op_tdata.readlines():
        lineArr = line.split(',')
        if lineArr[0] == 'poiID':
            continue
        poiIDs.append(int(lineArr[0]))

    print(poiIDs)

    for line in ot_tdata.readlines():
        lineArr = line.split(',')
        temp_line = list()
        if lineArr[0] == 'userID':
            continue
        for i in range(len(lineArr)):
            temp_line.append(lineArr[i].strip('\n'))
        Trajectory.append(temp_line)

    DATA = {}

    for index in range(len(Trajectory)):
        if int(Trajectory[index][-2]) >= 3:
            DATA.setdefault(Trajectory[index][0] + '-' + Trajectory[index][1], []).append(
                [Trajectory[index][2], Trajectory[index][3], Trajectory[index][4]])

    pop_list = []
    for k, v in DATA.items():
        v_new = sorted(v, key=lambda item: item[1])
        v_new_2 = []
        for i in range(len(v_new)):
            poi = v_new[i][0]
            got_before = 0
            for j in range(len(v_new_2)):
                if v_new_2[j][0] == poi:
                    got_before = 1
                    break

            if got_before == 0:
                v_new_2.append(v_new[i])

        if len(v_new_2) >= 3:
            v_new = v_new_2
            DATA[k] = v_new
            for i in range(len(v_new)):
                v_new[i][0] = int(v_new[i][0])
                v_new[i][1] = int(time.strftime("%H:%M:%S", time.localtime(int(v_new[i][1]))).split(":")[0])
                v_new[i][2] = int(time.strftime("%H:%M:%S", time.localtime(int(v_new[i][2]))).split(":")[0])

        else:
            pop_list.append(k)

    for key in pop_list:
        DATA.pop(key, None)

    ALL_TRAJ = []
    ALL_TRAJID = []
    ALL_USER = []
    ALL_TIME = []

    for k, v in DATA.items():
        traj = []
        traj_time = []

        for i in range(len(v)):
            traj.append(v[i][0])
            traj_time.append(v[i][1])

        ALL_TRAJ.append(traj)
        ALL_TIME.append(traj_time)

        str_k = k.split("-")

        ALL_TRAJID.append(int(str_k[1]))
        ALL_USER.append(str_k[0])

    return embedding_name, ALL_TRAJ, ALL_TRAJID, ALL_TIME, ALL_USER, poiIDs


def poi_transition_matrix(ALL_TRAJ, ALL_TRAJ_TIME, poiIDs):
    transition_mat = np.zeros((max(poiIDs) + 1, max(poiIDs) + 1))
    time_dict = dict()
    for j in range(len(ALL_TRAJ)):
        traj = ALL_TRAJ[j]
        traj_time = ALL_TRAJ_TIME[j]
        for i in range(len(traj) - 1):
            transition_mat[traj[i]][traj[i + 1]] += 1
            time_dict.setdefault(str(traj[i]) + "-" + str(traj[i + 1]), []).append(
                (traj_time[i + 1] - traj_time[i] + 24) % 24)

    transition_time_mat = np.zeros((max(poiIDs) + 1, max(poiIDs) + 1))
    for k, v in time_dict.items():
        str_k = str(k).split("-")
        poi_a = int(str_k[0])
        poi_b = int(str_k[1])
        v_avg = np.average(list(v))
        if v_avg < 0:
            print(v_avg)
            print(v)
        transition_time_mat[poi_a][poi_b] = int(np.round(v_avg))

    return transition_mat, transition_time_mat


def perturb_traj(traj, traj_time, deviation_limit, poiIDs, transition_mat, transition_time_mat):
    def insert(new_traj, new_traj_time):

        new_traj_temp_ = list(new_traj)
        new_traj_temp_time_ = list(new_traj_time)

        place_ = np.random.randint(0, len(new_traj) - 1)
        poi_before_ = new_traj[place_]
        poi_after_ = new_traj[place_ + 1]
        eligible_poi_ = []
        for j in range(len(poiIDs)):
            poi = poiIDs[j]
            if poiIDs[j] not in new_traj \
                    and transition_mat[poi_before_][poi] > 0 \
                    and transition_mat[poi][poi_after_] > 0 \
                    and transition_time_mat[poi_before_][poi] <= \
                    ((new_traj_time[place_ + 1] - new_traj_time[place_] + 24) % 24):
                eligible_poi_.append(poi)

        if len(eligible_poi_) > 0:
            poi_index_choice = np.random.randint(0, len(eligible_poi_))
            poi_choice = eligible_poi_[poi_index_choice]
            new_traj_temp_.insert(place_ + 1, poi_choice)
            new_traj_temp_time_.insert(place_ + 1,
                                       int((new_traj_time[place_] + transition_time_mat[poi_before_][
                                           poi_choice]) % 24))
            for t in range(place_ + 2, len(new_traj_temp_time_)):
                new_traj_temp_time_[t] = int((new_traj_temp_time_[t] +
                                              transition_time_mat[poi_before_][poi_choice] +
                                              transition_time_mat[poi_before_][poi_choice] -
                                              ((new_traj_temp_time_[place_ + 1] - new_traj_temp_time_[
                                                  place_] + 24) % 24)) % 24)

        return new_traj_temp_, new_traj_temp_time_

    new_traj = list(traj)
    new_traj_time = list(traj_time)

    max_perturb = int(np.ceil(len(traj) * deviation_limit))

    for i in range(max_perturb):
        choice = np.random.randint(0, 1)

        if choice == 0:
            new_traj_temp, new_traj_temp_time = insert(new_traj, new_traj_time)
            if metric.calc_F1(traj, new_traj_temp) < 1 - deviation_limit:
                break
            else:
                new_traj = new_traj_temp
                new_traj_time = new_traj_temp_time

        elif choice == 1:

            new_traj_temp = list(new_traj)
            new_traj_temp_time = list(new_traj_time)

            place = np.random.randint(len(new_traj))
            poi_before = -1
            poi_after = -1
            if place >= 1:
                poi_before = new_traj[place - 1]
            if place < len(new_traj) - 1:
                poi_after = new_traj[place + 1]

            eligible_poi = []
            for j in range(len(poiIDs)):
                poi = poiIDs[j]
                c_a = poiIDs[j] not in new_traj
                c_b = poi_before == -1 or transition_mat[poi_before][poi] > 0
                c_c = poi_after == -1 or transition_mat[poi][poi_after] > 0

                if c_a & c_b & c_c:
                    eligible_poi.append(poi)

            if len(eligible_poi) > 0:
                poi_index_choice = np.random.randint(0, len(eligible_poi))
                poi_choice = eligible_poi[poi_index_choice]
                new_traj_temp[place] = poi_choice

            if metric.calc_F1(traj, new_traj_temp) < 1 - deviation_limit:
                break
            else:
                new_traj = new_traj_temp
                new_traj_time = new_traj_temp_time

        elif choice == 2 and len(traj) >= 4:

            new_traj_temp = list(new_traj)
            new_traj_temp_time = list(new_traj_time)

            place = np.random.randint(len(new_traj))

            new_traj_temp.pop(place)
            new_traj_temp_time.pop(place)

            if metric.calc_F1(traj, new_traj_temp) < 1 - deviation_limit:
                break
            else:
                new_traj = new_traj_temp
                new_traj_time = new_traj_temp_time

    return new_traj, new_traj_time


def perturb_all_traj(ALL_TRAJ, ALL_TRAJ_ID, ALL_USER, ALL_TRAJ_TIME, poiIDs, copy_no, deviation_limit):
    traj_counter = max(ALL_TRAJ_ID)
    ALL_PERTURBED_TRAJ = []
    ALL_PERTURBED_TRAJID = []
    ALL_PERTURBED_USER = []
    ALL_PERTURBED_TIME = []

    if copy_no != 0:
        for j in range(len(ALL_TRAJ)):
            curr_traj = ALL_TRAJ[j]
            curr_traj_time = ALL_TRAJ_TIME[j]
            curr_user = ALL_USER[j]

            for traj_len in range(3, len(curr_traj)):
                for idx in range(len(curr_traj) - traj_len + 1):
                    new_traj = curr_traj[idx:idx + traj_len].copy()
                    new_traj_time = curr_traj_time[idx:idx + traj_len].copy()
                    new_user = curr_user
                    new_trajid = traj_counter + 1
                    traj_counter += 1

                    ALL_PERTURBED_TRAJ.append(new_traj)
                    ALL_PERTURBED_TIME.append(new_traj_time)
                    ALL_PERTURBED_USER.append(new_user)
                    ALL_PERTURBED_TRAJID.append(new_trajid)

    transition_mat, transition_time_mat = poi_transition_matrix(ALL_TRAJ, ALL_TRAJ_TIME, poiIDs)

    for j in range(len(ALL_TRAJ)):
        traj = ALL_TRAJ[j]
        traj_time = ALL_TRAJ_TIME[j]
        user = ALL_USER[j]
        trajid = ALL_TRAJ_ID[j]

        ALL_PERTURBED_TRAJ.append(traj)
        ALL_PERTURBED_TIME.append(traj_time)
        ALL_PERTURBED_USER.append(user)
        ALL_PERTURBED_TRAJID.append(trajid)

        for i in range(copy_no):
            perturbed_trajectory, perturbed_trajectory_time = perturb_traj(traj,
                                                                           traj_time,
                                                                           deviation_limit,
                                                                           poiIDs,
                                                                           transition_mat,
                                                                           transition_time_mat)
            ALL_PERTURBED_TRAJ.append(perturbed_trajectory)
            ALL_PERTURBED_TIME.append(perturbed_trajectory_time)
            ALL_PERTURBED_USER.append(user)
            ALL_PERTURBED_TRAJID.append(traj_counter + 1)
            traj_counter += 1
    return ALL_PERTURBED_TRAJ, ALL_PERTURBED_TRAJID, ALL_PERTURBED_USER, ALL_PERTURBED_TIME


def generate_ds(dat_ix, KFOLD, test_index, copy_no):
    files = glob.glob('processed_data/*')
    for f in files:
        os.remove(f)

    embedding_name, ALL_TRAJ, ALL_TRAJID, ALL_TIME, ALL_USER, poiIDs = trajectory_extractor(dat_ix)
    ALL_TRAJ, ALL_TRAJID, ALL_USER, ALL_TIME = perturb_all_traj(ALL_TRAJ,
                                                                ALL_TRAJID,
                                                                ALL_USER,
                                                                ALL_TIME,
                                                                poiIDs,
                                                                copy_no=copy_no, deviation_limit=0.5)

    query_set_dict_traj = {}
    query_set_dict_user = {}
    query_set_dict_time = {}
    query_set_dict_tid = {}

    for i in range(len(ALL_TRAJ)):
        q_str = str(ALL_TRAJ[i][0]) + "-" + str(ALL_TRAJ[i][-1])
        query_set_dict_traj.setdefault(q_str, []).append(ALL_TRAJ[i])
        query_set_dict_user.setdefault(q_str, []).append(ALL_USER[i])
        query_set_dict_time.setdefault(q_str, []).append(ALL_TIME[i])
        query_set_dict_tid.setdefault(q_str, []).append(ALL_TRAJID[i])

    # pp.pprint(query_set_dict_traj)

    qkeys = list(query_set_dict_traj.keys())
    np.random.shuffle(qkeys)

    keys_per_fold = len(qkeys) // KFOLD

    kfold_qkeys = {}
    for i in range(KFOLD - 1):
        kfold_qkeys[i + 1] = qkeys[i * keys_per_fold: (i + 1) * keys_per_fold]
    kfold_qkeys[KFOLD] = qkeys[(KFOLD - 1) * keys_per_fold:]

    for k, v in kfold_qkeys.items():
        to_traj_csv_train = []

        for i in range(len(v)):
            trajectories = query_set_dict_traj[v[i]]
            users = query_set_dict_user[v[i]]
            traj_ids = query_set_dict_tid[v[i]]
            traj_times = query_set_dict_time[v[i]]

            for j in range(len(trajectories)):
                to_traj_csv_train.append(trajectories[j])
                to_traj_csv_train.append(traj_times[j])
                to_traj_csv_train.append([users[j], traj_ids[j]])

        with open("processed_data/" + embedding_name + '_set_part_' + str(k) + '.csv', mode='w',
                  newline="") as csv_file:
            csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in to_traj_csv_train:
                csv_file_writer.writerow(row)

    def generate_train_test_data(test_index, KFOLD):

        train_lines = []
        for i in range(1, KFOLD + 1):
            if i == test_index:
                continue
            with open("processed_data/" + embedding_name + '_set_part_' + str(i) + '.csv', mode='r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    train_lines.append(row)

        with open("processed_data/" + embedding_name + '_train_set.csv', mode='w', newline="") as csv_file:
            csv_file_writer_ = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in train_lines:
                csv_file_writer_.writerow(row)

        test_lines = []
        with open("processed_data/" + embedding_name + '_set_part_' + str(test_index) + '.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                test_lines.append(row)

        with open("processed_data/" + embedding_name + '_test_set.csv', mode='w', newline="") as csv_file:
            csv_file_writer_ = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in test_lines:
                csv_file_writer_.writerow(row)

    generate_train_test_data(test_index, KFOLD)
