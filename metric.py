import numpy as np


def calc_F1(traj_act, traj_rec, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert (isinstance(noloop, bool))
    assert (len(traj_act) > 0)
    assert (len(traj_rec) > 0)

    if noloop == True:
        intersize = len(set(traj_act) & set(traj_rec))
    else:
        match_tags = np.zeros(len(traj_act), dtype=np.bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if match_tags[j] == False and poi == traj_act[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(traj_act)
    precision = intersize * 1.0 / len(traj_rec)
    Denominator = recall + precision
    if Denominator == 0:
        Denominator = 1
    F1 = 2 * precision * recall * 1.0 / Denominator
    return F1


# cpdef float calc_pairsF1(y, y_hat):
def calc_pairsF1(y, y_hat):
    assert (len(y) > 0)
    # assert (len(y) == len(set(y)))  # no loops in y
    # cdef int n, nr, nc, poi1, poi2, i, j
    # cdef double n0, n0r
    n = len(y)
    nr = len(y_hat)
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]: nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        F1 = 0
    else:
        F1 = 2. * precision * recall / (precision + recall)
    return float(F1)


def popularity_metric(traj_rec, poi_pairs_frequency):
    nr = len(traj_rec)

    if (nr <= 2):
        return 0

    sum_popularity = 0

    for i in range(nr):
        poi1 = traj_rec[i]
        for j in range(i + 1, nr):
            poi2 = traj_rec[j]
            sum_popularity += poi_pairs_frequency[poi1, poi2]

    popularity = sum_popularity / ((nr * (nr - 1) / 2) ** 0.7)

    return popularity


def popularity_K_traj(trajectories, poi_pairs_frequency):
    pop = []

    for traj in trajectories:
        popularity = popularity_metric(traj, poi_pairs_frequency)
        pop.append(popularity)

    pop = np.array(pop)
    return pop


def likability_score_3(GT_set, GT_freq, recommended_set):
    scores = []
    for i in range(len(GT_set)):
        max_score = max([GT_freq[i] * calc_F1(GT_set[i], rec_traj) for rec_traj in recommended_set])
        scores.append(max_score)

    return np.sum(scores) / float(np.sum(GT_freq))


def f1_evaluation(GT_set, GT_freq, recommended_set):
    total_score = 0

    for i in range(len(GT_set)):
        total_score += calc_F1(GT_set[i], recommended_set[0]) * GT_freq[i]

    return total_score


def pairsf1_evaluation(GT_set, GT_freq, recommended_set):
    total_score = 0

    for i in range(len(GT_set)):
        total_score += calc_pairsF1(GT_set[i], recommended_set[0]) * GT_freq[i]

    return total_score


def total_f1_evaluation(route, recommended_set):
    total_score = 0
    for i in range(len(recommended_set)):
        total_score += calc_F1(route, recommended_set[i])

    return total_score


# def tot_f1_evaluation(GT_set, GT_freq, recommended_set):
#     total_score = 0
#
#     for i in range(len(GT_set)):
#         for j in range(len(recommended_set)):
#             total_score += calc_F1(GT_set[i], recommended_set[j]) * GT_freq[i]
#
#     return

def tot_f1_evaluation(GT_set, GT_freq, recommended_set):
    total_score = 0
    scores = []
    for i in range(len(GT_set)):
        for j in range(len(recommended_set)):
            scores = scores + [calc_F1(GT_set[i], recommended_set[j])] * GT_freq[i]
            total_score += calc_F1(GT_set[i], recommended_set[j]) * GT_freq[i]
    #print(scores)
    return total_score


def tot_pf1_evaluation(GT_set, GT_freq, recommended_set):
    total_score = 0

    for i in range(len(GT_set)):
        for j in range(len(recommended_set)):
            total_score += calc_pairsF1(GT_set[i], recommended_set[j]) * GT_freq[i]

    return total_score




def coverage_iou(GT_set, rec_set):
    total_gt_set = set()

    for route in GT_set:
        total_gt_set = total_gt_set | set(route[1:-1])

    total_rec_set = set()

    for route in rec_set:
        total_rec_set = total_rec_set | set(route[1:-1])

    ratio = len(total_gt_set & total_rec_set) / len(total_gt_set | total_rec_set)

    return ratio


def intra_div_F1(set):
    div_scores = []
    for i in range(len(set)):
        for j in range(i + 1, len(set)):
            div_scores.append(1 - calc_F1(set[i], set[j]))

    return np.average(np.array(div_scores))
