import numpy as np
import lstm_model
import torch
import copy
import random
import pprint

np.random.seed(1234567890)
torch.manual_seed(1234567890)
random.seed(1234567890)

pp = pprint.PrettyPrinter(indent=4, width=180)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_probs(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_f = list(input_seq)
    input_seq_b = list(reversed(list(input_seq)))

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length, input_seq_sd), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length, input_seq_sd), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_f, output_prb_b


def get_model_probs_f(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_f = list(input_seq)

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length, input_seq_sd), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    return output_prb_f


def get_model_probs_b(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_b = list(reversed(list(input_seq)))

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length, input_seq_sd), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_b


def get_prob_in_idx(input_seq, I):
    S = len(list(input_seq))
    output_prb_f, output_prb_b = get_model_probs(input_seq)

    final_op = np.zeros((output_prb_f.shape[1]))

    for i in range(len(final_op)):
        final_op[i] = (I * output_prb_f[I - 1][i] + (S - I - 1) * output_prb_b[S - I - 2][i]) / (S - 1)

    candidates_replacement = (np.argsort(-final_op))

    return final_op, candidates_replacement


def get_prob_in_idx_2(input_seq, I):
    S = len(list(input_seq))

    if I > S / 2:
        output_prb_f = get_model_probs_f(input_seq)
        final_op = output_prb_f[I - 1]
    else:
        output_prb_b = get_model_probs_b(input_seq)
        final_op = output_prb_b[S - I - 2]

    return final_op


def get_traj_perplexity(input_seq):
    if len(input_seq) == 0:
        return 1000

    output_prb_f = get_model_probs_f(input_seq)
    fwd_perplexity = 0

    for i in range(1, len(input_seq)):
        prob_idx = output_prb_f[i - 1][input_seq[i]]
        if prob_idx != 0:
            fwd_perplexity += np.log(prob_idx)
        else:
            fwd_perplexity += -100

    fwd_perplexity = (-1.0) * fwd_perplexity

    return fwd_perplexity


def normalize(x):
    tem = copy.copy(x)
    tem = np.array(tem)
    if np.max(tem) == 0:
        return tem
    return list(tem / np.sum(tem))


def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i] + c[i - 1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def get_refined_traj(input_seq, prominent_poi_pos):
    prominent_poi = input_seq[prominent_poi_pos]
    N = len(input_seq)

    prominent_poi_probs = [0.0]
    for i in range(1, N - 1):
        final_prob, _ = get_prob_in_idx(input_seq, i)
        final_prob += 1e-6
        prominent_poi_probs.append(final_prob[prominent_poi])
        for input_seq_idx in range(len(input_seq)):
            final_prob[input_seq[input_seq_idx]] = 0

        sampled_idx = sample_from_candidate(final_prob)
        if i != prominent_poi_pos:
            input_seq[i] = sampled_idx

    prominent_poi_idx = np.argmax(prominent_poi_probs)
    input_seq[prominent_poi_idx] = prominent_poi
    if int(prominent_poi_idx) != prominent_poi_pos:
        final_prob, _ = get_prob_in_idx(input_seq, prominent_poi_pos)
        final_prob += 1e-6
        for input_seq_idx in range(len(input_seq)):
            final_prob[input_seq[input_seq_idx]] = 0
        sampled_idx = sample_from_candidate(final_prob)
        input_seq[prominent_poi_pos] = sampled_idx

    return input_seq, prominent_poi_idx


def get_refined_traj_2(input_seq, non_replacable_poi_pos):
    N = len(input_seq)

    new_input_seq = None
    new_non_replacable_poi_pos = None

    forbidden_poi_list = []
    for i in range(len(non_replacable_poi_pos)):
        if non_replacable_poi_pos[i] == 1:
            forbidden_poi_list.append(input_seq[i])

    def replace_poi(input_seq_, idx, forbidden_pois):
        final_prob, _ = get_prob_in_idx(input_seq_, idx)
        final_prob += 1e-6

        for poi in input_seq_:
            if poi != input_seq_[idx]:
                final_prob[poi] = 0

        for forbidden_poi in forbidden_pois:
            final_prob[forbidden_poi] = 0

        sampled_idx = sample_from_candidate(final_prob)
        input_seq_[idx] = sampled_idx
        return input_seq_

    def swap_poi(input_seq_, idx, non_replacable_poi_pos_):
        if idx >= N - 2:
            return input_seq_, non_replacable_poi_pos_

        another_random_poi_idx = idx
        while another_random_poi_idx == idx:
            another_random_poi_idx = np.random.randint(1, N - 2)

        temp = input_seq_[idx]
        input_seq_[idx] = input_seq_[another_random_poi_idx]
        input_seq_[another_random_poi_idx] = temp

        temp = non_replacable_poi_pos_[idx]
        non_replacable_poi_pos_[idx] = non_replacable_poi_pos_[another_random_poi_idx]
        non_replacable_poi_pos_[another_random_poi_idx] = temp

        return input_seq_, non_replacable_poi_pos_

    i = np.random.randint(1, N - 1)
    r = np.random.random()
    if non_replacable_poi_pos[i] == 0:

        if r < (1 / 2) or i >= N - 2:
            input_seq = replace_poi(input_seq, i, forbidden_poi_list)
        else:
            input_seq, non_replacable_poi_pos = swap_poi(input_seq, i, non_replacable_poi_pos)
    elif non_replacable_poi_pos[i] == 1:
        if r < (1 / 2):
            input_seq, non_replacable_poi_pos = swap_poi(input_seq, i, non_replacable_poi_pos)
            if non_replacable_poi_pos[i] == 0:
                input_seq = replace_poi(input_seq, i, forbidden_poi_list)

    return input_seq, non_replacable_poi_pos


def sampling_algo_2(barebone_seq, N_max=5):
    N = N_max

    dummy_candidate_seq = np.arange(N)
    dummy_non_replacable_poi_pos = np.zeros(dummy_candidate_seq.shape)

    counter = N
    for i in range(1, len(dummy_candidate_seq)):
        if dummy_candidate_seq[i] in barebone_seq:
            dummy_candidate_seq[i] = counter
            counter += 1

    dummy_candidate_seq[0] = barebone_seq[0]
    dummy_candidate_seq[-1] = barebone_seq[-1]

    dummy_non_replacable_poi_pos[0] = 1
    dummy_non_replacable_poi_pos[-1] = 1

    best_perp = 1000
    best_traj = dummy_candidate_seq

    prominent_poi_idx = np.random.randint(1, N - 1)
    dummy_candidate_seq[prominent_poi_idx] = barebone_seq[1]

    dummy_non_replacable_poi_pos[prominent_poi_idx] = 1

    curr_candidate_seq = dummy_candidate_seq
    curr_non_replacable_poi_pos = dummy_non_replacable_poi_pos
    curr_candidate_seq_perp = 1000
    unchanged = 2

    for j in range(5*(N-2)):
        curr_candidate_seq_cand, curr_non_replacable_poi_pos_cand = get_refined_traj_2(curr_candidate_seq.copy(),
                                                                                       curr_non_replacable_poi_pos.copy())

        curr_candidate_seq_cand_perp = get_traj_perplexity(curr_candidate_seq)
        if curr_candidate_seq_cand_perp < curr_candidate_seq_perp or unchanged >= 2:
            curr_candidate_seq = curr_candidate_seq_cand
            curr_non_replacable_poi_pos = curr_non_replacable_poi_pos_cand
            curr_candidate_seq_perp = curr_candidate_seq_cand_perp
            unchanged = 0
        else:
            unchanged += 1

        if best_perp == 1000 or best_perp > curr_candidate_seq_cand_perp:
            best_perp = curr_candidate_seq_cand_perp
            best_traj = curr_candidate_seq_cand

    return best_traj
