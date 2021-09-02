import numpy as np
import lstm_model
import torch
import random

np.random.seed(1234567890)
torch.manual_seed(1234567890)
random.seed(1234567890)

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


def get_backward_sequence(middle_poi, start_poi, end_poi, max_length):
    input_seq = [middle_poi]
    start_poi_probs = [0.0]

    while len(input_seq) < max_length:
        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]
        input_seq_batch_sd = [input_seq[0], input_seq[-1]]

        input_seq_batch_n = np.array(input_seq_batch).astype(int)
        input_seq_batch_len_n = np.array(input_seq_batch_len).astype(int)
        input_seq_batch_sd_n = np.array(input_seq_batch_sd).astype(int)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)
        input_seq_batch_sd_t = torch.LongTensor(input_seq_batch_sd_n).to(device)

        model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_bwd(input_seq_batch_t, input_seq_batch_len_t, input_seq_batch_sd_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == 1:

            start_poi_prob = output_prb[start_poi]
            start_poi_probs.append(start_poi_prob)

            output_prb[start_poi] = 0
            output_prb[end_poi] = 0
            for poi in input_seq:
                output_prb[int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)

        else:

            start_poi_prob = output_prb[-1][start_poi]
            start_poi_probs.append(start_poi_prob)

            output_prb[-1][start_poi] = 0
            output_prb[-1][end_poi] = 0
            for poi in input_seq:
                output_prb[-1][int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)[-1]

        input_seq.append(selected_poi)

    start_poi_idx = np.argmax(np.array(start_poi_probs))
    input_seq[int(start_poi_idx)] = start_poi
    candidate_seq = input_seq[:start_poi_idx + 1]

    return candidate_seq


def get_forward_sequence(middle_poi, start_poi, end_poi, max_length):
    input_seq = [middle_poi]
    end_poi_probs = [0.0]

    while len(input_seq) < max_length:

        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]
        input_seq_batch_sd = [input_seq[0], input_seq[-1]]

        input_seq_batch_n = np.array(input_seq_batch).astype(int)
        input_seq_batch_len_n = np.array(input_seq_batch_len).astype(int)
        input_seq_batch_sd_n = np.array(input_seq_batch_sd).astype(int)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)
        input_seq_batch_sd_t = torch.LongTensor(input_seq_batch_sd_n).to(device)

        model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_fwd(input_seq_batch_t, input_seq_batch_len_t, input_seq_batch_sd_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == 1:

            end_poi_prob = output_prb[end_poi]
            end_poi_probs.append(end_poi_prob)

            output_prb[start_poi] = 0
            output_prb[end_poi] = 0
            for poi in input_seq:
                output_prb[int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)

        else:

            end_poi_prob = output_prb[-1][end_poi]
            end_poi_probs.append(end_poi_prob)

            output_prb[-1][start_poi] = 0
            output_prb[-1][end_poi] = 0
            for poi in input_seq:
                output_prb[-1][int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)[-1]

        input_seq.append(selected_poi)

    end_poi_idx = np.argmax(np.array(end_poi_probs))
    input_seq[int(end_poi_idx)] = end_poi
    candidate_seq = input_seq[:end_poi_idx + 1]

    return candidate_seq


def get_fixed_length_sequence(start_poi, middle_poi, end_poi, seq_length):
    candidate_sequences = []
    candidate_seq_logprobs = []

    backward_seq = get_backward_sequence(middle_poi, start_poi, end_poi, seq_length - 1)

    input_seq = list(reversed(backward_seq))

    while len(input_seq) != seq_length:
        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]
        input_seq_batch_sd = [input_seq[0], input_seq[-1]]

        input_seq_batch_n = np.array(input_seq_batch).astype(int)
        input_seq_batch_len_n = np.array(input_seq_batch_len).astype(int)
        input_seq_batch_sd_n = np.array(input_seq_batch_sd).astype(int)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)
        input_seq_batch_sd_t = torch.LongTensor(input_seq_batch_sd_n).to(device)

        model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_fwd(input_seq_batch_t, input_seq_batch_len_t, input_seq_batch_sd_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == seq_length - 1:

            candidate_seq = input_seq.copy()
            candidate_seq.append(end_poi)
            candidate_sequences.append(candidate_seq)
            candidate_seq_logprobs.append(get_traj_perplexity(candidate_seq))
            break

        else:
            output_prb[-1][end_poi] = 0
            for poi in input_seq:
                output_prb[-1][int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)[-1]
            input_seq.append(selected_poi)

    forward_seq = get_forward_sequence(middle_poi, start_poi, end_poi, seq_length - 1)

    input_seq = list(reversed(forward_seq))

    while len(input_seq) != seq_length:
        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]
        input_seq_batch_sd = [input_seq[0], input_seq[-1]]

        input_seq_batch_n = np.array(input_seq_batch).astype(int)
        input_seq_batch_len_n = np.array(input_seq_batch_len).astype(int)
        input_seq_batch_sd_n = np.array(input_seq_batch_sd).astype(int)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)
        input_seq_batch_sd_t = torch.LongTensor(input_seq_batch_sd_n).to(device)

        model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_bwd(input_seq_batch_t, input_seq_batch_len_t, input_seq_batch_sd_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == seq_length - 1:

            candidate_seq = input_seq.copy()
            candidate_seq.append(start_poi)
            candidate_seq = list(reversed(candidate_seq))
            candidate_sequences.append(candidate_seq)
            candidate_seq_logprobs.append(get_traj_perplexity(candidate_seq))
            break

        else:
            output_prb[-1][start_poi] = 0
            for poi in input_seq:
                output_prb[-1][int(poi)] = 0

            selected_poi = np.argmax(output_prb, axis=-1)[-1]
            input_seq.append(selected_poi)

    selected_traj_idx = np.argmin(candidate_seq_logprobs)

    return candidate_sequences[int(selected_traj_idx)], candidate_seq_logprobs[int(selected_traj_idx)]


def get_sequence(start_poi, middle_poi, end_poi, min_seq_length, max_seq_length):
    candidate_sequences = []
    candidate_seq_logprobs = []

    for seq_length in range(min_seq_length, max_seq_length + 1):
        candidate_sequence, candidate_seq_logprob = get_fixed_length_sequence(start_poi, middle_poi, end_poi,
                                                                              seq_length)
        candidate_sequences.append(candidate_sequence)
        candidate_seq_logprobs.append(candidate_seq_logprob)

    selected_traj_idx = np.argmin(candidate_seq_logprobs)
    return candidate_sequences[int(selected_traj_idx)]
