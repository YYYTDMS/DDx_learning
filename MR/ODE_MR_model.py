import time
import math
import pickle as pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
import random
import warnings
import dill
import json
import pandas as pd
import os
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
device = torch.device("cuda:0")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

with open('./binary_train_codes_x.pkl', 'rb') as f0:
    binary_train_codes_x = pickle.load(f0)
with open('./binary_test_codes_x.pkl', 'rb') as f1:
    binary_test_codes_x = pickle.load(f1)
with open('./binary_val_codes_x.pkl', 'rb') as f2:
    binary_val_codes_x = pickle.load(f2)
with open('./binary_diag_train_codes_x.pkl', 'rb') as f_m0:
    binary_train_codes_diags = pickle.load(f_m0)
with open('./binary_diag_test_codes_x.pkl', 'rb') as f_m1:
    binary_test_codes_diags = pickle.load(f_m1)
with open('./binary_diag_val_codes_x.pkl', 'rb') as f_m2:
    binary_val_codes_diags = pickle.load(f_m2)
with open('./binary_proc_train_codes_x.pkl', 'rb') as f_p0:
    binary_train_codes_procs = pickle.load(f_p0)
with open('./binary_proc_test_codes_x.pkl', 'rb') as f_p1:
    binary_test_codes_procs = pickle.load(f_p1)
with open('./binary_proc_val_codes_x.pkl', 'rb') as f_p2:
    binary_val_codes_procs = pickle.load(f_p2)
with open('./patient_time_duration_encoded.pkl', 'rb') as f80:
    patient_time_duration_encoded = pickle.load(f80)
with open('./ddi_A_final.pkl', "rb") as f:
    ddi_adj = dill.load(f)
with open('./voc_final.pkl', 'rb') as voc_file:
    voc_data = dill.load(voc_file)

train_visit_lens = np.load('./train_visit_lens.npy')
val_visit_lens = np.load('./val_visit_lens.npy')
test_visit_lens = np.load('./test_visit_lens.npy')
train_pids = np.load('./train_pids.npy')
val_pids = np.load('./val_pids.npy')
test_pids = np.load('./test_pids.npy')
train_codes_y = np.load('./train_codes_y.npy')
val_codes_y = np.load('./val_codes_y.npy')
test_codes_y = np.load('./test_codes_y.npy')

DIAG_VOCAB = 1958
MED_VOCAB = 131
PROC_VOCAB = 1430

ddi_adj_tensor = torch.tensor(ddi_adj, dtype=torch.float32).to(device)
diag_voc = voc_data['diag_voc']
idx_to_icd = diag_voc.idx2word
icd_to_idx = diag_voc.word2idx

df_diag = pd.read_csv('./filter_diagnosis_icd9_ontology.csv')
icd2name = dict(zip(df_diag['code'], df_diag['name']))
name_to_icd = dict(zip(df_diag['name'].str.lower().str.strip(), df_diag['code']))

def get_index_from_name(name):
    clean_name = name.strip().lower()
    icd_code = name_to_icd.get(clean_name)
    if icd_code is None:
        return None
    idx = icd_to_idx.get(icd_code)
    return idx

def get_name_from_index(idx):
    icd = idx_to_icd.get(idx)
    if icd:
        return icd2name.get(icd, "")
    return ""

ode_DP_test_pred_path = './result/DDx_0.05/ode_predictions_probs_DDx_0.05.pkl'
with open(ode_DP_test_pred_path, 'rb') as f:
    ode_dp_test_probs_dict = pickle.load(f)

json_test_file_path = './result/potential_diagnosis_top10_test.json'
with open(json_test_file_path, 'r', encoding='utf-8') as f_json:
    llm_predictions_test = json.load(f_json)

llm_potential_dict = {item['pid']: item for item in llm_predictions_test}

processed_count_test = 0
filtered_out_count_test = 0
injected_disease_count_test = 0

for i in range(len(test_pids)):
    pid = test_pids[i]
    if pid not in ode_dp_test_probs_dict:
        continue

    dp_probs = ode_dp_test_probs_dict[pid]
    current_visit_vector = binary_test_codes_diags[i][-1]
    target_diag_indices = set(np.where(current_visit_vector > 0)[0])
    topk_dp_indices = np.argsort(dp_probs)[-10:][::-1]

    valid_candidate_names_set = set()

    for idx in topk_dp_indices:
        if idx not in target_diag_indices:
            name = get_name_from_index(idx)
            if name:
                valid_candidate_names_set.add(name.lower().strip())

    if pid not in llm_potential_dict:
        continue

    raw_llm_str = llm_potential_dict[pid].get("llm_pred_potential_diseases", "")
    if not raw_llm_str or "Answer: None" in raw_llm_str:
        continue

    clean_str = raw_llm_str.replace("Answer:", "").strip()
    llm_diseases = clean_str.split(";")
    indices_to_inject = []

    for disease_raw in llm_diseases:
        d_name = disease_raw.strip()
        d_name_clean = d_name.replace('"', '').replace("'", "").strip().lower()

        if not d_name_clean:
            continue

        if d_name_clean in valid_candidate_names_set:
            idx = get_index_from_name(d_name)
            if idx is not None:
                indices_to_inject.append(idx)
        else:
            filtered_out_count_test += 1

    if indices_to_inject:
        for new_idx in indices_to_inject:
            if new_idx < len(current_visit_vector):
                if binary_test_codes_diags[i][-1][new_idx] == 0.0:
                    binary_test_codes_diags[i][-1][new_idx] = 1.0
                    injected_disease_count_test += 1
        processed_count_test += 1

ode_DP_train_pred_path = './result/DDx_0.05_train/ode_predictions_probs_DDx_0.05.pkl'
with open(ode_DP_train_pred_path, 'rb') as f:
    ode_dp_train_probs_dict = pickle.load(f)

json_file_path = './result/potential_diagnosis_top10_train.json'
with open(json_file_path, 'r', encoding='utf-8') as f_json:
    llm_predictions_train = json.load(f_json)

llm_potential_dict = {item['pid']: item for item in llm_predictions_train}

processed_count_train = 0
filtered_out_count_train = 0
injected_disease_count_train = 0

for i in range(len(train_pids)):
    pid = train_pids[i]
    if pid not in ode_dp_train_probs_dict:
        continue

    dp_probs = ode_dp_train_probs_dict[pid]
    current_visit_vector = binary_train_codes_diags[i][-1]
    target_diag_indices = set(np.where(current_visit_vector > 0)[0])
    topk_dp_indices = np.argsort(dp_probs)[-10:][::-1]

    valid_candidate_names_set = set()

    for idx in topk_dp_indices:
        if idx not in target_diag_indices:
            name = get_name_from_index(idx)
            if name:
                valid_candidate_names_set.add(name.lower().strip())

    if pid not in llm_potential_dict:
        continue

    raw_llm_str = llm_potential_dict[pid].get("llm_pred_potential_diseases", "")
    if not raw_llm_str or "Answer: None" in raw_llm_str:
        continue

    clean_str = raw_llm_str.replace("Answer:", "").strip()
    llm_diseases = clean_str.split(";")
    indices_to_inject = []

    for disease_raw in llm_diseases:
        d_name = disease_raw.strip()
        d_name_clean = d_name.replace('"', '').replace("'", "").strip().lower()

        if not d_name_clean:
            continue

        if d_name_clean in valid_candidate_names_set:
            idx = get_index_from_name(d_name)
            if idx is not None:
                indices_to_inject.append(idx)
        else:
            filtered_out_count_train += 1

    if indices_to_inject:
        for new_idx in indices_to_inject:
            if new_idx < len(current_visit_vector):
                if binary_train_codes_diags[i][-1][new_idx] == 0.0:
                    binary_train_codes_diags[i][-1][new_idx] = 1.0
                    injected_disease_count_train += 1
        processed_count_train += 1

def transform_and_pad_input(x):
    tempX = []
    for ele in x:
        tempX.append(torch.tensor(ele).to(torch.float32))
    x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
    return x_padded

trans_y_train = torch.tensor(train_codes_y)
trans_y_val = torch.tensor(val_codes_y)
trans_y_test = torch.tensor(test_codes_y)
padded_X_train_drugs = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
padded_X_val_drugs = torch.transpose(transform_and_pad_input(binary_val_codes_x), 1, 2)
padded_X_test_drugs = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
padded_X_train_diags = torch.transpose(transform_and_pad_input(binary_train_codes_diags), 1, 2)
padded_X_val_diags = torch.transpose(transform_and_pad_input(binary_val_codes_diags), 1, 2)
padded_X_test_diags = torch.transpose(transform_and_pad_input(binary_test_codes_diags), 1, 2)
padded_X_train_procs = torch.transpose(transform_and_pad_input(binary_train_codes_procs), 1, 2)
padded_X_val_procs = torch.transpose(transform_and_pad_input(binary_val_codes_procs), 1, 2)
padded_X_test_procs = torch.transpose(transform_and_pad_input(binary_test_codes_procs), 1, 2)
class_num = train_codes_y.shape[1]
total_pids = list(train_pids) + list(val_pids) + list(test_pids)
cur_max = 0
for pid in total_pids:
    duration = patient_time_duration_encoded[pid]
    ts = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
    if cur_max < max(ts):
        cur_max = max(ts)

class ProHealth_Dataset(data.Dataset):
    def __init__(self, hyperG, diags, procs, data_label, pid, data_len):
        self.hyperG = hyperG
        self.diags = diags
        self.procs = procs
        self.data_label = data_label
        self.pid = pid
        self.data_len = data_len

    def __len__(self):
        return len(self.hyperG)

    def __getitem__(self, idx):
        return self.hyperG[idx], self.diags[idx], self.procs[idx], self.data_label[idx], self.pid[idx], self.data_len[idx]

class HierarchicalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(HierarchicalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.level_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.level_embeddings.weight)

    def forward(self):
        return self.level_embeddings.weight

class Encoder(nn.Module):
    def __init__(self, visit_dim, hdim, med_vocab, diag_vocab, proc_vocab):
        super(Encoder, self).__init__()
        self.med_embed_layer = HierarchicalEmbedding(med_vocab, 128)
        self.diag_embed_layer = HierarchicalEmbedding(diag_vocab, 128)
        self.proc_embed_layer = HierarchicalEmbedding(proc_vocab, 128)
        self.med_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.diag_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.proc_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.med_attention_context = nn.Linear(hdim, 1, bias=False)
        self.diag_attention_context = nn.Linear(hdim, 1, bias=False)
        self.proc_attention_context = nn.Linear(hdim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hdim * 3, hdim * 2),
            nn.ReLU(),
            nn.Linear(hdim * 2, hdim),
            nn.LayerNorm(hdim)
        )

    def forward(self, H_med, H_diag, H_proc):
        X_m = self.med_embed_layer()
        X_d = self.diag_embed_layer()
        X_p = self.proc_embed_layer()
        visit_emb_m = torch.matmul(H_med.T.to(torch.float32), X_m)
        visit_emb_d = torch.matmul(H_diag.T.to(torch.float32), X_d)
        visit_emb_p = torch.matmul(H_proc.T.to(torch.float32), X_p)
        hidden_m, _ = self.med_gru(visit_emb_m)
        hidden_d, _ = self.diag_gru(visit_emb_d)
        hidden_p, _ = self.proc_gru(visit_emb_p)
        alpha_m = self.softmax(torch.squeeze(self.med_attention_context(hidden_m), 1))
        alpha_d = self.softmax(torch.squeeze(self.diag_attention_context(hidden_d), 1))
        alpha_p = self.softmax(torch.squeeze(self.proc_attention_context(hidden_p), 1))
        h_m = torch.sum(torch.matmul(torch.diag(alpha_m), hidden_m), 0)
        h_d = torch.sum(torch.matmul(torch.diag(alpha_d), hidden_d), 0)
        h_p = torch.sum(torch.matmul(torch.diag(alpha_p), hidden_p), 0)
        h_concat = torch.cat([h_d, h_m, h_p], dim=-1)
        h_final = self.fusion_mlp(h_concat)
        return h_final

class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))
        dh = (1 - z) * (n - h)
        return dh

class ODEFunc(nn.Module):
    def __init__(self, hdim, ode_hid):
        super().__init__()
        self.func = nn.Sequential(nn.Linear(hdim, ode_hid),
                                  nn.Tanh(),
                                  nn.Linear(ode_hid, hdim))
        for m in self.func.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        output = self.func(y)
        return output

class ODE_VAE_Decoder(nn.Module):
    def __init__(self, hdim, dist_dim, nclass, ODE_Func):
        super(ODE_VAE_Decoder, self).__init__()
        self.fc_mu = nn.Linear(hdim, dist_dim)
        self.fc_var = nn.Linear(hdim, dist_dim)
        self.fc_mu0 = nn.Linear(hdim, dist_dim)
        self.fc_var0 = nn.Linear(hdim, dist_dim)
        self.relu = nn.ReLU()
        self.odefunc = ODE_Func
        self.final_layer = nn.Linear(hdim * 1, nclass)
        self.softmax = nn.Softmax(dim=-1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()

    def forward(self, z, timestamps):
        pred_z = odeint(func=self.odefunc, y0=z, t=timestamps, method='rk4', options=dict(step_size=0.1))
        output = self.final_layer(pred_z)
        return output

class ProHealth_VAE(nn.Module):
    def __init__(self, visit_dim, hdim, nclass, dist_dim, ode_hid, drug_emb, diag_emb, proc_emb):
        super(ProHealth_VAE, self).__init__()
        self.encoder = Encoder(visit_dim, hdim, drug_emb, diag_emb, proc_emb)
        self.ODE_Func = GRUODECell_Autonomous(hdim * 1)
        self.decoder = ODE_VAE_Decoder(hdim, dist_dim, nclass, self.ODE_Func)
        self.softmax = nn.Softmax()

    def forward(self, Hs_drug, Hs_diag, Hs_proc, timestamps, seq_lens):
        h_list = []
        for ii in range(len(Hs_diag)):
            h_encoded = self.encoder(
                Hs_drug[ii][:, 0:int(seq_lens[ii])],
                Hs_diag[ii][:, 0:int(seq_lens[ii])+1],
                Hs_proc[ii][:, 0:int(seq_lens[ii])+1]
            )
            h_list.append(h_encoded)
        h = torch.vstack(h_list)
        mu = self.decoder.fc_mu(h)
        log_var = self.decoder.fc_var(h)
        z = self.decoder.reparameterize(mu, log_var)
        zi = z
        pred_z = odeint(func=self.decoder.odefunc, y0=zi, t=timestamps, method='rk4', options=dict(step_size=0.1))
        pred2 = self.decoder.final_layer(pred_z)
        pred2 = torch.swapaxes(pred2, 0, 1)
        mug = mu
        log_varg = log_var
        ELBO = torch.mean(-0.5 * torch.sum(1 + log_varg - mug ** 2 - log_varg.exp(), dim=1))
        return 0, 0, pred2, mug, log_varg, ELBO

    def predict(self, Hs_drug, Hs_diag, Hs_proc, timestamps, seq_lens):
        h_list = []
        for ii in range(len(Hs_diag)):
            h_encoded = self.encoder(
                Hs_drug[ii][:, 0:int(seq_lens[ii])],
                Hs_diag[ii][:, 0:int(seq_lens[ii])+1],
                Hs_proc[ii][:, 0:int(seq_lens[ii])+1]
            )
            h_list.append(h_encoded)
        h = torch.vstack(h_list)
        diff_loss = 0
        mu = self.decoder.fc_mu(h)
        log_var = self.decoder.fc_var(h)
        z = self.decoder.reparameterize(mu, log_var)
        zi = z
        pred_z = odeint(func=self.decoder.odefunc, y0=zi, t=timestamps, method='rk4', options=dict(step_size=0.1))
        pred2 = self.decoder.final_layer(pred_z)
        pred2 = torch.swapaxes(pred2, 0, 1)
        return diff_loss, pred2

def ProHealth_loss(pred, truth, past, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max, ELBO, ddi_adj_matrix):
    criterion = nn.BCELoss()
    pred_prob = torch.sigmoid(pred)
    target_pred_prob = None
    loss_classification = 0
    if not ode_gate:
        loss = criterion(pred, truth)
    else:
        reconstruct_loss = 0
        last_visits_logits = []
        for i, traj in enumerate(pred):
            duration = duration_dict[pids[i].item()]
            temp = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            ts = [stamp / cur_max for stamp in temp]
            idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
            visit_lens = len(ts)
            traj_prob = torch.sigmoid(traj)
            last_visits_logits.append(traj[idx[-1], :])
            reconstruct_loss += criterion(traj_prob[idx[:-1], :], torch.swapaxes(past[i][:, 0:(visit_lens - 1)], 0, 1))
        last_visits_logits = torch.stack(last_visits_logits)
        reconstruct_loss = (reconstruct_loss / len(pred))
        target_pred_prob = torch.sigmoid(last_visits_logits)
        loss_bce = criterion(target_pred_prob, truth)
        truth_indices = []
        device = truth.device
        for row in truth:
            indices = torch.nonzero(row).squeeze(1)
            padded = torch.full((truth.shape[1],), -1, dtype=torch.long, device=device)
            if len(indices) > 0:
                padded[:len(indices)] = indices
            truth_indices.append(padded)
        truth_indices = torch.stack(truth_indices)
        loss_multi = F.multilabel_margin_loss(target_pred_prob, truth_indices)
        loss_classification = 0.8 * loss_bce + 0.2 * loss_multi + balance * ELBO + reconstruct_loss
        mm = torch.mm(target_pred_prob, ddi_adj_matrix)
        loss_ddi_batch = (mm * target_pred_prob).sum(dim=1)
        loss_ddi = loss_ddi_batch.mean() * 0.005
    return loss_classification + loss_ddi

def ddi_rate_score(record, ddi_A):
    if ddi_A is None:
        return 0.0
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

def calculate_metrics(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)
    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score
    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score
    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2 * average_prc[idx] * average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score
    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)
    ja = jaccard(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def train(model, lrate, num_epoch, train_loader, val_loader, model_directory, ode_gate, duration_dict, balance, cur_max):
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    best_metric_ja = 0
    val_loss_per_epoch = []
    train_average_loss_per_epoch = []
    ja_list = []
    ddi_list = []
    f1_list = []
    global ddi_adj_tensor
    for epoch in range(num_epoch):
        start = time.time()
        one_epoch_train_loss = []
        for i, (hyperGs, diags, procs, labels, pids, seq_lens) in enumerate(train_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            diags_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    diags_vae.append(diags[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            diags_vae = torch.stack(diags_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            timestamps = []
            for pid in pids_vae:
                duration = duration_dict[pid.item()]
                timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            temp = [stamp / cur_max for stamp in list(set(timestamps))]
            timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
            loss3, pred1, pred2, mu, log_var, ELBO = model(hyperGs_vae, diags_vae, procs_vae,
                timestamps, seq_lens_vae)
            loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, mu, log_var,
                                   duration_dict, timestamps, ode_gate, 0, cur_max, ELBO, ddi_adj_tensor)
            loss = loss3 * 0.01 + loss2
            one_epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        end = time.time() - start
        train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
        print('Epoch: [{}/{}], Average Loss: {}'.format(epoch + 1, num_epoch, round(train_average_loss_per_epoch[-1], 9)))
        model.eval()
        one_epoch_val_loss = []
        val_data_len = 0
        pred_list = []
        truth_list = []
        temp_val_loss_per_epoch = []
        total_med_cnt = 0
        total_visit_cnt = 0
        for i, (hyperGs, diags, procs, labels, pids, seq_lens) in enumerate(val_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            diags_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    diags_vae.append(diags[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            diags_vae = torch.stack(diags_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            with torch.no_grad():
                timestamps = []
                for pid in pids_vae:
                    duration = duration_dict[pid.item()]
                    timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
                temp = [stamp / cur_max for stamp in list(set(timestamps))]
                timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
                val_diff_loss, pred2 = model.predict(hyperGs_vae, diags_vae, procs_vae,
                    timestamps, seq_lens_vae)
                val_loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, 0, 0,
                                            duration_dict, timestamps, ode_gate, 0, cur_max, 0, ddi_adj_tensor)
                val_loss = val_diff_loss * 0.01 + val_loss2
                temp_val_loss_per_epoch.append(val_loss.item())
            val_data_len += len(pids_vae)
            truth_list.append(labels_vae)
            if ode_gate:
                for jj, traj in enumerate(pred2):
                    duration = duration_dict[pids_vae[jj].item()]
                    ts1 = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
                    ts = [stamp / cur_max for stamp in ts1]
                    idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
                    pred_list.append(traj[idx[-1], :])
            else:
                pred_list.append(pred2)
        val_loss_per_epoch.append(sum(temp_val_loss_per_epoch))
        pred = torch.vstack(pred_list)
        truth = torch.vstack(truth_list)
        pred_prob = torch.sigmoid(pred).cpu().numpy()
        y_gt = truth.cpu().numpy().astype(int)
        y_pred_bin = (pred_prob > 0.5).astype(int)
        wrapped_for_ddi = [[np.where(row == 1)[0]] for row in y_pred_bin]
        cur_ddi_rate = ddi_rate_score(wrapped_for_ddi, ddi_adj)
        ja, prauc, avg_p, avg_r, avg_f1 = calculate_metrics(y_gt, y_pred_bin, pred_prob)
        total_med_cnt = y_pred_bin.sum()
        total_visit_cnt = y_pred_bin.shape[0]
        avg_med = total_med_cnt / total_visit_cnt
        ddi_list.append(cur_ddi_rate)
        ja_list.append(ja)
        f1_list.append(avg_f1)
        print(f"DDI Rate: {cur_ddi_rate:.4f}, Jaccard: {ja:.4f}, PRAUC: {prauc:.4f}, F1: {avg_f1:.4f}, AVG_MED: {avg_med:.2f}")
        if ja > best_metric_ja:
            best_metric_ja = ja
            torch.save(model.state_dict(), f'{model_directory}/m3_5_MR_best_model.pth')
            print("New Best Model Saved!")
        model.train()
    with open(f"{model_directory}/m3_5_MR_nn_train_average_loss_per_epoch.pkl", "wb") as f:
        pickle.dump(train_average_loss_per_epoch, f)
    with open(f"{model_directory}/m3_5_MR_nn_val_average_loss_per_epoch.pkl", "wb") as f:
        pickle.dump(val_loss_per_epoch, f)
    return ddi_list, ja_list, f1_list

def Test(model, test_loader, duration_dict, cur_max, ode_gate, ddi_adj, ddi_adj_tensor):
    model.eval()
    pred_list = []
    truth_list = []
    output_results = {}
    all_pids_order = []
    with torch.no_grad():
        for i, (hyperGs, diags, procs, labels, pids, seq_lens) in enumerate(test_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            diags_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    diags_vae.append(diags[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            if not hyperGs_vae: continue
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            diags_vae = torch.stack(diags_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            timestamps = []
            for pid in pids_vae:
                duration = duration_dict[pid.item()]
                timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            temp = [stamp / cur_max for stamp in list(set(timestamps))]
            timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
            test_diff_loss, pred2 = model.predict(hyperGs_vae, diags_vae, procs_vae, timestamps, seq_lens_vae)
            truth_list.append(labels_vae)
            all_pids_order.extend([p.item() for p in pids_vae])
            if ode_gate:
                for jj, traj in enumerate(pred2):
                    duration = duration_dict[pids_vae[jj].item()]
                    ts1 = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
                    ts = [stamp / cur_max for stamp in ts1]
                    idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
                    pred_list.append(traj[idx[-1], :])
            else:
                pred_list.append(pred2)
    if not pred_list:
        print("Warning: No predictions made.")
        return
    pred = torch.vstack(pred_list)
    truth = torch.vstack(truth_list)
    pred_prob = torch.sigmoid(pred).cpu().numpy()
    for idx, pid_val in enumerate(all_pids_order):
        output_results[pid_val] = pred_prob[idx]
    y_gt = truth.cpu().numpy().astype(int)
    y_pred_bin = (pred_prob > 0.5).astype(int)
    wrapped_for_ddi = [[np.where(row == 1)[0]] for row in y_pred_bin]
    cur_ddi_rate = ddi_rate_score(wrapped_for_ddi, ddi_adj)
    ja, prauc, avg_p, avg_r, avg_f1 = calculate_metrics(y_gt, y_pred_bin, pred_prob)
    total_med_cnt = y_pred_bin.sum()
    total_visit_cnt = y_pred_bin.shape[0]
    avg_med = total_med_cnt / total_visit_cnt
    print(
        f"DDI Rate: {cur_ddi_rate:.4f}, Jaccard: {ja:.4f}, PRAUC: {prauc:.4f}, F1: {avg_f1:.4f}, AVG_MED: {avg_med:.4f}")
    save_path = './result/ode_predictions_probs_m3_0.05.pkl'
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(save_path, 'wb') as f:
        pickle.dump(output_results, f)
    return

model = ProHealth_VAE(128, 128, class_num, 128, 128, MED_VOCAB, DIAG_VOCAB, PROC_VOCAB).to(device)
training_data = ProHealth_Dataset(padded_X_train_drugs, padded_X_train_diags, padded_X_train_procs, trans_y_train, train_pids, train_visit_lens)
train_loader = DataLoader(training_data, batch_size=128, shuffle=False)
val_data = ProHealth_Dataset(padded_X_val_drugs, padded_X_val_diags, padded_X_val_procs, trans_y_val, val_pids, val_visit_lens)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
test_data = ProHealth_Dataset(padded_X_test_drugs, padded_X_test_diags, padded_X_test_procs, trans_y_test, test_pids, test_visit_lens)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
model_directory = './log/MR_0.05'
r1_list, r2_list, n1_list = train(model, 0.0001 / 2, 500, train_loader, val_loader, model_directory, True, patient_time_duration_encoded, 0.5, cur_max)
best_model_path = f'{model_directory}/m3_5_MR_best_model.pth'
model.load_state_dict(torch.load(best_model_path))
Test(model, test_loader, patient_time_duration_encoded, cur_max, True, ddi_adj, ddi_adj_tensor)
print("ALL Done!")