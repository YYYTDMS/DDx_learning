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
from sklearn.metrics import f1_score

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

set_seed(2026)

with open('./binary_train_codes_x.pkl', 'rb') as f0:
    binary_train_codes_x = pickle.load(f0)
with open('./binary_test_codes_x.pkl', 'rb') as f1:
    binary_test_codes_x = pickle.load(f1)
with open('./binary_drug_train_codes_x.pkl', 'rb') as f_m0:
    binary_train_codes_meds = pickle.load(f_m0)
with open('./binary_drug_test_codes_x.pkl', 'rb') as f_m1:
    binary_test_codes_meds = pickle.load(f_m1)
with open('./binary_proc_train_codes_x.pkl', 'rb') as f_p0:
    binary_train_codes_procs = pickle.load(f_p0)
with open('./binary_proc_test_codes_x.pkl', 'rb') as f_p1:
    binary_test_codes_procs = pickle.load(f_p1)
with open('./patient_time_duration_encoded.pkl', 'rb') as f80:
    patient_time_duration_encoded = pickle.load(f80)

train_codes_y = np.load('./train_codes_y.npy')
train_visit_lens = np.load('./train_visit_lens.npy')
test_codes_y = np.load('./test_codes_y.npy')
test_visit_lens = np.load('./test_visit_lens.npy')
train_pids = np.load('./train_pids.npy')
test_pids = np.load('./test_pids.npy')

DIAG_VOCAB = 1958
MED_VOCAB = 131
PROC_VOCAB = 1430

def transform_and_pad_input(x):
    tempX = []
    for ele in x:
        tempX.append(torch.tensor(ele).to(torch.float32))
    x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
    return x_padded

trans_y_train = torch.tensor(train_codes_y)
trans_y_test = torch.tensor(test_codes_y)
full_raw_x = list(binary_train_codes_x) + list(binary_test_codes_x)
full_raw_meds = list(binary_train_codes_meds) + list(binary_test_codes_meds)
full_raw_procs = list(binary_train_codes_procs) + list(binary_test_codes_procs)
padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
padded_X_test_meds = torch.transpose(transform_and_pad_input(binary_test_codes_meds), 1, 2)
padded_X_test_procs = torch.transpose(transform_and_pad_input(binary_test_codes_procs), 1, 2)
padded_X_full = torch.transpose(transform_and_pad_input(full_raw_x), 1, 2)
padded_X_full_meds = torch.transpose(transform_and_pad_input(full_raw_meds), 1, 2)
padded_X_full_procs = torch.transpose(transform_and_pad_input(full_raw_procs), 1, 2)
trans_y_full = torch.cat((trans_y_train, trans_y_test), dim=0)
full_pids = np.concatenate((train_pids, test_pids))
full_visit_lens = np.concatenate((train_visit_lens, test_visit_lens))
class_num = train_codes_y.shape[1]
total_pids = list(train_pids) + list(test_pids)
cur_max = 0
for pid in total_pids:
    duration = patient_time_duration_encoded[pid]
    ts = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
    if cur_max < max(ts):
        cur_max = max(ts)

class ProHealth_Dataset(data.Dataset):
    def __init__(self, hyperG, meds, procs, data_label, pid, data_len):
        self.hyperG = hyperG
        self.meds = meds
        self.procs = procs
        self.data_label = data_label
        self.pid = pid
        self.data_len = data_len

    def __len__(self):
        return len(self.hyperG)

    def __getitem__(self, idx):
        return self.hyperG[idx], self.meds[idx], self.procs[idx], self.data_label[idx], self.pid[idx], self.data_len[idx]

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
    def __init__(self, visit_dim, hdim, diag_vocab, med_vocab, proc_vocab):
        super(Encoder, self).__init__()
        self.diag_embed_layer = HierarchicalEmbedding(diag_vocab, 128)
        self.med_embed_layer = HierarchicalEmbedding(med_vocab, 128)
        self.proc_embed_layer = HierarchicalEmbedding(proc_vocab, 128)
        self.diag_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.med_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.proc_gru = nn.GRU(visit_dim, hdim, 1, batch_first=True)
        self.diag_attention_context = nn.Linear(hdim, 1, bias=False)
        self.med_attention_context = nn.Linear(hdim, 1, bias=False)
        self.proc_attention_context = nn.Linear(hdim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hdim * 3, hdim * 2),
            nn.ReLU(),
            nn.Linear(hdim * 2, hdim),
            nn.LayerNorm(hdim)
        )

    def forward(self, H_diag, H_med, H_proc):
        X_d = self.diag_embed_layer()
        X_m = self.med_embed_layer()
        X_p = self.proc_embed_layer()
        visit_emb_d = torch.matmul(H_diag.T.to(torch.float32), X_d)
        visit_emb_m = torch.matmul(H_med.T.to(torch.float32), X_m)
        visit_emb_p = torch.matmul(H_proc.T.to(torch.float32), X_p)
        hidden_d, _ = self.diag_gru(visit_emb_d)
        hidden_m, _ = self.med_gru(visit_emb_m)
        hidden_p, _ = self.proc_gru(visit_emb_p)
        alpha_d = self.softmax(torch.squeeze(self.diag_attention_context(hidden_d), 1))
        alpha_m = self.softmax(torch.squeeze(self.med_attention_context(hidden_m), 1))
        alpha_p = self.softmax(torch.squeeze(self.proc_attention_context(hidden_p), 1))
        h_d = torch.sum(torch.matmul(torch.diag(alpha_d), hidden_d), 0)
        h_m = torch.sum(torch.matmul(torch.diag(alpha_m), hidden_m), 0)
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
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()

    def forward(self, z, timestamps):
        pred_z = odeint(func=self.odefunc, y0=z, t=timestamps, method='rk4', options=dict(step_size=0.1))
        output = self.softmax(self.final_layer(pred_z))
        return output

class ProHealth_VAE(nn.Module):
    def __init__(self, visit_dim, hdim, nclass, dist_dim, ode_hid, diag_emb, drug_emb, proc_emb):
        super(ProHealth_VAE, self).__init__()
        self.encoder = Encoder(visit_dim, hdim, diag_emb, drug_emb, proc_emb)
        self.ODE_Func = GRUODECell_Autonomous(hdim * 1)
        self.decoder = ODE_VAE_Decoder(hdim, dist_dim, nclass, self.ODE_Func)
        self.softmax = nn.Softmax()

    def forward(self, Hs_diag, Hs_drug, Hs_proc, timestamps, seq_lens):
        h_list = []
        for ii in range(len(Hs_diag)):
            h_encoded = self.encoder(
                Hs_diag[ii][:, 0:int(seq_lens[ii])],
                Hs_drug[ii][:, 0:int(seq_lens[ii])],
                Hs_proc[ii][:, 0:int(seq_lens[ii])]
            )
            h_list.append(h_encoded)
        h = torch.vstack(h_list)
        mu = self.decoder.fc_mu(h)
        log_var = self.decoder.fc_var(h)
        z = self.decoder.reparameterize(mu, log_var)
        zi = z
        pred_z = odeint(func=self.decoder.odefunc, y0=zi, t=timestamps, method='rk4', options=dict(step_size=0.1))
        pred2 = self.decoder.softmax(self.decoder.final_layer(pred_z))
        pred2 = torch.swapaxes(pred2, 0, 1)
        mug = mu
        log_varg = log_var
        ELBO = torch.mean(-0.5 * torch.sum(1 + log_varg - mug ** 2 - log_varg.exp(), dim=1))
        return 0, 0, pred2, mug, log_varg, ELBO

    def predict(self, Hs_diag, Hs_drug, Hs_proc, timestamps, seq_lens):
        h_list = []
        for ii in range(len(Hs_diag)):
            h_encoded = self.encoder(
                Hs_diag[ii][:, 0:int(seq_lens[ii])],
                Hs_drug[ii][:, 0:int(seq_lens[ii])],
                Hs_proc[ii][:, 0:int(seq_lens[ii])]
            )
            h_list.append(h_encoded)
        h = torch.vstack(h_list)
        diff_loss = 0
        mu = self.decoder.fc_mu(h)
        log_var = self.decoder.fc_var(h)
        z = self.decoder.reparameterize(mu, log_var)
        zi = z
        pred2 = self.decoder(zi, timestamps)
        pred2 = torch.swapaxes(pred2, 0, 1)
        return diff_loss, pred2

def ProHealth_loss(pred, truth, past, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max, ELBO):
    criterion = nn.BCELoss()
    if not ode_gate:
        loss = criterion(pred, truth)
    else:
        reconstruct_loss = 0
        last_visits = []
        for i, traj in enumerate(pred):
            duration = duration_dict[pids[i].item()]
            temp = [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            ts = [stamp / cur_max for stamp in temp]
            idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
            visit_lens = len(ts)
            last_visits.append(traj[idx[-1], :])
            reconstruct_loss += criterion(traj[idx[:-1], :], torch.swapaxes(past[i][:, 0:(visit_lens - 1)], 0, 1))
        last_visits = torch.stack(last_visits)
        reconstruct_loss = (reconstruct_loss / len(pred))
        pred_loss = criterion(last_visits, truth)
        loss = pred_loss + balance * ELBO + reconstruct_loss
    return loss

def f1(y_true_hot, y_pred):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average='weighted', zero_division=0)

def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)

def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20, 30]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))
    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds

def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks

def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, balance, cur_max):
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    best_metric_r1 = 0
    test_loss_per_epoch = []
    train_average_loss_per_epoch = []
    r1_list = []
    r2_list = []
    n1_list = []
    code_10_list = []
    code_20_list = []
    code_30_list = []
    visit_10_list = []
    visit_20_list = []
    visit_30_list = []
    for epoch in range(num_epoch):
        start = time.time()
        one_epoch_train_loss = []
        for i, (hyperGs, meds, procs, labels, pids, seq_lens) in enumerate(train_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            meds_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    meds_vae.append(meds[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            meds_vae = torch.stack(meds_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            timestamps = []
            for pid in pids_vae:
                duration = duration_dict[pid.item()]
                timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            temp = [stamp / cur_max for stamp in list(set(timestamps))]
            timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
            loss3, pred1, pred2, mu, log_var, ELBO = model(hyperGs_vae, meds_vae, procs_vae, timestamps, seq_lens_vae)
            loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, mu, log_var, duration_dict, timestamps, ode_gate, 0, cur_max, ELBO)
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
        test_data_len = 0
        pred_list = []
        truth_list = []
        temp_test_loss_per_epoch = []
        for i, (hyperGs, meds, procs, labels, pids, seq_lens) in enumerate(test_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            meds_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    meds_vae.append(meds[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            meds_vae = torch.stack(meds_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            with torch.no_grad():
                timestamps = []
                for pid in pids_vae:
                    duration = duration_dict[pid.item()]
                    timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
                temp = [stamp / cur_max for stamp in list(set(timestamps))]
                timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
                test_diff_loss, pred2 = model.predict(hyperGs_vae, meds_vae, procs_vae, timestamps, seq_lens_vae)
                test_loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae, pids_vae, 0, 0, duration_dict, timestamps, ode_gate, 0, cur_max, 0)
                test_loss = test_diff_loss * 0.01 + test_loss2
                temp_test_loss_per_epoch.append(test_loss.item())
            test_data_len += len(pids_vae)
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
        test_loss_per_epoch.append(sum(temp_test_loss_per_epoch))
        pred = torch.vstack(pred_list)
        truth = torch.vstack(truth_list)
        pred = torch.argsort(pred, dim=-1, descending=True)
        preds = pred.detach().cpu().numpy()
        labels = truth.detach().cpu().numpy()
        f1_score1 = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        code_scores = code_level(labels, preds)
        visit_scores = visit_level(labels, preds)
        code_10_list.append(code_scores[0])
        code_20_list.append(code_scores[1])
        code_30_list.append(code_scores[2])
        visit_10_list.append(visit_scores[0])
        visit_20_list.append(visit_scores[1])
        visit_30_list.append(visit_scores[2])
        r1_list.append(recall[0])
        r2_list.append(recall[1])
        n1_list.append(f1_score1)
        print("cur:", "Recall@10", r1_list[-1], "Recall@20:", r2_list[-1], "F1:", n1_list[-1])
        print("Code-level Recall@10/20/30:", code_scores)
        print("Visit-level Precision@10/20/30:", visit_scores)
        if recall[0] > best_metric_r1:
            best_metric_r1 = recall[0]
            best_index = len(r1_list) - 1
            torch.save(model.state_dict(), f'{model_directory}/m3_5_DDx_best_model.pth')
        print("best:", "Recall@10", r1_list[best_index], "Recall@20:", r2_list[best_index], "F1:", n1_list[best_index],
              "| Code@10/20/30:",
              code_10_list[best_index],
              code_20_list[best_index],
              code_30_list[best_index],
              "| Visit@10/20/30:",
              visit_10_list[best_index],
              visit_20_list[best_index],
              visit_30_list[best_index], )
        print("time:", end)
        model.train()
    with open(f"{model_directory}/m3_5_DDx_nn_train_average_loss_per_epoch.pkl", "wb") as f:
        pickle.dump(train_average_loss_per_epoch, f)
    with open(f"{model_directory}/m3_5_DDx_nn_test_average_loss_per_epoch.pkl", "wb") as f:
        pickle.dump(test_loss_per_epoch, f)
    return r1_list, r2_list, n1_list


def Test(model, test_loader, duration_dict, cur_max, ode_gate, save_path=None):
    model.eval()
    pred_list = []
    truth_list = []
    output_results = {}
    all_pids_order = []
    with torch.no_grad():
        for i, (hyperGs, meds, procs, labels, pids, seq_lens) in enumerate(test_loader):
            hyperGs = hyperGs.to(device)
            labels = labels.to(device)
            hyperGs_vae = []
            meds_vae = []
            procs_vae = []
            labels_vae = []
            pids_vae = []
            seq_lens_vae = []
            for patient_num in range(len(labels)):
                if seq_lens[patient_num] > 1:
                    hyperGs_vae.append(hyperGs[patient_num])
                    meds_vae.append(meds[patient_num])
                    procs_vae.append(procs[patient_num])
                    labels_vae.append(labels[patient_num])
                    pids_vae.append(pids[patient_num])
                    seq_lens_vae.append(seq_lens[patient_num])
            if not hyperGs_vae: continue
            hyperGs_vae = torch.stack(hyperGs_vae).to(device)
            meds_vae = torch.stack(meds_vae).to(device)
            procs_vae = torch.stack(procs_vae).to(device)
            labels_vae = torch.stack(labels_vae).to(device)
            timestamps = []
            for pid in pids_vae:
                duration = duration_dict[pid.item()]
                timestamps += [sum(duration[0:gap + 1]) for gap in range(len(duration))]
            temp = [stamp / cur_max for stamp in list(set(timestamps))]
            timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
            _, pred2 = model.predict(hyperGs_vae, meds_vae, procs_vae, timestamps, seq_lens_vae)
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
    pred = torch.vstack(pred_list)
    truth = torch.vstack(truth_list)
    pred_prob = pred.cpu().numpy()
    for idx, pid_val in enumerate(all_pids_order):
        output_results[pid_val] = pred_prob[idx]
    y_gt = truth.cpu().numpy().astype(int)
    y_pred_indices = np.argsort(-pred_prob, axis=1)
    recall, precision = top_k_prec_recall(y_gt, y_pred_indices, [10, 20])
    code_acc = code_level(y_gt, y_pred_indices)
    visit_prec = visit_level(y_gt, y_pred_indices)
    f1_val = f1(y_gt, y_pred_indices)
    print("-" * 50)
    print(f"Recall@10, 20: {recall}")
    print(f"Precision@10, 20: {precision}")
    print(f"Code Level Acc@10, 20, 30: {code_acc}")
    print(f"Visit Level Prec@10, 20, 30: {visit_prec}")
    print(f"F1 Score: {f1_val:.4f}")
    print("-" * 50)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(output_results, f)
    return

full_train_dataset = ProHealth_Dataset(padded_X_full, padded_X_full_meds, padded_X_full_procs, trans_y_full, full_pids, full_visit_lens)
full_train_loader = DataLoader(full_train_dataset, batch_size=128, shuffle=False)
test_dataset = ProHealth_Dataset(padded_X_test, padded_X_test_meds, padded_X_test_procs, trans_y_test, test_pids, test_visit_lens)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
model = ProHealth_VAE(128, 128, class_num, 128, 128, DIAG_VOCAB, MED_VOCAB, PROC_VOCAB).to(device)
te_directory = None
model_directory = './log/DDx_0.05'
r1_list, r2_list, n1_list = train(model, 0.0001 / 2, 500, full_train_loader, test_loader, model_directory, True, patient_time_duration_encoded, 0.5, cur_max)

best_model_path = f'{model_directory}/m3_5_DDx_best_model.pth'
model.load_state_dict(torch.load(best_model_path))
save_file_path = f'./result/ode_predictions_probs_DDx_0.05.pkl'
Test(model, test_loader, patient_time_duration_encoded, cur_max, True, save_path=save_file_path)
print("All Done!")