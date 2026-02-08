import pickle as pickle
import numpy as np
import pandas as pd
import inflect
import dill
import json
import os
import math
import re
from collections import Counter

base_path = "../data/MR_0.05/"
ode_pred_path = '../ODE_result/MR_0.05/ode_predictions_probs_m3_0.05.pkl'
ode_pred_train_path = '../ODE_result/MR_0.05_train/ode_predictions_probs_m3_0.05.pkl'
ode_DP_pred_path = '../ODE_result/DP_0.05/ode_predictions_probs_DP_0.05.pkl'
DP_pred_path = './LLM_result/DDX_0.05/DDX_List_test.json'
potential_diseases_path = "LLM_result/DDX_0.05/potential_diagnosis_top10_test.json"
ddi_path = "../data/MR_0.05/ddi_A_final.pkl"
output_json_path = "prompt_data/DDxMed_prompt.json"
train_pids = np.load(base_path + 'train_pids.npy')
test_pids = np.load(base_path + 'test_pids.npy')

with open(base_path + 'patient_time_duration_encoded.pkl', 'rb') as f80:
    patient_time_duration_encoded = pickle.load(f80)
with open(ode_pred_path, 'rb') as f:
    ode_probs_dict = pickle.load(f)
with open(ode_pred_train_path, 'rb') as f:
    ode_probs_dict_train = pickle.load(f)
with open(ode_DP_pred_path, 'rb') as f:
    ode_dp_probs_dict = pickle.load(f)
with open(base_path + 'records_final_more_time_sorted.pkl', 'rb') as f1:
    record = pickle.load(f1)
with open(base_path + 'voc_final.pkl', 'rb') as voc_file:
    voc_data = dill.load(voc_file)
with open(potential_diseases_path, "r", encoding="utf-8") as f:
    llm_potential_data = json.load(f)
with open(DP_pred_path, "r", encoding="utf-8") as f:
    dp_results_data = json.load(f)
with open(ddi_path, 'rb') as f_ddi:
    ddi_adj = dill.load(f_ddi)

llm_potential_dict = {item['pid']: item for item in llm_potential_data}
dp_pred_map = {item['pid']: item.get('DDx_list', []) for item in dp_results_data}
diag_voc = voc_data['diag_voc']
med_voc = voc_data['med_voc']
pro_voc = voc_data['pro_voc']
idx_to_atc = med_voc.idx2word
idx_to_icd = diag_voc.idx2word
idx_to_pro = pro_voc.idx2word

with open(base_path + 'med_introduction.json', "r", encoding="utf-8") as f:
    data_drug = json.load(f)
atc2name = {item["ATC3"]: item["name"] for item in data_drug}
df_diag = pd.read_csv(base_path + 'filter_diagnosis_icd9_ontology.csv')
icd2name = dict(zip(df_diag['code'], df_diag['name']))
df_pro = pd.read_csv(base_path + 'filter_procedure_icd9_ontology.csv')
pro2name = dict(zip(df_pro['code'], df_pro['name']))
all_drug_code = list(med_voc.word2idx.keys())


def fun1(idx):
    result = []
    for id in idx:
        result.append(idx_to_atc[id])
    return result

def get_drug_str(drug_list):
    temp_str_list = []
    for drug in drug_list:
        if drug in atc2name:
            temp_str_list.append(f'"{atc2name[drug]}"')
        else:
            temp_str_list.append(f'"{drug}"')
    return ', '.join(temp_str_list)

def get_diag_str(diag_list):
    temp_str_list = []
    for diag in diag_list:
        if diag in icd2name:
            temp_str_list.append(f'"{icd2name[diag]}"')
        else:
            temp_str_list.append(f'"{diag}"')
    return ', '.join(temp_str_list)

def get_pro_str(pro_list):
    temp_str_list = []
    for pro in pro_list:
        if pro in pro2name:
            temp_str_list.append(f'"{pro2name[pro]}"')
        else:
            temp_str_list.append(f'"{pro}"')
    return ', '.join(temp_str_list)

def number_to_capitalized_ordinal(n):
    p = inflect.engine()
    return p.ordinal(p.number_to_words(n)).capitalize()


def parse_llm_pred_to_str(ddx_list):
    if not isinstance(ddx_list, list) or not ddx_list:
        return "None"
    formatted_list = []
    for item in ddx_list:
        clean_name = str(item).strip()
        clean_name = clean_name.replace('"', '').replace("'", "")
        if clean_name:
            formatted_list.append(f'"{clean_name}"')
    if not formatted_list:
        return "None"
    return ', '.join(formatted_list)


def get_drug_with_prob_str(drug_list, prob_list):
    zipped = sorted(zip(drug_list, prob_list), key=lambda x: x[1], reverse=True)
    temp_str_list = []
    for drug, prob in zipped:
        if drug in atc2name:
            name = atc2name[drug]
        else:
            name = drug
        temp_str_list.append(f'"{name} ({prob:.2f})"')
    return ', '.join(temp_str_list)

def get_ddi_warnings(drug_code_list, ddi_matrix, atc2idx_map, atc2name_map, strict_filter_pairs=None):
    warnings = []
    indices = []
    valid_drugs = []
    for code in drug_code_list:
        if code in atc2idx_map:
            indices.append(atc2idx_map[code])
            valid_drugs.append(code)
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = indices[i]
            idx_j = indices[j]
            if ddi_matrix[idx_i, idx_j] == 1:
                drug_i_code = valid_drugs[i]
                drug_j_code = valid_drugs[j]
                drug_i_name = atc2name_map.get(drug_i_code, drug_i_code)
                drug_j_name = atc2name_map.get(drug_j_code, drug_j_code)
                if strict_filter_pairs is not None:
                    current_pair_set = frozenset([drug_i_name, drug_j_name])
                    if current_pair_set not in strict_filter_pairs:
                        continue
                warnings.append(f"[{drug_i_name}, {drug_j_name}]")
    if not warnings:
        return "None"
    return "; ".join(warnings)

def np_encoder(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


ode_ddi_counter = Counter()
true_ddi_counter = Counter()
analyzed_count = 0
for i in range(len(train_pids)):
    pid = train_pids[i]
    if isinstance(pid, (np.ndarray, list)):
        if len(pid) > 0:
            pid = pid[0]
        else:
            continue
    if isinstance(pid, (np.integer, np.floating)):
        pid = int(pid)
    if pid not in ode_probs_dict_train:
        continue
    analyzed_count += 1
    visits = record[pid]
    target_drug_indices = visits[-1][2]
    probs = ode_probs_dict_train[pid]
    dynamic_threshold = 0.5
    selected_indices = np.where(probs >= dynamic_threshold)[0]
    ode_list = sorted(selected_indices.tolist())
    n_ode = len(ode_list)
    for idx_a in range(n_ode):
        for idx_b in range(idx_a + 1, n_ode):
            u, v = ode_list[idx_a], ode_list[idx_b]
            if ddi_adj[u, v] == 1:
                ode_ddi_counter[(u, v)] += 1
    true_list = sorted(list(set(target_drug_indices)))
    n_true = len(true_list)
    for idx_a in range(n_true):
        for idx_b in range(idx_a + 1, n_true):
            u, v = true_list[idx_a], true_list[idx_b]
            if ddi_adj[u, v] == 1:
                true_ddi_counter[(u, v)] += 1

all_observed_pairs = set(ode_ddi_counter.keys()) | set(true_ddi_counter.keys())
diff_list = []
for pair in all_observed_pairs:
    u_idx, v_idx = pair
    count_ode = ode_ddi_counter[pair]
    count_true = true_ddi_counter[pair]
    diff = count_ode - count_true
    u_code = idx_to_atc[u_idx]
    v_code = idx_to_atc[v_idx]
    u_name = atc2name.get(u_code, u_code)
    v_name = atc2name.get(v_code, v_code)
    diff_list.append({
        "pair_indices": pair,
        "drug_A": u_name,
        "drug_B": v_name,
        "count_ode": count_ode,
        "count_true": count_true,
        "diff": diff,
        "freq_ode": count_ode / analyzed_count if analyzed_count > 0 else 0,
        "freq_true": count_true / analyzed_count if analyzed_count > 0 else 0
    })
diff_list.sort(key=lambda x: x["diff"], reverse=True)
TARGET_DDI_PAIRS = set()
top_k = 1
for i in range(min(top_k, len(diff_list))):
    item = diff_list[i]
    pair_set = frozenset([item['drug_A'], item['drug_B']])
    TARGET_DDI_PAIRS.add(pair_set)
top1_pair_names = next(iter(TARGET_DDI_PAIRS)) if TARGET_DDI_PAIRS else None

results = []
for i in range(len(test_pids)):
    pid = test_pids[i]
    if pid not in ode_probs_dict:
        continue
    visits = record[pid]
    hist_drug = [visit[2] for visit in visits[:-1]]
    target_drug = visits[-1][2]
    target_diag_indices = set(visits[-1][0])
    diag = [visit[0] for visit in visits]
    procedure = [visit[1] for visit in visits]
    diag_code = [[idx_to_icd[idx] for idx in visit_diag] for visit_diag in diag]
    pro_code = [[idx_to_pro[idx] for idx in visit_pro] for visit_pro in procedure]
    hist_drug_code = [fun1(v) for v in hist_drug]
    target_drug_code_atc = fun1(target_drug)
    target_drug_code_name = get_drug_str(target_drug_code_atc)
    actual_disease_list = []
    for idx in target_diag_indices:
        icd = idx_to_icd[idx]
        name = icd2name.get(icd, "")
        actual_disease_list.append(name)
    probs = ode_probs_dict[pid]
    dynamic_threshold = 0.5
    selected_indices = np.where(probs >= dynamic_threshold)[0]
    candidate_codes = [idx_to_atc[idx] for idx in selected_indices]
    candidate_probs = [probs[idx] for idx in selected_indices]  # 获取对应的概率

    if top1_pair_names:
        ode_drug_names = set([atc2name.get(c, c) for c in candidate_codes])
        gt_drug_names = set([atc2name.get(c, c) for c in target_drug_code_atc])
        has_pair_in_ode = top1_pair_names.issubset(ode_drug_names)
        has_pair_in_gt = top1_pair_names.issubset(gt_drug_names)
        if has_pair_in_ode and not has_pair_in_gt:
            over_recommended_pids.append(pid)
        elif has_pair_in_ode and has_pair_in_gt:
            correctly_recommended_pids.append(pid)
    hist_candidate_codes = sorted(list(set([item for sublist in hist_drug_code for item in sublist])))
    hist_candidate_probs = []
    for code in hist_candidate_codes:
        if code in med_voc.word2idx:
            idx = med_voc.word2idx[code]
            hist_candidate_probs.append(probs[idx])
        else:
            hist_candidate_probs.append(0.0)
    potential_disease_str = "None"
    potential_disease_list = []
    dp_probs = ode_dp_probs_dict[pid]
    topk_dp_indices = np.argsort(dp_probs)[-10:][::-1]
    valid_candidate_names_set = set()
    for idx in topk_dp_indices:
        if idx not in target_diag_indices:
            icd = idx_to_icd[idx]
            name = icd2name.get(icd, "")
            if name:
                valid_candidate_names_set.add(name.lower().strip())
    num_before = len(valid_candidate_names_set)
    total_potential_before_llm += num_before
    potential_disease_str = "None"
    potential_disease_list = []
    has_potential_disease = False
    if pid in llm_potential_dict:
        raw_llm_str = llm_potential_dict[pid].get("llm_pred_potential_diseases", "Answer: None")
    else:
        raw_llm_str = "Answer: None"
    cleaned_str = raw_llm_str.replace("Answer:", "").strip()
    if cleaned_str.lower() != "none" and cleaned_str != "":
        temp_diseases = [d.strip() for d in cleaned_str.split(';')]
        formatted_names = []
        for d_raw in temp_diseases:
            d_name_clean = d_raw.replace('"', '').replace("'", "").strip()
            if d_name_clean.lower() in valid_candidate_names_set:
                potential_disease_list.append(d_name_clean)
                formatted_names.append(f'"{d_name_clean}"')
        num_after = len(potential_disease_list)
        total_potential_after_llm += num_after
        if len(formatted_names) > 0:
            potential_disease_str = ", ".join(formatted_names)
            has_potential_disease = True
        else:
            has_potential_disease = False
    else:
        has_potential_disease = False

    ddi_warning_msg = get_ddi_warnings(candidate_codes, ddi_adj, med_voc.word2idx, atc2name, strict_filter_pairs=TARGET_DDI_PAIRS)
    current_ddi_count = 0
    if ddi_warning_msg and ddi_warning_msg != "None":
        current_ddi_count = ddi_warning_msg.count(';') + 1
    total_ddi_pairs_count += current_ddi_count
    if not has_potential_disease:
        instruction = "You are an experienced clinical pharmacology expert specializing in longitudinal patient histories, medication continuity, and therapeutic decision-making.\n\nTo support your decision-making, you are provided with the following information:\n1. Patient History: The patient's comprehensive clinical profile, encompassing established diagnoses, procedures, and medication regimens that reflect their long-term treatment plan.\n2. Differential Diagnosis List: Differential diagnosis is an iterative clinical reasoning process in which physicians synthesize longitudinal clinical evidence and prioritize diagnoses to determine the dominant drivers of a patient’s current clinical state. The resulting prioritized diagnosis list is referred to as the Differential Diagnosis List.\n3. ODE Model Suggested Medications: Suggestions from a neural Ordinary Differential Equation (ODE) model. This model analyzes longitudinal electronic health records to capture continuous patient trajectories and predict medication with probabilities.\n4. Historical Medications: A list of medications that the patient has been prescribed in previous visits. These medications are important to consider in the final therapeutic decision.\n5. Potential Drug–Drug Interactions (DDIs): Based on a DDI knowledge base, some recommended medications may have potential drug–drug interactions.\n\nTask Rules:\n- Medications with a final probability ≥ 0.5 will be recommended to the patient.\n- You must revise the medication recommendation probabilities of both ODE Model Suggested Medications and Historical Medications based on the information above and clinical reasoning.\n\nAnalyze the Patient History, Historical Medications, ODE Model Suggested Medications and Potential Drug–Drug Interactions under standard pharmacotherapy principles. Based on this analysis, adjust your outputs to meet the following goals:\n- Raise the probability of clinically necessary medications to the [0.50–1.00] range.\n- Lower the probability of medications that lack evidence or have low necessity.\n"
    else:
        instruction = "You are an experienced clinical pharmacology expert specializing in longitudinal patient histories, medication continuity, and therapeutic decision-making.\n\nTo support your decision-making, you are provided with the following information:\n1. Patient History: The patient's comprehensive clinical profile, encompassing established diagnoses, procedures, and medication regimens that reflect their long-term treatment plan.\n2. Differential Diagnosis List: Differential diagnosis is an iterative clinical reasoning process in which physicians synthesize longitudinal clinical evidence, identify both explicit and latent conditions, and prioritize diagnoses to determine the dominant drivers of a patient’s current clinical state. The resulting prioritized diagnosis list is referred to as the Differential Diagnosis List.\n3. Potential Diseases: Latent or subclinical conditions identified by the physician during the differential diagnosis process. These conditions may influence the prioritization of existing therapies or support the inclusion of low-risk supportive medications, but MUST NOT justify the initiation of aggressive, high-risk, or acute-care treatments.\n4. ODE Model Suggested Medications: Suggestions from a neural Ordinary Differential Equation (ODE) model. This model analyzes longitudinal electronic health records to capture continuous patient trajectories and predict medication with probabilities.\n5. Historical Medications: A list of medications that the patient has been prescribed in previous visits. These medications are important to consider in the final therapeutic decision.\n6. Potential Drug–Drug Interactions (DDIs): Based on a DDI knowledge base, some recommended medications may have potential drug–drug interactions.\n\nTask Rules:\n- Medications with a final probability ≥ 0.5 will be recommended to the patient.\n- You must revise the medication recommendation probabilities of both ODE Model Suggested Medications and Historical Medications based on the information above and clinical reasoning.\n\nAnalyze the Patient History, Potential Diseases, Historical Medications, ODE Model Suggested Medications and Potential Drug–Drug Interactions under standard pharmacotherapy principles. Based on this analysis, adjust your outputs to meet the following goals:\n- Raise the probability of clinically necessary medications to the [0.50–1.00] range.\n- Lower the probability of medications that lack evidence or have low necessity.\n"

    duration = patient_time_duration_encoded[pid]
    input_text = ""
    for idx, code_list in enumerate(hist_drug_code):
        if idx == 0:
            cur_duration = 0
            temp = "Patient History:\n" + number_to_capitalized_ordinal(idx + 1) + f" Visit:"
        else:
            cur_duration += duration[idx]
            temp = number_to_capitalized_ordinal(idx + 1) + f" Visit: ({cur_duration} days later):"

        temp += f"\n- Diagnoses: " + "{" + get_diag_str(diag_code[idx]) + "}"
        temp += f"\n- Procedures: " + "{" + get_pro_str(pro_code[idx]) + "}"
        temp += f"\n- Medication: " + "{" + get_drug_str(code_list) + "}"
        input_text += temp + "\n\n"
    temp = f"Final Visit ({sum(duration)} days later):"
    dp_ddx_list = dp_pred_map.get(pid, [])
    dp_formatted_str = parse_llm_pred_to_str(dp_ddx_list)
    temp += f"\n- Diagnoses: " + "{" + get_diag_str(diag_code[-1]) + "}"
    temp += f"\n- Procedures: " + "{" + get_pro_str(pro_code[-1]) + "}"
    temp += f"\n\nAfter clinical assessment, the physician performs a differential diagnosis for the Final Visit, resulting in the following Differential Diagnosis List:\n"
    temp += "{" + dp_formatted_str + "}"

    if has_potential_disease:
        temp += f"\n\n- Potential Diseases: {{{potential_disease_str}}}"

    hist_cand_str = get_drug_str(hist_candidate_codes)
    cand_str = get_drug_with_prob_str(candidate_codes, candidate_probs)
    temp += "\n\n- Historical Medications: " + "{" + hist_cand_str + "}"
    temp += "\n- ODE Model Suggested Medications: " + "{" + cand_str + "}\n\n"

    if ddi_warning_msg and ddi_warning_msg != "None":
        temp += "- Potential Drug-Drug Interactions (DDIs): {" + ddi_warning_msg + "}\n\n"
    else:
        temp += "- Potential Drug-Drug Interactions (DDIs): None\n\n"

    if not has_potential_disease:
        input_text += temp + f"Your task:\n- Synthesize Historical Medications and ODE Model Suggested Medications to assign a final clinical probability to each medication. Adjust the provided ODE probabilities and incorporate Historical Medications into the final decision-making process based on your pharmacologic reasoning to output a precise value between 0.00 and 1.00.\n- Lower the probability of medications that lack evidence or have low necessity.\n- Incorporate the provided Potential Drug-Drug Interactions (DDIs) into the final medication recommendation as clinical considerations rather than absolute exclusions, allowing context-aware judgment based on patient-specific factors and therapeutic necessity.\n- Directly provide the reordered list of medication names in descending order of likelihood.\nOutput format:\nAnswer: (\"Medication 1\", \"Probability 1\"), (\"Medication 2\", \"Probability 2\"),...\n\nNote: Strictly follow the required output format, one by one output each medication in the modified medication list and its corresponding probability, do not output the analysis process."
    else:
        input_text += temp + f"Your task:\n- Synthesize Historical Medications and ODE Model Suggested Medications to assign a final clinical probability to each medication. Adjust the provided ODE probabilities and incorporate Historical Medications into the final decision-making process based on your pharmacologic reasoning to output a precise value between 0.00 and 1.00.\n- Treat Potential Diseases as latent risk factors that may influence the prioritization of existing therapies or the inclusion of low-risk supportive medications, but must NOT justify introducing aggressive or high-risk treatments.\n- Lower the probability of medications that lack evidence or have low necessity.\n- Incorporate the provided Potential Drug-Drug Interactions (DDIs) into the final medication recommendation as clinical considerations rather than absolute exclusions, allowing context-aware judgment based on patient-specific factors and therapeutic necessity.\n- Directly provide the reordered list of medication names in descending order of likelihood.\nOutput format:\nAnswer: (\"Medication 1\", \"Probability 1\"), (\"Medication 2\", \"Probability 2\"),...\n\nNote: Strictly follow the required output format, one by one output each medication in the modified medication list and its corresponding probability, do not output the analysis process."

    result = {
        "pid": pid,
        "target_drug_code": target_drug_code_atc,
        "target_drug_name":target_drug_code_name,
        "actual_diseases": actual_disease_list,
        "potential_diseases": potential_disease_list,
        "llm_pred_potential_diseases": raw_llm_str,
        "instruction": instruction,
        "hist_code": hist_candidate_codes,
        "all_code": all_drug_code,
        "candidate_code": candidate_codes,
        "input": input_text
    }
    results.append(result)

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, default=np_encoder, indent=2)