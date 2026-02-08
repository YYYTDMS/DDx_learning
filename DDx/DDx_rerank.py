import pickle as pickle
import numpy as np
import pandas as pd
import inflect
import dill
import json
import os

base_path = "../data/MR_0.05/"
ode_DP_pred_path = '../ODE_result/DP_0.05/ode_predictions_probs_DP_0.05.pkl'
output_json_path = "prompt_data/DDX_rerank_test.json"

with open(base_path + 'patient_time_duration_encoded.pkl', 'rb') as f80:
    patient_time_duration_encoded = pickle.load(f80)

test_pids = np.load(base_path + 'test_pids.npy')

with open(ode_DP_pred_path, 'rb') as f:
    ode_dp_probs_dict = pickle.load(f)
with open(base_path + 'records_final_more_time_sorted.pkl', 'rb') as f1:
    record = pickle.load(f1)
with open(base_path + 'voc_final.pkl', 'rb') as voc_file:
    voc_data = dill.load(voc_file)

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

def np_encoder(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

results = []
candidate_num_sum = 0

for i in range(len(test_pids)):
    pid = test_pids[i]
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
    actual_disease_names = []
    for idx in target_diag_indices:
        icd = idx_to_icd[idx]
        name = icd2name.get(icd, "")
        actual_disease_names.append(name)
    dp_probs = ode_dp_probs_dict[pid]
    topk_dp_indices = np.argsort(dp_probs)[-10:][::-1]
    candidate_potential_indices = [idx for idx in topk_dp_indices if idx not in target_diag_indices]
    potential_names_formatted = []
    potential_disease_list_clean = []
    if len(candidate_potential_indices) > 0:
        for idx in candidate_potential_indices:
            raw_icd = idx_to_icd[idx]
            name = icd2name.get(raw_icd, "")
            if name:
                potential_disease_list_clean.append(name)
                potential_names_formatted.append(f'"{name}"')
        potential_disease_str = ", ".join(potential_names_formatted)
    else:
        potential_disease_str = "None"

    instruction = (
        "You are an experienced clinical pharmacology and diagnosis expert specializing in longitudinal patient histories and therapeutic decision-making.\n\n"
        "To support your decision-making, you are provided with the following information sources:\n"
        "1. Patient History: The patient's comprehensive clinical profile, encompassing established diagnoses, procedures, and medication regimens.\n"
        "2. Candidate Potential Diseases: Latent or subclinical conditions predicted by a neural ODE model based on patient trajectories.\n"
        "3. Re-ranking Logic: The Final Diagnoses list must be strictly sorted based on the following Hierarchy of Clinical Priority (from highest to lowest importance):\n"
        "    (1) Critical Threats & Root Causes: Immediate life-threatening risks or the primary disease driving the current condition.\n"
        "    (2) Underlying Causes: The root problems that trigger multiple other complications.\n"
        "    (3) Active Treatment Context: Conditions strictly tied to the current hospitalization status.\n"
        "    (4) Chronic Conditions & Risks: Long-term background diseases or general risk factors.\n\n"
        "Your goal is to first select valid potential diseases from the candidates based on patient history, combine them with the confirmed diagnoses of the Final Visit, and then re-rank the complete list."
    )
    duration = patient_time_duration_encoded[pid]
    input_text = ""
    for idx, code_list in enumerate(hist_drug_code):
        if idx == 0:
            cur_duration = 0
            temp = number_to_capitalized_ordinal(idx + 1) + f" Visit:"
        else:
            cur_duration += duration[idx]
            temp = number_to_capitalized_ordinal(idx + 1) + f" Visit: ({cur_duration} days later):"

        temp += f"\n- Diagnoses: " + "{" + get_diag_str(diag_code[idx]) + "}"
        temp += f"\n- Procedures: " + "{" + get_pro_str(pro_code[idx]) + "}"
        temp += f"\n- Medication: " + "{" + get_drug_str(code_list) + "}"
        input_text += temp + "\n\n"

    temp = f"Final Visit ({sum(duration)} days later):"
    temp += f"\n- Diagnoses (Confirmed): " + "{" + get_diag_str(diag_code[-1]) + "}"
    temp += f"\n- Procedures: " + "{" + get_pro_str(pro_code[-1]) + "}"
    temp += f"\n\n- Candidate Potential Diseases: {{{potential_disease_str}}}"
    temp += f"\n\nYour task:\n"
    temp += "- Synthesize the patient's historical profile to select the most clinically appropriate potential diseases from Candidate Potential Diseases for the final visit, based on your standard pathological reasoning.\n"
    temp += "- Combine the selected valid potential diseases with the Diagnoses explicitly listed in the Final Visit.\n"
    temp += "- Re-rank this combined list of diseases based on the Re-ranking Logic and your medical reasoning.\n"
    temp += "- Directly provide the reordered list of disease names in descending order of likelihood. \n"
    temp += "Output format:\nAnswer: <Disease 1>; <Disease 2>; <Disease 3>, ..."
    input_text += temp
    result = {
        "pid": pid,
        "target_drug_code": target_drug_code_atc,
        "actual_diseases": actual_disease_names,
        "candidate_potential_diseases": potential_disease_list_clean,
        "instruction": instruction,
        "input": input_text
    }
    results.append(result)

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, default=np_encoder, indent=2)