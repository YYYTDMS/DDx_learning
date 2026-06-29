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
        'You are an experienced medical diagnosis expert who understands disease progression, '
        'comorbidity patterns, and temporal dependencies across multiple visits.\n\n'
        'Differential Diagnosis (DDx) is a structured clinical reasoning process that identifies '
        "potential underlying diseases by carefully examining the patient's observed clinical evidence, "
        'longitudinal disease progression, comorbidity patterns, procedures, and medication history. '
        'In this task, DDx is used to distinguish which diseases from the Candidate Potential Diseases '
        'are clinically plausible latent or missing conditions for the final visit. The process should '
        "repeatedly compare each candidate against the patient's historical trajectory and exclude "
        'candidates that are weakly supported, inconsistent with the clinical course, or unlikely to '
        'affect the current disease state.\n\n'
        'To support your decision-making, you are provided with the following information sources:\n'
        "1. Patient History: The patient's comprehensive clinical profile, encompassing established "
        'diagnoses, procedures, and medication regimens that reflect their long-term treatment plan.\n'
        '2. Candidate Potential Diseases: Latent or subclinical conditions predicted by a neural '
        'Ordinary Differential Equation (ODE) model. This model analyzes longitudinal electronic '
        'health records to capture continuous patient trajectories and predict disease progression.\n\n'
        'Your goal is to perform DDx-oriented screening: select the potential diseases from the '
        'Candidate Potential Diseases that remain clinically reasonable after differential reasoning, '
        "are medically consistent with the patient's historical profile, and can serve as valid "
        'potential diagnoses for the final visit.\n'
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

    input_text += temp + (
        'Your task:\n'
        '- Perform DDx-oriented reasoning over the Patient History and Candidate Potential Diseases. '
        "For each candidate disease, internally examine whether it is supported by the patient's "
        'longitudinal diagnoses, procedures, medication history, disease progression, and known '
        'comorbidity patterns.\n'
        '- Repeatedly compare and distinguish the candidate diseases, retaining only those that are '
        'clinically plausible latent or missing conditions for the final visit.\n'
        '- Select the most clinically appropriate potential diseases from Candidate Potential Diseases '
        'based on standard pathological reasoning.\n'
        '- Directly provide the names of the selected valid potential diseases.\n'
        '- If none of the candidate diseases are deemed clinically appropriate, output "Answer: None".\n'
        'Output format:\n'
        'Answer: <Disease 1>; <Disease 2>; <Disease 3>, ...'
    )
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
