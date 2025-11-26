# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import dill
from random import shuffle
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data'))
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
from collections import defaultdict
import numpy as np

proc_file = 'PROCEDURES_ICD.csv' 
diag_file = 'DIAGNOSES_ICD.csv'

patient_info_file = './gather_firstday.csv'


def process_proc():
    print('process_proc')
    proc_pd = pd.read_csv(proc_file, dtype={'ICD9_CODE': 'category'})
    
    # CHỈ GIỮ LẠI 3 CỘT CẦN THIẾT
    proc_pd = proc_pd[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]
    
    proc_pd.drop(index=proc_pd[proc_pd['ICD9_CODE'] == '0'].index, axis=0, inplace=True)
    proc_pd.ffill(inplace=True)
    proc_pd.dropna(inplace=True)
    proc_pd.drop_duplicates(inplace=True)
    
    # SORT ĐƠN GIẢN
    proc_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    proc_pd = proc_pd.reset_index(drop=True)

    def filter_first24hour_proc(proc_pd):
        proc_pd_new = proc_pd.drop(columns=['ICD9_CODE'])
        proc_pd_new = proc_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID']).head(1).reset_index(drop=True) 
        proc_pd_new = pd.merge(proc_pd_new, proc_pd, on=['SUBJECT_ID', 'HADM_ID'])
        return proc_pd_new
        
    proc_pd = filter_first24hour_proc(proc_pd)
    proc_pd = proc_pd.drop_duplicates()

    return proc_pd.reset_index(drop=True)


def process_diag():
    print('process_diag')

    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def process_side():
    print('process_side')

    side_pd = pd.read_csv(patient_info_file)
    # just use demographic information to avoid future information leak such as lab test and lab measurements
    side_pd = side_pd[['subject_id', 'hadm_id', 'icustay_id',
                       'gender_male', 'admission_type', 'first_icu_stay', 'admission_age',
                       'ethnicity', 'weight', 'height']]

    # process side_information
    side_pd = side_pd.dropna(thresh=4)
    side_pd.fillna(side_pd.mean(), inplace=True)
    side_pd = side_pd.groupby(by=['subject_id', 'hadm_id']).head(
        [1]).reset_index(drop=True)
    side_pd = pd.concat(
        [side_pd, pd.get_dummies(side_pd['ethnicity'])], axis=1)
    side_pd.drop(columns=['ethnicity', 'icustay_id'], inplace=True)
    side_pd.rename(columns={'subject_id': 'SUBJECT_ID',
                            'hadm_id': 'HADM_ID'}, inplace=True)
    return side_pd.reset_index(drop=True)



def filter_proc(proc_pd):
    proc_count = proc_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={
        0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    proc_pd = proc_pd[proc_pd['ICD9_CODE'].isin(proc_count.loc[:299, 'ICD9_CODE'])]

    return proc_pd.reset_index(drop=True)


def filter_diag(diag_pd, num=128):
    print('filter diag')
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(
        diag_count.loc[:num, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


# visit filter


def filter_by_visit_range(data_pd, v_range=(1, 2)):
    a = data_pd[['SUBJECT_ID', 'HADM_ID']].groupby(
        by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
    a = a[(a['HADM_ID_Len'] >= v_range[0]) & (a['HADM_ID_Len'] < v_range[1])]
    data_pd_filter = a.reset_index(drop=True)
    data_pd = data_pd.merge(
        data_pd_filter[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    return data_pd.reset_index(drop=True)


def process_all(visit_range=(1, 2)):
    # get proc and diag (visit>=2)
    proc_pd = process_proc()
    # XÓA: proc_pd = ndc2atc4(proc_pd)  # KHÔNG CẦN CHO THỦ THUẬT
    proc_pd = filter_by_visit_range(proc_pd, visit_range)

    diag_pd = process_diag()
    diag_pd = filter_diag(diag_pd, num=1999)

    proc_pd_key = proc_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = proc_pd_key.merge(
        diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    proc_pd = proc_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ICD9_CODE'].unique().reset_index()
    proc_pd = proc_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE': 'PROC_CODE'})
        
    diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(lambda x: list(x))
    proc_pd['PROC_CODE'] = proc_pd['PROC_CODE'].map(lambda x: list(x))
        
    data = diag_pd.merge(proc_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    return data


def filter_patient(data, dx_range=(2, np.inf), proc_range=(2, np.inf)):  # ĐỔI: rx_range → proc_range
    print('filter_patient')

    drop_subject_ls = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]

        for index, row in item_data.iterrows():
            dx_len = len(list(row['ICD9_CODE']))
            proc_len = len(list(row['PROC_CODE']))  # ĐỔI: ATC4 → PROC_CODE
            if dx_len < dx_range[0] or dx_len > dx_range[1]:
                drop_subject_ls.append(subject_id)
                break
            if proc_len < proc_range[0] or proc_len > proc_range[1]:
                drop_subject_ls.append(subject_id)
                break
    data.drop(index=data[data['SUBJECT_ID'].isin(
        drop_subject_ls)].index, axis=0, inplace=True)
    return data.reset_index(drop=True)



def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    proc = data['PROC_CODE'].values  # ĐỔI: med → proc

    unique_diag = set([j for i in diag for j in list(i)])
    unique_proc = set([j for i in proc for j in list(i)])  # ĐỔI: unique_med → unique_proc

    print('#diagnosis ', len(unique_diag))
    print('#procedure ', len(unique_proc))  # ĐỔI: #med → #procedure

    avg_diag = 0
    avg_proc = 0  # ĐỔI: avg_med → avg_proc
    max_diag = 0
    max_proc = 0  # ĐỔI: max_med → max_proc
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['PROC_CODE']))
        x = set(x)
        y = set(y)
        avg_diag += len(x)
        avg_proc += len(y)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_proc:
            max_proc = len(y)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of procedures ', avg_proc / cnt)  # ĐỔI: medicines → procedures
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of procedures ', max_proc)  # ĐỔI: medicines → procedures
    print('#max of visit ', max_visit)


def run(visit_range=(1, 2)):
    data = process_all(visit_range)
    data = filter_patient(data)

    # unique code save
    diag = data['ICD9_CODE'].values
    proc = data['PROC_CODE'].values  # ĐỔI: med → proc
    unique_diag = set([j for i in diag for j in list(i)])
    unique_proc = set([j for i in proc for j in list(i)])  # ĐỔI: unique_med → unique_proc

    return data, unique_diag, unique_proc 


def load_gamenet_multi_visit_data(file_name='data_gamenet.pkl'):
    data = pd.read_pickle(file_name)
    data.rename(columns={'NDC': 'PROC_CODE'}, inplace=True)  # ĐỔI: ATC4 → PROC_CODE
    data.drop(columns=['PRO_CODE', 'NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag = data['ICD9_CODE'].values
    proc = data['PROC_CODE'].values  # ĐỔI: med → proc
    unique_diag = set([j for i in diag for j in list(i)])
    unique_proc = set([j for i in proc for j in list(i)])  # ĐỔI: unique_med → unique_proc
    return data, unique_diag, unique_proc  # ĐỔI: unique_med → unique_proc


def load_gamenet_multi_visit_data_with_pro(file_name='data_gamenet.pkl'):
    data = pd.read_pickle(file_name)
    data.rename(columns={'NDC': 'PROC_CODE'}, inplace=True)  # ĐỔI: ATC4 → PROC_CODE
    data.drop(columns=['NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag = data['ICD9_CODE'].values
    proc = data['PROC_CODE'].values  # ĐỔI: med → proc
    pro = data['PRO_CODE'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_proc = set([j for i in proc for j in list(i)])  # ĐỔI: unique_med → unique_proc
    unique_pro = set([j for i in pro for j in list(i)])

    return data, unique_pro, unique_diag, unique_proc  # ĐỔI: unique_med → unique_proc

def create_real_multi_visit_data():
    """Tạo multi-visit data thực sự từ raw data"""
    print("Creating REAL multi-visit data from raw files...")
    
    # Load raw procedures và diagnoses
    proc_raw = pd.read_csv('PROCEDURES_ICD.csv')
    diag_raw = pd.read_csv('DIAGNOSES_ICD.csv')
    
    # CHUYỂN ĐỔI ICD9_CODE THÀNH STRING
    proc_raw['ICD9_CODE'] = proc_raw['ICD9_CODE'].astype(str)
    diag_raw['ICD9_CODE'] = diag_raw['ICD9_CODE'].astype(str)
    
    # Tìm patients có nhiều admissions (HADM_ID khác nhau)
    patient_visit_counts = diag_raw.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    multi_visit_patients = patient_visit_counts[patient_visit_counts >= 2].index
    
    print(f"Found {len(multi_visit_patients)} patients with 2+ visits")
    
    multi_visit_records = []
    
    for patient_id in multi_visit_patients:
        # Lấy tất cả admissions của patient
        patient_diag = diag_raw[diag_raw['SUBJECT_ID'] == patient_id]
        patient_proc = proc_raw[proc_raw['SUBJECT_ID'] == patient_id]
        
        admissions = patient_diag['HADM_ID'].unique()
        
        # Tạo record cho mỗi admission
        for adm_id in admissions:
            adm_diag_codes = patient_diag[patient_diag['HADM_ID'] == adm_id]['ICD9_CODE'].unique().tolist()
            adm_proc_codes = patient_proc[patient_proc['HADM_ID'] == adm_id]['ICD9_CODE'].unique().tolist()
            
            # Chỉ lấy admissions có cả diagnosis và procedure
            if adm_diag_codes and adm_proc_codes:
                record = {
                    'SUBJECT_ID': patient_id,
                    'HADM_ID': adm_id,
                    'ICD9_CODE': adm_diag_codes,
                    'PROC_CODE': adm_proc_codes
                }
                multi_visit_records.append(record)
    
    multi_visit_df = pd.DataFrame(multi_visit_records)
    
    # Lọc patients có ít nhất 2 visits hợp lệ
    visit_counts = multi_visit_df.groupby('SUBJECT_ID').size()
    valid_patients = visit_counts[visit_counts >= 2].index
    multi_visit_df = multi_visit_df[multi_visit_df['SUBJECT_ID'].isin(valid_patients)]
    
    print(f"Final multi-visit data: {multi_visit_df.shape}")
    print(f"Patients with 2+ visits: {multi_visit_df['SUBJECT_ID'].nunique()}")
    
    # DEBUG: Kiểm tra chi tiết
    if len(multi_visit_df) > 0:
        sample_patient = multi_visit_df['SUBJECT_ID'].iloc[0]
        patient_visits = multi_visit_df[multi_visit_df['SUBJECT_ID'] == sample_patient]
        print(f"Sample patient {sample_patient} has {len(patient_visits)} visits")
        print(f"Sample ICD9_CODE type: {type(multi_visit_df['ICD9_CODE'].iloc[0][0])}")
        print(f"Sample PROC_CODE type: {type(multi_visit_df['PROC_CODE'].iloc[0][0])}")
    
    return multi_visit_df

def main():
    print('-'*20 + '\ndata-single processing')
    data_single_visit, diag1, proc1 = run(visit_range=(1, 2))
    
    print('-'*20 + '\ndata-multi processing - USING SINGLE VISIT DATA')
    # THAY THẾ: Tạo multi-visit data thực sự
    data_multi_visit = create_real_multi_visit_data()
    
    # Kiểm tra critical: data multi-visit có patients với 2+ visits không?
    if len(data_multi_visit) == 0:
        print("❌ CRITICAL ERROR: No multi-visit data created!")
        # Fallback: dùng single-visit data nhưng cảnh báo
        print("⚠️  Using single-visit data as fallback (NOT IDEAL FOR GBert)")
        data_multi_visit = data_single_visit.copy()
        diag2 = diag1
        proc2 = proc1
        pro = diag1
    else:
        visit_counts = data_multi_visit.groupby('SUBJECT_ID').size()
        print(f"✅ Multi-visit validation: {len(visit_counts)} patients, min visits: {visit_counts.min()}, max visits: {visit_counts.max()}")
        
        if visit_counts.min() < 2:
            print("❌ WARNING: Multi-visit data contains patients with < 2 visits!")
        
        # Tạo vocabulary từ real data
        diag2 = set([code for codes in data_multi_visit['ICD9_CODE'] for code in codes])
        proc2 = set([code for codes in data_multi_visit['PROC_CODE'] for code in codes])
        pro = proc2  # Dùng procedure codes cho px-vocab

    unique_diag = diag1 | diag2
    unique_proc = proc1 | proc2
    
    with open('dx-vocab.txt', 'w') as fout:
        for code in unique_diag:
            fout.write(code + '\n')
    with open('proc-vocab.txt', 'w') as fout:
        for code in unique_proc:
            fout.write(code + '\n')

    with open('proc-vocab-multi.txt', 'w') as fout:
        for code in proc2:
            fout.write(code + '\n')
    with open('dx-vocab-multi.txt', 'w') as fout:
        for code in diag2:
            fout.write(code + '\n')
    with open('px-vocab-multi.txt', 'w') as fout:
        for code in pro:
            fout.write(code + '\n')

    # save data
    data_single_visit.to_pickle('data-single-visit.pkl')
    data_multi_visit.to_pickle('data-multi-visit.pkl')

    print('-'*20 + '\ndata-single stat')
    statistics(data_single_visit)
    print('-'*20 + '\ndata_multi stat')
    statistics(data_multi_visit)

    return data_single_visit, data_multi_visit


data_single_visit, data_multi_visit = main()
data_multi_visit.head(10)


# %%
# split train, eval and test dataset
random.seed(1203)


def split_dataset(data_path='data-multi-visit.pkl'):
    data = pd.read_pickle(data_path)
    sample_id = data['SUBJECT_ID'].unique()

    random_number = [i for i in range(len(sample_id))]
#     shuffle(random_number)

    train_id = sample_id[random_number[:int(len(sample_id)*2/3)]]
    eval_id = sample_id[random_number[int(
        len(sample_id)*2/3): int(len(sample_id)*5/6)]]
    test_id = sample_id[random_number[int(len(sample_id)*5/6):]]

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    ls2file(train_id, 'train-id.txt')
    ls2file(eval_id, 'eval-id.txt')
    ls2file(test_id, 'test-id.txt')

    print('train size: %d, eval size: %d, test size: %d' %
          (len(train_id), len(eval_id), len(test_id)))


split_dataset()


# %%
# generate ehr graph for gamenet


def generate_proc_graph():  # ĐỔI: generate_ehr_graph → generate_proc_graph
    data_multi = pd.read_pickle('data-multi-visit.pkl')
    data_single = pd.read_pickle('data-single-visit.pkl')

    proc_voc_size = 0  # ĐỔI: rx_voc → proc_voc
    proc_voc = {}  # ĐỔI: rx_voc → proc_voc
    with open('proc-vocab.txt', 'r') as fin:  # ĐỔI: rx-vocab → proc-vocab
        for line in fin:
            proc_voc[line.rstrip('\n')] = proc_voc_size  # ĐỔI: rx_voc → proc_voc
            proc_voc_size += 1

    proc_adj = np.zeros((proc_voc_size, proc_voc_size))  # ĐỔI: ehr_adj → proc_adj

    for idx, row in data_multi.iterrows():
        proc_set = list(map(lambda x: proc_voc[x], row['PROC_CODE']))  # ĐỔI: med_set → proc_set, ATC4 → PROC_CODE
        for i, proc_i in enumerate(proc_set):  # ĐỔI: med_i → proc_i
            for j, proc_j in enumerate(proc_set):  # ĐỔI: med_j → proc_j
                if j <= i:
                    continue
                proc_adj[proc_i, proc_j] = 1  # ĐỔI: ehr_adj → proc_adj
                proc_adj[proc_j, proc_i] = 1  # ĐỔI: ehr_adj → proc_adj

    for idx, row in data_single.iterrows():
        proc_set = list(map(lambda x: proc_voc[x], row['PROC_CODE']))  # ĐỔI: med_set → proc_set, ATC4 → PROC_CODE
        for i, proc_i in enumerate(proc_set):  # ĐỔI: med_i → proc_i
            for j, proc_j in enumerate(proc_set):  # ĐỔI: med_j → proc_j
                if j <= i:
                    continue
                proc_adj[proc_i, proc_j] = 1  # ĐỔI: ehr_adj → proc_adj
                proc_adj[proc_j, proc_i] = 1  # ĐỔI: ehr_adj → proc_adj

    print('avg procedure for one ', np.mean(np.sum(proc_adj, axis=-1)))  # ĐỔI: med → procedure

    return proc_adj  # ĐỔI: ehr_adj → proc_adj


proc_adj = generate_proc_graph()  # ĐỔI: ehr_adj → proc_adj
dill.dump(proc_adj, open('proc_adj.pkl', 'wb'))  # ĐỔI: ehr_adj → proc_adj


# %%
# max len procedure codes - ĐỔI: medical codes → procedure codes
data = data_multi_visit

max_len = 0
for subject_id in data['SUBJECT_ID'].unique():
    item_df = data[data['SUBJECT_ID'] == subject_id]
    len_tmp = 0
    for index, row in item_df.iterrows():
        len_tmp += (len(row['ICD9_CODE']) + len(row['PROC_CODE']))  # ĐỔI: ATC4 → PROC_CODE
    if len_tmp > max_len:
        max_len = len_tmp
print(max_len)


# %%
print(max_len)


# %%
# pd.read_pickle(file_name)





# %%
data.shape


# %%
data_dir = './'

print('multi visit')
multi_file = data_dir + 'data-multi-visit.pkl'
multi_pkl = pd.read_pickle(multi_file)
print("Multi visit columns:", multi_pkl.columns.tolist())  # THÊM DÒNG NÀY
multi_pkl.iloc[0, :]  # ĐỔI: [0, 4:] → [0, :] ĐỂ XEM TẤT CẢ COLUMNS

# %%
# stat - ĐỔI THÀNH PROCEDURE STATS
proc_cnt_ls = []  # ĐỔI: rx_cnt_ls → proc_cnt_ls
dx_cnt_ls = []
visit_cnt_ls = []
for subject_id in multi_pkl['SUBJECT_ID'].unique():
    visit_cnt = 0
    for idx, visit in multi_pkl[multi_pkl['SUBJECT_ID'] == subject_id].iterrows():
        proc_cnt_ls.append(len(visit['PROC_CODE']))  # ĐỔI: ATC4 → PROC_CODE
        dx_cnt_ls.append(len(visit['ICD9_CODE']))
        visit_cnt += 1
    visit_cnt_ls.append(visit_cnt)

print('mean')
print('dx', np.mean(dx_cnt_ls))
print('procedure', np.mean(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.mean(visit_cnt_ls))

print('max')
print('dx', np.max(dx_cnt_ls))
print('procedure', np.max(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.max(visit_cnt_ls))

print('min')
print('dx', np.min(dx_cnt_ls))
print('procedure', np.min(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.min(visit_cnt_ls))


print('single visit')
# %%
single_file = data_dir + 'data-single-visit.pkl'
single_pkl = pd.read_pickle(single_file)
print("Single visit columns:", single_pkl.columns.tolist())
single_pkl.head()
# %%

proc_cnt_ls = []  # ĐỔI: rx_cnt_ls → proc_cnt_ls
dx_cnt_ls = []
visit_cnt_ls = []
for subject_id in single_pkl['SUBJECT_ID'].unique():
    visit_cnt = 0
    for idx, visit in single_pkl[single_pkl['SUBJECT_ID'] == subject_id].iterrows():
        proc_cnt_ls.append(len(visit['PROC_CODE']))  # ĐỔI: ATC4 → PROC_CODE
        dx_cnt_ls.append(len(visit['ICD9_CODE']))
        visit_cnt += 1
    visit_cnt_ls.append(visit_cnt)

print('mean')
print('dx', np.mean(dx_cnt_ls))
print('procedure', np.mean(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.mean(visit_cnt_ls))

print('max')
print('dx', np.max(dx_cnt_ls))
print('procedure', np.max(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.max(visit_cnt_ls))

print('min')
print('dx', np.min(dx_cnt_ls))
print('procedure', np.min(proc_cnt_ls))  # ĐỔI: rx → procedure
print('visit', np.min(visit_cnt_ls))


# %%
data_dir = './data/'

single_pkl.head()


# %%
