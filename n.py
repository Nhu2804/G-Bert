# check_data_reality.py
import pandas as pd

def check_real_data():
    print("=== KIỂM TRA DATA THỰC TẾ ===")
    
    # Load raw data
    proc_raw = pd.read_csv('data/PROCEDURES_ICD.csv')
    diag_raw = pd.read_csv('data/DIAGNOSES_ICD.csv')
    
    print(f"Raw procedures: {proc_raw.shape}")
    print(f"Raw diagnoses: {diag_raw.shape}")
    
    # Kiểm tra multi-visit patients thực sự
    patient_visit_counts = diag_raw.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    multi_visit_patients = patient_visit_counts[patient_visit_counts >= 2]
    
    print(f"\nPatients with 2+ visits: {len(multi_visit_patients)}")
    print(f"Visit distribution: {patient_visit_counts.value_counts().sort_index()}")
    
    if len(multi_visit_patients) == 0:
        print("❌ KHÔNG CÓ MULTI-VISIT PATIENTS!")
        print("👉 Cần giải pháp thay thế")
        return False
    else:
        print("✅ Có multi-visit patients")
        return True

def check_processed_data():
    print("\n=== KIỂM TRA PROCESSED DATA ===")
    
    try:
        multi_data = pd.read_pickle('data-multi-visit.pkl')
        single_data = pd.read_pickle('data-single-visit.pkl')
        
        print(f"Multi-visit data shape: {multi_data.shape}")
        print(f"Single-visit data shape: {single_data.shape}")
        
        # Kiểm tra visits per patient
        multi_visits = multi_data.groupby('SUBJECT_ID').size()
        single_visits = single_data.groupby('SUBJECT_ID').size()
        
        print(f"\nMulti-visit - Patients: {len(multi_visits)}, Min visits: {multi_visits.min()}, Max visits: {multi_visits.max()}")
        print(f"Single-visit - Patients: {len(single_visits)}, Min visits: {single_visits.min()}, Max visits: {single_visits.max()}")
        
        if multi_visits.max() == 1:
            print("❌ MULTI-VISIT DATA CHỈ CÓ 1 VISIT PER PATIENT!")
            return False
        else:
            print("✅ Multi-visit data có sequential visits")
            return True
            
    except FileNotFoundError:
        print("Processed data files not found")
        return False

if __name__ == "__main__":
    has_raw_multi_visit = check_real_data()
    has_processed_multi_visit = check_processed_data()
    
    if not has_raw_multi_visit:
        print("\n🚨 GIẢI PHÁP: Dùng single-visit data với data augmentation")