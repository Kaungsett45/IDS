import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load and preprocess data (simplified)
def load_and_preprocess():
    print("Loading data...")
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
               'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
               'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
               'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty']
    
    train_data = pd.read_csv('nsl-kdd/KDDTrain+.txt', names=columns)
    test_data = pd.read_csv('nsl-kdd/KDDTest+.txt', names=columns)
    
    # Attack mapping
    attack_categories = {
        'normal': 'Normal',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
        'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
    }
    
    train_data['attack_category'] = train_data['attack_type'].map(attack_categories)
    test_data['attack_category'] = test_data['attack_type'].map(attack_categories)
    
    train_data = train_data.drop(['difficulty', 'attack_type'], axis=1)
    test_data = test_data.drop(['difficulty', 'attack_type'], axis=1)
    
    # Encode categorical
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        combined = pd.concat([train_data[col], test_data[col]])
        le.fit(combined)
        train_data[col] = le.transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
    
    # Separate features and labels
    X_train = train_data.drop(['attack_category'], axis=1)
    y_train = train_data['attack_category']
    X_test = test_data.drop(['attack_category'], axis=1)
    y_test = test_data['attack_category']
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    # SMOTE
    print("\nApplying SMOTE...")
    from imblearn.over_sampling import SMOTE
    print(f"Before SMOTE: {Counter(y_train)}")
    
    class_counts = Counter(y_train)
    majority_count = max(class_counts.values())
    target_count = int(majority_count * 0.2)
    
    sampling_strategy = {}
    for label, count in class_counts.items():
        if count < target_count:
            sampling_strategy[label] = target_count
    
    if sampling_strategy:
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {Counter(y_train)}")
    
    return X_train, X_test, y_train, y_test, label_encoder

# Main comparison
def compare_models():
    print("="*70)
    print("INDIVIDUAL MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess()
    
    results = {}
    
    # 1. Train XGBoost
    print("\n" + "="*70)
    print("1. TRAINING XGBOOST (ALONE)")
    print("="*70)
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.2,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        print("Training XGBoost...")
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = xgb_acc
        print(f"\nXGBoost Accuracy: {xgb_acc*100:.2f}%")
        print("\nXGBoost Classification Report:")
        print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))
    except Exception as e:
        print(f"XGBoost error: {e}")
        xgb_model = None
    
    # 2. Train LightGBM
    print("\n" + "="*70)
    print("2. TRAINING LIGHTGBM (ALONE)")
    print("="*70)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=3,
            num_leaves=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1
        )
        print("Training LightGBM...")
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        results['LightGBM'] = lgb_acc
        print(f"\nLightGBM Accuracy: {lgb_acc*100:.2f}%")
        print("\nLightGBM Classification Report:")
        print(classification_report(y_test, lgb_pred, target_names=label_encoder.classes_))
    except Exception as e:
        print(f"LightGBM error: {e}")
        lgb_model = None
    
    # 3. Ensemble (Weighted Voting)
    print("\n" + "="*70)
    print("3. ENSEMBLE (WEIGHTED VOTING: XGB 60% + LGBM 40%)")
    print("="*70)
    if xgb_model and lgb_model:
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
            voting='soft',
            weights=[0.6, 0.4],
            n_jobs=-1
        )
        print("Training Ensemble...")
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        results['Ensemble'] = ensemble_acc
        print(f"\nEnsemble Accuracy: {ensemble_acc*100:.2f}%")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=label_encoder.classes_))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Accuracy':<15} {'Improvement':<15}")
    print("-"*50)
    
    baseline = results.get('XGBoost', 0)
    for model, acc in results.items():
        improvement = ((acc - baseline) / baseline * 100) if baseline > 0 else 0
        print(f"{model:<20} {acc*100:>6.2f}%        {improvement:>+6.2f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    if 'Ensemble' in results:
        ensemble_boost = ((results['Ensemble'] - baseline) / baseline * 100)
        print(f"1. Ensemble improves accuracy by {ensemble_boost:.2f}% over single XGBoost")
        print(f"2. XGBoost weight: 60% (higher accuracy, more influence)")
        print(f"3. LightGBM weight: 40% (faster, complements XGBoost)")
        print(f"4. Soft voting averages probabilities for better predictions")
    print("="*70)

if __name__ == "__main__":
    compare_models()
