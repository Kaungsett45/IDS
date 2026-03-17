import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class IntrusionDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.categorical_encoders = {}
        
        # Attack type mappings
        self.attack_categories = {
            'normal': 'Normal',
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 
            'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
            'processtable': 'DoS', 'mailbomb': 'DoS',
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
            'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 
            'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 
            'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
            'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 
            'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
            'xterm': 'U2R'
        }
        
        self.attack_descriptions = {
            'Normal': 'Legitimate network traffic - No attack detected',
            'DoS': 'Denial of Service - Attempts to make resources unavailable',
            'Probe': 'Probing/Scanning - Surveillance to gather information',
            'R2L': 'Remote to Local - Unauthorized access from remote machine',
            'U2R': 'User to Root - Privilege escalation attacks'
        }
    
    def load_data(self, train_file, test_file):
        print("Loading NSL-KDD Dataset...")
        
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
        
        # Use ALL training data
        train_data = pd.read_csv(train_file, names=columns)
        test_data = pd.read_csv(test_file, names=columns)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data):
        print("\nPreprocessing data...")
        
        # Map specific attacks to categories
        train_data['attack_category'] = train_data['attack_type'].apply(
            lambda x: self.attack_categories.get(x, 'Unknown')
        )
        test_data['attack_category'] = test_data['attack_type'].apply(
            lambda x: self.attack_categories.get(x, 'Unknown')
        )
        
        # Drop difficulty and attack_type columns
        train_data = train_data.drop(['difficulty', 'attack_type'], axis=1)
        test_data = test_data.drop(['difficulty', 'attack_type'], axis=1)
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        for col in categorical_cols:
            combined = pd.concat([train_data[col], test_data[col]])
            le = LabelEncoder()
            le.fit(combined)
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            self.categorical_encoders[col] = le
        
        # Separate features and labels
        X_train = train_data.drop(['attack_category'], axis=1)
        y_train = train_data['attack_category']
        X_test = test_data.drop(['attack_category'], axis=1)
        y_test = test_data['attack_category']
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(y_train)
        y_test = self.label_encoder.transform(y_test)
        
        # Apply SMOTE strategically for minority classes
        print("\nApplying SMOTE for minority classes...")
        try:
            from imblearn.over_sampling import SMOTE
            from collections import Counter
            print(f"Before SMOTE: {Counter(y_train)}")
            
            # Calculate target: boost minorities to 20% of majority
            class_counts = Counter(y_train)
            majority_count = max(class_counts.values())
            target_count = int(majority_count * 0.2)
            
            # Only oversample classes below target
            sampling_strategy = {}
            for label, count in class_counts.items():
                if count < target_count:
                    sampling_strategy[label] = target_count
            
            if sampling_strategy:
                smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE: {Counter(y_train)}")
            else:
                print("Classes already balanced")
        except Exception as e:
            print(f"SMOTE error: {e}")
        
        print("Preprocessing complete!")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        print("\nTraining Optimized 2-Model Weighted Voting Ensemble...")
        print("Using: XGBoost (Best performers for imbalanced data)\n")
        
        # 1. XGBoost - Tuned for IDS
        xgb_model = None
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
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                tree_method='hist'
            )
            print("XGBoost configured (weight: 0.6)")
        except ImportError:
            print("XGBoost not available - using fallback")
        
        # 2. LightGBM - Tuned for IDS
        lgb_model = None
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
            print("LightGBM configured (weight: 0.4)")
        except ImportError:
            print("LightGBM not available - using fallback")
        
        # Fallback: Random Forest if XGBoost/LightGBM unavailable
        if not xgb_model and not lgb_model:
            print("\nUsing Random Forest fallback...")
            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced_subsample',
                max_samples=0.7
            )
            self.model.fit(X_train, y_train)
            print("Random Forest training complete!")
            return
        
        # Build Weighted Voting Ensemble
        estimators = []
        weights = []
        
        if xgb_model:
            estimators.append(('xgb', xgb_model))
            weights.append(0.6)  # XGBoost gets 60% weight
        
        if lgb_model:
            estimators.append(('lgb', lgb_model))
            weights.append(0.4)  # LightGBM gets 40% weight
        
        # Normalize weights if only one model available
        if len(estimators) == 1:
            weights = [1.0]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        print(f"\nTraining weighted ensemble: {[name for name, _ in estimators]}")
        print(f"  Weights: {weights}")
        print("  Strategy: Soft voting with optimized hyperparameters\n")
        
        self.model.fit(X_train, y_train)
        print("Model training complete!")
    
    def evaluate_model(self, X_test, y_test):
        print("\nEvaluating model performance...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=self.label_encoder.classes_))
        
        return y_pred, y_test
    
    def visualize_results(self, y_test, y_pred):
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Intrusion Detection System - Attack Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('Actual Attack Type')
        axes[0, 0].set_xlabel('Predicted Attack Type')
        
        # 2. Attack Distribution
        attack_counts = pd.Series(y_test).value_counts()
        attack_labels = [self.label_encoder.classes_[i] for i in attack_counts.index]
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']
        axes[0, 1].bar(attack_labels, attack_counts.values, color=colors)
        axes[0, 1].set_title('Attack Type Distribution in Test Data', fontweight='bold')
        axes[0, 1].set_xlabel('Attack Category')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        elif hasattr(self.model, 'estimators_'):
            # For ensemble, use RF importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.estimators_[0][1].feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        else:
            feature_importance = pd.DataFrame({'feature': [], 'importance': []})
        
        axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
        axes[1, 0].set_title('Top 10 Important Features', fontweight='bold')
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].invert_yaxis()
        
        # 4. Attack Type Descriptions
        axes[1, 1].axis('off')
        description_text = "ATTACK TYPE EXPLANATIONS:\n\n"
        for attack, desc in self.attack_descriptions.items():
            description_text += f"• {attack}:\n  {desc}\n\n"
        
        axes[1, 1].text(0.1, 0.9, description_text, fontsize=11, 
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('ids_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'ids_analysis.png'")
        plt.show()

def main():
    print("="*60)
    print("INTRUSION DETECTION SYSTEM (IDS)")
    print("="*60)
    
    ids = IntrusionDetectionSystem()
    
    # Load data
    train_data, test_data = ids.load_data(
        'nsl-kdd/KDDTrain+.txt',
        'nsl-kdd/KDDTest+.txt'
    )
    
    # Preprocess
    X_train, X_test, y_train, y_test = ids.preprocess_data(train_data, test_data)
    
    # Train
    ids.train_model(X_train, y_train)
    
    # Evaluate
    y_pred, y_test = ids.evaluate_model(X_test, y_test)
    
    # Visualize
    ids.visualize_results(y_test, y_pred)
    
    print("\n" + "="*60)
    print("IDS System Ready for Detection!")
    print("="*60)

if __name__ == "__main__":
    main()
