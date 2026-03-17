import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ImprovedIDS:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = []
        self.selected_features = []
        self.categorical_encoders = {}
        
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
        print("📂 Loading NSL-KDD Dataset...")
        
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
        
        train_data = pd.read_csv(train_file, names=columns)
        test_data = pd.read_csv(test_file, names=columns)
        
        print(f"✅ Training samples: {len(train_data)}")
        print(f"✅ Testing samples: {len(test_data)}")
        
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data):
        print("\n🔧 Preprocessing data...")
        
        # Map attacks to categories
        train_data['attack_category'] = train_data['attack_type'].apply(
            lambda x: self.attack_categories.get(x, 'Unknown')
        )
        test_data['attack_category'] = test_data['attack_type'].apply(
            lambda x: self.attack_categories.get(x, 'Unknown')
        )
        
        train_data = train_data.drop(['difficulty'], axis=1)
        test_data = test_data.drop(['difficulty'], axis=1)
        
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
        X_train = train_data.drop(['attack_type', 'attack_category'], axis=1)
        y_train = train_data['attack_category']
        X_test = test_data.drop(['attack_type', 'attack_category'], axis=1)
        y_test = test_data['attack_category']
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(y_train)
        y_test = self.label_encoder.transform(y_test)
        
        print("✅ Preprocessing complete!")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train):
        print("\n⚖️ Applying SMOTE for class balancing...")
        
        # Show original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nOriginal class distribution:")
        for label, count in zip(unique, counts):
            print(f"  {self.label_encoder.classes_[label]}: {count}")
        
        # Apply SMOTE with sampling strategy
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=42,
            k_neighbors=5
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Show new distribution
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print("\nBalanced class distribution:")
        for label, count in zip(unique, counts):
            print(f"  {self.label_encoder.classes_[label]}: {count}")
        
        print(f"\n✅ Training samples increased: {len(X_train)} → {len(X_train_balanced)}")
        
        return X_train_balanced, y_train_balanced
    
    def select_features(self, X_train, y_train, X_test):
        print("\n🎯 Performing feature selection...")
        
        # Train a preliminary model for feature importance
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        temp_model.fit(X_train, y_train)
        
        # Select features with importance > threshold
        self.feature_selector = SelectFromModel(temp_model, threshold='median', prefit=True)
        
        X_train_selected = self.feature_selector.transform(X_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [feat for feat, selected in zip(self.feature_names, selected_mask) if selected]
        
        print(f"✅ Features reduced: {len(self.feature_names)} → {len(self.selected_features)}")
        print(f"\nTop selected features:")
        feature_importance = temp_model.feature_importances_[selected_mask]
        sorted_idx = np.argsort(feature_importance)[::-1][:10]
        for idx in sorted_idx:
            print(f"  - {self.selected_features[idx]}: {feature_importance[idx]:.4f}")
        
        return X_train_selected, X_test_selected
    
    def train_model(self, X_train, y_train):
        print("\n🤖 Training Optimized Random Forest Classifier...")
        
        # Tuned hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=200,           # Increased trees
            criterion='gini',
            max_depth=30,               # Deeper trees
            min_samples_split=5,        # More aggressive splitting
            min_samples_leaf=2,         # Smaller leaves
            max_features='sqrt',        # sqrt for better generalization
            bootstrap=True,
            oob_score=True,
            class_weight='balanced',    # Handle remaining imbalance
            random_state=42,
            n_jobs=-1,
            max_samples=0.8             # Subsample for diversity
        )
        
        self.model.fit(X_train, y_train)
        print("✅ Model training complete!")
        
        if hasattr(self.model, 'oob_score_'):
            print(f"📊 Out-of-bag score: {self.model.oob_score_*100:.2f}%")
    
    def evaluate_model(self, X_test, y_test):
        print("\n📊 Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=self.label_encoder.classes_,
                                    digits=4))
        
        return y_pred, y_test
    
    def visualize_results(self, y_test, y_pred):
        print("\n📈 Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Improved IDS - SMOTE + Feature Selection', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # Per-class accuracy
        class_accuracy = []
        for i in range(len(self.label_encoder.classes_)):
            mask = y_test == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum() * 100
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']
        axes[0, 1].bar(self.label_encoder.classes_, class_accuracy, color=colors)
        axes[0, 1].set_title('Per-Class Accuracy', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Feature importance (top 15)
        if len(self.selected_features) > 0:
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
            axes[1, 0].set_title('Top 15 Selected Features', fontweight='bold')
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].invert_yaxis()
        
        # Improvements summary
        axes[1, 1].axis('off')
        summary_text = "IMPROVEMENTS APPLIED:\n\n"
        summary_text += "✓ SMOTE Oversampling\n"
        summary_text += "  - Balanced minority classes\n"
        summary_text += f"  - R2L and U2R boosted\n\n"
        summary_text += "✓ Feature Selection\n"
        summary_text += f"  - {len(self.feature_names)} → {len(self.selected_features)} features\n"
        summary_text += "  - Removed low-importance features\n\n"
        summary_text += "✓ Hyperparameter Tuning\n"
        summary_text += "  - 200 trees (was 100)\n"
        summary_text += "  - max_depth=30 (was 20)\n"
        summary_text += "  - class_weight='balanced'\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, fontsize=11, 
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('improved_ids_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'improved_ids_analysis.png'")
        plt.show()

def main():
    print("="*80)
    print("🛡️  IMPROVED INTRUSION DETECTION SYSTEM")
    print("    SMOTE + Feature Selection + Hyperparameter Tuning")
    print("="*80)
    
    ids = ImprovedIDS()
    
    # Load data
    train_data, test_data = ids.load_data(
        'nsl-kdd/KDDTrain+.txt',
        'nsl-kdd/KDDTest+.txt'
    )
    
    # Preprocess
    X_train, X_test, y_train, y_test = ids.preprocess_data(train_data, test_data)
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = ids.apply_smote(X_train, y_train)
    
    # Feature Selection
    X_train_selected, X_test_selected = ids.select_features(X_train_balanced, y_train_balanced, X_test)
    
    # Train
    ids.train_model(X_train_selected, y_train_balanced)
    
    # Evaluate
    y_pred, y_test = ids.evaluate_model(X_test_selected, y_test)
    
    # Visualize
    ids.visualize_results(y_test, y_pred)
    
    print("\n" + "="*80)
    print("✅ Improved IDS System Ready!")
    print("="*80)

if __name__ == "__main__":
    main()
