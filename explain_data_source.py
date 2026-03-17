import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from ids_system import IntrusionDetectionSystem

print("="*80)
print("📂 WHERE DO CONFUSION MATRIX VALUES COME FROM?")
print("="*80)

print("\n" + "="*80)
print("SOURCE 1: NSL-KDD TEST FILE")
print("="*80)
print("\nFile location: nsl-kdd/KDDTest+.txt")
print("This file contains REAL network traffic data with KNOWN attack labels")

# Read the raw test file
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

test_data_raw = pd.read_csv('nsl-kdd/KDDTest+.txt', names=columns)

print(f"\nTotal rows in file: {len(test_data_raw)}")
print("\nFirst 10 rows showing ACTUAL ATTACK LABELS from the file:")
print("-"*80)
print(f"{'Row':<6} {'Attack Type (from file)':<25} {'Features (first 3)'}")
print("-"*80)
for i in range(10):
    attack = test_data_raw.iloc[i]['attack_type']
    features = f"duration={test_data_raw.iloc[i]['duration']}, protocol={test_data_raw.iloc[i]['protocol_type']}, service={test_data_raw.iloc[i]['service']}"
    print(f"{i+1:<6} {attack:<25} {features}")

print("\n" + "="*80)
print("SOURCE 2: ATTACK TYPE DISTRIBUTION IN FILE")
print("="*80)
print("\nCounting how many times each attack appears in the file:")
print("-"*80)
attack_counts = test_data_raw['attack_type'].value_counts()
print(attack_counts.head(20))
print(f"\n... and {len(attack_counts) - 20} more attack types")

print("\n" + "="*80)
print("SOURCE 3: GROUPING ATTACKS INTO 5 CATEGORIES")
print("="*80)

attack_categories = {
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

test_data_raw['attack_category'] = test_data_raw['attack_type'].apply(
    lambda x: attack_categories.get(x, 'Unknown')
)

print("\nAfter grouping, we have 5 categories:")
category_counts = test_data_raw['attack_category'].value_counts()
print(category_counts)
print(f"\nTotal: {category_counts.sum()} samples")

print("\n" + "="*80)
print("SOURCE 4: MODEL PREDICTIONS")
print("="*80)
print("\nNow we train the model and let it PREDICT the attack types...")

# Train the model
ids = IntrusionDetectionSystem()
train_data, test_data = ids.load_data('nsl-kdd/KDDTrain+.txt', 'nsl-kdd/KDDTest+.txt')
X_train, X_test, y_train, y_test = ids.preprocess_data(train_data, test_data)
ids.train_model(X_train, y_train)

# Get predictions
y_pred = ids.model.predict(X_test)

print("\nModel has made predictions for all 22,544 samples!")
print("\nComparing ACTUAL (from file) vs PREDICTED (from model):")
print("-"*80)
print(f"{'Sample':<8} {'ACTUAL (from file)':<20} {'PREDICTED (by model)':<20} {'Match?'}")
print("-"*80)

attack_names = ids.label_encoder.classes_

for i in range(30):
    actual_label = attack_names[y_test[i]]
    predicted_label = attack_names[y_pred[i]]
    match = "✓ CORRECT" if y_test[i] == y_pred[i] else "✗ WRONG"
    print(f"{i+1:<8} {actual_label:<20} {predicted_label:<20} {match}")

print("\n" + "="*80)
print("SOURCE 5: BUILDING THE CONFUSION MATRIX")
print("="*80)
print("\nNow we COUNT how many times each (ACTUAL, PREDICTED) pair occurs:")

cm = confusion_matrix(y_test, y_pred)

print("\nExample: Let's count DoS attacks manually from the 30 samples above:")
dos_index = list(attack_names).index('DoS')
dos_actual = sum(1 for i in range(30) if y_test[i] == dos_index)
dos_correct = sum(1 for i in range(30) if y_test[i] == dos_index and y_pred[i] == dos_index)
print(f"  - DoS samples in first 30: {dos_actual}")
print(f"  - Correctly predicted as DoS: {dos_correct}")

print("\nDoing this for ALL 22,544 samples gives us the confusion matrix:")
print("\n" + " "*15 + "PREDICTED")
print(" "*15, end="")
for name in attack_names:
    print(f"{name:>10}", end="")
print()
print("-"*80)

for i, actual_name in enumerate(attack_names):
    print(f"ACTUAL {actual_name:<8}", end="")
    for j in range(len(attack_names)):
        print(f"{cm[i][j]:>10}", end="")
    print()

print("\n" + "="*80)
print("EXPLANATION OF EACH CELL")
print("="*80)
print("\nEach number in the matrix means:")
print(f"  CM[DoS][Normal] = {cm[0][1]} means:")
print(f"    → {cm[0][1]} samples were ACTUALLY DoS attacks (from file)")
print(f"    → But the model PREDICTED them as Normal")
print(f"    → This is a MISTAKE (False Negative for DoS)")

print(f"\n  CM[DoS][DoS] = {cm[0][0]} means:")
print(f"    → {cm[0][0]} samples were ACTUALLY DoS attacks (from file)")
print(f"    → And the model CORRECTLY PREDICTED them as DoS")
print(f"    → This is CORRECT! ✓")

print("\n" + "="*80)
print("COMPLETE DATA FLOW")
print("="*80)
print("""
1. NSL-KDD Test File (KDDTest+.txt)
   ↓
   Contains 22,544 rows of network traffic
   Each row has 41 features + 1 ACTUAL attack label
   
2. Read File
   ↓
   Load into pandas DataFrame
   Extract ACTUAL labels (y_test) from 'attack_type' column
   
3. Train Model
   ↓
   Model learns patterns from training data (125,973 samples)
   
4. Model Predicts
   ↓
   For each of 22,544 test samples:
     - Model looks at 41 features
     - Predicts what attack type it thinks it is
     - Stores prediction in y_pred
   
5. Compare
   ↓
   For each sample i (from 0 to 22,543):
     - ACTUAL = y_test[i] (from file)
     - PREDICTED = y_pred[i] (from model)
     - Count this combination in confusion matrix
   
6. Confusion Matrix
   ↓
   CM[actual][predicted] = count of samples with this combination
   
7. Calculate Accuracy
   ↓
   Accuracy = (Sum of diagonal) / (Total samples)
   Accuracy = (Correct predictions) / 22,544
""")

print("\n" + "="*80)
print("SAVE DETAILED TRACE")
print("="*80)

# Save detailed trace
with open('data_source_explanation.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("WHERE CONFUSION MATRIX DATA VALUES COME FROM\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA SOURCE: nsl-kdd/KDDTest+.txt\n")
    f.write("-"*80 + "\n")
    f.write(f"This file contains {len(test_data_raw)} rows of REAL network traffic data\n")
    f.write("Each row has:\n")
    f.write("  - 41 network features (duration, bytes, protocols, etc.)\n")
    f.write("  - 1 attack_type label (the GROUND TRUTH)\n\n")
    
    f.write("ACTUAL LABELS (y_test):\n")
    f.write("-"*80 + "\n")
    f.write("These come DIRECTLY from the 'attack_type' column in KDDTest+.txt\n")
    f.write("Distribution:\n")
    for attack, count in category_counts.items():
        f.write(f"  {attack}: {count} samples\n")
    f.write(f"Total: {category_counts.sum()}\n\n")
    
    f.write("PREDICTED LABELS (y_pred):\n")
    f.write("-"*80 + "\n")
    f.write("These come from the Random Forest model's predictions\n")
    f.write("The model:\n")
    f.write("  1. Was trained on 125,973 samples from KDDTrain+.txt\n")
    f.write("  2. Learned patterns of each attack type\n")
    f.write("  3. Now predicts attack type for each test sample\n\n")
    
    f.write("CONFUSION MATRIX CALCULATION:\n")
    f.write("-"*80 + "\n")
    f.write("For each of the 22,544 test samples:\n")
    f.write("  1. Get ACTUAL label from file (y_test[i])\n")
    f.write("  2. Get PREDICTED label from model (y_pred[i])\n")
    f.write("  3. Increment CM[actual][predicted] by 1\n\n")
    
    f.write("EXAMPLE TRACE:\n")
    f.write("-"*80 + "\n")
    for i in range(20):
        actual = attack_names[y_test[i]]
        predicted = attack_names[y_pred[i]]
        f.write(f"Sample {i+1}: ACTUAL={actual}, PREDICTED={predicted} → CM[{actual}][{predicted}] += 1\n")
    
    f.write("\n\nFINAL CONFUSION MATRIX:\n")
    f.write("-"*80 + "\n")
    f.write(" "*15 + "PREDICTED\n")
    f.write(" "*15)
    for name in attack_names:
        f.write(f"{name:>10}")
    f.write("\n" + "-"*80 + "\n")
    
    for i, actual_name in enumerate(attack_names):
        f.write(f"ACTUAL {actual_name:<8}")
        for j in range(len(attack_names)):
            f.write(f"{cm[i][j]:>10}")
        f.write("\n")
    
    f.write("\n\nACCURACY CALCULATION:\n")
    f.write("-"*80 + "\n")
    diagonal_sum = sum(cm[i][i] for i in range(len(cm)))
    f.write(f"Diagonal sum (correct predictions) = {diagonal_sum}\n")
    f.write(f"Total samples = {len(y_test)}\n")
    f.write(f"Accuracy = {diagonal_sum} / {len(y_test)} = {diagonal_sum/len(y_test)*100:.2f}%\n")

print("✅ Detailed explanation saved to: data_source_explanation.txt")

print("\n" + "="*80)
print("✅ COMPLETE! YOU CAN NOW EXPLAIN TO YOUR TEACHER:")
print("="*80)
print("""
1. Data comes from NSL-KDD dataset file (KDDTest+.txt)
2. File has 22,544 rows with ACTUAL attack labels
3. Model predicts attack type for each row
4. Confusion matrix COUNTS how many times each (actual, predicted) pair occurs
5. Diagonal = correct predictions
6. Accuracy = diagonal sum / total samples
""")
print("="*80)
