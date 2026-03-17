import pandas as pd
from collections import Counter

def analyze_dataset():
    """
    Analyze NSL-KDD dataset to show all attack types
    """
    print("="*80)
    print("🔍 NSL-KDD DATASET ANALYSIS - Attack Types")
    print("="*80)
    
    # Column names
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
    
    # Load datasets
    print("\n📂 Loading Training Dataset...")
    train_data = pd.read_csv('nsl-kdd/KDDTrain+.txt', names=columns)
    
    print("📂 Loading Testing Dataset...")
    test_data = pd.read_csv('nsl-kdd/KDDTest+.txt', names=columns)
    
    # Attack type mapping
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
    
    print("\n" + "="*80)
    print("📊 TRAINING DATASET ANALYSIS")
    print("="*80)
    print(f"Total Training Samples: {len(train_data):,}")
    
    # Get unique attack types in training data
    train_attacks = train_data['attack_type'].value_counts()
    
    print("\n🎯 Attack Types in Training Data:")
    print("-" * 80)
    
    # Group by category
    categories = {'Normal': [], 'DoS': [], 'Probe': [], 'R2L': [], 'U2R': []}
    
    for attack, count in train_attacks.items():
        category = attack_categories.get(attack, 'Unknown')
        categories[category].append((attack, count))
    
    # Display by category
    for category in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        if categories[category]:
            total = sum([count for _, count in categories[category]])
            print(f"\n{'='*80}")
            print(f"📌 {category} Category - Total: {total:,} samples")
            print(f"{'='*80}")
            
            for attack, count in sorted(categories[category], key=lambda x: x[1], reverse=True):
                percentage = (count / len(train_data)) * 100
                print(f"   • {attack:20s} : {count:6,} samples ({percentage:5.2f}%)")
    
    print("\n" + "="*80)
    print("📊 TESTING DATASET ANALYSIS")
    print("="*80)
    print(f"Total Testing Samples: {len(test_data):,}")
    
    # Get unique attack types in testing data
    test_attacks = test_data['attack_type'].value_counts()
    
    print("\n🎯 Attack Types in Testing Data:")
    print("-" * 80)
    
    # Group by category
    categories_test = {'Normal': [], 'DoS': [], 'Probe': [], 'R2L': [], 'U2R': []}
    
    for attack, count in test_attacks.items():
        category = attack_categories.get(attack, 'Unknown')
        categories_test[category].append((attack, count))
    
    # Display by category
    for category in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        if categories_test[category]:
            total = sum([count for _, count in categories_test[category]])
            print(f"\n{'='*80}")
            print(f"📌 {category} Category - Total: {total:,} samples")
            print(f"{'='*80}")
            
            for attack, count in sorted(categories_test[category], key=lambda x: x[1], reverse=True):
                percentage = (count / len(test_data)) * 100
                print(f"   • {attack:20s} : {count:6,} samples ({percentage:5.2f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    train_unique = len(train_attacks)
    test_unique = len(test_attacks)
    
    print(f"\n✅ Training Dataset:")
    print(f"   • Total Samples: {len(train_data):,}")
    print(f"   • Unique Attack Types: {train_unique}")
    print(f"   • Categories: 5 (Normal, DoS, Probe, R2L, U2R)")
    
    print(f"\n✅ Testing Dataset:")
    print(f"   • Total Samples: {len(test_data):,}")
    print(f"   • Unique Attack Types: {test_unique}")
    print(f"   • Categories: 5 (Normal, DoS, Probe, R2L, U2R)")
    
    # Category distribution
    print(f"\n📊 Category Distribution:")
    print("-" * 80)
    
    train_data['category'] = train_data['attack_type'].apply(
        lambda x: attack_categories.get(x, 'Unknown')
    )
    test_data['category'] = test_data['attack_type'].apply(
        lambda x: attack_categories.get(x, 'Unknown')
    )
    
    train_cat_counts = train_data['category'].value_counts()
    test_cat_counts = test_data['category'].value_counts()
    
    print(f"\n{'Category':<15} {'Training':<15} {'Testing':<15}")
    print("-" * 45)
    for cat in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        train_count = train_cat_counts.get(cat, 0)
        test_count = test_cat_counts.get(cat, 0)
        print(f"{cat:<15} {train_count:>10,}     {test_count:>10,}")
    
    print("\n" + "="*80)
    print("✅ CONCLUSION")
    print("="*80)
    print("""
Your IDS model is trained on ALL these attack types from the NSL-KDD dataset.

The model learns to recognize patterns in network traffic and classify them into
5 main categories:

1. Normal   - Legitimate traffic
2. DoS      - Denial of Service attacks (10 types)
3. Probe    - Scanning/Reconnaissance attacks (6 types)
4. R2L      - Remote to Local attacks (16 types)
5. U2R      - User to Root privilege escalation (8 types)

When you run the live demo, the system detects these attacks based on the
patterns it learned during training!
    """)
    
    print("="*80)

if __name__ == "__main__":
    analyze_dataset()
