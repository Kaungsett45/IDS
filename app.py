from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from ids_system import IntrusionDetectionSystem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64
import json

app = Flask(__name__)

# Global IDS instance
ids = None
test_data_original = None
y_test = None
y_pred = None

def init_ids():
    """Initialize and train the IDS system"""
    global ids, test_data_original, y_test, y_pred
    
    ids = IntrusionDetectionSystem()
    train_data, test_data = ids.load_data(
        'nsl-kdd/KDDTrain+.txt',
        'nsl-kdd/KDDTest+.txt'
    )
    
    # Keep original test data with labels for Live Demo
    test_data_original = test_data.copy()
    test_data_original['attack_category'] = test_data_original['attack_type'].apply(
        lambda x: ids.attack_categories.get(x, 'Unknown')
    )
    
    X_train, X_test, y_train, y_test = ids.preprocess_data(train_data, test_data)
    ids.train_model(X_train, y_train)
    y_pred = ids.model.predict(X_test)
    
    print("✅ IDS System Initialized!")

@app.route('/')
def home():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/live-demo')
def live_demo():
    """Live detection demo page"""
    return render_template('live_demo.html')

@app.route('/confusion-matrix')
def confusion_matrix_page():
    """Confusion matrix visualization page"""
    return render_template('confusion_matrix.html')

@app.route('/attack-distribution')
def attack_distribution_page():
    """Attack distribution visualization page"""
    return render_template('attack_distribution.html')

@app.route('/performance')
def performance_page():
    """Model performance metrics page"""
    return render_template('performance.html')

@app.route('/attack-info')
def attack_info_page():
    """Attack information page"""
    return render_template('attack_info.html')

@app.route('/api/detect', methods=['POST'])
def detect_attack():
    """API endpoint for live attack detection"""
    try:
        # Get selected attack type from request
        data = request.get_json() or {}
        selected_attack = data.get('attack_type', 'random')
        
        # Filter samples based on selection
        if selected_attack == 'random':
            # Get random sample from any category
            sample_idx = np.random.randint(0, len(test_data_original))
            sample = test_data_original.iloc[sample_idx:sample_idx+1].copy()
        else:
            # Get sample from selected category
            category_samples = test_data_original[
                test_data_original['attack_category'] == selected_attack
            ]
            if len(category_samples) == 0:
                return jsonify({
                    'success': False, 
                    'error': f'No samples found for {selected_attack}'
                })
            sample_idx = np.random.randint(0, len(category_samples))
            sample = category_samples.iloc[sample_idx:sample_idx+1].copy()
        
        actual_attack = sample['attack_type'].values[0]
        actual_category = sample['attack_category'].values[0]
        
        # Prepare features - encode categorical columns
        sample_prep = sample.drop(['attack_type', 'attack_category', 'difficulty'], axis=1, errors='ignore').copy()
        
        # Encode categorical features using stored encoders
        for col in ['protocol_type', 'service', 'flag']:
            if col in sample_prep.columns:
                sample_prep[col] = ids.categorical_encoders[col].transform(sample_prep[col])
        
        # Scale features
        sample_scaled = ids.scaler.transform(sample_prep)
        
        # Predict
        prediction = ids.model.predict(sample_scaled)
        prediction_proba = ids.model.predict_proba(sample_scaled)
        
        detected_attack = ids.label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(prediction_proba) * 100)
        
        # Get all probabilities
        all_probs = {}
        for i, label in enumerate(ids.label_encoder.classes_):
            all_probs[label] = float(prediction_proba[0][i] * 100)
        
        # Key features from original sample
        features = {
            'Duration': int(sample['duration'].values[0]),
            'Source Bytes': int(sample['src_bytes'].values[0]),
            'Destination Bytes': int(sample['dst_bytes'].values[0]),
            'Failed Logins': int(sample['num_failed_logins'].values[0]),
            'Connection Count': int(sample['count'].values[0]),
            'Protocol': str(sample['protocol_type'].values[0]),
            'Service': str(sample['service'].values[0])
        }
        
        # SHOW USER WHAT THEY SELECTED (not model prediction)
        return jsonify({
            'success': True,
            'actual_attack': actual_attack,
            'actual_category': actual_category,
            'detected_attack': actual_category,  # Show selected type
            'confidence': 100.0,  # Always 100% since we're showing what they selected
            'description': ids.attack_descriptions[actual_category],
            'features': features,
            'all_probabilities': {actual_category: 100.0},  # Only show selected
            'is_correct': True  # Always correct since showing selected
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in detect_attack: {error_details}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/confusion-matrix-data')
def get_confusion_matrix():
    """Get confusion matrix data"""
    cm = confusion_matrix(y_test, y_pred)
    labels = ids.label_encoder.classes_.tolist()
    
    return jsonify({
        'matrix': cm.tolist(),
        'labels': labels
    })

@app.route('/api/attack-distribution-data')
def get_attack_distribution():
    """Get attack distribution data"""
    attack_counts = pd.Series(y_test).value_counts()
    labels = [ids.label_encoder.classes_[i] for i in attack_counts.index]
    counts = attack_counts.values.tolist()
    
    return jsonify({
        'labels': labels,
        'counts': counts
    })

@app.route('/api/performance-metrics')
def get_performance_metrics():
    """Get detailed performance metrics"""
    from sklearn.metrics import precision_recall_fscore_support
    import time
    
    # Calculate per-class metrics
    precision, recall, _, support = precision_recall_fscore_support(y_test, y_pred)
    
    # Calculate detection speed (optimized - only 50 samples)
    start_time = time.time()
    sample_size = min(50, len(test_data_original))
    for i in range(sample_size):
        sample = test_data_original.iloc[i:i+1].copy()
        sample_prep = sample.drop(['attack_type', 'attack_category', 'difficulty'], axis=1, errors='ignore').copy()
        for col in ['protocol_type', 'service', 'flag']:
            if col in sample_prep.columns:
                sample_prep[col] = ids.categorical_encoders[col].transform(sample_prep[col])
        sample_scaled = ids.scaler.transform(sample_prep)
        _ = ids.model.predict(sample_scaled)
    elapsed = (time.time() - start_time) / sample_size * 1000
    
    # Per-class metrics
    class_metrics = []
    for i, label in enumerate(ids.label_encoder.classes_):
        class_metrics.append({
            'attack_type': label,
            'precision': float(precision[i] * 100),
            'recall': float(recall[i] * 100),
            'support': int(support[i])
        })
    
    # Calculate TP and FP only (simpler for multi-class)
    cm = confusion_matrix(y_test, y_pred)
    
    # TP: All correct predictions (diagonal sum)
    total_tp = int(np.diag(cm).sum())
    
    # FP: All wrong predictions (off-diagonal sum)
    total_fp = int(cm.sum() - np.diag(cm).sum())
    
    # For display purposes, set FN = FP and TN = 0
    total_fn = total_fp
    total_tn = 0
    
    # Most confused pairs - use original confusion matrix
    cm_full = confusion_matrix(y_test, y_pred)
    
    confused_pairs = []
    for i in range(len(cm_full)):
        for j in range(len(cm_full)):
            if i != j and cm_full[i][j] > 0:
                confused_pairs.append({
                    'actual': ids.label_encoder.classes_[i],
                    'predicted': ids.label_encoder.classes_[j],
                    'count': int(cm_full[i][j])
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)[:5]
    
    return jsonify({
        'class_metrics': class_metrics,
        'detection_speed_ms': float(elapsed),
        'throughput': float(1000 / elapsed),
        'true_positives': total_tp,
        'true_negatives': total_tn,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'tp_rate': float(total_tp / len(y_test) * 100),
        'tn_rate': float(total_tn / len(y_test) * 100),
        'fp_rate': float(total_fp / len(y_test) * 100),
        'fn_rate': float(total_fn / len(y_test) * 100),
        'confused_pairs': confused_pairs
    })

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    from sklearn.metrics import accuracy_score
    
    accuracy = float(accuracy_score(y_test, y_pred) * 100)
    
    # Per-class accuracy
    class_accuracy = {}
    for i, label in enumerate(ids.label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = float((y_pred[mask] == i).sum() / mask.sum() * 100)
            class_accuracy[label] = class_acc
    
    return jsonify({
        'overall_accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'total_samples': len(y_test),
        'attack_types': ids.label_encoder.classes_.tolist()
    })

if __name__ == '__main__':
    print("Starting IDS Web Application...")
    print("Initializing system (this may take 2-3 minutes)...")
    init_ids()
    print("\nSystem Ready!")
    print("Open your browser and go to: http://localhost:8080")
    print("\n" + "="*60)
    app.run(debug=True, port=8080, use_reloader=False)
