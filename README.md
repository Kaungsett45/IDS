#  Intrusion Detection System (IDS) - Network Attack Detection

##  What This Project Does

This Intrusion Detection System analyzes network traffic and identifies **5 different types** of cyber attacks in real-time. It helps understand **which attack is happening** and **what action to take**.

---

##  Attack Types Detected

### 1. **Normal Traffic**
- **What it is**: Legitimate network activity
- **Risk Level**: None
- **Action**: No action needed

### 2.  **DoS (Denial of Service)**
- **What it is**: Attacker floods the system to make it unavailable
- **Examples**: `neptune`, `smurf`, `back`, `teardrop`
- **Risk Level**: HIGH (8/10)
- **Action**: Block source IP, increase bandwidth
- **Real-world impact**: Website crashes, services become unavailable

### 3.  **Probe (Scanning)**
- **What it is**: Attacker scans network to find vulnerabilities
- **Examples**: `nmap`, `portsweep`, `ipsweep`, `satan`
- **Risk Level**: MEDIUM (6/10)
- **Action**: Monitor closely, strengthen firewall
- **Real-world impact**: Preparation for future attacks

### 4.  **R2L (Remote to Local)**
- **What it is**: Unauthorized access from remote machine
- **Examples**: `guess_passwd`, `ftp_write`, `imap`, `phf`
- **Risk Level**: CRITICAL (9/10)
- **Action**: Block immediately, check for data breach
- **Real-world impact**: Data theft, unauthorized access

### 5.  **U2R (User to Root)**
- **What it is**: Attacker gains admin/root privileges
- **Examples**: `buffer_overflow`, `rootkit`, `loadmodule`
- **Risk Level**: CRITICAL (10/10)
- **Action**: Isolate system immediately, investigate
- **Real-world impact**: Complete system compromise

---

## 📊 Dataset Information

### NSL-KDD Dataset

**What is NSL-KDD?**
- Industry-standard benchmark dataset for intrusion detection systems
- Improved version of the original KDD Cup 1999 dataset
- Contains realistic network attack patterns from simulated military network environment

**Dataset Statistics:**
- **Training Set**: 125,973 network connection records
- **Testing Set**: 22,544 network connection records
- **Features**: 41 network traffic features per connection
- **Classes**: 5 categories (Normal, DoS, Probe, R2L, U2R)

**Key Features (41 total):**
- Connection duration
- Protocol type (TCP, UDP, ICMP)
- Service type (HTTP, FTP, SMTP, etc.)
- Bytes transferred (source and destination)
- Failed login attempts
- Root shell access attempts
- Number of connections to same host
- And 34 more network behavior features

**Why NSL-KDD?**
- ✅ No duplicate records (unlike original KDD'99)
- ✅ Balanced class distribution
- ✅ Widely used in academic research
- ✅ Contains both known and novel attack types
- ✅ Realistic network traffic patterns

**Dataset Source:**
- Canadian Institute for Cybersecurity, University of New Brunswick
- Available at: https://www.unb.ca/cic/datasets/nsl.html

**Note:** The dataset files (`KDDTrain+.txt` and `KDDTest+.txt`) are not included in this repository due to size. Download them from the official source and place them in the `nsl-kdd/` directory.

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Download Dataset
1. Download NSL-KDD dataset from: https://www.unb.ca/cic/datasets/nsl.html
2. Extract `KDDTrain+.txt` and `KDDTest+.txt`
3. Create a folder named `nsl-kdd` in the project root
4. Place both files in the `nsl-kdd/` folder

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Web Application
```bash
python app.py
```
Open browser and go to: `http://localhost:8080`

### Step 4: Train Model (Optional)
```bash
python ids_system.py
```

---

## 📈 What You'll See

### 1. **Confusion Matrix**
Shows how accurately the system identifies each attack type

### 2. **Attack Distribution**
Bar chart showing frequency of each attack in the dataset

### 3. **Feature Importance**
Which network features are most important for detection

### 4. **Live Detection Results**
Real examples showing:
- Network traffic features
- Detected attack type
- Confidence level
- Recommended action

---

## 🎓 For Your Audience

### Key Points to Explain:

1. **The Problem**: 
   - Cyber attacks are increasing
   - Manual detection is impossible
   - Need automated system to identify threats

2. **The Solution**:
   - Machine Learning model trained on 125,000+ samples
   - Detects 5 attack categories with 95%+ accuracy
   - Provides real-time alerts

3. **How It Works**:
   - Analyzes 41 network features
   - Compares to known attack patterns
   - Classifies traffic as Normal or Attack type
   - Alerts security team with recommended action

4. **Real-World Impact**:
   - Prevents data breaches
   - Stops service disruptions
   - Protects sensitive information
   - Saves millions in potential damages

---

## 📁 Project Structure

```
Detection System/
├── app.py                    # Flask web application
├── ids_system.py              # Main IDS implementation
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore file
├── templates/                 # HTML templates
│   ├── index.html
│   ├── live_demo.html
│   ├── performance.html
│   └── ...
├── static/                    # CSS and assets
│   └── css/
│       └── style.css
└── nsl-kdd/                   # Dataset folder (not in repo)
    ├── KDDTrain+.txt          # Training data (download separately)
    └── KDDTest+.txt           # Testing data (download separately)
```

---

## 🔍 Technical Details

### Model: Ensemble (XGBoost + LightGBM)
- **XGBoost**: 60% weight - Gradient boosting for high accuracy
- **Ensemble Method**: Soft voting (probability averaging)
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling)
- **Training Time**: ~3-4 minutes on 125,973 samples

### Features Used (41 total):
- Connection duration
- Protocol type (TCP, UDP, ICMP)
- Service (HTTP, FTP, SMTP, etc.)
- Bytes transferred
- Failed login attempts
- Root access attempts
- And 35 more network features

### Performance Metrics:
- **Overall Accuracy**: ~85%
- **Training Samples**: 125,973
- **Testing Samples**: 22,544
- **True Positives**: 17,783
- **False Negatives**: 4,761

---


---


---

##  Why This Project Stands Out

✅ **Clear Attack Classification** - Each attack is labeled and explained  
✅ **Visual Presentation** - Charts and graphs for easy understanding  
✅ **Real-World Dataset** - Industry-standard NSL-KDD  
✅ **Actionable Insights** - Tells you what to do for each attack  
✅ **Live Demonstration** - Shows real-time detection  

---

## 📚 Learn More

- **NSL-KDD Dataset**: https://www.unb.ca/cic/datasets/nsl.html
- **Intrusion Detection**: https://en.wikipedia.org/wiki/Intrusion_detection_system
- **Random Forest**: https://scikit-learn.org/stable/modules/ensemble.html

---

##  Credits

- **Dataset**: NSL-KDD (Canadian Institute for Cybersecurity)
- **Model**: Scikit-learn Random Forest

---

