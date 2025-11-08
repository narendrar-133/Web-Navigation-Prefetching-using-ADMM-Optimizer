# 🚀 Web Page Prefetching for Latency Reduction

This project implements an intelligent web content prefetching mechanism aimed at **significantly reducing user-perceived latency** by accurately predicting and loading the user's next intended web request.

The core of this system is a **Prefetching Model** integrated with an **Optimizer** to minimize request latency and maximize cache hit rates, thereby enhancing the overall Quality of Experience (QoE).

## 💡 Core Functionality

### **Prefetching Mechanism**

* **Next-Request Prediction:** Uses a learned model to predict $\mathbf{o'(t+1)}$ based on current request $\mathbf{c(t)}$ and history.
* **Intelligent Caching:** Prefetches the predicted content $\mathbf{o'(t+1)}$ into a **high-speed cache**.
* **Latency Mitigation:** Serves $\mathbf{c(t+1)}$ instantly from the cache if $\mathbf{c(t+1)} = \mathbf{o'(t+1)}$, bypassing traditional database lookup time.

### **Optimization & Feedback**

* **Optimal Selection:** The **Optimizer** determines the best content to prefetch, balancing prediction accuracy and the cost of bandwidth usage.
* **Continuous Learning:** The system incorporates a feedback loop, logging both the prediction $\mathbf{o'(t+1)}$ and the actual user request $\mathbf{c(t+1)}$ for iterative model improvement.

## 💾 Dataset & Data Preparation

The model is trained on a large-scale server log dataset (web traces), capturing extensive user clickstreams and access patterns.

### **Dataset Files**

Due to the large file size of the original logs, the processed data has been converted to optimized Excel formats for efficient storage.

* **`raw_logs_processed.xlsx`**: Cleaned and parsed event log data (`visitorid`, `timestamp`, `itemid`).
* **`user_sessions_final.xlsb`**: The final, **sessionized** dataset (`session_id`, `timestamp`, `url`). **.xlsb** provides significant file size reduction.

### **Data Pipeline Steps**

* **Log Parsing**: Extraction of essential features (Client ID, Timestamp, URL).
* **Static File Filtering**: Removal of common static content requests (images, CSS, JS) that do not represent user intent.
* **Sessionization**: Grouping events into distinct user visits using a **30-minute inactivity timeout**.

## ⚙️ Execution & Dependencies

### **Prerequisites**

* **Python 3.x**
* **Core Libraries**: `pandas`, `numpy`, `scikit-learn` (or equivalent ML library).
* **Excel Readers**: `openpyxl` (for XLSX) and `pyxlsb` (for XLSB).

### **Installation**

# Clone the repository
git clone [YOUR_REPO_URL]

# Install dependencies
pip install pandas numpy scikit-learn openpyxl pyxlsb


### **Running Scripts**

* **Model Training**:
    ```bash
    python model_train.py --data_file user_sessions_final.xlsb
    ```
* **Simulation/Evaluation**:
    ```bash
    python simulation.py --model_path [path/to/trained/model]
    ```

## 📈 Performance & Results

### **Key Metrics**

* **Latency Reduction (ms)**
* **Cache Hit Ratio (%)**
* **Bandwidth Overhead (%)**

### **Preliminary Findings**

*(***Note:*** *Replace these placeholders with your actual simulation results.*)

* **Achieved Hit Rate**: The model demonstrated an average **45% cache hit rate** for next-click prediction.
* **Observed Gain**: This resulted in an average **250ms latency reduction** for cached content.
* **Efficiency**: Bandwidth consumption from incorrect prefetches was maintained below **15%**.

## 🤝 Contribution

This project is open to contributions, bug reports, and suggestions. Feel free to fork the repository and submit a pull request!
