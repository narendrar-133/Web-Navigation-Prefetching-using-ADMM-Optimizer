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

## 🎯 Accuracy and Hit Rate Metrics

The table below details the predictive accuracy, showing how often the correct next-click was found in the top recommendations.

| Model | Top-1 Accuracy (%) | Top-3 Hit Rate (%) | Top-5 Hit Rate (%) |
| :--- | :--- | :--- | :--- |
| **Constrained Markov** | $51.02\%$ | $59.66\%$ | $61.59\%$ |
| **LSTM** | $72.3\%$ | $82.1\%$ | **$84.6\%$** |
| **BiLSTM** | **$73.0\%$** | $81.5\%$ | $83.7\%$ |

### Key Accuracy Findings
* **Best Top-5 Hit Rate:** The **LSTM** model achieved the highest overall predictive accuracy with a **$84.6\%$ Top-5 Hit Rate**, making it the most reliable for providing a list of strong recommendations.
* **Best Top-1 Accuracy:** The **BiLSTM** model narrowly outperformed the others in predicting the single most likely next-click with a **$73.0\%$ Top-1 Accuracy**.

---

## ⏱️ Prediction Latency Metrics (ms)

Prediction latency measures the time required for the model to generate a prediction, critical for real-time applications.

| Model | Average Prediction Latency (ms) | P95 Latency (ms) | P99 Latency (ms) |
| :--- | :--- | :--- | :--- |
| **Constrained Markov** | **$0.03$** | $0.00$ | $0.92$ |
| **LSTM** | $0.86$ | $1.08$ | $1.48$ |
| **BiLSTM** | $1.44$ | $1.57$ | **$1.98$** |

### Key Latency Findings
* **Lowest Latency:** The **Constrained Markov Model** is significantly faster, boasting an average prediction latency of only **$0.03ms$**, which is ideal for extremely low-latency requirements.
* **Worst-Case Latency (P99):** The **Constrained Markov Model** also maintains the best worst-case performance, with $99\%$ of predictions completing in under $0.92ms$.

---

## 📊 Summary and Trade-Off

| Model | Advantage | Trade-Off |
| :--- | :--- | :--- |
| **LSTM / BiLSTM** | **High Accuracy** (Top-5 Hit Rate $\sim 84\%$) | Higher Latency ($\sim 0.86ms$ to $1.44ms$ average) |
| **Constrained Markov** | **Extremely Low Latency** ($0.03ms$ average) | Lower Accuracy (Top-5 Hit Rate $\sim 61.6\%$) |


This project is open to contributions, bug reports, and suggestions. Feel free to fork the repository and submit a pull request!
