"""Shared paths and settings for the CyberSift project."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
CLEAN_DATA_PATH = PROCESSED_DIR / "clean.parquet"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"
FEATURES_PATH = MODELS_DIR / "features.json"
RULE_THRESHOLDS_PATH = MODELS_DIR / "rule_thresholds.json"
DEMO_SAMPLE_PATH = SAMPLE_DIR / "cybersift_demo.csv"
RF_MODEL_PATH = MODELS_DIR / "cybersift_rf.joblib"
DAYSHIFT_RF_MODEL_PATH = MODELS_DIR / "rf_dayshift.joblib"
BASELINES_MODEL_PATH = MODELS_DIR / "baselines.joblib"
TUNING_CSV_PATH = TABLES_DIR / "tuning.csv"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_CSV_PATH = TABLES_DIR / "model_comparison.csv"
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"
PR_CURVE_FIGURE_PATH = FIGURES_DIR / "pr_curve.png"
ROC_CURVE_FIGURE_PATH = FIGURES_DIR / "roc_curve.png"
THRESHOLD_TRADEOFF_FIGURE_PATH = FIGURES_DIR / "threshold_tradeoff.png"
DETECTION_BY_ATTACK_CSV_PATH = TABLES_DIR / "detection_by_attack_type.csv"
DETECTION_BY_ATTACK_FIGURE_PATH = FIGURES_DIR / "detection_by_attack_type.png"
FEATURE_IMPORTANCE_CSV_PATH = TABLES_DIR / "feature_importance_top20.csv"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "feature_importance_top20.png"
DAYSHIFT_DETECTION_CSV_PATH = TABLES_DIR / "dayshift_detection_by_attack_type.csv"
DAYSHIFT_DETECTION_FIGURE_PATH = FIGURES_DIR / "dayshift_detection_by_attack_type.png"
ABLATION_CSV_PATH = TABLES_DIR / "ablation_destination_port.csv"
WORKLOAD_REDUCTION_CSV_PATH = TABLES_DIR / "workload_reduction.csv"
FALSE_NEGATIVES_CSV_PATH = TABLES_DIR / "errors_false_negatives.csv"
FALSE_POSITIVES_CSV_PATH = TABLES_DIR / "errors_false_positives.csv"
EDA_CLASS_BALANCE_CSV_PATH = TABLES_DIR / "eda_class_balance.csv"
EDA_ATTACK_TYPE_COUNTS_CSV_PATH = TABLES_DIR / "eda_attack_type_counts.csv"
TEST_LOG_PATH = REPORTS_DIR / "test_log.txt"
TEST_CASES_CSV_PATH = TABLES_DIR / "test_cases.csv"

RANDOM_STATE = 42
LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"
TARGET_COL = "is_attack"
DAY_COL = "day"
SPLIT_COL = "split"
DAYSHIFT_COL = "dayshift"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
DESTINATION_PORT_FEATURE = "Destination Port"
DEV_SAMPLE_FRAC = 0.1
EXPECTED_FEATURE_COUNT = 69
DEMO_SAMPLE_ROWS = 20_000
RARE_ATTACK_MAX_TEST_ROWS = 200
RULE_FLOOD_PERCENTILE = 0.99

DEV_TRAIN_FRAC = 0.1
TRAIN_CAP = 500_000
TUNE_CAP = 150_000
LOGREG_MAX_ITER = 300
DECISION_TREE_MAX_DEPTH = 10
CLASS_WEIGHT_BALANCED = "balanced"
RF_N_ESTIMATORS = 100
RF_TUNING_N_ESTIMATORS = 50
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_MAX_DEPTH_GRID = [12, 20]
RF_MIN_SAMPLES_LEAF_GRID = [1, 5]
RF_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
RF_SMALL_N_ESTIMATORS = 50
RF_SMALL_MAX_DEPTH = 15
RF_TINY_N_ESTIMATORS = 25
RF_TINY_MAX_DEPTH = 10
RF_MIN_N_ESTIMATORS = 10
RF_MIN_MAX_DEPTH = 8
ABLATION_TRAIN_ROWS = 300_000
ERROR_SAMPLE_ROWS = 20
RARE_LABEL_ROWS = 100
WORKLOAD_TOP_K = [1_000, 10_000]
RISK_LOW_MAX = 0.3
RISK_HIGH_MIN = 0.7
PRESET_LABELS = {
    "Normal browsing": ["BENIGN"],
    "Port scan": ["PortScan"],
    "DDoS flood": ["DDoS"],
    "Password guessing": ["FTP-Patator", "SSH-Patator"],
    "Bot": ["Bot"],
}

APP_PAGE_TITLE = "CyberSift"
APP_TAGLINE = "Filtering Network Noise, Finding Threats"
COURSE_LABEL = "AI2002 · Path A: Software (ML + Streamlit)"
DATASET_SOURCE = "CIC-IDS2017 labeled flow CSVs"
PREPROCESSING_STEPS = [
    "Remove exact duplicate flows before splitting.",
    "Drop duplicate and constant columns found in the dataset.",
    "Replace infinite values with missing values and convert features to numeric.",
    "Impute missing values with medians learned on the training split.",
]
REPORT_LIMITATIONS = [
    "The dataset comes from a 2017 laboratory environment.",
    "The attacks are simulated and may not match current real-world traffic.",
    "Some CIC-IDS2017 labels may be noisy or inconsistent.",
    "Very rare attacks have too few examples for a reliable score.",
    "The model sees flow summaries only; it does not inspect packet payloads.",
    "The model needs retraining over time as traffic and attack patterns change.",
]
GLOSSARY_ENTRIES = [
    ("Flow", "One recorded conversation between two network endpoints."),
    ("Feature", "A measured value describing one flow."),
    ("Label", "The dataset's recorded class for a flow."),
    ("Attack type", "A named category of malicious traffic in the dataset."),
    ("Imputation", "Filling a missing feature value with a learned replacement."),
    ("Random Forest", "A collection of decision trees whose outputs are combined."),
    ("Tree vote", "One tree's estimate of whether a flow is an attack."),
    ("Risk score", "The model's estimated attack probability from 0 to 1."),
    ("Threshold", "The risk score above which a flow becomes an alert."),
    ("Recall", "The share of known attacks that the model catches."),
]
APP_LAYOUT = "wide"
APP_TABS = [
    ":material/shield: Threat Desk",
    ":material/search: Inspect a Flow",
    ":material/monitoring: Live Monitor",
    ":material/assessment: Model Report Card",
    ":material/info: How it works",
]
DATA_SOURCE_OPTIONS = ["Demo traffic", "Upload CSV"]
MAX_UPLOAD_ROWS = 200_000
ALERT_THRESHOLD_MIN = 0.05
ALERT_THRESHOLD_MAX = 0.95
ALERT_THRESHOLD_STEP = 0.01
THREAT_DESK_FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "SYN Flag Count",
    "RST Flag Count",
]
THREAT_NORMAL_MAX_PCT = 5.0
THREAT_ELEVATED_MAX_PCT = 20.0
PRIORITY_QUEUE_SIZE = 5
PRIORITY_CRITICAL_MIN = 0.90
PRIORITY_HIGH_MIN = 0.70
FLOW_MAP_MAX_POINTS = 3_000
FLOW_MAP_TOP_RISK_LIMIT = 1_000
FLOW_MAP_PCA_COMPONENTS = 2
FLOW_MAP_CACHE_ENTRIES = 4
SCORED_INPUT_CACHE_ENTRIES = 2
UPLOAD_INPUT_CACHE_ENTRIES = 2
SERVICE_CHART_TOP_PORTS = 8
ALERT_LOG_LIMIT = 10
INSPECT_IMPORTANT_FEATURE_COUNT = 10
INSPECT_EXPLANATION_FEATURE_COUNT = 5
LIVE_MONITOR_BATCH_SIZE = 100
LIVE_MONITOR_ROWS_PER_PHASE = 400
LIVE_MONITOR_PHASES = [
    ("Normal traffic", "BENIGN"),
    ("DDoS burst", "DDoS"),
    ("Normal traffic", "BENIGN"),
    ("Port-scan burst", "PortScan"),
]
LIVE_MONITOR_SPEED_SECONDS = {"Slow": 0.65, "Normal": 0.25, "Fast": 0.05}
PORT_SERVICE_NAMES = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    123: "NTP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP",
    8080: "HTTP-alt",
}

# These columns are known to be duplicate or constant in CIC-IDS2017.
# clean_raw also checks the current data and drops any other constants it finds.
DROP_COLUMNS = [
    "Fwd Header Length.1",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
EDA_ATTACK_LABELS = ["DDoS", "PortScan", "DoS Hulk", "FTP-Patator", "Bot"]
EDA_BOXPLOT_SAMPLE_PER_LABEL = 20_000
EDA_CORRELATION_SAMPLE_ROWS = 100_000
FIGURE_DPI = 150

EDA_BOXPLOT_FEATURES = [
    "Flow Packets/s",
    "Flow Duration",
    "Total Fwd Packets",
    "SYN Flag Count",
    "Average Packet Size",
    "Flow IAT Mean",
]

EDA_MEDIAN_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Average Packet Size",
    "SYN Flag Count",
]
