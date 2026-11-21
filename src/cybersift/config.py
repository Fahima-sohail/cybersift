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
APP_LAYOUT = "wide"
APP_TABS = [
    "Threat Desk",
    "Inspect a Flow",
    "Live Monitor",
    "Model Report Card",
    "How it works",
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
APP_HISTOGRAM_COLOR = "#00E5A8"
APP_THRESHOLD_LINE_COLOR = "#F2CC60"

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
