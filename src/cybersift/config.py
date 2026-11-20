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

RANDOM_STATE = 42
LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"
DEV_SAMPLE_FRAC = 0.1
EXPECTED_FEATURE_COUNT = 69
DEMO_SAMPLE_ROWS = 20_000
RARE_ATTACK_MAX_TEST_ROWS = 200
RULE_FLOOD_PERCENTILE = 0.99

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
