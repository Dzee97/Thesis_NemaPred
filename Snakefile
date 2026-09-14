from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])
OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

rule preprocess_sequences:
    input:
        input_fasta = DATA_DIR / "raw" / "arb-silva.de_2026-09-03_id1507588.tgz"
    output:
        output_database = DATA_DIR / config["preprocess"]["output_database"],
    script:
        "scripts/02_preprocess_sequences.py"