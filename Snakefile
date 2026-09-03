from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])
OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

rule download_silva:
    output:
        output_fasta = DATA_DIR / config["download"]["output_fasta"]
    params:
        base_url = config["download"]["base_url"],
        file_name = config["download"]["file_name"],
        target_taxon = config["download"]["target_taxon"]
    script:
        "scripts/01_download_silva.py"

rule preprocess_sequences:
    input:
        input_fasta = DATA_DIR / config["download"]["output_fasta"]
    output:
        output_aligned_fasta = DATA_DIR / config["preprocess"]["output_aligned_fasta"],
        output_ungapped_fasta = DATA_DIR / config["preprocess"]["output_ungapped_fasta"]
    params:
        min_ungapped_length = config["preprocess"]["min_ungapped_length"],
        max_ambiguity_fraction = config["preprocess"]["max_ambiguity_fraction"]
    script:
        "scripts/02_preprocess_sequences.py"