from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAXML_DATA_DIR = PROCESSED_DATA_DIR / "raxml"

OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

SEED = config["project"]["seed"]

rule create_metabarcoding_18s_fasta:
    input:
        input_otus = RAW_DATA_DIR / "Project_data_Nematode_traits" / "Metabarcoding_data" / "18S_metabarcoding" / "18S_Metabarcoding_OTU_table_Indonesia.xlsx"
    output:
        output_fasta = PROCESSED_DATA_DIR / "Metabarcoding_18S_OTUs.fasta"
    script:
        "scripts/preprocess_otus.py"

rule create_metabarcoding_18s_alignment:
    input:
        input_arb = RAW_DATA_DIR / "SILVA" / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_DATA_DIR / "Metabarcoding_18S_OTUs.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "Metabarcoding_18S_OTUs_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta}"

rule cluster_marnemafundiv_18s_sequences:
    input:
        input_fasta = RAW_DATA_DIR / "Project_data_Nematode_traits" / "18S_references" / "MarNemaFunDiv_18S_sequences_combined.fasta"
    params:
        id = 0.99
    output:
        output_uc = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_combined.uc"
    shell:
        "vsearch --cluster_fast {input.input_fasta} --id {params.id} --uc {output.output_uc}"

rule create_marnemafundiv_18s_alignment:
    input:
        input_arb = RAW_DATA_DIR / "SILVA" / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = RAW_DATA_DIR / "Project_data_Nematode_traits" / "18S_references" / "MarNemaFunDiv_18S_sequences_combined.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta}"

rule process_marnemafundiv_18s_alignment:
    input:
        input_fasta = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_aligned.fasta",
        input_uc = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_combined.uc",
        input_outgroup = RAW_DATA_DIR / "SILVA" / "outgroups.fasta"
    output:
        output_parquet = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_processed.parquet"
    script:
        "scripts/preprocess_sequences.py"

rule filter_marnemafundiv_18s_alignment:
    input:
        input_parquet = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_processed.parquet"
    params:
        no_duplicates = False,
        no_contained = False,
        only_cluster_centroids = True,
        min_length = 500,
        dist_model = "tn93",
        model_gamma = 1.0,
        max_z_score = 1.0
    output:
        output_fasta = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_filtered.fasta"
    script:
        "scripts/filter_sequences.py"


rule tree_marnemafundiv_18s_alignment:
    input:
        input_fasta = PROCESSED_DATA_DIR / "MarNemaFunDiv_18S_sequences_filtered.fasta"
    params:
        model = "GTR+G+I",
        raxml_dir = RAXML_DATA_DIR,
        prefix_final = RAXML_DATA_DIR  / "Final_MarNemaFunDiv_18S_sequences"
    output:
        out_best_tree = RAXML_DATA_DIR / "Eval_MarNemaFunDiv_18S_sequences.raxml.bestTree"
    shell:
        """
        mkdir -p {params.raxml_dir}
        raxml-ng --search --model {params.model} --msa {input.input_fasta} --prefix {params.prefix_final} --seed {SEED}
        """
