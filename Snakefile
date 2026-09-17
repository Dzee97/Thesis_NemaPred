from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])
OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

SEED = config["project"]["seed"]

rule create_silva_nematode_18s_alignment:
    input:
        input_arb = DATA_DIR / "raw" / "SILVA" / "arb-silva.de_2026-09-16_id1509977_trunc.arb"
    output:
        output_fasta = DATA_DIR / "processed" / "SILVA_Nematode_18S_sequences_aligned.fasta"
    shell:
        "sina -i {input.input_arb} --prealigned -o {output.output_fasta}"

rule create_marnemafundiv_18s_alignment:
    input:
        input_arb = DATA_DIR / "raw" / "SILVA" / "arb-silva.de_2026-09-16_id1509977_trunc.arb",
        input_fasta = DATA_DIR / "raw" / "Project_data_Nematode_traits" / "18S_references" / "MarNemaFunDiv_18S_sequences_combined.fasta"
    output:
        output_fasta = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta}"

rule process_marnemafundiv_18s_alignment:
    input:
        input_fasta = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_aligned.fasta"
    params:
        header_type = "ncbi"
    output:
        output_parquet = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_processed.parquet"
    script:
        "scripts/01_preprocess_sequences.py"

rule filter_marnemafundiv_18s_alignment:
    input:
        input_parquet = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_processed.parquet"
    params:
        allow_duplicates = False,
        allow_contained = False,
        top_k_species_length = 1,
        top_k_genus_length = 5
    output:
        output_fasta = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_filtered.fasta"
    script:
        "scripts/02_filter_sequences.py"

rule phylogeny_marnemafundiv_18s_alignment:
    input:
        input_fasta = DATA_DIR / "processed" / "MarNemaFunDiv_18S_sequences_filtered.fasta"
    params:
        model = "GTR+G",
        prefix_start = DATA_DIR / "processed" / "Start_MarNemaFunDiv_18S_sequences",
        prefix_eval = DATA_DIR / "processed" / "Eval_MarNemaFunDiv_18S_sequences"
    #output:
        #output_tree = DATA_DIR / "processed" / "Tree_MarNemaFunDiv_18S_sequences.bestTree"
    shell:
        """
        raxml-ng --start --model {params.model} --msa {input.input_fasta} --tree pars{{10}} --prefix {params.prefix_start}
        raxml-ng --evaluate --model {params.model} --msa {input.input_fasta} --tree {params.prefix_start}.raxml.startTree --prefix {params.prefix_eval}
        """
        #"raxml-ng --msa {input.input_fasta} --model {params.model} --prefix {params.prefix} --seed {SEED}
    