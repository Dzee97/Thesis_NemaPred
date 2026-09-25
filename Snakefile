from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAXML_DATA_DIR = PROCESSED_DATA_DIR / "raxml"

OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

SEED = config["project"]["seed"]

rule create_otus_18s_fasta:
    input:
        input_otus = RAW_DATA_DIR / "Project_data_Nematode_traits" / "Metabarcoding_data" / "18S_metabarcoding" / "18S_Metabarcoding_OTU_table_Indonesia.xlsx"
    output:
        output_fasta = PROCESSED_DATA_DIR / "OTUs_18S.fasta"
    script:
        "scripts/preprocess_otus.py"

rule create_ncbi_18s_fasta:
    input:
        input_fasta = RAW_DATA_DIR / "Project_data_Nematode_traits" / "18S_references" / "MarNemaFunDiv_18S_sequences_combined.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "NCBI_18S.fasta"
    script:
        "scripts/preprocess_ncbi.py"

rule create_silva_18s_fasta:
    input:
        input_fasta = RAW_DATA_DIR / "SILVA" / "SILVA_Nematodes_SQ70_AQ70.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "SILVA_18S.fasta"
    script:
        "scripts/preprocess_silva.py"

rule create_outgroup_18s_fasta:
    input:
        input_fasta = RAW_DATA_DIR / "SILVA" / "outgroups.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "Outgroups_18S.fasta"
    script:
        "scripts/preprocess_silva.py"

rule create_bold_18s_fasta:
    input:
        input_tsv = RAW_DATA_DIR / "BOLD" / "BOLD_Nematodes.tsv"
    output:
        output_fasta = PROCESSED_DATA_DIR / "BOLD_18s.fasta"
    script:
        "scripts/preprocess_bold.py"


rule create_and_cluster_ref_18s_fasta:
    input:
        input_ncbi_fasta = PROCESSED_DATA_DIR / "NCBI_18S.fasta",
        input_silva_fasta = PROCESSED_DATA_DIR / "SILVA_18S.fasta"
        #input_bold_fasta = PROCESSED_DATA_DIR / "BOLD_18s.fasta"
    params:
        id = 0.99
    output:
        output_ref_fasta = PROCESSED_DATA_DIR / "Ref_18S.fasta",
        output_ref_uc = PROCESSED_DATA_DIR / "Ref_18S.uc"
    shell:
        """
        cat {input.input_ncbi_fasta} {input.input_silva_fasta} > {output.output_ref_fasta}
        vsearch --cluster_fast {output.output_ref_fasta} -id {params.id} -uc {output.output_ref_uc} --strand both
        """

rule create_otus_18s_alignment:
    input:
        input_arb = RAW_DATA_DIR / "SILVA" / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_DATA_DIR / "OTUs_18S.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "OTUs_18S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"


rule create_ref_18s_alignment:
    input:
        input_arb = RAW_DATA_DIR / "SILVA" / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_DATA_DIR / "Ref_18S.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "Ref_18S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"

rule process_18s_alignments:
    input:
        input_ref_fasta = PROCESSED_DATA_DIR / "Ref_18S_aligned.fasta",
        input_ref_uc = PROCESSED_DATA_DIR / "Ref_18S.uc",
        input_otu_fasta = PROCESSED_DATA_DIR / "OTUs_18S_aligned.fasta",
        input_out_fasta = PROCESSED_DATA_DIR / "Outgroups_18S.fasta",
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl"
    output:
        output_parquet = PROCESSED_DATA_DIR / "Sequence_store.parquet"
    script:
        "scripts/process_sequences.py"

rule filter_18s_sequences:
    input:
        input_parquet = PROCESSED_DATA_DIR / "Sequence_store.parquet",
        input_trait_genera = RAW_DATA_DIR / "Project_data_Nematode_traits" / "MarNemaFunDiv_Genera.xlsx"
    params:
        no_duplicates = False,
        no_contained = False,
        only_centroids = True,
        min_length = 300,
        max_ambiguity = 0.01,
        otu_coverage = 0.8,
        dist_model = "tn93",
        model_gamma = 0.4,
        max_z_score = 3,
        k_closest = 20
    output:
        output_fasta = PROCESSED_DATA_DIR / "Filtered_18S_aligned.fasta",
        output_outgroup = PROCESSED_DATA_DIR / "Filtered_18S_aligned.outgroup",
        output_genus_cov = PROCESSED_DATA_DIR / "Filtered_18s_genus_coverage.csv"
    script:
        "scripts/filter_sequences.py"


rule create_18s_tree:
    input:
        input_fasta = PROCESSED_DATA_DIR / "Filtered_18S_aligned.fasta",
        input_outgroup = PROCESSED_DATA_DIR / "Filtered_18S_aligned.outgroup"
    params:
        model = "GTR+G+I",
        raxml_dir = RAXML_DATA_DIR,
        prefix_final = RAXML_DATA_DIR  / "Tree_18S"
    output:
        out_best_tree = RAXML_DATA_DIR / "Tree_18S.raxml.bestTree"
    shell:
        """
        mkdir -p {params.raxml_dir}
        raxml-ng --search --model {params.model} --msa {input.input_fasta} --prefix {params.prefix_final} --seed {SEED} --outgroup $(cat {input.input_outgroup}) --tree pars{{25}},rand{{25}}
        """
