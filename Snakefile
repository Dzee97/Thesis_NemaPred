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


rule create_and_cluster_ref_18s_fasta:
    input:
        input_ncbi_fasta = PROCESSED_DATA_DIR / "NCBI_18S.fasta",
        input_silva_fasta = PROCESSED_DATA_DIR / "SILVA_18S.fasta"
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

rule process_trait_data:
    input:
        input_trait_genera = RAW_DATA_DIR / "Project_data_Nematode_traits" / "MarNemaFunDiv_Genera.xlsx",
        input_trait_data = RAW_DATA_DIR / "Project_data_Nematode_traits" / "MarNemaFunDiv_nematodes_traits (1).xlsx",
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl"
    output:
        output_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    script:
        "scripts/process_traits.py"

rule filter_18s_sequences:
    input:
        input_parquet = PROCESSED_DATA_DIR / "Sequence_store.parquet",
        input_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    params:
        min_length = 400,
        max_ambiguity = 0.01,
        otu_coverage = 0.7,
        dist_model = "tn93",
        model_gamma = 0.4,
        max_z_score = 3,
        genus_clusters = 3,
        min_length_frac = 0.8
    output:
        output_fasta = PROCESSED_DATA_DIR / "Filtered_18S_aligned.fasta",
        output_genus_cov = PROCESSED_DATA_DIR / "Filtered_18s_genus_coverage.csv"
    script:
        "scripts/filter_sequences.py"


rule create_18s_candidate_tree:
    input:
        input_fasta = PROCESSED_DATA_DIR / "Filtered_18S_aligned.fasta",
    params:
        model = "GTR+G+I",
        raxml_dir = RAXML_DATA_DIR,
        prefix_final = RAXML_DATA_DIR  / "Tree_18S"
    output:
        out_best_tree = RAXML_DATA_DIR / "Tree_18S.raxml.bestTree"
    shell:
        """
        mkdir -p {params.raxml_dir}
        raxml-ng --search --model {params.model} --msa {input.input_fasta} --prefix {params.prefix_final} --seed {SEED} 
        """

rule final_18s_sequences:
    input:
        input_parquet = PROCESSED_DATA_DIR / "Sequence_store.parquet",
        input_tree = RAXML_DATA_DIR / "Tree_18S.raxml.bestTree",
        input_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    params:
        num_genus_neighbors = 5
    output:
        output_fasta = PROCESSED_DATA_DIR / "Final_18S_aligned.fasta",
        output_selection = PROCESSED_DATA_DIR / "Final_18S_selection.csv",
        output_outgroup = PROCESSED_DATA_DIR / "Final_18S_aligned.outgroup",
        output_itol_class = PROCESSED_DATA_DIR / "Final_18S_class.annotation",
        output_itol_order = PROCESSED_DATA_DIR / "Final_18S_order.annotation",
        output_itol_family_dir = directory(PROCESSED_DATA_DIR / "Final_18S_family_annotation")
    script:
        "scripts/final_sequences.py"

rule create_18s_final_tree:
    input:
        input_fasta = PROCESSED_DATA_DIR / "Final_18S_aligned.fasta",
        input_outgroup = PROCESSED_DATA_DIR / "Final_18S_aligned.outgroup"
    params:
        model = "GTR+G+I",
        raxml_dir = RAXML_DATA_DIR,
        prefix_final = RAXML_DATA_DIR  / "Final_tree_18S",
        pars_trees = 50,
        rand_trees = 50
    output:
        out_best_tree = RAXML_DATA_DIR / "Final_tree_18S.raxml.bestTree"
    shell:
        """
        mkdir -p {params.raxml_dir}
        raxml-ng --search --model {params.model} --msa {input.input_fasta} --prefix {params.prefix_final} --seed {SEED} --outgroup $(cat {input.input_outgroup}) --tree pars{{{params.pars_trees}}},rand{{{params.rand_trees}}}
        """