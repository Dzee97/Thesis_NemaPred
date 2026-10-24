from pathlib import Path

configfile: "config.yaml"

DATA_DIR = Path(config["project"]["data_dir"])

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TRAIT_DATA_DIR = RAW_DATA_DIR / "Project_data_Nematode_traits"

RAW_18S_METABARCODING_DATA_DIR = RAW_DATA_DIR / "Project_data_Nematode_traits" / "Metabarcoding_data" / "18S_metabarcoding"
RAW_COI_METABARCODING_DATA_DIR = RAW_DATA_DIR / "Project_data_Nematode_traits" / "Metabarcoding_data" / "COI_metabarcoding"

RAW_18S_NCBI_DATA_DIR = RAW_DATA_DIR / "Project_data_Nematode_traits" / "18S_references"
RAW_18S_SILVA_DATA_DIR = RAW_DATA_DIR / "SILVA"

RAW_COI_NCBI_DATA_DIR = RAW_DATA_DIR / "Project_data_Nematode_traits" / "COI_references"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_18S_DATA_DIR = PROCESSED_DATA_DIR / "18S"
PROCESSED_COI_DATA_DIR = PROCESSED_DATA_DIR / "COI"

RAXML_DATA_DIR = PROCESSED_DATA_DIR / "raxml"
PASTML_DATA_DIR = PROCESSED_DATA_DIR / "pastml"

OUTPUT_DIR = Path(config["project"]["output_dir"])
LOG_DIR = Path(config["project"]["log_dir"])

SEED = config["project"]["seed"]

# --------------------------------
# 18S Pipeline
# --------------------------------

rule download_ncbi_data:
    input:
        input_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl",
        entrez_cache = PROCESSED_DATA_DIR / "Entrez_cache.pkl"
    output:
        output_parquet = PROCESSED_DATA_DIR / "Sequence_store.parquet",
        output_18S_fasta = PROCESSED_DATA_DIR / "NCBI_18S.fasta",
        output_28S_fasta = PROCESSED_DATA_DIR / "NCBI_28S.fasta",
        output_COI_template_fasta = PROCESSED_DATA_DIR / "NCBI_COI_template.fasta",
        output_COI_fragment_fasta = PROCESSED_DATA_DIR / "NCBI_COI_fragment.fasta"
    script:
        "scripts/download_ncbi.py"

rule align_18s_references:
    input:
        input_arb = RAW_18S_SILVA_DATA_DIR / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_DATA_DIR / "NCBI_18S.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "NCBI_18S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"

rule align_28s_references:
    input:
        input_arb = RAW_18S_SILVA_DATA_DIR / "SILVA_138.2_LSURef_NR99_03_07_24_opt.arb",
        input_fasta = PROCESSED_DATA_DIR / "NCBI_28S.fasta"
    output:
        output_fasta = PROCESSED_DATA_DIR / "NCBI_28S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"

rule align_coi_references:
    input:
        input_template_fasta = PROCESSED_DATA_DIR / "NCBI_COI_template.fasta",
        input_fragment_fasta = PROCESSED_DATA_DIR / "NCBI_COI_fragment.fasta"
    output:
        output_template_fasta = PROCESSED_DATA_DIR / "NCBI_COI_template_aligned.fasta",
        output_all_fasta = PROCESSED_DATA_DIR / "NCBI_COI_all_aligned.fasta"
    shell:
        """
        mafft --localpair --maxiterate 1000 --preservecase --adjustdirection {input.input_template_fasta} > {output.output_template_fasta}
        mafft --localpair --maxiterate 1000 --addfragments {input.input_fragment_fasta} --preservecase --adjustdirection {output.output_template_fasta} > {output.output_all_fasta}
        """

rule create_18s_otus_fasta:
    input:
        input_otus = RAW_18S_METABARCODING_DATA_DIR / "18S_Metabarcoding_OTU_table_Indonesia.xlsx"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "OTUs_18S.fasta"
    script:
        "scripts/preprocess_otus.py"

rule create_18s_ncbi_fasta:
    input:
        input_fasta = RAW_18S_NCBI_DATA_DIR / "MarNemaFunDiv_18S_sequences_combined.fasta"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "NCBI_18S.fasta"
    script:
        "scripts/preprocess_ncbi.py"

rule create_18s_silva_fasta:
    input:
        input_fasta = RAW_18S_SILVA_DATA_DIR / "SILVA_Nematodes_SQ70_AQ70.fasta"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "SILVA_18S.fasta"
    script:
        "scripts/preprocess_silva.py"

rule create_18s_outgroup_fasta:
    input:
        input_fasta = RAW_18S_SILVA_DATA_DIR / "outgroups.fasta"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "Outgroups_18S.fasta"
    script:
        "scripts/preprocess_silva.py"

rule create_and_cluster_18s_ref_fasta:
    input:
        input_ncbi_fasta = PROCESSED_18S_DATA_DIR / "NCBI_18S.fasta",
        input_silva_fasta = PROCESSED_18S_DATA_DIR / "SILVA_18S.fasta"
    params:
        id = 0.99
    output:
        output_ref_fasta = PROCESSED_18S_DATA_DIR / "Ref_18S.fasta",
        output_ref_uc = PROCESSED_18S_DATA_DIR / "Ref_18S.uc"
    shell:
        """
        cat {input.input_ncbi_fasta} {input.input_silva_fasta} > {output.output_ref_fasta}
        vsearch --cluster_fast {output.output_ref_fasta} -id {params.id} -uc {output.output_ref_uc} --strand both
        """

rule create_18s_otus_alignment:
    input:
        input_arb = RAW_18S_SILVA_DATA_DIR / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_18S_DATA_DIR / "OTUs_18S.fasta"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "OTUs_18S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"

rule create_18s_ref_alignment:
    input:
        input_arb = RAW_18S_SILVA_DATA_DIR / "SILVA_144_SSURef_NR99_opt.arb",
        input_fasta = PROCESSED_18S_DATA_DIR / "Ref_18S.fasta"
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "Ref_18S_aligned.fasta"
    shell:
        "sina -i {input.input_fasta} -r {input.input_arb} -o {output.output_fasta} --turn"

rule process_18s_alignments:
    input:
        input_ref_fasta = PROCESSED_18S_DATA_DIR / "Ref_18S_aligned.fasta",
        input_ref_uc = PROCESSED_18S_DATA_DIR / "Ref_18S.uc",
        input_otu_fasta = PROCESSED_18S_DATA_DIR / "OTUs_18S_aligned.fasta",
        input_out_fasta = PROCESSED_18S_DATA_DIR / "Outgroups_18S.fasta",
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl"
    output:
        output_parquet = PROCESSED_18S_DATA_DIR / "Sequence_store.parquet"
    script:
        "scripts/process_sequences.py"

rule process_trait_data:
    input:
        input_trait_data = RAW_TRAIT_DATA_DIR / "MarNemaFunDiv_nematodes_traits (1).xlsx",
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl"
    output:
        output_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    script:
        "scripts/process_traits.py"

rule filter_18s_sequences:
    input:
        input_parquet = PROCESSED_18S_DATA_DIR / "Sequence_store.parquet",
        input_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv"
    params:
        min_length = 400,
        max_ambiguity = 0.01,
        otu_coverage = 0.7,
        genus_length_frac = 0.6,
        dist_model = "tn93",
        model_gamma = 0.4,
        max_z_score = 3,
        prune_min_dist = 0.05,
        family_silhouette_margin = 0.3,
    output:
        output_fasta = PROCESSED_18S_DATA_DIR / "Filtered_18S_aligned.fasta",
        output_outgroup = PROCESSED_18S_DATA_DIR / "Filtered_18S_aligned.outgroup",
        output_genus_cov = PROCESSED_18S_DATA_DIR / "Filtered_18s_genus_coverage.csv",
        output_ref_stats = PROCESSED_18S_DATA_DIR / "Filtered_18s_ref_stats.csv"
    script:
        "scripts/filter_sequences.py"

rule create_18s_tree:
    input:
        input_fasta = PROCESSED_18S_DATA_DIR / "Filtered_18S_aligned.fasta",
        input_outgroup = PROCESSED_18S_DATA_DIR / "Filtered_18S_aligned.outgroup"
    params:
        model = "GTR+G",
        raxml_dir = RAXML_DATA_DIR,
        prefix_final = RAXML_DATA_DIR  / "Tree_18S",
        pars_trees = 50,
        rand_trees = 50
    output:
        out_best_tree = RAXML_DATA_DIR / "Tree_18S.raxml.support"
    shell:
        """
        mkdir -p {params.raxml_dir}
        raxml-ng --all --model {params.model} --msa {input.input_fasta} --prefix {params.prefix_final} --seed {SEED} --outgroup $(cat {input.input_outgroup}) --tree pars{{{params.pars_trees}}},rand{{{params.rand_trees}}}
        """

rule run_18s_pastml:
    input:
        input_trait_data = PROCESSED_DATA_DIR / "Trait_data.csv",
        input_ml_tree = RAXML_DATA_DIR / "Final_tree_18S.raxml.support",
        input_bootstrap_trees = RAXML_DATA_DIR / "Final_tree_18S.raxml.bootstraps"
    params:
        output_acr_dir = PASTML_DATA_DIR
    output:
        output_tree = PASTML_DATA_DIR / "Final_tree_18S_annotated.nwk",
        output_tree_support = PASTML_DATA_DIR / "Final_tree_18S_support.csv"
    script:
        "scripts/run_pastml.py"

# --------------------------------
# COI Pipeline
# --------------------------------

rule create_coi_otus_fasta:
    input:
        input_otus = RAW_COI_METABARCODING_DATA_DIR / "COI_group_OTU_table_filtered_minSample_10000_minOTU_2_relative_0p001pct.xlsx"
    output:
        output_fasta = PROCESSED_COI_DATA_DIR / "OTUs_COI.fasta"
    script:
        "scripts/preprocess_otus.py"

rule create_coi_ncbi_fasta:
    input:
        input_fasta = RAW_COI_NCBI_DATA_DIR / "MarNemaFunDiv_mitochondrial_COI_sequences_combined.fasta"
    output:
        output_template_fasta = PROCESSED_COI_DATA_DIR / "NCBI_COI_Template.fasta",
        output_fragments_fasta = PROCESSED_COI_DATA_DIR / "NCBI_COI_Fragments.fasta"
    script:
        "scripts/preprocess_coi.py"

rule create_coi_ref_alignment:
    input:
        input_template_fasta = PROCESSED_COI_DATA_DIR / "NCBI_COI_Template.fasta",
        input_fragments_fasta = PROCESSED_COI_DATA_DIR / "NCBI_COI_Fragments.fasta"
    output:
        output_fasta = PROCESSED_COI_DATA_DIR / "Ref_COI_aligned.fasta"
    shell:
        "mafft --auto --preservecase --adjustdirection --addfragments {input.input_fragments_fasta} {input.input_template_fasta} > {output.output_fasta}"

rule create_and_cluster_coi_ref_fasta:
    input:
        input_fasta = PROCESSED_COI_DATA_DIR / "NCBI_COI_Fragments.fasta"
    params:
        id = 0.99
    output:
        output_ref_uc = PROCESSED_COI_DATA_DIR / "Ref_COI.uc"
    shell:
        "vsearch --cluster_fast {input.input_fasta} -id {params.id} -uc {output.output_ref_uc} --strand both"
    
rule create_coi_otus_alignment:
    input:
        input_ref_fasta = PROCESSED_COI_DATA_DIR / "Ref_COI_aligned.fasta",
        input_otus_fasta = PROCESSED_COI_DATA_DIR / "OTUs_COI.fasta"
    output:
        output_fasta = PROCESSED_COI_DATA_DIR / "OTUs_COI_aligned.fasta"
    shell:
        """
        mafft --auto --preservecase --adjustdirection --keeplength --addfragments {input.input_otus_fasta} {input.input_ref_fasta} > {output.output_fasta}
        sed -i -n '/^>OTU/,$p' {output.output_fasta}
        """

rule process_coi_alignments:
    input:
        input_ref_fasta = PROCESSED_COI_DATA_DIR / "Ref_COI_aligned.fasta",
        input_ref_uc = PROCESSED_COI_DATA_DIR / "Ref_COI.uc",
        input_otu_fasta = PROCESSED_COI_DATA_DIR / "OTUs_COI_aligned.fasta",
    params:
        worms_cache = PROCESSED_DATA_DIR / "WoRMS_cache.pkl"
    output:
        output_parquet = PROCESSED_DATA_DIR / "COI_Sequence_store.parquet"
    script:
        "scripts/process_sequences.py"