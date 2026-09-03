import hashlib
import gzip
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

if "snakemake" not in globals():
    from snakemake.script import Snakemake
    snakemake: Snakemake = None

DNA = set("ACGT")
IUPAC_CHARS = "ACGTRYSWKMBDHVN"
GAP_CHARS = "-.~_"

trans_dict = {}
# Standard DNA/IUPAC entries
for c in IUPAC_CHARS:
    trans_dict[ord(c)] = c
    trans_dict[ord(c.lower())] = c
# Fix RNA Uracil to Thymine
trans_dict[ord('U')] = 'T'
trans_dict[ord('u')] = 'T'
# Normalize Gaps
for c in GAP_CHARS:
    trans_dict[ord(c)] = '-'

# Create the translation table with 'N' as the fallback for anything else
NORMALIZE_TRANS = str.maketrans(trans_dict)
NORMALIZE_TRANS.update({i: ord('N') for i in range(256) if i not in trans_dict})

def ambiguity_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    dna_count = sequence.count('A') + sequence.count('C') + sequence.count('G') + sequence.count('T')
    return (len(sequence) - dna_count) / len(sequence)

def main() -> None:
    min_ungapped_length = snakemake.params.min_ungapped_length
    max_ambiguity_fraction = snakemake.params.max_ambiguity_fraction
    input_fasta = Path(snakemake.input.input_fasta)
    output_aligned_fasta = Path(snakemake.output.output_aligned_fasta)
    output_ungapped_fasta = Path(snakemake.output.output_ungapped_fasta)

    scanned = kept = dropped_length = dropped_ambiguity = dropped_duplicate = 0
    seen_hashes = set()

    try:
        with gzip.open(input_fasta, "rt", encoding="utf-8") as in_gz, \
             gzip.open(output_aligned_fasta, "wt", encoding="utf-8") as out_aligned_gz, \
             gzip.open(output_ungapped_fasta, "wt", encoding="utf-8") as out_ungapped_gz:

            for record in SeqIO.parse(in_gz, "fasta"):
                scanned += 1

                normalized_seq = str(record.seq).translate(NORMALIZE_TRANS)
                ungapped_seq = normalized_seq.replace("-", "")

                if len(ungapped_seq) < min_ungapped_length:
                    dropped_length += 1
                    continue

                if ambiguity_fraction(ungapped_seq) > max_ambiguity_fraction:
                    dropped_ambiguity += 1
                    continue

                seq_hash = hashlib.sha256(ungapped_seq.encode("utf-8")).hexdigest()
                if seq_hash in seen_hashes:
                    dropped_duplicate += 1
                    continue
                seen_hashes.add(seq_hash)

                # Step 5: High-speed Raw Writes 
                # Avoids object instantiation overhead of SeqRecord and Bio.SeqIO.write
                header = f">{record.description}\n"
                
                out_aligned_gz.write(header)
                out_aligned_gz.write(normalized_seq)
                out_aligned_gz.write("\n")
                
                out_ungapped_gz.write(header)
                out_ungapped_gz.write(ungapped_seq)
                out_ungapped_gz.write("\n")
                
                kept += 1

                if scanned % 5000 == 0:
                    print(
                        f"Processed {scanned:,} records | Kept: {kept:,} | "
                        f"Dropped (Len: {dropped_length:,}, Amb: {dropped_ambiguity:,}, Dup: {dropped_duplicate:,})", 
                        flush=True
                    )

        print("\n=== Filtering Summary ===")
        print(f"Total Scanned:         {scanned:,}")
        print(f"Dropped (Too Short):   {dropped_length:,}")
        print(f"Dropped (Ambiguity):   {dropped_ambiguity:,}")
        print(f"Dropped (Duplicates):  {dropped_duplicate:,}")
        print(f"Total Retained:        {kept:,}")
        print("=========================")

    except Exception:
        output_aligned_fasta.unlink(missing_ok=True)
        output_ungapped_fasta.unlink(missing_ok=True)
        raise

if __name__ == "__main__":
    main()