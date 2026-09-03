import requests
import gzip
import io
from urllib.parse import urljoin
from pathlib import Path
from Bio import SeqIO

if "snakemake" not in globals():
    from snakemake.script import Snakemake
    snakemake: Snakemake = None

def main() -> None:
    base_url = snakemake.params.base_url
    file_name = snakemake.params.file_name
    target_taxon = snakemake.params.target_taxon
    output_fasta = Path(snakemake.output.output_fasta)

    url = urljoin(base_url, file_name)
    print(f"Streaming SILVA alignment: {url}")
    print(f"Keeping records matching taxon: {target_taxon}")

    scanned = kept = 0
    keep_current = first_content_seen = False
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            
            with gzip.GzipFile(fileobj=response.raw, mode="rb") as gz_in, \
                 io.TextIOWrapper(gz_in, encoding="utf-8", errors="replace") as text_in, \
                 gzip.open(output_fasta, "wt", encoding="utf-8") as out:

                for line in text_in:
                    if line.startswith(">"):
                        scanned += 1
                        first_content_seen = True

                        keep_current = target_taxon in line

                        if keep_current:
                            out.write(line)
                            kept += 1

                        if scanned % 100000 == 0:
                            print(f"  scanned {scanned:,}; kept {kept:,}", flush=True)

                    elif keep_current:
                        out.write(line)
                        
    except Exception:
        output_fasta.unlink(missing_ok=True)
        raise

    if not first_content_seen or kept == 0:
        output_fasta.unlink(missing_ok=True)
        raise RuntimeError("No valid FASTA data found or zero matching records kept.")

if __name__ == "__main__":
    main()
