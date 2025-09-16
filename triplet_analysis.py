# -*- coding: utf-8 -*-
"""
Modified Code (v9): Per‐primer repeat selection, multithreaded batch‐level processing,
filtering, plotting with conditional labels, and summary.
"""
import os
import sys
import gzip
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from Bio import SeqIO
from Bio.Seq import Seq
import regex
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --- Logging Setup ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("analysis.log", mode="w")
        ]
    )

# --- Interactive Primer & Repeat Selection ---
def select_primers_and_repeats(primers: Dict[str, List[Tuple[str,str]]]) -> Dict[str,str]:
    names = list(primers.keys())
    print("Available primers:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    choice = input("Select primers by numbers (comma-separated) or 'all': ").strip()
    if choice.lower() == "all":
        selected = names
    else:
        idxs = [int(x)-1 for x in choice.split(",") if x.strip().isdigit()]
        selected = [names[i] for i in idxs if 0 <= i < len(names)]
        if not selected:
            selected = names

    primer_repeats: Dict[str,str] = {}
    for p in selected:
        default = "GAA" if p == "FXN" else "CAG"
        rpt = input(f"Enter repeat for {p} (default={default}): ").strip().upper() or default
        primer_repeats[p] = rpt
    return primer_repeats

# --- Primer Pattern Compilation ---
class PrimerProcessor:
    def __init__(self, primers: Dict[str,List[Tuple[str,str]]], repeats: Dict[str,str]):
        self.primers = primers
        self.repeats = repeats

    def compile_primer_patterns(self, max_mismatches: int = 2
                               ) -> Dict[str,List[Tuple[regex.Pattern,str,str,str]]]:
        """
        Returns mapping:
          primer_name -> list of (pattern, primer1, primer2, repeat_unit)
        """
        primer_patterns: Dict[str,List[Tuple[regex.Pattern,str,str,str]]] = {}
        for pname, rpt in self.repeats.items():
            cfgs = self.primers[pname]
            compiled = []
            for fwd, rev in cfgs:
                pat = regex.compile(
                    f"({fwd}){{e<={max_mismatches}}}(.*?)({rev}){{e<={max_mismatches}}}",
                    flags=regex.BESTMATCH | regex.IGNORECASE
                )
                compiled.append((pat, fwd, rev, rpt))
            primer_patterns[pname] = compiled
        return primer_patterns

# --- Sequence Processor with Batch‐level Parallelism ---
class SequenceProcessor:
    def __init__(self, primer_patterns: Dict[str,List[Tuple[regex.Pattern,str,str,str]]]):
        self.primer_patterns = primer_patterns

    @staticmethod
    def calculate_repeat_percentage(seq: str, repeat: str) -> float:
        return (seq.count(repeat) * len(repeat) / len(seq)) * 100 if seq else 0

    def find_flanked_sequences(self, seq: str, pattern: regex.Pattern, repeat: str):
        m = pattern.search(seq)
        if not m:
            return None
        core = m.group(2)
        if not core:
            return None
        pct = self.calculate_repeat_percentage(core, repeat)
        if pct > 70:
            formatted = ">>>" + core.replace(repeat, "~") + "<<<"
            return core, formatted
        return None

    def process_single_record(self, record, rpt: str) -> Tuple[int,float,str,str,str,str]:
        """
        Returns (length, avg_quality, formatted_seq, orientation, primer_name, rpt)
        or None if no match.
        """
        seq = str(record.seq)
        qual_vals = record.letter_annotations["phred_quality"]
        avg_q = sum(qual_vals) / len(qual_vals) if qual_vals else 0
        seq_rc = None

        for pname, patterns in self.primer_patterns.items():
            for pat, fwd, rev, repeat in patterns:
                # forward
                res = self.find_flanked_sequences(seq, pat, repeat)
                if res:
                    length, formatted = len(res[0]), res[1]
                    return length, avg_q, formatted, "forward", pname, repeat
                # reverse
                if seq_rc is None:
                    seq_rc = str(Seq(seq).reverse_complement())
                res = self.find_flanked_sequences(seq_rc, pat, repeat)
                if res:
                    length, formatted = len(res[0]), res[1]
                    return length, avg_q, formatted, "reverse", pname, repeat
        return None

    def process_fastq(self, file_path: str,
                      batch_size: int = 1000) -> Tuple[List[int],List[float],List[Tuple[int,str,str,str]]]:
        """
        Process a single FASTQ/GZ file in parallel on record batches.
        Returns (lengths, qualities, formatted_tuples).
        """
        results = defaultdict(list)
        opener = gzip.open if file_path.endswith(".gz") else open
        with opener(file_path, "rt") as handle:
            records = SeqIO.parse(handle, "fastq")
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                batch = []
                for rec in records:
                    batch.append(rec)
                    if len(batch) >= batch_size:
                        futures.append(executor.submit(self._process_batch, batch))
                        batch = []
                if batch:
                    futures.append(executor.submit(self._process_batch, batch))

                for fut in as_completed(futures):
                    for out in fut.result():
                        length, avg_q, formatted, orient, pname, rpt = out
                        results["lengths"].append(length)
                        results["qualities"].append(avg_q)
                        results["formatted"].append((length, formatted, orient, pname, rpt))

        # sort by length
        results["formatted"].sort(key=lambda x: x[0])
        return results["lengths"], results["qualities"], results["formatted"]

    def _process_batch(self, batch: List, ) -> List[Tuple[int,float,str,str,str,str]]:
        out = []
        for rec in batch:
            try:
                res = self.process_single_record(rec, None)  # rpt is included in patterns
                if res:
                    out.append(res)
            except Exception:
                continue
        return out

# --- CSV & Plot Helpers ---
def calculate_proportions(counts: np.ndarray, window_radius: int = 3) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    kernel = np.ones(2*window_radius + 1)
    sums = np.convolve(counts, kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        props = counts / sums
        props[~np.isfinite(props)] = 0
    return props

def create_length_distribution_csv(lengths: List[int],
                                   qualities: List[float],
                                   output_path: str):
    arr_l = np.array(lengths)
    arr_q = np.array(qualities)
    bins = np.arange(arr_l.min(), arr_l.max()+2, 1)
    counts, edges = np.histogram(arr_l, bins=bins)
    props = calculate_proportions(counts)

    avg_q = []
    for i in range(len(edges)-1):
        mask = (arr_l >= edges[i]) & (arr_l < edges[i+1])
        avg_q.append(arr_q[mask].mean() if mask.any() else np.nan)

    df = pd.DataFrame({
        "Length": edges[:-1].astype(int),
        "Count": counts,
        "Proportion": np.round(props, 3),
        "Average_Quality": np.round(avg_q, 2)
    })
    df.to_csv(output_path, index=False)
    logging.info(f"Saved CSV: {output_path}")

def plot_combined_distributions(read_lengths, read_qualities, formatted, output_path):
    from pathlib import Path
    barcode_name = Path(output_path).stem
    total_reads = len(read_lengths)

    arr_l = np.array(read_lengths)
    arr_q = np.array(read_qualities)

    # 1) bins
    bins = np.arange(arr_l.min(), arr_l.max() + 2, 1)
    counts, edges = np.histogram(arr_l, bins=bins)

    # 2) avg quality per bin
    avg_q = []
    for i in range(len(edges)-1):
        mask = (arr_l >= edges[i]) & (arr_l < edges[i+1])
        avg_q.append(arr_q[mask].mean() if mask.any() else 0)

    # 3) build figure
    fig, ax = plt.subplots(figsize=(12,6))
    cmap = LinearSegmentedColormap.from_list("rg", ["red","yellow","green"])
    norm = Normalize(vmin=min(avg_q), vmax=max(avg_q))

    bars = ax.bar(edges[:-1], counts, width=1,
                  color=cmap(norm(avg_q)), align="edge", edgecolor="none")

    # 4) colorbar legend for quality
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Average Quality")

    # 5) (length,count) labels on non-zero bars
    for left, height in zip(edges[:-1], counts):
        if height > 0:
            ax.text(
                left + 0.5,
                height * 1.01,
                f"({int(left)},{int(height)})",
                ha="center",
                va="bottom",
                rotation=90,    # <— make label vertical
                fontsize=8
            )

    # 6) y-axis ticks at each order of magnitude
    max_count = counts.max() if counts.size>0 else 1
    max_order = int(np.ceil(np.log10(max_count)))
    yticks = [10**i for i in range(1, max_order+2)]
    ax.set_yscale("log")
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # 7) x-axis up to max_length + 5
    max_len = edges.max()
    ax.set_xlim(0, max_len + 5)

    ax.set_xlabel("Repeat Length")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{barcode_name}  •  Total Reads: {total_reads}")

    # 8) info box (median quality + top primer)
    med_q = np.median(arr_q)
    primer_counts = Counter(pname for (_,_,_,pname,_) in formatted)
    top_p, top_ct = primer_counts.most_common(1)[0]
    info = f"Median Q: {med_q:.2f}\nTop Primer: {top_p} ({top_ct})"
    ax.text(0.05, 0.95, info,
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))

    # 9) inset quality histogram
    axins = inset_axes(ax, width="20%", height="20%", loc="upper right",
                       bbox_to_anchor=(0,0,1,1),
                       bbox_transform=ax.transAxes)
    axins.hist(arr_q, bins=30)
    axins.set_title("Quality Dist.", fontsize=8)
    axins.set_yticks([])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved plot: {output_path}")

# --- Main Pipeline ---
def main(directory: str):
    setup_logging()
    output_dir = Path(directory)/"analysis_results"
    output_dir.mkdir(exist_ok=True)

    # primer dict inlined for clarity
    primers = {
        "SCA1": [("CGCCGGGACACAAGGCTGAG","CACCTCAGCAGGGCTCCGGG"),
                 ("CCCGGAGCCCTGCTGAGGTG","CTCAGCCTTGTGTCCCGGCG")],
        "SCA2": [("TCACCATGTCGCTGAAGCCC","CCGCCGCCCGCGGCTGCCAA"),
                 ("TTGGCAGCCGCGGGCGGCGG","GGGCTTCAGCGACATGGTGA")],
        "SCA3": [("TCACTTTTGAATGTTTCAGA","GGGGACCTATCAGGACAGAG"),
                 ("CTCTGTCCTGATAGGTCCCC","TCTGAAACATTCAAAAGTGA")],
        "SCA6": [("CCGGCGGCTCGGGGCCCCCG","GCGGTGGCCAGGCCGGGCCG"),
                 ("CGGCCCGGCCTGGCCACCGC","CGGGGGCCCCGAGCCGCCGG")],
        "SCA7": [("CAGCGGCCGCGGCCGCCCGG","CCGCCGCCTCCGCAGCCCCA"),
                 ("TGGGGCTGCGGAGGCGGCGG","CCGGGCGGCCGCGGCCGCTG")],
        "SCA12":[("CAGCCGCCTCCAGCCTCCTG","CTGCGAGTGCGCGCGTGTGG"),
                 ("CCACACGCGCGCACTCGCAG","CAGGAGGCTGGAGGCGGCTG")],
        "SCA17":[("TTTTGGAAGAGCAACAAAGG","GCAGTGGCAGCTGCAGCCGT"),
                 ("ACGGCTGCAGCTGCCACTGC","CCTTTGTTGCTCTTCCAAAA")],
        "HTT": [("TCGAGTCCCTCAAGTCCTTC","CAACAGCCGCCACCGCCGCC"),
                ("GGCGGCGGTGGCGGCTGTTG","GAAGGACTTGAGGGACTCGA")],
        "AR":  [("GCGCCAGTTTGCTGCTGCTG","CAAGAGACTAGCCCCAGGCA"),
                ("TGCCTGGGGCTAGTCTCTTG","CAGCAGCAGCAAACTGGCGC")],
        "FXN": [("TACAAAAAAAAAAAAAAAA","AATAAAGAAAAGTTAGCCGG"),
                ("CCGGCTAACTTTTCTTTATT","TTTTTTTTTTTTTTTTGTA")]
    }

    # select primers & repeats
    primer_repeats = select_primers_and_repeats(primers)

    # compile patterns
    pp = PrimerProcessor(primers, primer_repeats)
    primer_patterns = pp.compile_primer_patterns(max_mismatches=1)

    seqp = SequenceProcessor(primer_patterns)
    logging.info(f"Selected primers: {', '.join(primer_repeats.keys())}")

    fastq_files = [f for f in os.listdir(directory)
                   if f.endswith(".fastq") or f.endswith(".fastq.gz")]
    if not fastq_files:
        logging.error("No FASTQ files found.")
        sys.exit(1)

    start_time = time.time()
    for fq in fastq_files:
        fp = str(Path(directory)/fq)
        logging.info(f"Processing {fq}")
        lengths, quals, formatted = seqp.process_fastq(fp)
        if not lengths:
            logging.warning(f"No reads retained in {fq}")
            continue

        # filtering
        total = len(lengths)
        uniq, cnts = np.unique(lengths, return_counts=True)
        mask = cnts >= total * 0.001
        keep = set(uniq[mask])
        filt = [(l,q,f) for (l,q,f) in zip(lengths,quals,formatted) if l in keep]
        if not filt:
            logging.warning(f"All reads filtered out in {fq}")
            continue
        lengths, quals, formatted = zip(*filt)

        # basename + top primer
        stem = Path(fq).stem
        pc = Counter(p for (_,_,_,p,_) in formatted)
        top_p,_ = pc.most_common(1)[0]
        base = f"{stem}_{top_p}"

        # CSV
        csv_p = output_dir / f"{base}_lengths.csv"
        create_length_distribution_csv(list(lengths),
                                       list(quals),
                                       str(csv_p))

        # plot
        png_p = output_dir / f"{base}_dist.png"
        plot_combined_distributions(list(lengths),
                                    list(quals),
                                    list(formatted),
                                    str(png_p))

        # summary
        txt_p = output_dir / f"{base}_summary.txt"
        with open(txt_p, "w") as out:
            out.write("Length\tFormatted\tOrient\tPrimer\tRepeat\n")
            for l, fmt, orient, pname, rpt in formatted:
                out.write(f"{l}\t{fmt}\t{orient}\t{pname}\t{rpt}\n")
        logging.info(f"Wrote summary: {txt_p}")

    elapsed = time.time() - start_time
    logging.info(f"Total time: {elapsed:.2f}s")


if __name__=="__main__":
    setup_logging()
    target_directory = r""
    if not os.path.isdir(target_directory):
        logging.error(f"The directory {target_directory} does not exist.")
        sys.exit(1)
    main(target_directory)