# Triplet Repeat Analysis Pipeline

This repository provides a Python pipeline for analyzing triplet repeat expansions from sequencing data (FASTQ/FASTQ.GZ).
It performs per-primer repeat selection, multithreaded batch processing, quality filtering, visualization, and summary generation.

## Features

* Interactive selection of primers and repeat motifs.
* Efficient parallel processing of FASTQ/FASTQ.GZ files.
* Automatic filtering of low-abundance repeat lengths.
* Export of:

  * **CSV files** with length distributions, proportions, and average quality.
  * **Plots** of repeat length distributions with quality-based coloring.
  * **Summary tables** of detected repeats with orientation, primer, and sequence context.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/triplet-repeat-analysis.git
cd triplet-repeat-analysis
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`:

* numpy
* pandas
* matplotlib
* biopython
* regex

## Usage

Run the script on a directory containing FASTQ/FASTQ.GZ files:

```bash
python triplet_analysis.py
```

You will be prompted to:

1. Select primers of interest (from built-in set).
2. Provide the repeat motif (defaults provided).

Results are saved in a subdirectory:

* `*_lengths.csv` → length distributions with counts, proportions, and quality.
* `*_dist.png` → visualization of repeat length distributions.
* `*_summary.txt` → tab-delimited summary of retained reads.

## Example

If you run:

```bash
python triplet_analysis.py
```

on a directory with `sample.fastq.gz`, outputs will be created in:

```
analysis_results/
  ├── sample_SCA1_lengths.csv
  ├── sample_SCA1_dist.png
  └── sample_SCA1_summary.txt
```

## Citation

If you use this code in an academic publication, please cite:

```
Your Name, Triplet Repeat Analysis Pipeline, GitHub repository,
https://github.com/YOUR_USERNAME/triplet-repeat-analysis
```

(Once linked with Zenodo, replace with the DOI here.)

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)).
