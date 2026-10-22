# Printable retained-study presentation draft

This A3 portrait PDF turns the existing modular poster text and retained figures into a single review layout. The author list, affiliations and final venue dimensions remain to be confirmed. It has not been submitted.

The original analysis, figures, reports and abstract are unchanged. The builder checks their exact input hashes before rendering. It performs no model execution, training, outcome generation or checkpoint access. The parent README and poster text contain the full scientific scope and limitations.

## Build

Use Python with ReportLab and a local DejaVu Sans font directory. From this directory:

```bash
python build_poster.py --output /tmp/FIM_Historical_Study_Poster.pdf
```

The default font directory is `/usr/share/fonts/truetype/dejavu`; use `--font-dir` to specify another installation. The builder embeds regular and bold TrueType fonts, requires a new output path and writes a receipt alongside the PDF. Source identities, tool versions and PDF digest are recorded in the receipt.

AI assistance was used for this layout and its condensed wording. Authors must review attribution, claims, source limitations and venue requirements before presentation. The A3 layout is a review artifact; no final conference poster-size compliance is asserted.
