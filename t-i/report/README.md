# CVPR-format RIPL assignment report

This directory contains the report source for the RIPL prospective-student assignment. It uses
the official CVPR 2026 LaTeX style. The current version contains the completed aligned Push-T T-I
experiment; T-II--T-IV sections should be extended as their experiments finish.

All headline T-I values are centralized as macros in `preamble.tex` so the abstract, result table,
status table, and conclusion cannot silently diverge. Plot measurements are stored in
`make_plots.py` and must match the backed-up JSON/TensorBoard artifacts.

The build requires a local Tectonic installation in addition to the Python `report` extra.

Generate the plots and build with:

```bash
cd t-i
python -m pip install -e '.[report]'
cd report
python make_plots.py
tectonic main.tex
cp main.pdf ../output/pdf/ripl_assignment_report.pdf
```

The verified PDF is copied to `../output/pdf/ripl_assignment_report.pdf`.
