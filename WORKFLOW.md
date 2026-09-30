# Portable DIS Dijet Generation and H1 Reweighting Workflow

This guide describes how to move this project to another computer, rebuild the
C++ event generator, generate a schema-v4 DIS event sample, reweight the sample,
and produce the charged-particle observables compared with H1 data.

The commands below use three shell variables:

```bash
export PROJECT_ROOT=/absolute/path/to/ep
export PYTHIA_PREFIX=/absolute/path/to/pythia8310
export PYTHON_ENV="$PROJECT_ROOT/.venv"
```

Replace the first two values for the new computer. Do not copy the literal
`/home/congyi/...` paths from the original machine.

## 1. Portability verdict

The source code, model checkpoints, numerical tables, configuration files, and
text event output can be transferred. The copied project is **not immediately
runnable without rebuilding and configuring the new machine**.

The important findings from the portability audit are:

| Location | Machine-specific content | Effect and action |
|---|---|---|
| `DIS/src/Makefile_tmd:6-7` | `/home/congyi/pythia8310/include` and `lib` | Active build defaults. Override `PYTHIAINC` and `PYTHIALIB` on every build, as shown below. Editing the Makefile is not required. |
| `DIS/src/main.cc:106` | `/home/congyi/pythia8310/share/Pythia8/xmldoc` | Active runtime fallback. Set `DIS_PYTHIA8DATA` when running. Setting only the standard `PYTHIA8DATA` variable is insufficient because this program overwrites it. |
| `DIS/DIS_dijet_tmd.x` | Original Pythia and Anaconda RPATHs, plus the Linux ELF loader | The existing executable is tied to the original x86-64 Linux environment. Do not reuse it; rebuild it. |
| `DIS/src/Readme.md` | Original Pythia paths and an original Anaconda Python path | Documentation examples only. Use this workflow instead. |
| `DIS/H1reweight.ipynb` saved outputs | Old `/home/congyi/...` display strings | Historical notebook output only; the executable code cells contain no absolute paths. Rerunning or clearing the outputs removes them. |
| `__pycache__/*.pyc` | Original absolute Python source filenames | Disposable cache files. Do not transfer them. |
| `.matplotlib-emulator/fontlist-*.json` | Original system font paths | Disposable Matplotlib cache. Do not transfer it. |

A string-level audit found no hard-coded absolute path inside the two `.pt`
checkpoints or the `FWW_TMD_TABLE.bin` payload.

There are also two working-directory assumptions:

- Run `DIS_dijet_tmd.x` from `$PROJECT_ROOT/DIS`; it opens `src/...` and
  `results/...` relative to the current directory.
- The first code cell of `DIS/H1reweight.ipynb` currently assumes that
  `Path.cwd()` is `$PROJECT_ROOT`. Standard Jupyter execution instead starts
  this notebook in `$PROJECT_ROOT/DIS`. Apply the small edit in section 7
  before running it on the new machine.

## 2. Files to transfer

For a from-scratch run, transfer at least:

```text
ep/
├── HankelIntegrator.py
├── NN_model.py
├── dipole_function.py
├── model_ciBK_best.pt
└── DIS/
    ├── H1reweight.ipynb
    ├── corss_section.py
    ├── dijet_emulator/
    │   ├── __init__.py
    │   ├── dijet_emulator.py
    │   └── model.pt
    └── src/
        ├── Makefile_tmd
        ├── *.cc
        ├── *.cpp
        ├── *.hpp
        ├── dis_config.txt
        ├── dis_config_dijet.txt
        ├── FWW_TMD_TABLE.bin
        └── Qs2_table.dat
```

The supplied `FWW_TMD_TABLE.bin` should normally be transferred rather than
regenerated. It is a required runtime input.

These generated files may be omitted:

```text
DIS/DIS_dijet_tmd.x
**/__pycache__/
*.pyc
.matplotlib-emulator/
DIS/results/
```

Keep `DIS/results/dis_dijet_tmd_output.txt` only if the new computer should skip
event generation and reweight the existing sample. The audited production file
is approximately 2.15 GB.

To verify a transfer, create a checksum manifest on the old computer:

```bash
cd "$PROJECT_ROOT"
sha256sum \
  model_ciBK_best.pt \
  DIS/dijet_emulator/model.pt \
  DIS/src/FWW_TMD_TABLE.bin \
  DIS/src/Qs2_table.dat \
  > transfer.sha256
```

Transfer `transfer.sha256` with the project, then run this on the new computer:

```bash
cd "$PROJECT_ROOT"
sha256sum -c transfer.sha256
```

## 3. System and Pythia environment

### Supported baseline

The current project was tested with:

- x86-64 Linux;
- GNU Make 4.4.1;
- a C++17 compiler (`g++` 15.2.0 in the audited environment);
- Pythia 8.310 headers, shared library, and XML data;
- Python 3.12.13 and PyTorch 2.x.

Other modern C++17 compilers and Pythia 8.3 releases may work. A smoke test also
compiled and ran successfully with an unpatched Pythia 8.312 installation.
However, use the same Pythia build for scientific reproducibility.

### Important Pythia reproducibility limitation

`Makefile_tmd` says that the original Pythia 8.310 library was patched for
per-beam primordial-\(k_T\) control. That patched source or patch file is **not
stored in this project**. The code in this directory uses the public Pythia API,
so a standard Pythia 8.3 installation can compile and run it, but bitwise or
physics-level equivalence to the original patched installation is not
guaranteed.

For a strict reproduction, obtain and archive the original Pythia 8.310 source
plus its patch, rebuild it on the new machine, and set `PYTHIA_PREFIX` to that
installation. For a functional run, install a compatible Pythia 8.3 release.
The official Pythia manual documents the standard `./configure`, `make`, and
`make install` flow:

<https://www.pythia.org/pdfdoc/pythia8300.pdf>

An example installation layout is:

```bash
cd /path/to/unpacked/pythia8310
./configure --prefix="$PYTHIA_PREFIX"
make -j"$(nproc)"
make install
```

Before building this project, verify the three paths it needs:

```bash
test -f "$PYTHIA_PREFIX/include/Pythia8/Pythia.h"
test -f "$PYTHIA_PREFIX/lib/libpythia8.so"
test -d "$PYTHIA_PREFIX/share/Pythia8/xmldoc"
```

On a platform where the shared library has a different suffix or layout, adapt
the Makefile link settings. The procedure in this guide was validated on Linux.

## 4. Python environment

Create the environment from the project root:

```bash
cd "$PROJECT_ROOT"
python3.12 -m venv "$PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib jupyterlab ipykernel
```

Install PyTorch separately so that its build matches the new computer's CPU,
NVIDIA CUDA, or AMD ROCm environment. Use the command generated by the official
selector:

<https://docs.pytorch.org/get-started/locally/>

For a CPU-only installation, a normal pip installation is sufficient:

```bash
python -m pip install torch
```

JupyterLab installation and launch commands are also documented at:

<https://jupyter.org/install>

Registering a named kernel is optional but makes the intended environment clear:

```bash
python -m ipykernel install \
  --user \
  --name ep-emulator \
  --display-name "Python (ep-emulator)"
```

Verify imports and hardware visibility:

```bash
python - <<'PY'
import matplotlib
import numpy
import pandas
import scipy
import torch

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

CUDA is optional. The model loaders and notebook fall back to the CPU, although
full production reweighting will be considerably slower there. For the audited
1,200,000-event production size, a CUDA-capable GPU, approximately 32 GB of
system memory, and several GB of free disk space are practical recommendations
rather than hard code requirements. Disk and memory needs grow with `nEvents`.

## 5. Build the event generator

Do not edit the original absolute paths into new absolute paths. Override them
at build time:

```bash
cd "$PROJECT_ROOT"
make -C DIS/src -f Makefile_tmd clean
make -C DIS/src -f Makefile_tmd \
  PYTHIAINC="$PYTHIA_PREFIX/include" \
  PYTHIALIB="$PYTHIA_PREFIX/lib"
```

The result is:

```text
DIS/DIS_dijet_tmd.x
```

Check that all shared libraries resolve on the new machine:

```bash
ldd DIS/DIS_dijet_tmd.x
```

There must be no `not found` entry, and `libpythia8.so` should resolve to
`$PYTHIA_PREFIX/lib`. The Makefile embeds the selected Pythia library directory
as an RPATH.

## 6. Configure and run event generation

### Choose the correct configuration

Create a local production configuration:

```bash
cd "$PROJECT_ROOT"
cp DIS/src/dis_config.txt DIS/src/dis_config.local.txt
```

Edit `DIS/src/dis_config.local.txt`. For the current H1 notebook, the following
values are required because the notebook validates them:

```text
Q2_min = 5.0
Q2_max = 10.0
y_min = 0.05
y_max = 0.6
back_to_back = 1
non_back_to_back = 1
shower_on = 1
output_frame = 1
E_lepton = 27.6
E_nucleon = 920.0
```

Keep the other validated production settings from `dis_config.txt`, including
`z_reg`, `k_max`, table-support limits, and fragmentation settings, unless the
analysis is intentionally being changed.

`nEvents` is the number of accepted events **per topology and polarization**.
With both topology switches enabled, the target output contains
`4 * nEvents` event headers. The audited configuration uses
`nEvents = 300000`, producing 1,200,000 event headers and a roughly 2.15 GB text
file.

Do not use `dis_config_dijet.txt` with the current H1 notebook: it uses a
different \(Q^2\), \(y\), and beam-energy range, so notebook validation will
stop.

### Small smoke test

First make a temporary configuration with `nEvents = 2` or `10`, then run from
the `DIS` directory:

```bash
cd "$PROJECT_ROOT"
cp DIS/src/dis_config.local.txt /tmp/dis_config_smoke.txt
sed -i -E 's/^nEvents[[:space:]]*=.*/nEvents = 2/' \
  /tmp/dis_config_smoke.txt

cd "$PROJECT_ROOT/DIS"
mkdir -p results
DIS_CONFIG=/tmp/dis_config_smoke.txt \
DIS_OUTPUT=results/dis_dijet_tmd_smoke.txt \
DIS_PYTHIA8DATA="$PYTHIA_PREFIX/share/Pythia8/xmldoc" \
MC_SEED=12345 \
./DIS_dijet_tmd.x
```

A successful smoke test ends with nonzero output for all four strata:

```text
Inelastic transverse
Inelastic longitudinal
Coherent transverse
Coherent longitudinal
```

### Production run

Use a fixed positive `MC_SEED` for reproducibility:

```bash
cd "$PROJECT_ROOT/DIS"
mkdir -p results
DIS_CONFIG=src/dis_config.local.txt \
DIS_OUTPUT=results/dis_dijet_tmd_output.txt \
DIS_PYTHIA8DATA="$PYTHIA_PREFIX/share/Pythia8/xmldoc" \
MC_SEED=12345 \
./DIS_dijet_tmd.x | tee results/dis_dijet_tmd.log
```

The generator does not create a missing parent directory for `DIS_OUTPUT`, so
run `mkdir -p results` first.

Validate the event file:

```bash
cd "$PROJECT_ROOT/DIS"
head -n 7 results/dis_dijet_tmd_output.txt
tail -n 4 results/dis_dijet_tmd_output.txt
rg -c '^EV ' results/dis_dijet_tmd_output.txt
```

Required properties are:

- the first line reports `schema_version=4`;
- the sampling metadata matches the H1 settings above;
- the last four lines contain generation statistics for both modes and both
  polarizations;
- with both modes enabled, the `EV` count is `4 * nEvents`.

## 7. Make the reweight notebook independent of its launch directory

In the first code cell of `DIS/H1reweight.ipynb`, replace:

```python
# Resolve paths relative to the notebook working directory (project root).
PROJECT_DIR = Path(".").resolve()
DIS_DIR = PROJECT_DIR / "DIS"
```

with:

```python
def find_project_root(start):
    """Find the transferred project root from the current notebook directory."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "DIS" / "src").is_dir()
            and (candidate / "model_ciBK_best.pt").is_file()
        ):
            return candidate
    raise FileNotFoundError("Could not locate the ep project root.")


PROJECT_DIR = find_project_root(Path.cwd().resolve())
DIS_DIR = PROJECT_DIR / "DIS"
```

This works whether Jupyter starts the kernel in `$PROJECT_ROOT` or
`$PROJECT_ROOT/DIS`, and it avoids inserting another machine-specific absolute
path.

The notebook prints a path named `H1 paper`. `H1Pt.pdf` is not present in this
project and is not read by the current code; the H1 Tables 8 and 9 values are
encoded in the notebook. The PDF is therefore optional documentation, not a
runtime dependency.

## 8. Run reweighting and produce observables

The notebook expects exactly:

```text
$PROJECT_ROOT/DIS/results/dis_dijet_tmd_output.txt
```

Activate the Python environment and start Jupyter:

```bash
cd "$PROJECT_ROOT"
source "$PYTHON_ENV/bin/activate"
export MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib-emulator"
jupyter lab DIS/H1reweight.ipynb
```

Select the `Python (ep-emulator)` kernel if it was registered, then run all
cells in order.

For a non-interactive run after applying the section 7 edit:

```bash
cd "$PROJECT_ROOT"
source "$PYTHON_ENV/bin/activate"
export MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib-emulator"
jupyter execute DIS/H1reweight.ipynb \
  --timeout=-1 \
  --output=H1reweight.executed
```

The first run computes hard cross sections and writes a cache. Later runs reuse
the cache only if the event-file SHA256, event count, model checksums,
quadrature order, and continuation prescription still match.

If GPU memory is limited, reduce these values before launching Jupyter:

```bash
export H1_WEIGHT_BATCH_SIZE=2000
export CROSS_SECTION_DIPOLE_BATCH_SIZE=64
```

The notebook defaults `CROSS_SECTION_N_RADIAL` to `64`. Changing it changes the
cache identity and recomputes weights.

### Observable and selections

The output observable is the absolute weighted charged-particle density

\[
\frac{1}{N}\frac{dn}{dp_T^*}
\]

for:

- \(5 < Q^2 < 10\ \mathrm{GeV}^2\);
- \(0.05 < y < 0.6\);
- three Bjorken-\(x_B\) bins;
- \(0 < \eta^* < 1.5\) and \(1.5 < \eta^* < 5\);
- particle cuts \(-2 < \eta_{\rm lab} < 2.5\),
  \(p_{T,\rm lab} > 0.15\ \mathrm{GeV}\), and
  \(0 < p_T^* < 10\ \mathrm{GeV}\).

The H1 event bins use \(x_B\). The elastic and inelastic hard cross sections use
\(x_g\), capped at `0.01` with the continuation documented in the notebook.
There is no unit-area normalization or post-hoc fit to H1.

### Produced files

The notebook creates:

```text
DIS/results/H1/h1_event_weights.npz
DIS/results/H1/h1_event_weights_metadata.json
DIS/results/H1/h1_q2_5_10_spectra.csv
DIS/results/H1/h1_q2_5_10_selection_summary.csv
DIS/results/H1/h1_q2_5_10_metrics.csv
DIS/results/H1/h1_q2_5_10_pt_comparison.png
DIS/results/H1/h1_q2_5_10_pt_comparison.pdf
```

Interpretation:

- `h1_q2_5_10_spectra.csv` contains all 49 model/H1 points, model Monte Carlo
  errors, H1 errors, ratios, and diagonal pulls.
- `h1_q2_5_10_selection_summary.csv` records selected-event and effective-event
  statistics.
- `h1_q2_5_10_metrics.csv` reports diagnostic diagonal chi-squared values and
  mean model/H1 ratios for the six spectra.
- The PNG and PDF are the final comparison figures.

The final notebook cell should report:

```text
Validated 49 model/data points in six H1 spectra.
```

Check the output inventory from the shell:

```bash
cd "$PROJECT_ROOT"
ls -lh DIS/results/H1/
python - <<'PY'
import pandas as pd

table = pd.read_csv("DIS/results/H1/h1_q2_5_10_spectra.csv")
assert len(table) == 49
print(table.groupby(["region", "x_bin"]).size())
PY
```

## 9. Optional FWW table regeneration

Regeneration is unnecessary for a transferred production run and is much more
expensive than copying the supplied 16 MB table. If the physics input is
intentionally changed, regenerate it from the project root with the active
Python environment:

```bash
cd "$PROJECT_ROOT"
source "$PYTHON_ENV/bin/activate"
python -B DIS/src/fww_table_generator.py \
  --device cuda \
  --qs2-table DIS/src/Qs2_table.dat \
  --work-directory fww_table_work \
  --output DIS/src/FWW_TMD_TABLE.bin
```

Use `--device cpu` only if CUDA is unavailable and the longer runtime is
acceptable.

## 10. Troubleshooting

| Symptom | Cause and correction |
|---|---|
| `Failed to load TMD data file: src/FWW_TMD_TABLE.bin` | The executable was launched outside `$PROJECT_ROOT/DIS`, or the table was not transferred. |
| `Failed to load Q_s^2 table` | Same working-directory problem, or `Qs2_table.dat` is missing. |
| Pythia initialization fails | `DIS_PYTHIA8DATA` is unset, incorrect, or belongs to a different Pythia installation. |
| `libpythia8.so => not found` | Rebuild with the correct `PYTHIALIB`; inspect with `ldd`. |
| Output file cannot be opened | Create its parent directory before running. |
| Notebook looks for `DIS/DIS/...` | Apply the project-root discovery edit in section 7. |
| Notebook reports unexpected sampling metadata | Use the H1-compatible values in `dis_config.txt`, not `dis_config_dijet.txt`. |
| Notebook reports an unexpected schema or strata | Regenerate with the current executable, schema 4, and both topology switches enabled. |
| CUDA is unavailable | CPU fallback is supported; expect a longer reweighting runtime. |
| Weights are recomputed unexpectedly | The event file, a model, a numerical setting, or the cache metadata changed. |

## 11. Reproducibility record

For a publishable or long-lived production, archive these together:

- this project source and `WORKFLOW.md`;
- the exact `dis_config.local.txt`;
- `MC_SEED`;
- compiler and GNU Make versions;
- the Pythia version, source, patch, build options, and installation identity;
- `python -m pip freeze`;
- GPU model, driver, and CUDA/ROCm versions;
- SHA256 checksums for both model checkpoints, both numerical tables, and the
  generated event file;
- the executed notebook and all CSV/PDF outputs.

This record is necessary because the current project has no pinned Python lock
file and does not include the stated Pythia patch.
