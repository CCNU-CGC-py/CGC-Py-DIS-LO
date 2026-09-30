# DIS TMD Dijet Generator

C++ event generator for DIS dijets with a tabulated Sudakov-evolved
Weizsacker–Williams (WW) TMD gluon density and an optional PB-TMD shower.

## Prerequisites

- `g++` with C++17 support
- [Pythia8](https://pythia.org/) (headers + shared library)

Default Pythia paths in `Makefile_tmd`:

```text
PYTHIAINC = /home/congyi/pythia8310/include
PYTHIALIB = /home/congyi/pythia8310/lib
```

Override them if your install lives elsewhere:

```bash
make -f Makefile_tmd \
  PYTHIAINC=/path/to/pythia/include \
  PYTHIALIB=/path/to/pythia/lib
```

## Build

From this directory (`DIS/src`):

```bash
make -f Makefile_tmd clean
make -f Makefile_tmd
```

Or from the `DIS` directory:

```bash
make -C src -f Makefile_tmd
```

Successful builds produce:

```text
DIS/DIS_dijet_tmd.x
```

Object files stay in `DIS/src/*.o`. Remove them and the executable with:

```bash
make -f Makefile_tmd clean
```

## Runtime data

Run the executable from the `DIS` directory (paths in `main.cc` are relative
to that working directory):

```bash
cd /path/to/DIS
./DIS_dijet_tmd.x
```

Required inputs:

| File | Role |
|---|---|
| `src/FWW_TMD_TABLE.bin` | 3D WW TMD table (`x`, `k_T`, `Q^2`) |
| `src/Qs2_table.dat` | Saturation scale `Q_s^2(x)` (needed when `shower_on = 1`) |
| `src/dis_config.txt` | Run parameters (override with `DIS_CONFIG`) |

Example with a custom config:

```bash
DIS_CONFIG=src/dis_config.txt ./DIS_dijet_tmd.x
```

## Regenerate the TMD table (optional)

`FWW_TMD_TABLE.bin` is a `100 x 100 x 200` grid in `x`, `k_T`, and the
normalized logarithmic `Q^2` coordinate, with physical ranges

- `1e-7 <= x <= 1e-2`
- `1e-3 <= k_T <= 100 GeV`
- `Q_s^2(x) <= Q^2 <= 400 GeV^2`

From the project root, with a CUDA-enabled Python environment:

```bash
/home/congyi/anaconda3/envs/emulator/bin/python -B \
  DIS/src/fww_table_generator.py --device cuda \
  --qs2-table DIS/src/Qs2_table.dat \
  --work-directory fww_table_work \
  --output DIS/src/FWW_TMD_TABLE.bin
```

`TMDInterpolator` does trilinear interpolation in
`(log(x), log(k_T), eta)` and returns `-1` outside the table domain.

## Source layout

| File | Role |
|---|---|
| `main.cc` | Program entry point |
| `dis_event.*` | Event generation / hadronization glue |
| `dis_kinematics.hpp` | DIS kinematics helpers |
| `tmd_shower.*` | PB-TMD target-gluon shower |
| `TMDInterpolator.*` | Reader for `FWW_TMD_TABLE.bin` |
| `Qs2Interpolator.*` | Reader for `Qs2_table.dat` |
| `Makefile_tmd` | Build rules |
