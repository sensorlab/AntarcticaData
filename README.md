# AntarcticaData

Processing pipeline for a multi-band, multi-constellation GNSS dataset recorded
continuously aboard **R/V Laura Bassi** on a 216-day closed-loop voyage
**Trieste → Panama Canal → Ross Sea (Antarctica) → Cape Horn → Trieste**
(26 September 2025 to 29 April 2026). A u-blox **ZED-F9P-15B** receiver
(firmware HPG L1L5 1.40) with a Tersus AX4E02 antenna logged raw observables,
broadcast navigation, the RF spectrum of both bands (`MON-SPAN`) and receiver
health telemetry to one `.ubx` file per day (216 files, 176.9 GB).

- Data paper: Bertalanič, B., Dimc, F., and Blatnik, A.: *A multi-band,
  multi-constellation shipborne GNSS dataset from a 216-day closed-loop voyage
  between Europe and Antarctica*, Earth System Science Data (accepted).
  Discussion paper: <https://doi.org/10.5194/essd-2026-432>
- Dataset (Zenodo, CC-BY 4.0): <https://doi.org/10.5281/zenodo.20344100>
  (concept DOI, resolves to the latest version)
- Code: MIT licence. Version **v1.1** is the version used in the paper.

## Layout

```
.
├── ubx/          Binary UBX parser (mmap + structured NumPy dtypes, vectorised; no pyubx2)
├── analysis/     Trajectory, daily QC, multipath, RF spectrum, TEC, scintillation proxy,
│                 anomaly and spoofing catalogues, jamming confound
├── ppp/          PRIDE PPP-AR wrapper, ANTEX preparation, real-time vs PPP comparison
├── rinex/        convbin wrapper (RINEX 3.04 conversion)
├── figures/      Figure scripts of the paper
├── tables/       Table scripts of the paper (LaTeX + CSV)
├── verify/       Validation analyses of the paper, pytest tests and parser fixtures
├── config/       pipeline.yaml, receiver.yaml, ports.yaml
├── Snakefile     Per-day workflow over the 216 days (idempotent)
├── run_ppp.sh        RINEX conversion and kinematic PPP for all days
├── run_strengthen.sh PPP checks, real-time vs PPP comparison, PPP berth repeatability
└── run_revision.sh   Cruise-wide tables, statistics and figures of the paper
```

## Install

Python ≥ 3.11. Cartopy needs the GEOS and PROJ system libraries
(macOS: `brew install geos proj`; Debian/Ubuntu: `apt install libgeos-dev libproj-dev`).

```sh
git clone https://github.com/sensorlab/AntarcticaData.git
cd AntarcticaData
python -m venv .venv && source .venv/bin/activate
pip install -e .          # add [dev] for pytest, ruff and mypy
```

### External programs (RINEX and PPP)

Only the RINEX archive and the PPP comparison need two programs that cannot be
installed with `pip`. Parsing, the per-day analysis and most figures run with
Python alone. Both programs build with a standard Linux toolchain
(`gcc`, `gfortran`, `make`, `bc`) and must be on `PATH`.

| Program | Used for | Source | Notes |
|---|---|---|---|
| `convbin` | UBX → RINEX 3.04 (`rinex/convbin_runner.py`) | [RTKLIB demo5 fork (rtklibexplorer)](https://github.com/rtklibexplorer/RTKLIB) | The deposited RINEX files were written by `CONVBIN EX 2.5.0`. Mainline RTKLIB maps some F9P signals incorrectly (e.g. B2a as B2I). |
| `pdp3` | Kinematic PPP-AR (`ppp/pride_runner.py`) | [PRIDE PPP-AR](https://github.com/PrideLab/PRIDE-PPPAR) ([Geng et al., 2019](https://doi.org/10.1007/s10291-019-0888-1)) | The paper used version 3.2.8. Build with the upstream `install.sh`. |

```sh
# convbin (RTKLIB demo5, v2.5.0)
curl -sL https://github.com/rtklibexplorer/RTKLIB/archive/refs/tags/v2.5.0.tar.gz | tar xz
make -C RTKLIB-2.5.0/app/consapp/convbin/gcc CC="gcc -Wno-error=implicit-function-declaration"
# put RTKLIB-2.5.0/app/consapp/convbin/gcc/convbin on PATH

# pdp3 (PRIDE PPP-AR)
git clone https://github.com/PrideLab/PRIDE-PPPAR.git
cd PRIDE-PPPAR && ./install.sh
```

`ppp/pride_runner.py` passes the frequencies explicitly (`-frq G15 E15 C25 J15`),
because PRIDE's default needs GPS L2 and BeiDou B3I, which this receiver does not
track. It also passes the first epoch as the session start (`-s`). Otherwise pdp3
searches for the first epoch on a whole second; the receiver's epochs lie 9–20 ms
before the second, so the search takes tens of minutes and on some days starts the
session minutes before the end of the file.

**Receiver antenna calibration.** PRIDE takes the IGS20 antenna file named in each
day's clock product (for this cruise `igs20_2353` to `igs20_2415`) and downloads it
when it is missing. The downloaded file has no entry for the receiver antenna, and
PRIDE then applies zero phase-centre corrections without a warning. Run this once
before processing. It downloads those versions into PRIDE's table folder and appends
the `TRSAX4E02 NONE` calibration from the NGS file `ngs20.atx` (also in the dataset):

```sh
bash ppp/prepare_pride_tables.sh /path/to/ngs20.atx
```

`ppp/pride_runner.py` stops on any day whose antenna file still lacks the entry.

### Orbit, clock and ionosphere products

pdp3 downloads the orbit, clock and bias products itself (Wuhan University rapid
multi-GNSS products, `WUM0MGXRAP`, by default) and caches them in
`~/.PRIDE_PPPAR/data`. `verify/tec_vs_gim.py` downloads the CODE global ionosphere
maps (IONEX) from the anonymous AIUB server. Neither needs an account.

`ppp/igs_products.py` is an optional helper that downloads IGS final products from
[NASA CDDIS](https://cddis.nasa.gov/archive/gnss/products/). It is not needed for the
results of the paper. CDDIS requires a free [NASA Earthdata](https://urs.earthdata.nasa.gov)
account, configured once in `~/.netrc` (`chmod 600 ~/.netrc`):

```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

## Configure

The default `config/pipeline.yaml` expects the repository next to two folders, and the
run scripts write to the same `work/` folder:

```
parent/
├── AntarcticaData/          this repository
├── antarctica2026/data/     the daily YYYYMMDD.ubx files
└── work/                    all derived output (staging, derived, figures, tables, logs)
```

For another layout, edit the paths in `config/pipeline.yaml` (relative to the repository
folder unless absolute) and `UBX_DIR` at the top of `run_ppp.sh`.
`config/receiver.yaml` describes the receiver and antenna setup, and
`config/ports.yaml` lists the port stops used for the berth-repeatability analysis.

### Data access

The 216 daily `.ubx` files are on Zenodo (<https://doi.org/10.5281/zenodo.20344100>)
in eight monthly 7z archives (`ubx_YYYY_MM.7z`, 63.3 GB in total). Unpack them into
one folder and point `paths.ubx_dir` at it. The RINEX 3.04 files are in
`rinex3_all.tar`, one folder per day with `obs.crx.gz` (Hatanaka-compressed and
gzipped) and `nav.rnx.gz`.

## Quick start (one day)

```sh
# Parse one .ubx → 8 Parquet files + 1 Zarr block in work/staging/<yyyymmdd>/
python -m ubx.cli parse /path/to/20250930.ubx

# Per-day analysis
python -m analysis.qc_summary  --day 20250930
python -m analysis.trajectory  --day 20250930
python -m analysis.multipath   --day 20250930
python -m analysis.spectrum    --day 20250930
python -m figures.make_figures --day 20250930 --quick
```

## Full dataset

```sh
snakemake --cores 32 -p                  # default target: parse + analysis + figures
snakemake --cores 32 -p analysis_all     # only per-day analysis
snakemake --cores 32 -p figures          # only figure rendering
snakemake --cores 32 -p rinex_all        # RINEX 3.04 conversion (opt-in)
snakemake --cores 32 -p ppp_all          # kinematic PPP-AR (opt-in, heavy)
```

The workflow is idempotent (based on file modification times). Days are found by
scanning `paths.ubx_dir` for `YYYYMMDD.ubx`. On a 32-core workstation, parsing and
the per-day analysis of all files take about one hour.

### Products of the paper

After the per-day stages, three scripts reproduce the remaining products of the paper.
Run them from the repository folder with the environment active.

```sh
bash run_ppp.sh --jobs 8      # RINEX conversion and kinematic PPP for all days
bash run_strengthen.sh        # PPP checks, real-time vs PPP comparison (ppp.compare,
                              #   ppp.percentiles), PPP berth repeatability
bash run_revision.sh          # file inventory and gaps, framing integrity, observation
                              #   accuracy, multipath summaries, autocorrelation-aware
                              #   statistics, OGS cross-check and ship meteorology
```

PRIDE PPP-AR needs up to about 3 h per day of data, so the PPP of the whole cruise
takes two to three days with eight days processed in parallel.
`python -m verify.ppp_run_summary` lists, per day, the systems used, the antenna file
and any product fallbacks.

## Stages

1. **parse**: `.ubx` → per-message Parquet (`zstd-3`, 128 MB row groups) + Zarr for the MON-SPAN spectra.
2. **analysis**: trajectory, QC, code multipath (M1/M2 after Estey and Meertens), RF spectrum and RFI, TEC (validation), scintillation σ_φ proxy.
3. **figures and tables**: the figures and tables of the paper.
4. **PPP**: kinematic PRIDE PPP-AR per day, compared with the receiver's `NAV-HPPOSLLH` solution.
5. **RINEX**: `convbin` (RTKLIB demo5) → `obs.rnx` + `nav.rnx` (RINEX 3.04).

## Recorded UBX messages

| Class-ID | Name | Purpose | Rate |
|---|---|---|---|
| 0x02-0x15 | `RXM-RAWX` | Raw observables: pseudorange, carrier phase, Doppler, C/N₀, lock time, tracking status per signal | 2 Hz |
| 0x02-0x13 | `RXM-SFRBX` | Broadcast navigation subframes (ephemeris and almanac source) | per subframe |
| 0x02-0x14 | `RXM-MEASX` | Pre-PVT multi-band measurements | 2 Hz |
| 0x01-0x07 | `NAV-PVT` | Position, velocity and time fix | 2 Hz |
| 0x01-0x14 | `NAV-HPPOSLLH` | High-precision latitude, longitude and height | 2 Hz |
| 0x01-0x35 | `NAV-SAT` | Per-satellite C/N₀, elevation, azimuth, used flags | 2 Hz |
| 0x0A-0x38 | `MON-RF` | AGC, noise floor, jamming indicator, antenna status per RF block | 1 Hz |
| 0x0A-0x31 | `MON-SPAN` | RF spectrum, 2 RF blocks × 256 bins, 128 MHz span, 500 kHz per bin (RF0 centred at 1583.5 MHz, L1; RF1 at 1191.5 MHz, L2/L5) | 1 Hz |
| 0x0A-0x39 | `MON-SYS` | CPU load, run time, receiver temperature | 1 Hz |

The receiver tracks seven satellite systems (GPS, GLONASS, Galileo, BeiDou, QZSS,
NavIC and SBAS). GPS Block IIF/III, Galileo, BeiDou-3 and QZSS are recorded on L1 and
L5; GLONASS, SBAS, BeiDou-2 and the older GPS satellites on L1 only; NavIC on L5 only.

## Tests

```sh
pytest verify/                 # parser and schema tests
RUN_BIG=1 pytest verify/       # adds an end-to-end test on a full day of staged data
```

`verify/fixtures/sample_60s.ubx` is a 60-second clip for the parser round-trip and
schema tests. The validation analyses in `verify/` (e.g. `tec_vs_gim.py`,
`ogs_cross_validation.py`, `daily_inference.py`) need the full Parquet and Zarr
outputs and are not part of the default `pytest` run.

## References

Methods and external programs used by the pipeline:

- **RTKLIB demo5**: Everett, T. (rtklibexplorer fork), RINEX conversion (`convbin`).
  <https://github.com/rtklibexplorer/RTKLIB>
- **PRIDE PPP-AR**: Geng, J., Chen, X., Pan, Y., Mao, S., Li, C., Zhou, J., and Zhang, K.:
  PRIDE PPP-AR: an open-source software for GNSS PPP ambiguity resolution,
  GPS Solut., 23, 91, 2019. <https://doi.org/10.1007/s10291-019-0888-1>
- **u-blox documentation**: u-blox AG, *ZED-F9P-15B High Precision GNSS Module Data Sheet*
  (UBX-23009090) and *u-blox F9 HPG L1L5 1.40 Interface Description* (UBX-23006991).
- **Code multipath M1/M2**: Estey, L. H. and Meertens, C. M.: TEQC: The multi-purpose
  toolkit for GPS/GLONASS data, GPS Solut., 3, 42–49, 1999.
- **IGS products**: Johnston, G., Riddell, A., and Hausler, G.: The International GNSS
  Service, in: Springer Handbook of Global Navigation Satellite Systems, 967–982, 2017.

## Citation

If you use the code, please cite the paper and the dataset:

> Bertalanič, B., Dimc, F., and Blatnik, A.: A multi-band, multi-constellation
> shipborne GNSS dataset from a 216-day closed-loop voyage between Europe and
> Antarctica, Earth Syst. Sci. Data (accepted), discussion paper:
> <https://doi.org/10.5194/essd-2026-432>, 2026.

> Blatnik, A., Bertalanič, B., and Dimc, F.: A Comprehensive GNSS Dataset from the
> 41st Laura Bassi Expedition to Antarctica, Zenodo [data set],
> <https://doi.org/10.5281/zenodo.20344100>, 2026.

The DOI of the final paper will be added on publication.

## License

MIT, see `LICENSE`.
