# EO Embedding Leakage Study

How much object-level information leaks from compact Earth observation (EO) embeddings, and how does that leakage trade off against downstream utility as the embeddings are compressed?

Project statement: *SatSplit-Bench: accuracy-bytes-compute-privacy benchmark for split inference on EO tasks over orbit-realistic ISLs, with an object-level feature-inversion leakage metric.* Phases 0–7 cover accuracy, bytes and privacy, including the object-level inversion metric, for a split after the full encoder. Phase 8 adds compute, intermediate split points and orbit-realistic ISLs. All of these are done; Phase 9 is the paper.

## Research questions

- **RQ1, attribute leakage:** can a linear probe on an embedding detect objects it was not used for (solar farms), above chance?
- **RQ2, trade-off:** as embeddings are compressed (PCA dims, int8), does leakage fall faster or slower than utility?
- **RQ3, metric mismatch:** do pixel metrics (SSIM, LPIPS) of an inverted image rank encoders the same way as object-level leakage?
- **RQ4, split point** (Phase 8a/8b): how do leakage, utility, payload bytes and on-board compute change with the split point inside the encoder?
- **RQ5, ISL feasibility** (Phase 8c): which payloads fit real inter-satellite link rates and contact windows, and what utility and leakage do the feasible ones carry?

## Main findings (Phases 4b–8; details and CIs in `RESULTS.md`)

- **Presence leaks.** Every audited embedding reveals whether a tile contains solar panels (test AUC 0.90–0.95). Much of this is spectral (65 per-band pixel statistics alone reach 0.91), but adding the embedding to those statistics still raises AUC by +0.016 to +0.039 (fragile for ResNet18-MoCo across seeds). ViTs leak more than CNNs. Stronger attackers add ≤ 0.005 AUC; a CNN on raw pixels reaches 0.970.
- **Location does not leak from pooled embeddings.** Grid classifiers, a GAN inverter and a transformer localiser all land within 0.01 IoU of a position-only prior. Spatial payloads are different: feature maps sent at a split beat the prior by +0.07 to +0.19 IoU (Phase 8b), and beat the pooled vector on the same tiles in 8/8 tests. Most of this is the image layout itself, though. The raw tile pooled to the same 4×4 grid already gets +0.13, and the encoders add at most about +0.05 (Phase 8d, exploratory).
- **Compression is not a defence.** PCA and quantisation cut utility at least as fast as leakage.
- **Pixel metrics are not leakage metrics** (RQ3): LPIPS does not track object-level leakage; SSIM tracks only the reconstruction-based measures.
- **The split point does not control presence leakage** (RQ4, Phase 8b, pre-registered; replicated on 3 split seeds in 8d). Leakage is already near its maximum after the first stage, within ≤ 0.02 AUC across splits. Utility mostly rises with depth. Early ResNet maps are larger than the zlib-compressed tile (R50 layer1: 2.46× raw in fp32).
- **Real ISLs carry embeddings, but none of the feasible high-utility payloads is private** (RQ5, Phase 8c). Sentinel-2A↔2B have continuous line of sight at about 4,460 km, and LEO→GEO relays are visible about 70% of the day. At S-band only the 4-bit PCA codes fit live (32 B, EuroSAT 0.963). At Ka-band the embeddings fit while raw tiles don't. At 1.8 Gbit/s optical, even zlib-compressed raw tiles flow live. Every feasible high-utility payload leaks at AUC 0.92–0.94.
- **An adversarial defence cuts leakage at a cost** (Phase 7, pre-registered): −0.018 AUC against fresh attackers at full size, while BigEarthNet macro AP (−0.028) and EuroSAT accuracy (−0.010) fail non-inferiority. Combined with PCA-16 the reduction is −0.124 AUC (exploratory). The in-training adversary is fooled; fresh attackers are not.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Environment setup | Done |
| 1 | Data acquisition | Done |
| 2 | Embeddings | Done |
| 3 | Compression | Done |
| 4 | Probes and controls (linear + MLP) | Done |
| 4b | Learned-leakage follow-ups (pre-registered) | Done |
| 5 | Inversion: weak and strong inverters | Done |
| 6 | Localisation attack | Done |
| 7 | Own encoder and defence | Done: "leakage cut at a cost" (leakage −0.018 AUC, utility fails non-inferiority) |
| 8a | Compute per split (FLOPs, parameters, latency, bytes) | Done |
| 8b | Split points inside the encoder (pre-registered 18 Nov) | Done: depth adds presence leakage "not supported" (1/4); early splits leak location "mixed" (2/4, ResNets only) |
| 8c | Orbit-realistic ISLs (real orbits, link rates, contact windows) | Done |
| 8d | Follow-ups to 8b (exploratory): pixel-map control, `gap` vs `map4` location, split seeds 1–2 | Done: the significant 8b effects replicate in 3/3 seeds |
| 9 | Analysis and paper | Planned (deadline 21 Nov 2026, 23:59) |

---

## How to run

Phases 0–8d are done; results are in `RESULTS.md`. The Phase 8 commands are in [Phase 8 scripts and how to run them](#phase-8-scripts-and-how-to-run-them). The commands below re-run a phase: every runner skips runs that already
finished and resumes where it stopped. The runs use the GPU, so run them one after the other, and keep other heavy apps closed
(memory pressure made Phase 6 stall once). Start long jobs detached (`nohup ... &` from a system terminal, not an editor's
terminal), so closing the editor cannot kill them.

> **Before a long run: stop the machine from sleeping, and re-enable sleep afterwards.** `caffeinate -i` (macOS) and
> `systemd-inhibit` (Linux) only block *idle* sleep while the command runs; closing a laptop lid still sleeps it. To keep
> running with the lid closed, disable sleep system-wide (stay on AC power and keep the laptop on a hard surface).
>
> | OS | Disable sleep | Re-enable sleep | Check |
> |---|---|---|---|
> | macOS | `sudo pmset -a disablesleep 1` | `sudo pmset -a disablesleep 0` | `pmset -g \| grep -i sleepdisabled` (1 = disabled) |
> | Linux (systemd) | `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target` | `sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target` | `systemctl status sleep.target` |
> | Windows (PowerShell as admin) | `powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0` | `powercfg /change standby-timeout-ac 30; powercfg /change hibernate-timeout-ac 180` (or your previous values) | `powercfg /query SCHEME_CURRENT SUB_SLEEP` |
>
> Lid closed: on Linux also set `HandleLidSwitch=ignore` (and `HandleLidSwitchExternalPower=ignore`) in
> `/etc/systemd/logind.conf`, then `sudo systemctl restart systemd-logind`; undo by restoring `suspend`. On Windows:
> `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0; powercfg /setactive SCHEME_CURRENT`
> (0 = do nothing; set it back to `1` = sleep afterwards). On macOS, `disablesleep 1` already covers the lid.

### Platforms

| | macOS | Linux | Windows |
|---|---|---|---|
| Tested | **Yes** (Apple Silicon, 16 GB, MPS) | No, but the same commands apply | No; use **WSL2** (Ubuntu) and the Linux commands |
| Phase runners (`scripts/run_phase*.sh`) | bash | bash | bash inside WSL2 (they activate `eoenv/bin/activate`, which a native Windows venv does not have) |
| Single Python steps | bash/zsh | bash | WSL2, or native PowerShell |
| Keep the machine awake | `caffeinate -i` | `systemd-inhibit` | inside WSL2, set Windows sleep to "Never" while running |

**GPU device:** the scripts default to `--device mps` when available, otherwise `cpu`. On an NVIDIA GPU, pass `--device cuda`
to each Python step. The phase runners do not pass `--device`, so on Linux and Windows they currently run on the CPU, which is far
slower. Phase 4b's land-cover pseudo-labelling step (`analysis4b.py`) also asks for `mps` directly.

### Re-run a phase

**macOS**

```bash
cd ~/Desktop/projects/Satsplitbench
mkdir -p results/phase7
nohup caffeinate -i bash scripts/run_phase7.sh baseline > results/phase7/baseline.log 2>&1 &
tail -f results/phase7/baseline.log        # Ctrl+C stops watching only, not the run
# after the baseline (uses the pre-registered lambda = 0.5, then runs the Phase 7 analysis):
nohup caffeinate -i bash scripts/run_phase7.sh defended > results/phase7/defended.log 2>&1 &
```

**Linux** (and Windows inside WSL2)

```bash
cd /path/to/Satsplitbench                  # in WSL2, e.g. /mnt/c/Users/<you>/projects/Satsplitbench
mkdir -p results/phase7
nohup systemd-inhibit --what=idle:sleep bash scripts/run_phase7.sh baseline > results/phase7/baseline.log 2>&1 &
tail -f results/phase7/baseline.log
nohup systemd-inhibit --what=idle:sleep bash scripts/run_phase7.sh defended > results/phase7/defended.log 2>&1 &
```

If `systemd-inhibit` is not available (some WSL2 setups), drop it: `nohup bash scripts/run_phase7.sh baseline > ... &`.

**Windows, native PowerShell** (single steps only; the runners need bash)

```powershell
cd C:\path\to\Satsplitbench
.\eoenv\Scripts\Activate.ps1
python scripts\own_encoder.py --name own_vit_baseline --adv-weight 0 --ben --device cuda
Get-Content results\phase7\baseline.log -Wait      # follow a log file
```

The same pattern runs any phase (`run_phase4b.sh`, `run_phase5.sh`, `run_phase6.sh`, `run_phase7.sh baseline|defended`); each
phase section below lists its individual Python steps. On Windows, use `\` in paths, or forward slashes, which Python also accepts.

Results: `results/phase4b/analysis/`, `results/phase5/analysis/`, `results/phase6/analysis/` and `results/phase7/analysis/`.

## Phases 0–4: setup and baseline probes

### Phase 0: Environment setup

Python 3.12 virtual environment in `eoenv/`. Developed on a Mac (Apple Silicon, MPS available).

| Package | Version |
|---|---|
| torch | 2.15.0 |
| torchgeo | 0.10.1 |
| scikit-learn | 1.9.1 |
| tensorflow | 2.21.0 (only for reading GloSoFarID TFRecords) |
| timm / torchmetrics / lightgbm | 1.0.31 / 1.9.0 / 4.7.0 |
| numpy / scipy / pandas / rasterio | 2.5.4 / 1.18.1 / 3.1.1 / 1.5.2 |

**macOS / Linux / WSL2**

```bash
python3.12 -m venv eoenv
source eoenv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 torchgeo==0.10.1 timm==1.0.31 torchmetrics==1.9.0 \
  scikit-learn==1.9.1 lightgbm==4.7.0 tensorflow==2.21.0 numpy==2.5.4 scipy==1.18.1 pandas==3.1.1 rasterio==1.5.2
brew install libomp            # macOS only: LightGBM needs it
python -c "import torch, torchgeo; print('mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available(), torchgeo.__version__)"
```

**Windows (native PowerShell)**

```powershell
py -3.12 -m venv eoenv
.\eoenv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch==2.15.0 torchvision==0.29.2 torchgeo==0.10.1 timm==1.0.31 torchmetrics==1.9.0 `
  scikit-learn==1.9.1 lightgbm==4.7.0 tensorflow==2.21.0 numpy==2.5.4 scipy==1.18.1 pandas==3.1.1 rasterio==1.5.2
python -c "import torch, torchgeo; print('cuda', torch.cuda.is_available(), torchgeo.__version__)"
```

For an NVIDIA GPU, install the CUDA build of torch from the selector at pytorch.org instead of the default wheel.

### Phase 1: Data acquisition

All data lives under `data/` (git-ignored).

| Dataset | Role | Location | Contents | Verified |
|---|---|---|---|---|
| EuroSAT (MS) | Utility task (land cover) | `data/eurosat/` | 27,000 patches, 64×64, 13 bands, 10 classes | 27,000 .tif files extracted |
| GloSoFarID, 2023 global shard | Object leakage (solar farms) | `data/glosofarid/` | 3,463 patches, 256×256, 13 bands, masks | md5 `6c3d50e8728b88e53472ed40d7be29ee` matches Zenodo |
| BigEarthNet v2, S2 (optional) | Utility and leakage on the same images | `data/bigearthnet/` | 549,488 patches, 120×120, 12 bands, 19 labels | 549,488 patches + 549,488 reference maps extracted |

Large downloads drop. Each command below is wrapped in a loop that retries until it succeeds. The torchgeo downloads
(EuroSAT, BigEarthNet) restart a file from the beginning on failure, and `checksum=True` makes torchgeo verify the archive's
MD5, so a broken file is never used; if a loop keeps failing on the same partial archive, delete that archive and run it again.
The GloSoFarID `curl` download resumes where it stopped.

**EuroSAT** (about 2 GB)

macOS / Linux / WSL2:

```bash
until python -c "from torchgeo.datasets import EuroSAT; EuroSAT(root='data/eurosat', split='train', download=True, checksum=True)"; do
  echo "download failed, retrying in 10 s"; sleep 10
done
```

Windows PowerShell:

```powershell
do { python -c "from torchgeo.datasets import EuroSAT; EuroSAT(root='data/eurosat', split='train', download=True, checksum=True)"; if ($LASTEXITCODE) { Start-Sleep 10 } } until ($LASTEXITCODE -eq 0)
```

**BigEarthNet v2 (Sentinel-2 only)**

About 64 GB of archives (`BigEarthNet-S2.tar.gzaa` + `.gzab`, plus `Reference_Maps`), and more again once extracted. Check free disk
space first.

macOS / Linux / WSL2:

```bash
until python -c "from torchgeo.datasets import BigEarthNetV2; BigEarthNetV2(root='data/bigearthnet', split='train', bands='s2', download=True, checksum=True)"; do
  echo "download failed, retrying in 30 s"; sleep 30
done
```

Windows PowerShell:

```powershell
do { python -c "from torchgeo.datasets import BigEarthNetV2; BigEarthNetV2(root='data/bigearthnet', split='train', bands='s2', download=True, checksum=True)"; if ($LASTEXITCODE) { Start-Sleep 30 } } until ($LASTEXITCODE -eq 0)
```

BigEarthNet is Level-2A (surface reflectance) and has no B10 cirrus band, so it has 12 bands. EuroSAT and GloSoFarID have all 13.

**GloSoFarID**

The data is on Zenodo ([record 10939100](https://zenodo.org/records/10939100)) as five gzipped TFRecord shards (about 10.3 GB). The [GitHub repo](https://github.com/yzyly1992/GloSoFarID) holds only a converter script, which is not needed here. Only the 2023 global shard is downloaded.

macOS / Linux / WSL2:

```bash
mkdir -p data/glosofarid
# -C - resumes a partial download; the loop retries if the connection drops
until curl -L -C - --retry 5 --retry-delay 5 -o data/glosofarid/solar_2023_global.tfrecord.gz \
  "https://zenodo.org/records/10939100/files/solar_2023_global.tfrecord.gz?download=1"; do sleep 5; done
md5 data/glosofarid/solar_2023_global.tfrecord.gz        # macOS; on Linux: md5sum ...
# expected: 6c3d50e8728b88e53472ed40d7be29ee
```

Windows PowerShell (`curl.exe`, not the `curl` alias; the loop resumes after each drop):

```powershell
New-Item -ItemType Directory -Force data\glosofarid | Out-Null
do {
  curl.exe -L -C - --retry 5 --retry-delay 5 -o data\glosofarid\solar_2023_global.tfrecord.gz "https://zenodo.org/records/10939100/files/solar_2023_global.tfrecord.gz?download=1"
  if ($LASTEXITCODE) { Start-Sleep 5 }
} until ($LASTEXITCODE -eq 0)
Get-FileHash -Algorithm MD5 data\glosofarid\solar_2023_global.tfrecord.gz   # 6C3D50E8728B88E53472ED40D7BE29EE
```

`scripts/load_glosofarid.py` reads the TFRecords straight into numpy, skipping the 24 GB .tif conversion:

```bash
python scripts/load_glosofarid.py data/glosofarid/solar_2023_global.tfrecord.gz --out data/glosofarid/npy
```

| File | Shape, dtype | Meaning |
|---|---|---|
| `solar_2023_global_x.npy` | (3463, 13, 256, 256), uint16 | Bands B1–B12 including B8A, reflectance × 10,000 |
| `solar_2023_global_y.npy` | (3463,), uint8 | 1 if any solar-panel pixel: 1,774 positive, 1,689 negative |

Add `--masks` to also save per-pixel masks.

**Encoder weights** are not downloaded yet. torchgeo fetches them on first use into `~/.cache/torch/hub/checkpoints/`.

### Data notes

- **No resizing, anywhere.** Resizing changes the ground resolution (256→64 makes each pixel 40 m instead of 10 m), blurs small objects and would make leakage look lower than it is. Every image goes into the encoders at native 10 m.
- **Patch size:** GloSoFarID patches are 256×256, EuroSAT 64×64. Each GloSoFarID patch is tiled into 16 native 64×64 tiles (~55k tiles), each labelled from its mask. All tiles from one patch stay in the same train/test split. Tiling needs the masks: rerun the loader with `--masks`.
- **ViT input size (resolved):** ViT-S/14 needs sizes divisible by 14, which 64×64 is not, so the plan uses `ViTSmall16` (16-pixel patches; 64 ÷ 16 = 4) instead.
- **BigEarthNet bands:** the 13-band `SENTINEL2_ALL_*` encoders need B10 filled (for example with zeros between B09 and B11). BigEarthNet is L2A while the encoders were trained on L1C, so report its results separately.
- **Disk:** the BigEarthNet archives (`BigEarthNet-S2.tar.gzaa/ab`, 63.5 GB) and the EuroSAT zip (2 GB) can be deleted after confirming loading works with `download=False`.

### Phase 2: Embeddings

Each frozen encoder ran once over EuroSAT and GloSoFarID at native 64×64, 10 m. GloSoFarID was first reloaded with masks:

```bash
python scripts/load_glosofarid.py data/glosofarid/solar_2023_global.tfrecord.gz --out data/glosofarid/npy --masks
python scripts/embed.py --sanity     # all encoders + random-init control, both datasets
```

Detached alternative (macOS; on Linux use `systemd-inhibit --what=idle:sleep` instead of `caffeinate -i`):

```bash
mkdir -p results/logs
nohup caffeinate -i python scripts/embed.py --sanity > results/logs/phase2_embed.log 2>&1 &
tail -f results/logs/phase2_embed.log
```

Results (`--sanity` = linear probe, EuroSAT train → val; chance 10%):

| Encoder | Dim | EuroSAT val accuracy |
|---|---|---|
| ResNet50 DINO | 2048 | 97.2% |
| ResNet18 MoCo | 512 | 95.5% |
| ViT-S/16 DINO | 384 | 95.3% |
| ViT-S/16 MoCo | 384 | 94.1% |
| ResNet18 random (control) | 512 | 79.7% |

GloSoFarID tiles: 55,408, split by original patch 70/15/15.

| Split | Tiles | With solar |
|---|---|---|
| Train | 38,784 | 8,591 (22.2%) |
| Val | 8,304 | 1,818 (21.9%) |
| Test | 8,320 | 2,023 (24.3%) |

Outputs go to `data/embeddings/`: `splits_glosofarid.npz` (tile labels, patch ids, split by patch, 70/15/15) and per encoder `eurosat_{train,val,test}.npy`, `glosofarid.npy`, `meta.json`. EuroSAT is cached once to `data/eurosat/npy/` (official splits: 16,200 / 5,400 / 5,400), reordered to the encoders' band order (torchgeo's EuroSAT puts B8A last). The torchgeo weight `transforms` are not used because they resize to 256 and crop to 224; only the ÷ 10,000 normalisation is applied.

| Encoder (torchgeo) | Architecture | Pretraining | Embedding dim | Why |
|---|---|---|---|---|
| `ResNet18_Weights.SENTINEL2_ALL_MOCO` | CNN | MoCo | 512 | Baseline SSL4EO model |
| `ViTSmall16_Weights.SENTINEL2_ALL_MOCO` | Transformer | MoCo | 384 | Same method as ResNet18, different architecture |
| `ResNet50_Weights.SENTINEL2_ALL_DINO` | CNN | DINO | 2048 | Different SSL method |
| `ViTSmall16_Weights.SENTINEL2_ALL_DINO` | Transformer | DINO | 384 | Same method as ResNet50, different architecture |

All four take 13 bands. The ViTs use 16-pixel patches, so a 64×64 tile is 4×4 patches at native resolution. timm's `dynamic_img_size=True` resamples their positional embeddings from 14×14 to 4×4 at run time (a model change, not a data change); their EuroSAT accuracy above confirms they work at this size.

### Phase 3: Compression

```bash
python scripts/compress.py      # all encoders in data/embeddings, ~15 s
```

Detached alternative: `mkdir -p results/logs && nohup caffeinate -i python scripts/compress.py > results/logs/phase3_compress.log 2>&1 &`

- 8 variants per encoder: `full` (uncompressed reference), `pca16`, `pca64`, `pca256`, each also as `_int8`.
- PCA and int8 scales are fitted on training rows only: EuroSAT train + GloSoFarID train tiles, subsampled to equal numbers (16,200 each). Val/test rows are only transformed.
- int8: symmetric per-dimension scale = max |value| on training rows / 127. Round-trip error is at most half a quantisation step; under 0.006% of values clip.
- Outputs in `data/compressed/<encoder>/<variant>/` (int8 variants store int8 + `scale.npy`; multiply to dequantise), PCA fits in `<encoder>/pca<k>.npz`, a summary in `<encoder>/meta.json`.

| Encoder | Variance kept: 16 / 64 / 256 dims | EuroSAT val acc: pca16 / pca16_int8 / pca64 |
|---|---|---|
| ResNet18 MoCo | 55.2% / 79.3% / 95.4% | 93.0% / 92.9% / 95.1% |
| ResNet50 DINO | 63.1% / 87.0% / 94.9% | 91.7% / 91.7% / 95.4% |
| ViT-S/16 DINO | 35.1% / 68.0% / 96.2% | 90.8% / 90.8% / 94.8% |
| ViT-S/16 MoCo | 49.9% / 81.8% / 98.8% | 87.0% / 87.0% / 93.6% |
| ResNet18 random | 99.2% / 99.8% / 100.0% | 69.3% / 69.3% / 76.3% |

Sizes per embedding: 16 B (`pca16_int8`) up to 8,192 B (ResNet50 `full`). int8 costs almost no utility. The random encoder's variance sits almost entirely in 16 directions, so "variance kept" is not comparable across encoders.

### Phase 4: Probes and controls

```bash
python scripts/probes.py        # all encoders x variants + controls, ~30 min on MPS
```

Detached alternative:

```bash
mkdir -p results/logs
nohup caffeinate -i python scripts/probes.py > results/logs/phase4_probes.log 2>&1 &
tail -f results/logs/phase4_probes.log
```

- **Tasks:** utility = EuroSAT land cover (test accuracy, chance 10%); leakage = GloSoFarID tile "solar farm present" (test ROC-AUC, chance 0.5).
- **Probes:** standardised logistic regression (C chosen on val) and a one-hidden-layer MLP (256 units, early stopping on val, 3 seeds).
- **Uncertainty:** 95% bootstrap CIs, 1,000 resamples; leakage resamples whole patches, not tiles.
- **Controls:** `resnet18_random` and `pixelstats` (per-band mean, std, p10, p50, p90; 65 features, no resizing).
- Results in `results/probes.csv` (164 rows), log in `results/probes.log`.

| Encoder | Utility, full (linear) | Leakage AUC, full: linear / MLP | Leakage AUC, pca16: linear / MLP |
|---|---|---|---|
| ViT-S/16 DINO | 96.3% | 0.937 / 0.947 | 0.889 / 0.919 |
| ResNet50 DINO | 97.4% | 0.934 / 0.934 | 0.855 / 0.907 |
| ViT-S/16 MoCo | 95.3% | 0.922 / 0.932 | 0.861 / 0.892 |
| ResNet18 MoCo | 96.6% | 0.894 / 0.904 | 0.791 / 0.863 |
| Pixel statistics (control) | 89.8% | 0.836 / 0.907 | n/a |
| ResNet18 random (control) | 79.9% | 0.796 / 0.824 | 0.762 / 0.812 |

Findings:
- **Most leakage is spectral.** Pixel statistics reach AUC 0.907 (MLP). The best embedding (ViT-S/16 DINO, 0.947) adds about 4 points, with non-overlapping CIs; ResNet18 MoCo only matches the pixel baseline.
- **H1 supported, with a qualifier:** leakage stays far above chance at 16–64 dims, but compressed embeddings add little beyond pixel statistics.
- **H2 modestly supported:** ResNet18 MoCo and ViT-S/16 DINO have near-equal utility (96.6% vs 96.3%) but different leakage (0.894 vs 0.937, CIs disjoint). Not yet tested with paired comparisons.
- **RQ2:** for ResNet18, full → pca16 costs 2.9 points of utility and 10.3 points of linear leakage, but only 4.1 points of MLP leakage. Compression makes leakage less linearly accessible rather than removing it.
- **int8 makes no difference** to utility or leakage.

---

## Phases 4b–7: research-grade follow-ups

From Phase 4b on, every attacker and model is a research-grade model, tuned on val and evaluated on test. Linear probes stay only as a reference row, because they are the standard measure in the probing literature and carry the RQ2 finding above.

The central question from here: **what does an embedding reveal beyond the spectrum?** Every leakage number is reported against a control that removes the spectral shortcut (pixel statistics) or the location shortcut (a location prior: the average mask at that tile position, since GloSoFarID patches were likely collected centred on farms).

### Models by phase

Each model is chosen for what it lets the study measure:
- **Vector attackers span three model families** (neural, tree, kernel), so no leakage finding rests on one family.
- **Repeats are kept where they help a comparison:** the deep MLP extends the Phase 4 linear → MLP ladder; the upper-bound CNN is a ResNet-18 so it matches the audited ResNet18; the own encoder is a ViT so it can sit beside the audited ViT-S/16s.
- **Controls** use the same architecture as the model they control (the pixel-statistics inverter mirrors the strong inverter).
- **Measurement tools** stay identical across every encoder, compression level and task they compare (the 4b attackers score attribute, spatial and utility tasks alike; the panel segmenter scores every inverter).
- **Frozen loss network:** the strong inverter's perceptual loss uses ResNet50 MoCo, deliberately not an audited encoder, so the loss cannot favour one.

| Phase | Component | Family | Model | Training | Scored by |
|---|---|---|---|---|---|
| 4b | Attribute attackers | MLP, tree ensemble, kernel | Deep residual MLP (3 blocks, 512 units, LayerNorm, GELU, dropout); LightGBM; RBF-kernel SVM (Nyström approximation) | Hyperparameter search on val | Test ROC-AUC, paired bootstrap |
| 4b | Spatial probe | Same three | The 4b attackers, one per cell of a 4×4 grid inside a tile (reuse by design) | Same | Grid IoU vs location prior |
| 4b | Utility on BigEarthNet | Same three | The 4b attackers, one per label, 19 labels (reuse by design) | Same | Macro mAP |
| 4b | Image-level upper bound | CNN | ResNet-18 trained from scratch on raw 64×64×13 tiles (matches the audited ResNet18) | Supervised; flips and 90° rotations only | Leakage available in the image itself |
| 5 | Weak inverter | Non-parametric | k-nearest-neighbour retrieval: reconstruct a tile as the mean of the k training tiles closest in embedding space | k chosen on val | SSIM, LPIPS (RGB bands), spectral angle |
| 5 | Strong inverter | GAN (CNN) | Generator: decoder with residual blocks and self-attention at 8×8 and 16×16; discriminator: PatchGAN | L1 + multispectral perceptual loss (frozen ResNet50 MoCo, not audited) + adversarial loss | Same, plus mask IoU |
| 5 | Pixel-statistics inverter (control) | GAN (CNN) | Same GAN, input = pixel statistics (reuse by design) | Same | Learned leakage = embedding inverter minus this |
| 5 | Panel segmenter | CNN | U-Net on real GloSoFarID tiles and masks | Dice + BCE | Run on reconstructions: detection and mask IoU |
| 6 | Localisation attack | Transformer + CNN head | Query-based decoder (DETR / Mask2Former style): memory tokens from the embedding, 64 learnable queries (8×8), 4 cross-attention layers, conv upsampling head | Dice + focal loss | Mask IoU vs location prior and pixel-statistics control |
| 7 | Own encoder, baseline | Transformer | ViT-Small, 8-pixel patches on 64×64×13 (64 tokens) | Masked autoencoder (MAE), 75% masking, train-split tiles only | Phase 4b–6 attackers |
| 7 | Own encoder, defended | Transformer + MLP heads | Same ViT and budget, plus a solar-farm adversary behind a gradient-reversal layer | MAE + land-cover head − adversary | Phase 4b–6 attackers, retrained fresh |
| 7 | Defence filter (optional; not run, dropped for time) | MLP-based | Variational information bottleneck (stochastic encoder with a KL penalty) on frozen pretrained embeddings, plus the gradient-reversal adversary | Utility head + KL + adversary | Phase 4b–6 attackers |

Out of scope: diffusion-based inversion (too heavy for the timeline on a Mac). Named as a limitation: stronger attacks can only reveal more.

### Phase 4b: Learned-leakage follow-ups (pre-registered)

Written down before running; every one is reported whatever it shows.

- **Harder utility task (BigEarthNet):** EuroSAT accuracy is 95–97% for every encoder, too close to its ceiling to separate them, so "matched utility" for H2 is measured on BigEarthNet's 19-label land-cover task instead (macro mAP).
  - Subset of the official splits (237,871 / 122,342 / 119,825 patches available), for example 50k / 10k / 10k.
  - Centre-cropped to 112×112, which is cropping, not resizing: ViT-S/16 needs a multiple of 16, and 120 is not one. Labels cover the full 120×120 patch; the cropped 4-pixel border (40 m) is a small source of label noise.
  - B10 zero-filled between B09 and B11. BigEarthNet is L2A while the encoders were trained on L1C, so this shift affects all encoders and results are reported separately from EuroSAT.
  - Compressed with the existing Phase 3 PCA and int8 fits (transform only, no refit).
- **Incremental leakage:** attacker on pixel statistics alone vs pixel statistics + embedding. The AUC gain is the learned leakage.
- **Per-class leakage:** AUC within each land-cover class, removing the "bright built-up surface" shortcut. GloSoFarID tiles get land-cover pseudo-labels from the best EuroSAT utility attacker.
- **Location prior check:** test from the masks whether panels really concentrate at the patch centre before relying on that baseline.
- **Spatial probe:** 4×4 grid localisation inside a tile, against the location prior.
- **Image-level upper bound:** how much leakage the raw tile itself allows.
- **Paired tests for H2:** paired bootstrap of the leakage difference between encoders on the same test patches; leakage compared at matched utility on each encoder's trade-off curve.
- **Robustness:** 3 train/val/test split seeds, not only probe seeds.
- **Coarser quantisation:** int4, int2 and 1-bit, to find where quantisation starts to cut leakage and utility.

The full pre-registration, with primary and secondary analyses, Holm correction and its disclosure, is in `results/preregistration.md`.

**Status: done; results in `RESULTS.md` and `results/phase4b/analysis/`.**

Run everything, in pre-registered order (resumes if interrupted; roughly 15–18 h in total, best overnight):

```bash
mkdir -p results/phase4b && caffeinate -i bash scripts/run_phase4b.sh 2>&1 | tee results/phase4b/run.log
```

Detached alternative (recommended for long runs; survives closing the terminal or editor):

```bash
mkdir -p results/phase4b
nohup caffeinate -i bash scripts/run_phase4b.sh > results/phase4b/run.log 2>&1 &
tail -f results/phase4b/run.log        # Ctrl+C stops watching only, not the run
```

On Linux / WSL2, replace `caffeinate -i` with `systemd-inhibit --what=idle:sleep` (or drop it); on native Windows run the steps
in the table below one by one. See [How to run](#how-to-run), including how to disable sleep.

The script runs these steps; each can also be run on its own:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python scripts/probes.py --out results/probes_v2.csv --preds-dir results/preds/phase4` | Phase 4 probes again, saving per-tile predictions and validation scores (needed for paired tests) | ~30 min |
| 2 | `python scripts/quantise.py` then `python scripts/probes.py --comp-root data/quantised --out results/probes_quant.csv --preds-dir results/preds/phase4_quant` | Quantile codes at 8/4/2/1 bits, then linear + MLP probes on them | ~1 h |
| 3 | `python scripts/splits.py --seeds 1 2` | Two extra train/val/test splits by patch | ~2 min |
| 4 | `python scripts/bigearthnet.py` | BigEarthNet subset (30k / 10k / 10k): embed with all encoders, compress with the existing fits, pixel statistics | ~25 min |
| 5 | `python scripts/attack4b.py --tasks leakage leakage_incr utility_eurosat utility_ben spatial` | Deep MLP, LightGBM and RBF SVM on every task, encoder and float variant | ~10–14 h |
| 6 | `python scripts/upperbound.py` | ResNet-18 from scratch on raw tiles | ~45 min |
| 7 | `python scripts/attack4b.py --tasks leakage leakage_incr --variants full --split-seed 1` (and 2) | Primary comparisons on the extra splits | ~2.5 h |
| 8 | `python scripts/analysis4b.py` | All pre-registered tests, into `results/phase4b/analysis/` | ~5 min |

Implementation notes:
- **Attacker selection:** for each cell, the attacker with the best **validation** score is selected and its test score is the leakage estimate. Phase 4's linear and MLP probes are candidates too. (Amended before any 4b run: the first version selected on test, which inflates estimates.)
- **Quantisation scheme:** q8/q4/q2/q1 use per-dimension quantile (equal-frequency) bins with bin-mean reconstruction, fitted on the Phase 3 PCA fit rows. Phase 3's max-scaled int8 would collapse most values to zero at 4 bits or fewer. q8 matches float (bridge to Phase 3's int8).
- **BigEarthNet bands:** 20 m and 60 m bands are put on the 10 m grid by nearest-neighbour copying (each value repeated 2×2 or 6×6; no values invented, ground extent unchanged), matching how EuroSAT and GloSoFarID are delivered. Approved as consistent with the no-resizing rule.
- **Location prior (checked):** panels concentrate at patch centres. P(tile has panels) is 0.42 for the 4 centre tiles and 0.11–0.12 for the corners; tile position alone gives test AUC 0.659. Spatial baselines therefore use the average mask for that tile position, deliberately stricter than the attackers, who never see position.
- **LightGBM settings:** 63 histogram bins, learning rate 0.1, early stopping after 50 rounds: 3.6× faster than the defaults on 2048-dim inputs, test AUC 0.933 vs 0.935.
- **macOS:** LightGBM needs `brew install libomp` and must be imported before torch (both ship an OpenMP runtime). `attackers.py` handles the order.

Measured on the largest input (ResNet50 DINO, 2048 dims, full-size leakage):

| Attacker | Val AUC | Test AUC | Time | Peak memory |
|---|---|---|---|---|
| Deep MLP | 0.931 | 0.926 | 2.7 min | n/a |
| LightGBM | 0.931 | 0.933 | 4.3 min | 2.8 GB |
| RBF SVM | 0.931 | 0.937 | 1.4 min | 4.2 GB |

### Phase 5: Inversion

**Status: done; results in `RESULTS.md` and `results/phase5/analysis/`.**

Run after Phase 4b has finished (both use the GPU, and RQ3 reads Phase 4b's results). Roughly 14–20 h:

```bash
mkdir -p results/phase5 && caffeinate -i bash scripts/run_phase5.sh 2>&1 | tee results/phase5/run.log
```

Detached alternative (recommended for long runs; survives closing the terminal or editor):

```bash
mkdir -p results/phase5
nohup caffeinate -i bash scripts/run_phase5.sh > results/phase5/run.log 2>&1 &
tail -f results/phase5/run.log        # Ctrl+C stops watching only, not the run
```

On Linux / WSL2, replace `caffeinate -i` with `systemd-inhibit --what=idle:sleep` (or drop it); on native Windows run the steps
in the table below one by one. See [How to run](#how-to-run), including how to disable sleep.

The analysis plan is pre-registered in `results/preregistration.md` (Phase 5 section), written before any Phase 5 model was trained: 8 primary tests (learned and spatial leakage through GAN inversion at full size, Holm-corrected), with secondary analyses and RQ3's rank correlations.

Measured times:

| Step | Time |
|---|---|
| Segmenter | ~2 min per epoch, 40 max (~80 min) |
| Mean baseline, kNN | under 5 min per run |
| GAN | ~3.7 min per warm-up epoch, ~5.1 min per adversarial epoch; 20 max with early stopping, ~1–1.7 h per GAN |

Fixes found in testing: the perceptual network now loads the full pretrained ResNet50 MoCo and reads stages 1–3 directly (timm's `features_only` dropped stage 4 and broke weight loading).

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/segmenter.py` | U-Net panel segmenter on real tiles; threshold on val; real-image ceiling and location-prior baseline in `results/phase5/segmenter.json` |
| 2 | `python scripts/invert.py --method mean` | Mean-tile baseline |
| 3 | `python scripts/invert.py --method knn --encoder <enc> --variant <var>` | k-NN retrieval inverter (k chosen on val), all 5 encoders at `full` and `pca16`, plus pixel statistics |
| 4 | `python scripts/invert.py --method gan --encoder <enc> --variant <var>` | GAN inverter: 4 audited encoders at `full` and `pca16`, plus the pixel-statistics and random-ResNet18 controls |
| 5 | `python scripts/analysis5.py` | Pre-registered tests into `results/phase5/analysis/`: `summary.csv`, `primary.csv`, `secondary.csv`, `rq3.csv` |

Every run writes `results/phase5/<method>/<encoder>/<variant>/` (`metrics.json`, per-tile scores for paired tests, 16 example reconstructions) and a row in `results/phase5/inversion.csv`.

- Weak (k-nearest-neighbour retrieval) and strong (GAN with a self-attention generator) inverters, plus the pixel-statistics GAN as a control.
- The strong inverter's perceptual loss uses frozen ResNet50 MoCo, which is not one of the audited encoders, so the loss cannot favour any of them.
- Trained on train tiles only; never evaluated on tiles an inverter was trained on.
- Scored on pixel metrics (RQ3) and spatially, by running the panel segmenter on reconstructions (mask IoU vs location prior).

### Phase 6: Localisation attack

- Query-based transformer decoder from embedding to a 64×64 panel mask, on every encoder and compression level.
- Pixel statistics cannot localise, so localisation above the location prior is learned leakage.
- Compared with the inverters on the same mask-IoU score.

**Status: done; results in `RESULTS.md` and `results/phase6/analysis/`.** The analysis plan is pre-registered in `results/preregistration.md` (Phase 6 section): 8 primary tests at full size (vs the location prior and vs the pixel-statistics control), Holm-corrected.

```bash
mkdir -p results/phase6 && caffeinate -i bash scripts/run_phase6.sh 2>&1 | tee results/phase6/run.log
```

Detached alternative (recommended for long runs; survives closing the terminal or editor):

```bash
mkdir -p results/phase6
nohup caffeinate -i bash scripts/run_phase6.sh > results/phase6/run.log 2>&1 &
tail -f results/phase6/run.log        # Ctrl+C stops watching only, not the run
```

On Linux / WSL2, replace `caffeinate -i` with `systemd-inhibit --what=idle:sleep` (or drop it); on native Windows run the steps
in the table below one by one. See [How to run](#how-to-run), including how to disable sleep.

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/localise.py --encoder pixelstats --variant full` | Pixel-statistics control |
| 2 | `python scripts/localise.py --encoder <enc> --variant <var>` | 4 audited encoders × full, pca256, pca64, pca16 (full first: primary tests) |
| 3 | `python scripts/localise.py --encoder resnet18_random --variant full` | Random-weight control |
| 4 | `python scripts/analysis6.py` | Pre-registered tests into `results/phase6/analysis/` |

Model: 16 memory tokens from the embedding, 64 learned queries (8×8) with learned 2D positions, 4 pre-norm decoder layers (width 256, 8 heads), 3 upsampling conv stages to 64×64; 13.2M parameters for a 2048-dim input. Trained on all train tiles with Dice + focal loss, so it also gives a detection score comparable with Phase 5. About 1–2 min per epoch; ~30–70 min per run with early stopping; 18 runs.

### Phase 7: Own encoder and defence

- The same small ViT was trained twice with the same data and budget: baseline MAE, and MAE with the adversarial leakage penalty.
- **One script, one difference.** Both runs train MAE (75% masking), the same land-cover head, and the same solar-farm adversary behind a gradient-reversal layer. Only the reversal strength λ differs: the baseline uses λ = 0, so its adversary trains and monitors leakage but sends exactly zero gradient into the encoder (verified); the defended run uses λ > 0.
- Encoder: ViT-Small, 8-pixel patches on 64×64×13 (64 tokens), fixed 2D sin-cos positions (so BigEarthNet's 112×112 works unchanged), mean-pooled 384-dim embedding, the same size as the audited ViT-S/16s. 25.3M parameters including decoder and heads. Trained only on train-split tiles (EuroSAT train + GloSoFarID train).
- Each encoder then went through exactly the audited encoders' pipeline: compression, quantisation, BigEarthNet, all Phase 4b attackers, the Phase 6 localisation attack and the Phase 5 GAN, all trained fresh on its embeddings.
- The only difference between the two is the privacy objective, so any change in leakage is caused by the defence.
- Realistic target: remove the learned excess over pixel statistics, not reach AUC 0.5, since spectral leakage is inherent to any useful embedding. (Result: the excess over pixel statistics fell from +0.036 to +0.018 AUC.)
- Evaluated only against attackers trained fresh on its outputs (Phase 4b–6), never only against its own adversary.

**Status: done (13 Nov); results in `RESULTS.md` and `results/phase7/analysis/`. Verdict: "leakage cut at a cost".** The settings, recorded in `results/preregistration.md` (section "Phase 7 defence settings") on 12 Nov, before either encoder was trained:
- **λ = 0.5**, ramped linearly from 0 over the first 4 epochs (the baseline's λ = 0 stays exactly 0), a single untuned choice because the deadline allows one defended run. `run_phase7.sh defended` refuses any other value.
- **No spatial adversary.** Phase 6 found almost no location information to remove (mask IoU within 0.01 of the position-only prior), so the defence targets presence leakage only.
- **Five primary tests** (`scripts/analysis7.py`, Holm over 5): leakage must fall (Phase 4b best attacker, Phase 6 detection AUC, Phase 5 GAN detection AUC) while BigEarthNet macro AP and EuroSAT accuracy stay within 0.02 and 0.01 of the baseline. The verdict is one of "works", "leakage cut at a cost" or "no reduction", and all three are reported.

```bash
mkdir -p results/phase7 && nohup caffeinate -i bash scripts/run_phase7.sh baseline > results/phase7/baseline.log 2>&1 &
nohup caffeinate -i bash scripts/run_phase7.sh defended > results/phase7/defended.log 2>&1 &    # after the baseline; ends with analysis7.py
tail -f results/phase7/baseline.log      # or defended.log; Ctrl+C stops watching only
```

Foreground alternative (stops if the terminal closes):

```bash
mkdir -p results/phase7 && caffeinate -i bash scripts/run_phase7.sh baseline 2>&1 | tee results/phase7/baseline.log
caffeinate -i bash scripts/run_phase7.sh defended 2>&1 | tee results/phase7/defended.log
```

On Linux / WSL2, replace `caffeinate -i` with `systemd-inhibit --what=idle:sleep` (or drop it); on native Windows run the steps
in the table below one by one. See [How to run](#how-to-run), including how to disable sleep.

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/own_encoder.py --name own_vit_baseline --adv-weight 0 --ben` | Train (40 epochs, ~5–6 h) and export embeddings incl. BigEarthNet |
| 2 | `python scripts/compress.py --encoders <name>`, `python scripts/quantise.py --encoders <name>`, `python scripts/bigearthnet.py --compress-only --encoders <name>` | Same compression as the audited encoders |
| 3 | `python scripts/attack4b.py --tasks leakage leakage_incr utility_eurosat utility_ben spatial --encoders <name>` | All Phase 4b attackers |
| 4 | `python scripts/localise.py --encoder <name> --variant full` | Phase 6 localisation attack |
| 5 | `python scripts/invert.py --method gan --encoder <name> --variant full` | Phase 5 GAN inverter |
| 6 | `python scripts/analysis7.py` (run by the defended stage) | Baseline vs defended: 5 pre-registered tests, secondary analyses |

## Phase 8: SatSplit extensions (done 18 Nov; results in `RESULTS.md`)

Phases 0–7 measure accuracy, bytes and privacy for **one** split point: after the full encoder. Phase 8 adds the three parts
of the SatSplit-Bench statement that are still missing: **compute** (8a), **split points** (8b) and **orbit-realistic ISLs**
(8c). Build order: 8a → 8b → 8c. 8b uses 8a's split definitions, and 8c uses the compute numbers from 8a and the byte sizes
from 8a and 8b.

The setting: satellite A images a tile, runs the encoder **head** up to a split point and sends the **payload** over an
inter-satellite link (ISL). Satellite B (or a ground station) runs the **tail**. The privacy question is what an eavesdropper
or an untrusted relay learns from the payload.

### Shared definitions: split points and payloads

Input tile: 13 bands × 64 × 64 at 10 m, uint16 = **106,496 B**. That's the "send the raw image" baseline. The feature shapes
below were checked with timm's `forward_intermediates` at 64×64:

| Encoder | Split | Feature map | Full map, fp32 | Full map, int8 | fp32 vs raw | Pooled vector (dims) |
|---|---|---|---|---|---|---|
| ResNet18 (MoCo, random) | layer1 | 64 × 16 × 16 | 65,536 B | 16,384 B | 0.62× | 64 |
| | layer2 | 128 × 8 × 8 | 32,768 B | 8,192 B | 0.31× | 128 |
| | layer3 | 256 × 4 × 4 | 16,384 B | 4,096 B | 0.15× | 256 |
| | layer4 (= Phases 2–7) | 512 × 2 × 2 | 8,192 B | 2,048 B | 0.08× | 512 |
| ResNet50 (DINO) | layer1 | 256 × 16 × 16 | 262,144 B | 65,536 B | **2.46×** | 256 |
| | layer2 | 512 × 8 × 8 | 131,072 B | 32,768 B | 1.23× | 512 |
| | layer3 | 1024 × 4 × 4 | 65,536 B | 16,384 B | 0.62× | 1024 |
| | layer4 (= Phases 2–7) | 2048 × 2 × 2 | 32,768 B | 8,192 B | 0.31× | 2048 |
| ViT-S/16 (MoCo, DINO) | blocks 3 / 6 / 9 / 12 (12 = Phases 2–7) | 17 tokens × 384 | 26,112 B | 6,528 B | 0.25× | 384 |

Every split gets two payload forms:
- **`gap`, pooled vector:** global average pooling for ResNet stages, and the CLS token after the final LayerNorm for ViT
  blocks. At layer4 and block 12 this reproduces the existing Phase 2 embeddings exactly. The extraction checks that (max
  absolute difference < 1e-4) and stops if it fails.
- **`map4`, spatial map:** the feature map average-pooled to 4 × 4 (ResNet layer1–2; layer3 is natively 4 × 4, and layer4
  keeps its native 2 × 2), or the 16 patch tokens for the ViTs, flattened and stored as float16, on GloSoFarID only. An
  attacker that sees the full transmitted map can only learn more, so `map4` results are a lower bound.

BigEarthNet tiles are 112 × 112, so their maps are larger. The pooled vectors keep the same dimensions, so the BigEarthNet
utility task uses `gap` only.

New research questions:
- **RQ4, split point:** how do leakage, utility, payload bytes and on-board compute change with the split point inside the encoder?
- **RQ5, ISL feasibility:** which payloads fit real ISL rates and contact windows, and what utility and leakage do the feasible ones carry?

### 8a: Compute

**Question:** what does each split cost on board satellite A?

Measured for each encoder × split, plus the own ViT-S/8 (final split) and the raw-transmission baseline:

| Measure | How |
|---|---|
| Parameters, head and tail | counted per module, up to and after the split |
| FLOPs, head and tail | `torch.utils.flop_counter.FlopCounterMode`, batch 1, 13 × 64 × 64 input (a multiply-add counts as 2 FLOPs) |
| Peak activation memory of the head | largest intermediate tensor, in bytes |
| Head latency | MPS and CPU, batch 1 and 64; 20 warm-up runs and 100 timed runs; median and IQR; `torch.mps.synchronize()` before every timer read |
| Payload bytes | both forms × fp32 / fp16 / int8, from the table above |
| Raw baseline | raw bytes and losslessly compressed raw bytes (zlib, mean over 1,000 test tiles) |

**Sanity checks (the script fails if any is off):**
- The FLOP counter gives ≈ 3.6 GFLOPs (1.8 GMACs) for torchvision ResNet-18 at 3 × 224 × 224.
- Head + tail FLOPs equal the full model's FLOPs.
- At the final split, the head parameters equal the encoder's parameters.

**Limits:** laptop latency is not flight hardware, and radiation-tolerant on-board processors are much slower. FLOPs and bytes
are hardware-independent, so they are the primary compute measures. Latency is reported only as a relative ranking. Energy is
not measured; FLOPs serve as the proxy.

**Output:** `scripts/compute8a.py` → `results/phase8/compute.csv`. No training, under 1 h. Descriptive only, so it needs no
pre-registration. Run the latency timing when nothing else is using the GPU; FLOPs and bytes can be computed at any time.

### 8b: Split points (pre-registered before running)

**Question:** RQ4. Does an earlier split leak more or less, and does it give away location that the final embedding does not?

**Encoders:** the 4 audited encoders, plus `resnet18_random` as a control, at every split in the table: 15 new split points
(the 12 audited ones are tiers 1–2, the 3 random ones tier 2). The final split repeats the existing embeddings and is
re-checked against them, and is attacked again with the deep MLP so every split uses the same attacker. The own encoders
are left out for time.

**Pipeline:**
1. **Extract** (`scripts/splits8b.py`): one pass per dataset (EuroSAT train/val/test; all 55,408 GloSoFarID tiles; the
   BigEarthNet 30k/10k/10k subset with the Phase 4b preprocessing) collects every split at once. It writes
   `data/compressed/<encoder>@<split>/{gap,map4}/*.npy`, so `attack4b.py` and `localise.py` read the files unchanged.
   About 1.5–2.5 h, mostly reading BigEarthNet.
2. **Attack** (deep MLP only, with hyperparameters chosen on val, split seed 0). In Phase 4b the deep MLP was chosen, or within
   0.005 AUC of the chosen attacker, in almost every case. Using it alone cuts the run time by about 3×.
   - `gap`: `leakage`, `leakage_incr`, `utility_eurosat`, `utility_ben`
   - `map4`: `spatial` (16-cell grid, macro AP), `leakage`
   - Outputs: `results/phase8/runs8b.csv` and `results/preds/phase8b/`. About 5–8 h.
3. **Localise (optional, tier 3):** the Phase 6 localiser on `map4` at the earliest split of each audited encoder (4 runs,
   about 0.5–1 h each).
4. **Analyse** (`scripts/analysis8b.py`): primary tests, secondary curves, the frontier table and figures.

**Primary tests** (Holm over 8; cluster bootstrap over patches with 1,000 resamples; one-sided; leakage = test score of the
attacker chosen on validation):

| Tests | Hypothesis | Measure |
|---|---|---|
| P8b.1–4 | **Depth adds presence leakage:** final split − earliest split (layer1 / block 3) > 0, one test per audited encoder | leakage AUC, `gap` |
| P8b.5–8 | **Early splits leak location:** earliest split − final split > 0, one test per audited encoder | spatial macro AP, `map4` |

Why these two: Phase 4b found the leakage beyond pixel statistics is learned, so it should grow with depth. Phases 4b–6 found
almost no location information in final embeddings, so if location leaks anywhere, it should be at early splits, where the
maps still have spatial resolution.

**Verdict rule, per family of 4 tests:** supported if at least 3 are significant after Holm, not supported if at most 1 is,
mixed otherwise. Every test is reported, whatever the outcome.

**Secondary analyses (exploratory):**
- leakage and utility curves over all splits;
- `leakage_incr` at each split (does any split beat the pixel statistics?);
- the random-encoder control at each split;
- leakage per payload byte;
- the Pareto frontier over payload bytes, head FLOPs (from 8a), utility and leakage;
- `map4` presence leakage;
- localiser IoU against the location prior (tier 3).

**Pre-registered fallbacks:**
- If a `map4` matrix doesn't fit in memory (worst case: ResNet50 layer3 at 16,384 dims × about 39k train rows), fit PCA-2048
  on the train rows only and record that.
- If tier 1 hasn't finished by **19 Nov, 14:00**, report the encoders that finished and list the rest as not run.

**Tiers:**
- Tier 1: `gap` `leakage` + `utility_ben`, and `map4` `spatial`, for the 4 audited encoders.
- Tier 2: `leakage_incr`, `utility_eurosat`, `map4` `leakage`, the random control.
- Tier 3: the localiser.

Before 8b runs, this section is copied into `results/preregistration.md` under "Phase 8b split-point settings", with the date.

### 8c: Orbit-realistic ISLs

**Question:** RQ5. Given real orbits and link rates, which payloads can be sent, and what do they cost in utility and leakage?

**Orbits:** public TLEs from CelesTrak, fetched once and saved with their epoch in `data/orbits/`: Sentinel-2A/2B/2C, EDRS-A
(hosted on Eutelsat 9B) and EDRS-C. Check the exact catalog names when fetching. They are propagated with `skyfield` (SGP4)
over 24 h at 10 s steps. Line of sight counts as clear when the straight segment between the satellites stays at least
100 km above the Earth's surface (an atmosphere margin).

**Scenarios:**

| Scenario | Link | Notes |
|---|---|---|
| S1, LEO → GEO relay | Sentinel-2 → EDRS, optical, 1.8 Gbit/s nominal | Sentinel-2's real data-relay path; gives visibility windows and distances |
| S2, LEO crosslink in one plane | every real Sentinel-2 pair (2A, 2B, 2C); synthetic leader–follower pairs at 1°, 5° and 20° phase (about 125, 625 and 2,500 km) | From the 18 Nov 2026 TLEs: 2B and 2C are about 180° apart and never in line of sight, while 2A (no longer in its original slot) sits about 4,460 km from 2B with continuous line of sight. The script reports each pair's line-of-sight share rather than assuming it. The synthetic pairs give continuous contact. |
| S3, LEO constellation | synthetic Walker-delta at 786 km SSO (e.g. 24/3/1), links to neighbours in the same and adjacent planes | inter-plane distances and windows |

**Link classes** (nominal rates are input parameters, defaults `s_band` 2 Mbit/s, `ka_band` 500 Mbit/s, `optical_1g8`
1.8 Gbit/s, `optical_10g` 10 Gbit/s, `optical_100g` 100 Gbit/s, changed with `--links name=rate`; the paper cites a source for
each value used): S-band RF (about
1–10 Mbit/s), Ka-band RF (about 0.1–1 Gbit/s), optical (1.8 Gbit/s for EDRS; about 10–100 Gbit/s for commercial optical
ISLs). Optionally, RF rates scale with distance through free-space path loss, rate(d) = min(R_max, R_ref · (d_ref / d)²).

**Workload:** a Sentinel-2-like imager (290 km swath, about 6.6 km/s ground speed) produces about **4,700 tiles/s** of
640 m × 640 m tiles while imaging. That's about **4.0 Gbit/s raw** with all 13 bands on the 10 m grid. The real instrument
keeps its 20 m and 60 m bands at lower resolution, so this overstates raw volume; the paper reports it both ways.

**Outputs, per scenario × link class × payload** (payloads from 8a/8b plus the Phase 3/4b PCA and quantisation variants):
- the rate needed to send in real time (tiles/s × bytes × 8), and the share of imaged tiles a link can carry live;
- tiles per contact window and per day, from the visibility windows;
- per-tile latency T = T_head (8a) + 8 · bytes / rate + distance / c (queueing not modelled);
- **the ISL-feasible frontier:** for each link, the payloads that fit, the best utility among them, and the leakage that comes
  with it. That answers "what does a given link force you to leak?".

**Sanity checks:** the TLEs give a Sentinel-2 period of about 100.6 min and an altitude of about 786 km, EDRS-C sits at about
35,786 km, and the line of sight between 2A and 2B is never clear.

**Output:** `scripts/isl8c.py` → `results/phase8/isl_links.csv`, `isl_windows.csv`, `isl_payloads.csv`,
`isl_feasibility.csv`, `isl_frontier.csv` and `isl_README.json`. The frontier is computed twice: with BigEarthNet utility
(Phase 4b float variants and 8b splits) and with EuroSAT utility (which also covers the Phase 4b quantile codes, the only
payloads small enough for S-band). New dependency:
`pip install skyfield`. Runs in minutes. Descriptive only, so it needs no pre-registration.

**Limits:** these are nominal rates, not full link budgets, so pointing, acquisition time and terminal duty cycles are left
out. On-board storage and queueing are not modelled, and the constellations in S2 and S3 are synthetic.

### Phase 8 schedule (deadline 21 Nov 2026, 23:59)

| When | Work | Cut-off rule |
|---|---|---|
| 18 Nov, evening | ~~build 8a, 8b and 8c; write the 8b pre-registration; run `run_phase8.sh all`~~ (done, finished 21:24) | if 8b hasn't started by 23:59, move it to future work |
| 19 Nov | ~~check the run, read `results/phase8/analysis/` and `isl_*.csv`, write the Phase 8 section of `RESULTS.md`~~ (done 18 Nov) | 8b tier 1 by 14:00, otherwise report it as partial |
| 20–21 Nov | Phase 9: the paper | no new experiments after 20 Nov, 09:00 |

### Phase 8 scripts and how to run them

| Script | Phase | What it does |
|---|---|---|
| `scripts/splitpoints.py` | 8a, 8b | split points, stage functions and the `gap` / `map4` readouts (shared) |
| `scripts/compute8a.py` | 8a | parameters, FLOPs, peak activation, payload bytes, head latency → `results/phase8/compute.csv` |
| `scripts/splits8b.py` | 8b | extracts `gap` and `map4` at every split into `data/compressed/<encoder>@<split>/` (about 14 GB) |
| `scripts/attack4b.py` (unchanged) | 8b | deep-MLP attackers on the split variants → `results/phase8/runs8b.csv` |
| `scripts/analysis8b.py` | 8b | the 8 pre-registered tests, secondary tables, frontier, figure → `results/phase8/analysis/` |
| `scripts/isl8c.py` | 8c | TLEs, contact windows, payloads, feasibility, frontier → `results/phase8/isl_*.csv` |
| `scripts/deepen8.py` | 8d | exploratory follow-ups: pixel 4×4-map control, `gap` vs `map4` location, 8b contrasts on split seeds 1–2 → `results/phase8/deepen/` |
| `scripts/run_phase8.sh` | all | runs `8a`, `8b`, `8c`, or `all` (8a → 8b → 8c); every stage resumes where it stopped |

Checks built in, all of which passed in testing:
- Every encoder's stages reproduce its own output exactly (difference 0).
- On 300-row subsets of EuroSAT, GloSoFarID and BigEarthNet, the final-split `gap` equals the stored Phase 2 embeddings
  (difference 0).
- The FLOP counter gives 3.63 GFLOPs for torchvision ResNet-18 at 224 × 224.
- Head + tail FLOPs and parameters add up to the full model at every split.
- The Sentinel-2 orbit (100.6 min period, about 797 km altitude) and EDRS-C (35,793 km) check out.
- `run_phase8.sh 8b` refuses to start without the "Phase 8b split-point settings" section in `results/preregistration.md`
  (now written).

**One-time setup:** `pip install skyfield` (already done in `eoenv`). 8c fetches public TLEs from CelesTrak once a day into
`data/orbits/`.

**Run everything, detached** (macOS; about 7–10 h, mostly 8b). 8a goes first, because its latency timing needs an idle GPU:

```bash
cd ~/Desktop/projects/Satsplitbench
mkdir -p results/phase8
nohup caffeinate -i bash scripts/run_phase8.sh all > results/phase8/run.log 2>&1 &
tail -f results/phase8/run.log            # Ctrl+C stops watching only
```

Add the optional tier 3 (the localiser on the earliest split, +2–4 h) with
`TIER3=1 nohup caffeinate -i bash scripts/run_phase8.sh all > results/phase8/run.log 2>&1 &`.

**Or one stage at a time:**

```bash
nohup caffeinate -i bash scripts/run_phase8.sh 8a > results/phase8/8a.log 2>&1 &    # ~15-25 min
nohup caffeinate -i bash scripts/run_phase8.sh 8b > results/phase8/8b.log 2>&1 &    # ~6-9 h, after 8a
nohup caffeinate -i bash scripts/run_phase8.sh 8c > results/phase8/8c.log 2>&1 &    # ~1 min, after 8b
nohup caffeinate -i bash scripts/run_phase8.sh 8d > results/phase8/8d.log 2>&1 &    # ~1-1.5 h, after 8b; exploratory follow-ups
```

8d is not part of `all`. Its plan was written after the 8b results were seen, so its results are reported as exploratory
(`results/preregistration.md`, "Phase 8d follow-ups"). Outputs go to `results/phase8/deepen/`: `vs_pixels.csv`, `form.csv`,
`replication.csv` and `README.json`.

Foreground alternative (stops if the terminal closes): `caffeinate -i bash scripts/run_phase8.sh all 2>&1 | tee results/phase8/run.log`.

**Single steps** (as the runner calls them):

| Step | Command |
|---|---|
| 8a | `python scripts/compute8a.py` (add `--no-latency` for FLOPs, parameters and bytes only, ~1 min) |
| 8b extract | `python scripts/splits8b.py` (test: `--encoders resnet18_moco --limit 300 --out /tmp/t8b`) |
| 8b attack | `python scripts/attack4b.py --tasks leakage utility_ben --encoders resnet18_moco@layer1 --variants gap --attackers deep_mlp --out results/phase8/runs8b.csv --preds results/preds/phase8b` |
| 8b analysis | `python scripts/analysis8b.py` |
| 8c | `python scripts/isl8c.py` (offline: `--tle-file data/orbits/tle_<date>.txt`) |

On Linux / WSL2, replace `caffeinate -i` with `systemd-inhibit --what=idle:sleep`. On native Windows, run the single steps in
PowerShell; the runner needs bash. See [How to run](#how-to-run), including how to disable sleep.

## Phase 9: Analysis and paper (planned)

- **Main figure:** utility (x) vs leakage (y), one line per encoder, one point per compression level, with the pixel-statistics level marked. BigEarthNet macro AP is the utility axis (for H2), with EuroSAT as a second panel.
- **Phase 8 figures:** leakage, utility, bytes and head FLOPs against split point (8a/8b), and the ISL-feasible frontier: for each link class, the best utility that fits and the leakage it carries (8c).
- Report 95% CIs and, where seeds exist, mean ± std; paired tests for encoder comparisons. Label every result as pre-registered or exploratory.
- Answer H1–H3, RQ1–RQ5 and the Phase 4b–8 questions.
- Paper, about 6 pages: introduction, related work, setup, results, discussion and limits, conclusion. Secondary tables go to an appendix.
- Deadline: 21 Nov 2026, 23:59.

## Repository layout

```
data/                      # downloaded datasets (git-ignored)
  eurosat/
  glosofarid/
    solar_2023_global.tfrecord.gz
    npy/
  bigearthnet/
  embeddings/              # Phase 2 outputs (+ BigEarthNet ben_*.npy, split seeds 1-2)
  compressed/              # Phase 3 outputs (+ pixelstats control)
  quantised/               # Phase 4b q8/q4/q2/q1 variants
  orbits/                  # Phase 8c: CelesTrak TLEs, one file per download date
  compressed/<enc>@<split>/  # Phase 8b split-point variants (gap, map4)
eoenv/                     # Python virtual environment
scripts/
  load_glosofarid.py       # GloSoFarID TFRecords -> numpy (+ masks)
  embed.py                 # Phase 2: tiling, splits, encoders
  compress.py              # Phase 3: PCA + int8
  probes.py                # Phase 4: linear + MLP probes, controls
  common.py                # shared loading, tile cache, location prior, bootstrap, paired tests, Holm
  attackers.py             # Phase 4b: deep MLP, LightGBM, RBF SVM
  quantise.py              # Phase 4b: quantile codes at 8/4/2/1 bits
  splits.py                # Phase 4b: extra split seeds
  bigearthnet.py           # Phase 4b: BigEarthNet utility task
  attack4b.py              # Phase 4b: attackers on every task
  upperbound.py            # Phase 4b: ResNet-18 image-level upper bound
  analysis4b.py            # Phase 4b: pre-registered tests
  imagemodels.py           # Phase 4b/5: ResNet-18, U-Net, GAN generator and discriminator, perceptual net
  segmenter.py             # Phase 5: panel segmenter
  invert.py                # Phase 5: mean, kNN and GAN inverters + evaluation
  analysis5.py             # Phase 5: pre-registered tests
  localise.py              # Phase 6: query-based transformer decoder, embedding -> panel mask
  analysis6.py             # Phase 6: pre-registered tests
  own_encoder.py           # Phase 7: ViT-Small/8 MAE encoder, baseline and defended (gradient reversal)
  analysis7.py             # Phase 7: baseline vs defended, pre-registered tests
  run_phase4b.sh           # runs all of Phase 4b
  run_phase5.sh            # runs all of Phase 5
  run_phase6.sh            # runs all of Phase 6
  run_phase7.sh            # runs Phase 7: baseline, then defended
  splitpoints.py           # Phase 8: split points, stages, gap/map4 readouts (shared by 8a and 8b)
  compute8a.py             # Phase 8a: FLOPs, parameters, latency, payload bytes per split
  splits8b.py              # Phase 8b: extract pooled and spatial features at every split point
  analysis8b.py            # Phase 8b: pre-registered split-point tests, frontier, figure
  isl8c.py                 # Phase 8c: orbits, contact windows, ISL feasibility
  run_phase8.sh            # runs Phase 8: 8a, 8b, 8c or all
results/
  preregistration.md       # pre-registrations: Phases 4b, 5, 6, 7 and 8b
  probes.csv               # Phase 4 results
  probes.log
  phase4b/                 # runs.csv, analysis/
  phase5/                  # inversion.csv, per-run folders, analysis/
  phase6/                  # localisation.csv, per-run folders, analysis/
  phase7/                  # baseline.log, defended.log, training json, analysis/
  phase8/                  # compute.csv, runs8b.csv, isl_*.csv, analysis/, run.log
  preds/                   # per-tile test predictions
  models/                  # trained image models
```
