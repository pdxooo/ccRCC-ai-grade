# highres_a6000: CHANGES against the HPC originals

Component: A6000 (lab-a6000) versions of the high-resolution pilot scripts (stage 0 + stage 1 extraction).
The scientific recipe is unchanged: same locations (2048 per case, seed rule `stable_seed('highres_v1',20261021,patient_id)`),
same 2x2 sub-tiling, same L/H/Hd inputs, UNI bilinear / Virchow2 bicubic, ImageNet mean/std, fp16 autocast, float16
storage, same decode code. The originals in `originals/` are read-only reference copies (not used at run time).
Nothing has been deployed to, started on or changed on lab-a6000 or the HPC.

K = `/data/cluster/projects/korean_ccrcc_SBIAD2452`, P = `/data/workstation/he_ccrcc_isup_20261021`.

**Launcher: the resident scheduler (`a6000/scheduler/scheduler.py`) is the only launcher for stage 0 and stage 1.**
`run_a6000.sh` is a fallback for use only while the scheduler is not running (section 4). Never use both.

## 0. Review round 2 (2026-10-21): what changed and why

Blocking issues from the review, and their fixes:

| # | Issue | Fix |
|---|---|---|
| B1 | Stage 1 did not check that stage 0 passed (protocol section 7) | `stage1_extract.py` `stage0_gate()`: before any extraction, and before any CUDA use, `features/highres_pilot_v1_20261021_a6000/STAGE0_REPORT.json` must say `PASS_PENDING_HUMAN_VISUAL_REVIEW` with `code_sha256` == sha256 of the current `stage0_extract.py`, all automated checks `True`, the same helper-module sha256s, the same weight reference (smoke summary) sha256 and the same library versions. Not run / running / waiting / error => exit 3 (`waiting_for_stage0`); FAIL or any mismatch => exit 1 (`blocked_by_stage0`). Re-checked on every polling cycle. |
| B2 | Second stage-0 instance: uncaught `BlockingIOError` => exit 1 (= FAIL) | Caught: prints `BUSY`, writes nothing, exit 5. The lock moved out of the output dir to `K/features/highres_pilot_v1_20261021_a6000.lock`. |
| B3 | CUDA OOM while loading models in stage 1 => exit 1 | Model load (and `require_gpu`, the free-memory probe, `embedder`) wrapped: `torch.cuda.OutOfMemoryError` and CUDA/cuBLAS/cuDNN allocation `RuntimeError`s (`out of memory`, `CUBLAS_STATUS_ALLOC_FAILED`, `CUDNN_STATUS_ALLOC_FAILED`, `CUBLAS_STATUS_NOT_INITIALIZED`) => exit 3. Same classifier (`s0.is_gpu_oom`) for OOM during a case and anywhere in the run. |
| B4 | Builder's launch plan duplicated the scheduler's jobs and ran stage 0 next to shard 0 on cuda:0 | The manual launch plan is withdrawn. The scheduler is the launcher (stage 0 alone on its GPU, stage 1 after stage 0 exit 0). `run_a6000.sh` is a documented fallback: it refuses to start, and stops before every retry, while `P/logs/scheduler.lock` is held. Even in fallback mode stage 1 cannot overlap stage 0 any more (B1). |
| B5 | Stage-0 crashes exit 1 like a scientific FAIL; report left at `running` | Stage-0 exit codes: 0 PASS, **1 only for FAIL** (automated checks, weight or manifest sha256 mismatch), 2 crash (report `error` + traceback), 3 waiting (data not verified, GPU busy below `--min-free-gib`, CUDA OOM => report `waiting_gpu_out_of_memory`), 5 lock busy. SIGTERM => report `interrupted`, exit 143. Any uncaught exception outside `main()` (e.g. bad `HE_PATH_PREFIX`) exits 2 via `sys.excepthook` when run as a script. Free-memory check before model load (`--min-free-gib`, default 12). All edits landed before anything ran, so no stage-1 output exists under an older stage-0 sha256. |
| B6 | Builder's advice to add the 4 `uni.npy` files to the `stage0` transfer batch | **Withdrawn, not acted on.** `pull_batch.sh` / `lists/stage0.txt` were not touched. Lane B is executing `pull_batch.sh stage0`; overwriting a running bash script can kill the lane. Stage 0 is not on the critical path: `feat_uni` (lane A, ~2 h) lands well before `stage1_c01` (lane B, >= ~4 h), and stage 0 takes ~20-40 min. If an earlier visual review matters, a separate one-off batch (through `verify.py`, so `.verified` markers are written) is the transfer owner's call, with the user's approval. |

Minor issues fixed in the same round:
* Stage 1 exit policy: errored cases no longer stop a lane while other assigned cases still wait for data: exit 3 with
  state `waiting_with_errors` (errors stay recorded in the shard summary and `error.json`); exit 1 only once nothing is
  left to wait for, or for run-level errors (weight mismatch, recipe mismatch, stage-0 FAIL).
* Provenance: `s0.env_fingerprint()` (python, torch, torch CUDA, cuDNN, torchvision, timm, Pillow, numpy + its path,
  imagecodecs, czifile) is recorded by stage 0, is part of the stage-1 recipe (so a library change mid-run is refused,
  never mixed) and is written into every stage-1 `complete.json` together with the full `platform` (GPU name/memory,
  hostname, device, CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER) and the sha256 of the stage-0 report it ran under.
  The shard summary carries the platform from the start; each run appends start/exit/interrupted lines to
  `OUT/shards/<shard>.runs.jsonl` (the summary itself is still the latest state).
* One recipe for all 120 cases: the first extraction of any shard pins the recipe in `OUT/RECIPE.json` (atomic
  `os.link`); every later process with a different recipe refuses to run (exit 1). A recipe mismatch found in a case's
  `complete.json` is now run-level too (the shard stops instead of extracting further cases under the new recipe).
* Module hashes in decode workers: now actually checked. The spawn pool's initializer (`worker_check`) re-imports
  `czi_pilot_v2` / `full_patch_qc` in every worker and requires the same resolved path and sha256 the main process
  verified, otherwise the pool breaks before any decode. (The earlier report claimed this check; it did not exist.)
* `HE_PATH_PREFIX` unset/empty: the transfer-marker gate is now off (HPC data are local, as in the original). The
  `_a6000` output suffix is kept even without a prefix, deliberately: without it this script would write into the
  directory of the queued HPC job 5397630 (`features/highres_pilot_v1_20261021`). This is a deviation from "unset =>
  identical behaviour" that needs the user's sign-off (deviation C1).
* `verify_no_work_paths.py`: `*_path` lookups are flagged in every context (assignment, comparison, `.get()`), unless
  they are the direct argument of `hpc()` or carry an explicit `# prefixed:` waiver comment saying where the prefix
  comes from. The unpatched originals now give 9 findings (the extra one is the original's unprefixed `weight_path`).
* Stage 0 checks `patient_manifest.csv` against `qc/primary_technical_pool_v1_20261010/LOCK.json`
  `patient_manifest_sha256` (verified on the mirror: `dc87d09e...`, equal to the stage-1 lock's value); mismatch => FAIL.
* Stage 0 checks the transfer markers before any CUDA initialisation (a stage 0 waiting for `feat_uni` no longer
  creates a CUDA context on the shared card every retry).
* `QUARANTINE.json`: four release conditions (A6000 stage-0 automated PASS, visual review 7.4, A6000-vs-A800 parity,
  training check 7.6); rewritten whenever its content differs from the code's; states that nothing in the extraction
  code reads `RELEASE.json`, so every downstream consumer must check it.
* `run_a6000.sh`: writes its own `$$` to the pid file, traps TERM/INT (TERMs the python job first, which kills its own
  decode workers and records `interrupted`, then any leftover children), `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `nice 10` +
  `ionice -c2 -n7`, bounded retries for errors.
* Decode futures have a timeout (`DECODE_TIMEOUT`, 1800 s per batch; normal is seconds): a hung CZI read becomes a
  `DecodeTimeout` error, the workers are terminated, and the process exits instead of holding GPU memory forever.
  After any error, and on SIGTERM, decode workers are terminated rather than waited for.
* Visibility: the stage-1 shard summary has `waiting_for` (per case: `P/transfer/verified_stage0`, the CZI, `uni.npy`,
  `virchow2.npy`, whichever markers are missing) and `stage0_gate` (code + reason). Stage 0 prints one JSON line per case.
* Stage 0 no longer recomputes a PASS: if `STAGE0_REPORT.json` already records `PASS_PENDING_HUMAN_VISUAL_REVIEW` for
  the current code sha256 and cases, and every per-case output re-hashes to its `complete.json`, it exits 0 without
  touching anything (`--force` recomputes). The files under human review are therefore not rewritten by a re-launch.

Kept as is (with reasons):
* The stage-1 recipe hashes the whole `stage0_extract.py` (not only its scientific functions). Strict on purpose:
  every stage-0 edit must land before stage 0 runs on the A6000. A later edit makes stage 1 refuse (stage-0 gate and
  recipe pin) until stage 0 is re-run and the stage-1 output directory is moved aside. All edits of this round are in.
* A6000 outputs stay inside the mirror tree (`K/features/..._a6000`, `K/features/..._stage1_a6000`); the suffixes keep
  them apart from mirrored HPC data. Only the stage-0 run lock was moved out of the output directory.

## 1. Files and sha256

Originals vs patched:

| file | original sha256 | patched sha256 | original source |
|---|---|---|---|
| `stage0_extract.py` | `64cb8927dd1f011807443ca832bc29506a99d0586ddec50331f7ef6c9828e167` | `f7d43854f290eec74508c4916fa8ea569b56dfd1f901dd66009c8a30893ca556` | /Users/xlzhexue/Desktop/proj/lxh/he训练/jobs_20261021/highres_pilot_v1_20261021/stage0_extract.py (= HPC K/bib_phase1/highres_pilot_v1_20261021/stage0_extract.py, job 5397630; A6000 mirror copy has the same sha256; the local original was not modified) |
| `czi_pilot_v2.py` | `a1dd330b100d8fb779e2c4b3be2a965959bfaa98e329ff6b87d9c6640827bf6d` | `b776d23b53c5ddfbb7987fd515dbf561808d386d86a4f9b9857687ae8fa3a1d8` | A6000 mirror P/hpcroot + K/bib_phase1/czi_pilot_v2.py (ssh cat, read-only) |
| `validate.py` | `eeb56e45facf484fc8ddd8a9d436a2eeb73eb386912078d6f31f61168403b9eb` | `637fde1a91a80f3f8b74a4d312ee94c12d719058e1c3dfe8fe7b3627f42c2305` | A6000 mirror P/hpcroot/data/cluster/projects/gastritis_uni2h/outputs/encoder-validation-20261008/validate.py |
| `full_patch_qc.py` | `29e7ffa8623a910ff15cd09dd00966c52aa8c0d153fc00e73c83a6a683a4df9a` | not copied (used unchanged) | A6000 mirror P/hpcroot + K/bib_phase1/full_patch_qc.py (imported unchanged from the mirror at run time) |

New files:

| file | sha256 |
|---|---|
| `stage1_extract.py` | `a3c6740f80e7362e09e9dda2d34eb3b203027a1d0234c339a5dfd64b201b0d78` |
| `verify_no_work_paths.py` | `e2ac8b5135fa45cc636dbcb27e16760d82cc5019786a0f17b3a4384474007af5` |
| `run_a6000.sh` | `e772bb91635988b18ceb8db87c6cba3ca384bbb982d00c0281d94228710610eb` |
| `tests/fake_full_patch_qc.py` | `d9c8186f63abb365f3fd14aca24a899c71076c43c08fbe3ba4aa092e6a5fa80c` |
| `tests/fake_validate.py` | `ce44812ffc6fe8a3d8d3385506c57dfa087bf608fff7c70b9868eea4e1844017` |
| `tests/harness.py` | `3d30c3a98ce5d2273f6bf9018f841434cc5d528cc91d79f9f74ddbf16063d3b6` |
| `tests/layout_probe.py` | `3482603cd865d1b48836278dda6c9839f43eee6f62b34d100719d48fd442c95a` |
| `tests/make_world.py` | `d3f7512a8ca278f287d4f9949bfbaba65f5feeab4daa04c135131b26d4b5c3e9` |
| `tests/run_all.py` | `7397deed25a32fd85bc6f4e47b3f7b55e20ad2e7d6512775ac70f3eb67aacb57` |
| `tests/test_import_layout.py` | `ecb277331d5a099d67184b81f81449ec1e0178de958082c48cafd09f6e03abd9` |
| `tests/test_prefix_identity.py` | `95dc00a36a21900ba33337e014d798ac2bc1e43d0d3310ff593c9ee6b03fe52b` |
| `tests/test_runner.py` | `1ea768ff2657aac6467b8bf264bf91234fe4b8ce2716547422e4dcbb5cc8e3c3` |

`stage0_extract.py` pins the sha256 of the patched `czi_pilot_v2.py`, the patched `validate.py` and the unchanged mirror
`full_patch_qc.py` (`PINNED`), refuses to run if the modules actually imported differ (`check_import_layout()`), and
every decode worker re-checks path + sha256 in the pool initializer (`worker_check()`).

## 2. Deviations from the HPC originals (every one; nothing else differs)

### A. Path prefix (all three patched files)
* Every hard-coded HPC path gets `$HE_PATH_PREFIX` (trailing `/` stripped) prepended. Unset or empty => the exact HPC
  strings (`hpc(p)` returns `p` itself). czi_pilot_v2.py: `ROOT` (line 8; the `qc_deps` insert on line 9 derives from it).
  validate.py: `ROOT`, `ASSETS`, `BIN` (`RUN` derives from `ROOT`). stage0_extract.py: `ROOT`, `OLD`, `row['primary_path']`
  (via `hpc()`, both the sha256 check and the np.load); the slide path is built from `ROOT`.
* czi_pilot_v2.py and validate.py differ from the originals ONLY by the expression
  `os.environ.get('HE_PATH_PREFIX','').rstrip('/')+` and a trailing comment (verified by `tests/test_prefix_identity.py`).
* `verify_no_work_paths.py`: AST check that no `/work/` string literal remains unless it is the right operand of a
  prefix expression or the argument of `hpc()`, and that no `*_path` lookup (subscript or `.get()`, any context)
  bypasses `hpc()` without a `# prefixed:` waiver. Clean on all runtime files; 9 findings in the unpatched originals.
  Not covered: keys not ending in `_path`, and HPC paths that reach the code without such a key.

### B. Import layout (stage0_extract.py, inherited by stage1)
* After the original two inserts (`ROOT/qc_deps`, `ROOT/bib_phase1`) the component directory is inserted at
  `sys.path[0]`. Result: `czi_pilot_v2` resolves to the patched copy here; `full_patch_qc` (no copy here) resolves to
  the unchanged mirror file `ROOT/bib_phase1/full_patch_qc.py`, whose `from czi_pilot_v2 import ...` then binds the
  patched module; numpy/czifile/imagecodecs/tifffile still come from `ROOT/qc_deps` exactly as on the HPC. Spawn decode
  workers inherit the same `sys.path`; the pool initializer `worker_check()` verifies path + sha256 in each worker.
* `validate.py` is loaded from this directory (`VALIDATE=HERE/'validate.py'`) instead of `OLD/encoder-validation-20261008/`.
  `OLD` stays defined (prefixed) but is no longer used for loading. Only `build()`/`forward()` are used; its `main()`
  would write into the mirror's RUN directory and is never called.

### C. Stage 0 additions (no effect on feature values)
* C1 `OUT = ROOT/features/highres_pilot_v1_20261021_a6000` (HPC run uses `.../highres_pilot_v1_20261021`; never mixed),
  also without a prefix (needs the user's sign-off; see section 0). Run lock: `OUT` + `.lock` as a sibling file.
* C2 Transfer gate (only with a prefix): exit 3 (waiting, nothing written, no CUDA) unless
  `$HE_PROJECT_DIR/transfer/verified_stage0` exists and every stage-0 case has `.verified` markers on its CZI and on
  its stored `uni.npy` / `virchow2.npy` (the original would have crashed later on a missing `uni.npy`; the `stage0`
  transfer batch does not carry `uni.npy`, it arrives with `feat_uni`).
* C3 Integrity: `patient_manifest.csv` sha256 must equal the pool `LOCK.json` `patient_manifest_sha256`; each encoder's
  weight sha256 must equal `ROOT/features/nine_encoder_smoke_v1_20261010/summary_shard0.json` `models[name].weight_sha256`.
  Either mismatch => `STAGE0_REPORT.json` state `FAIL`, exit 1, before any extraction.
* C4 Shared GPU: `--min-free-gib` (default 12) checked before loading models (exit 3, nothing written); CUDA OOM
  anywhere => report `waiting_gpu_out_of_memory`, exit 3. `--device` (default `cuda` = current device, as before),
  `--workers` (default 6, as before), `--force`.
* C5 Exit codes and report states: see the module docstring and section 0 (B5). Original: 0 PASS / 1 everything else.
* C6 `STAGE0_REPORT.json` additionally records `platform` (= `env_fingerprint()` + hostname, GPU name/memory/capability,
  device, CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER, HE_PATH_PREFIX, HE_PROJECT_DIR, pid), `modules`, `weights_reference`,
  `manifest_sha256`, `transfer_gate`, `inputs_verified`, `workers`, `gpu_free_gib_at_start`, `scheduler_job`,
  `started_unix`/`ended_unix`/`exit_code`, per-encoder `weight_path` / `weight_sha256_matches_smoke_summary`.
* C7 Refactor without behaviour change so stage 1 can import instead of copy: the embedding closure -> `embedder()`,
  the per-case decode/embed loop -> `extract_locations()`, the stored-L cosine -> `cosine_to_stored()`, the geometry
  dict -> `geometry_record()`, model loading -> `load_models()`, the decode pool -> `decode_pool()`.
  `torch.cuda.synchronize()` -> `embed.synchronize()`. `buffered()` waits at most `DECODE_TIMEOUT` per batch.
  Verified byte-for-byte: the original algorithm (CPU substitutions only) and the patched stage 0 produce identical
  npy, png, complete.json and checks on the fake world (52 files). The per-case `complete.json` of stage 0 is unchanged.
* C8 Skip when a PASS for this code is already recorded and re-hashes (section 0); one JSON progress line per case.

### D. New: stage1_extract.py
* Case list: `$HE_PROJECT_DIR/transfer/lists/STAGE1_CASES_LOCK.json`; recomputed list sha256 must equal both the file's
  `list_sha256` and the pinned `264688b81e670a4b381df8a7b4caf43de64d2926584bfd621994541f94a66584`; 120 unique cases;
  manifest sha256 must equal the lock's `manifest_sha256`; `lists/stage1_cases.txt` must hold the same cases (order
  recorded). Label columns of the manifest are dropped unread; only fold x grade counts from the lock are logged.
* Stage-0 gate (B1). Partition: `--shard i --nshards n` = round-robin over lock order (n=2: every 12-case transfer
  batch gives 6+6).
* A case runs only when `transfer/verified_stage0` (weights) and the `.verified` markers of its CZI, stored `uni.npy`
  and `virchow2.npy` exist. Before extraction: primary coords sha256, CZI receipt sha256 == manifest, stored features
  sha256 == stored complete.json, stored n / coords sha / source sha consistent.
* Science = imported stage-0 functions (`location_indices`, `extract_locations` -> `decode`/`make_inputs`/`buffered`,
  `embedder`, `load_models`, `cosine_to_stored`, `geometry_record`, `decode_pool`). Output per case `OUT/<stem>/`:
  `<enc>_L.npy (n,d)`, `<enc>_H.npy (n,4,d)`, `<enc>_Hd.npy (n,4,d)`, `location_indices.npy`, `coords.npy` (each written
  to `.tmp` then renamed), then `complete.json` (sha256 of all 8 files, recipe sha256, L-vs-stored min/mean cosine per
  encoder, status PASS or FAIL (FAIL if min < 0.999), geometry, input hashes, platform, environment, stage-0 report
  sha256, quarantined=true).
* Recipe = locations, seed rule, batch, encoders (dim, interpolation, weight sha256), stage0_extract.py sha256, the three
  helper-module sha256s, library versions. Pinned for the whole output directory in `OUT/RECIPE.json`.
* Resume: a case with `complete.json` is skipped after re-hashing every file; a hash mismatch is a case error, a
  different recipe stops the shard; nothing is ever silently mixed or overwritten. Per-shard and per-case `flock`s.
* Shared GPU: `--min-free-gib` (default 12) => exit 3 before loading models; CUDA OOM at load or during a case => exit 3
  (`error.json` state `gpu_out_of_memory_retry_later` for the case). `--wait/--poll` keep a loaded process polling for
  newly verified cases (used by the fallback runner; the scheduler relaunches instead). Keeps >= 20 GiB disk free.
* Exit codes: 0 all assigned cases complete and PASS; 3 waiting (stage 0 not passed yet, data not verified, GPU busy /
  OOM; also `waiting_with_errors`); 4 all complete but >= 1 case FAIL; 5 another process holds this shard; 1 error
  (stage 0 FAILED / mismatch, recipe or weight mismatch, lock edited, case errors with nothing left to wait for).
* `OUT/QUARANTINE.json` (section 0).

### E. Protocol deviation that needs the user's consent
Protocol section 7 ("全部通过才进入第 1 阶段") requires ALL stage-0 checks, including the human visual review (7.4) and
the 8-case training check (7.6), before stage 1. With the default scheduler setting (`stage1_requires_approval: false`)
stage-1 **extraction** starts once the automated stage-0 checks (7.1-7.3, 7.5 timing) pass, so the cards keep working
while the human review is pending; 7.4 and 7.6 then gate the **use** of the features (QUARANTINE.json / RELEASE.json).
If the visual review finds no visible difference, the protocol says stop: the extracted stage-1 features are simply never
released. For strict protocol order instead, set `stage1_requires_approval: true` in the scheduler config and create
`P/HIGHRES_STAGE0_APPROVED` after the review (the scheduler then holds stage 1); this component needs no change for that.

## 3. Unified diffs

### czi_pilot_v2.py
```diff
--- original/czi_pilot_v2.py
+++ a6000/czi_pilot_v2.py
@@ -5,7 +5,7 @@
 import sys, math, csv, json, hashlib, time, traceback, collections
 from pathlib import Path
 import xml.etree.ElementTree as ET
-ROOT=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
+ROOT=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/projects/korean_ccrcc_SBIAD2452')  # A6000: HE_PATH_PREFIX mirror root; '' => HPC original
 sys.path.insert(0,str(ROOT/'qc_deps'))
 import numpy as np
 import czifile
```

### validate.py
```diff
--- original/validate.py
+++ a6000/validate.py
@@ -13,10 +13,10 @@
 import timm
 from torchvision import transforms as T
 
-ROOT=Path('/data/cluster/projects/gastritis_uni2h')
+ROOT=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/projects/gastritis_uni2h')  # A6000: HE_PATH_PREFIX mirror root; '' => HPC original
 RUN=ROOT/'outputs/encoder-validation-20261008'
-ASSETS=Path('/data/cluster/software/uni2h/models')
-BIN=Path('/data/cluster/bin')
+ASSETS=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/software/uni2h/models')
+BIN=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/bin')
 NAMES=['uni','uni2-h','phikon-v2','conch','virchow2','gigapath']
 EXPECTED={'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
 MEAN=(.485,.456,.406); STD=(.229,.224,.225)
```

### stage0_extract.py
```diff
--- original/stage0_extract.py
+++ a6000/stage0_extract.py
@@ -4,22 +4,61 @@
   L  : full 128 um crop resized to 224 (current pipeline, ~0.57 um/px)
   H  : 2x2 physical sub-tiles (64 um) each resized to 224 (~0.29 um/px)
   Hd : 2x2 quadrants of the L image, each upsampled to 224 (H geometry, L detail)
+
+A6000 port (lab-a6000, no Slurm; see CHANGES.md). Scientific recipe unchanged. Differences:
+  * every HPC absolute path is prefixed with $HE_PATH_PREFIX ('' or unset => byte-identical HPC paths);
+  * import shim: this directory's patched czi_pilot_v2.py / validate.py are the modules used,
+    full_patch_qc.py is imported unchanged from ROOT/bib_phase1 (path + sha256 checked in the main process at start
+    and again in every decode worker by the pool initializer);
+  * output directory features/highres_pilot_v1_20261021_a6000 (never mixes with the HPC run, job 5397630);
+  * with a prefix (A6000 mirror) it waits unless $HE_PROJECT_DIR/transfer/verified_stage0 and the per-file .verified
+    markers exist; patient_manifest.csv must match the pool LOCK.json; weight sha256s must match the HPC smoke summary;
+  * shared GPU: a busy card or a CUDA out-of-memory => exit 3 (retry later), never reported as FAIL; a crash =>
+    STAGE0_REPORT.json state 'error', exit 2; a PASS already recorded for this exact code is not recomputed (--force);
+  * the embedding loop is factored into module-level functions so stage1_extract.py imports (not copies) it.
+Exit codes: 0 PASS_PENDING_HUMAN_VISUAL_REVIEW (now, or already recorded for this code) | 1 FAIL: automated checks
+failed, or weight / manifest sha256 mismatch (a human must look) | 2 error: crash, state 'error' (retrying is safe) |
+3 waiting: transferred inputs not verified yet, or the shared GPU is busy / ran out of memory (retry later) |
+5 busy: another stage-0 process holds the run lock (nothing done).
 """
 import os
+if __name__=='__main__':  # A6000: exit 1 means FAIL only (the scheduler stops all high-res work on it); any uncaught crash exits 2
+    import sys as _sys,traceback as _tb
+    def _crash(*exc):
+        _tb.print_exception(*exc);_sys.stdout.flush();_sys.stderr.flush();os._exit(2)
+    _sys.excepthook=_crash
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
-import sys,csv,json,hashlib,time,importlib.util,collections,multiprocessing
+import sys,csv,json,hashlib,time,importlib,importlib.util,collections,multiprocessing,fcntl,signal,traceback
 from pathlib import Path
-ROOT=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
+PFX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')  # A6000 mirror root prepended to every HPC absolute path; '' => HPC original
+def hpc(p):return PFX+p if PFX else p
+ROOT=Path(hpc('/data/cluster/projects/korean_ccrcc_SBIAD2452'))
 sys.path.insert(0,str(ROOT/'qc_deps'));sys.path.insert(0,str(ROOT/'bib_phase1'))
-import numpy as np
-from concurrent.futures import ProcessPoolExecutor
 HERE=Path(__file__).resolve().parent
+sys.path.insert(0,str(HERE))  # A6000 import shim: patched czi_pilot_v2.py here shadows ROOT/bib_phase1's; full_patch_qc.py has no copy here
+import numpy as np
+from concurrent.futures import ProcessPoolExecutor,TimeoutError as FuturesTimeout
 POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
 STORED=ROOT/'features/nine_encoder_full_v1_20261010'
-OUT=ROOT/'features/highres_pilot_v1_20261021'
-OLD=Path('/data/cluster/projects/gastritis_uni2h/outputs')
+OUT=ROOT/'features/highres_pilot_v1_20261021_a6000'  # A6000: the HPC run (job 5397630) writes features/highres_pilot_v1_20261021; kept apart even without a prefix
+RUN_LOCK=OUT.with_name(OUT.name+'.lock')  # single-instance lock, outside OUT so a directory-level parity comparison never sees it
+OLD=Path(hpc('/data/cluster/projects/gastritis_uni2h/outputs'))
+VALIDATE=HERE/'validate.py'  # A6000: prefix-patched copy of OLD/'encoder-validation-20261008/validate.py'
+SMOKE=ROOT/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json'  # weights that produced the stored features
+PROJECT=Path(os.environ.get('HE_PROJECT_DIR','/data/workstation/he_ccrcc_isup_20261021'))
+TRANSFER=PROJECT/'transfer'
+GATED=bool(PFX)  # transfer markers exist only on the A6000 mirror; without a prefix (HPC) the data are local, as in the original
+# sha256 of the helper modules this script must import (patched copies here; full_patch_qc unchanged on the mirror).
+PINNED={'czi_pilot_v2.py':'b776d23b53c5ddfbb7987fd515dbf561808d386d86a4f9b9857687ae8fa3a1d8',
+        'validate.py':'637fde1a91a80f3f8b74a4d312ee94c12d719058e1c3dfe8fe7b3627f42c2305',
+        'full_patch_qc.py':'29e7ffa8623a910ff15cd09dd00966c52aa8c0d153fc00e73c83a6a683a4df9a'}
 ENCODERS={'uni':(1024,'bilinear'),'virchow2':(2560,'bicubic')}  # interpolation as in extract_nine_korea.py
 LOCATIONS=2048;SEED=20261021;STAGE0_G1=2;STAGE0_G2=2;BATCH=32;WORKERS=6
+WAITING=3
+EXIT_OK,EXIT_FAIL,EXIT_ERROR,EXIT_BUSY=0,1,2,5
+PASS_STATE='PASS_PENDING_HUMAN_VISUAL_REVIEW'
+MIN_FREE_GIB=12  # shared GPU: with less free memory than this, do not load models (exit 3, retry later)
+DECODE_TIMEOUT=1800  # seconds to wait for one decoded batch (normally seconds): a hung CZI read becomes an error, not a hang
 _reader=None;_path=None
 
 def sha(p):
@@ -67,13 +106,15 @@
             L,H,Hd=make_inputs(rgb,interp);Ls.append(L);Hs.append(H);Hds.append(Hd)
     return start,{k:tuple(np.stack(a) for a in v) for k,v in out.items()},native
 
+class DecodeTimeout(RuntimeError):pass
 def buffered(pool,tasks,depth=8):
     it=iter(tasks);pending=collections.deque()
     for _ in range(depth):
         try:pending.append(pool.submit(decode,next(it)))
         except StopIteration:break
     while pending:
-        result=pending.popleft().result()
+        try:result=pending.popleft().result(timeout=DECODE_TIMEOUT)  # A6000: bounded wait (original: unbounded)
+        except FuturesTimeout:raise DecodeTimeout(f'no decoded batch within {DECODE_TIMEOUT}s (hung CZI read or stalled I/O)') from None
         try:pending.append(pool.submit(decode,next(it)))
         except StopIteration:pass
         yield result
@@ -102,64 +143,258 @@
     finally:r.close()
     return files
 
-def main():
+# ---- A6000 additions: gating, provenance, weight verification, shared-GPU handling (no scientific effect) ----
+def marker(f):return Path(f).with_name(Path(f).name+'.verified')  # written by the transfer lanes after sha256 verification
+def case_inputs(row):
+    """Transferred files one case needs: the CZI and the stored features of every encoder (for the L parity check)."""
+    stem=Path(row['slide_filename']).stem
+    return [ROOT/'data/he'/row['slide_filename']]+[STORED/stem/f'{nm}.npy' for nm in ENCODERS]
+def missing_markers(files):
+    """Transfer gate: the files without a .verified marker. Always [] without a prefix (HPC: data are local, no transfer)."""
+    return [str(f) for f in files if not marker(f).exists()] if GATED else []
+
+def require_gpu(device):
     import torch
-    torch.set_num_threads(2);torch.manual_seed(SEED)
     assert torch.cuda.is_available(),'GPU is required'
-    t_start=time.monotonic()
-    rows=list(csv.DictReader((POOL/'patient_manifest.csv').open(encoding='utf-8-sig')))
-    lookup={r['patient_id']:r for r in rows};cases=select_cases(rows)
-    report=dict(state='running',stage=0,job_id=os.getenv('SLURM_JOB_ID'),code_sha256=sha(__file__),pool_lock_sha256=sha(POOL/'LOCK.json'),
-                cases=cases,locations_per_case=LOCATIONS,encoders={k:dict(dim=v[0],interpolation=v[1]) for k,v in ENCODERS.items()},checks={},timing={})
-    save(OUT/'STAGE0_REPORT.json',report)
-    pathology=module('pathology_adapters',OLD/'encoder-validation-20261008/validate.py')
-    models={}
+    d=torch.device(device);assert d.type=='cuda','GPU is required'
+    if d.index is not None:torch.cuda.set_device(d)
+
+def gpu_free_gib(device):
+    import torch
+    d=torch.device(device)
+    return torch.cuda.mem_get_info(d if d.index is not None else None)[0]/1024**3 if d.type=='cuda' else None
+
+OOM_MARKERS=('out of memory','CUBLAS_STATUS_ALLOC_FAILED','CUDNN_STATUS_ALLOC_FAILED','CUBLAS_STATUS_NOT_INITIALIZED')
+def is_gpu_oom(e):
+    """A CUDA allocation failure (another lab user's job took the shared card's memory): retry later, not a failure."""
+    return type(e).__name__=='OutOfMemoryError' or (isinstance(e,RuntimeError) and any(s in str(e) for s in OOM_MARKERS))
+
+def env_fingerprint():
+    """Versions of everything that can change decoded pixels or features; no CUDA initialisation. Recorded in the
+    stage-0 report; stage 1 puts it into its recipe and requires stage 0 to have passed under the same set."""
+    import platform,torch,PIL
+    def version(name):
+        try:return getattr(importlib.import_module(name),'__version__',None)
+        except Exception:return None
+    return dict(python=platform.python_version(),torch=torch.__version__,torch_cuda=torch.version.cuda,
+                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
+                torchvision=version('torchvision'),timm=version('timm'),pillow=PIL.__version__,numpy=np.__version__,
+                numpy_path=np.__file__,imagecodecs=version('imagecodecs'),czifile=version('czifile'))
+
+def platform_info(device,query_gpu=True):
+    import socket,torch
+    d=torch.device(device)
+    info=dict(env_fingerprint(),hostname=socket.gethostname(),device=str(d),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
+              CUDA_DEVICE_ORDER=os.environ.get('CUDA_DEVICE_ORDER'),HE_PATH_PREFIX=PFX,HE_PROJECT_DIR=str(PROJECT),pid=os.getpid(),gpu=None)
+    if query_gpu and d.type=='cuda':
+        p=torch.cuda.get_device_properties(d)
+        info.update(gpu=p.name,gpu_total_memory_gib=round(p.total_memory/1024**3,2),gpu_capability=f'{p.major}.{p.minor}')
+    return info
+
+def check_import_layout():
+    """The patched czi_pilot_v2 (this dir), the patched validate.py (this dir) and the unchanged mirror full_patch_qc are the modules used."""
+    import czi_pilot_v2,full_patch_qc
+    got={'czi_pilot_v2.py':Path(czi_pilot_v2.__file__).resolve(),'full_patch_qc.py':Path(full_patch_qc.__file__).resolve(),'validate.py':VALIDATE.resolve()}
+    want={'czi_pilot_v2.py':(HERE/'czi_pilot_v2.py').resolve(),'full_patch_qc.py':(ROOT/'bib_phase1/full_patch_qc.py').resolve(),'validate.py':(HERE/'validate.py').resolve()}
+    assert got==want,f'Import layout differs from the A6000 design: {got} != {want}'
+    assert czi_pilot_v2.ROOT==ROOT and full_patch_qc.ROOT==ROOT,(czi_pilot_v2.ROOT,ROOT)
+    files={k:dict(path=str(v),sha256=sha(v)) for k,v in got.items()}
+    for k,v in files.items():assert v['sha256']==PINNED[k],f'{k} sha256 {v["sha256"]} != pinned {PINNED[k]}'
+    return files
+
+def worker_check(expected):
+    """Pool initializer, runs once in every spawn decode worker: the worker must import the very files (path and
+    sha256) that check_import_layout() verified in the main process, else the pool breaks before any decode."""
+    import czi_pilot_v2,full_patch_qc
+    for k,m in (('czi_pilot_v2.py',czi_pilot_v2),('full_patch_qc.py',full_patch_qc)):
+        p=Path(m.__file__).resolve()
+        assert str(p)==expected[k]['path'] and sha(p)==expected[k]['sha256'],f'decode worker {os.getpid()} imports {k} from {p}; main process verified {expected[k]}'
+    assert czi_pilot_v2.ROOT==ROOT==full_patch_qc.ROOT,(czi_pilot_v2.ROOT,ROOT)
+
+def decode_pool(workers,modules):
+    """The spawn decode pool (as in the original), with the per-worker module check."""
+    return ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_check,initargs=(modules,))
+
+def close_pool(pool,kill=False):
+    """kill=True (after an error, a hung decode or a signal): terminate the workers first, so shutdown never blocks."""
+    if pool is None:return
+    if kill:
+        for p in list((getattr(pool,'_processes',None) or {}).values()):
+            try:p.terminate()
+            except Exception:pass
+    pool.shutdown(wait=not kill,cancel_futures=True)
+
+def load_pathology():return module('pathology_adapters',VALIDATE)
+
+class IntegrityError(RuntimeError):pass  # an input does not match its recorded sha256: FAIL (a human must look), not a crash
+class WeightMismatch(IntegrityError):pass
+def load_models(pathology,device='cuda'):
+    """Build each encoder; its weight file sha256 must equal the HPC smoke summary before any extraction."""
+    expected=json.loads(SMOKE.read_text())['models'];models={};infos={}
     for name in ENCODERS:
-        model,_,info=pathology.build(name);report['encoders'][name]['weight_sha256']=sha(info.get('weight_path',info.get('weight')))
-        models[name]=model.eval().requires_grad_(False).cuda()
-    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
+        model,_,info=pathology.build(name);weight=info.get('weight_path',info.get('weight'));got=sha(weight)  # prefixed: validate.build() builds it from the prefixed ASSETS/BIN
+        want=expected[name]['weight_sha256']
+        if got!=want:raise WeightMismatch(f'{name}: weight {weight} sha256 {got} != smoke summary {want}')
+        infos[name]=dict(weight_path=str(weight),weight_sha256=got,weight_sha256_matches_smoke_summary=True)
+        models[name]=model.eval().requires_grad_(False).to(device)
+    return models,infos
+
+def already_passed(cases,stems):
+    """The recorded report if STAGE0_REPORT.json is a PASS made by this exact code for these cases and every per-case
+    output still re-hashes to its complete.json; None otherwise (=> recompute)."""
+    try:r=json.loads((OUT/'STAGE0_REPORT.json').read_text())
+    except (OSError,ValueError):return None
+    if r.get('state')!=PASS_STATE or r.get('code_sha256')!=sha(__file__) or r.get('cases')!=cases:return None
+    for stem in stems:
+        try:c=json.loads((OUT/stem/'complete.json').read_text())
+        except (OSError,ValueError):return None
+        if c.get('code_sha256')!=r['code_sha256']:return None
+        for name,h in c.get('files',{}).items():
+            if not (OUT/stem/name).is_file() or sha(OUT/stem/name)!=h:return None
+    return r
+
+def _terminate(signum,frame):raise SystemExit(128+signum)  # SIGTERM -> unwind: kill the decode workers, mark the report
+
+# ---- shared extraction steps (moved out of main() unchanged so stage 1 imports them) ----
+def embedder(pathology,models,device='cuda'):
+    import torch
+    dev=torch.device(device)
+    mean=torch.tensor([.485,.456,.406],device=dev)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]
     def embed(name,arr):
-        x=(torch.from_numpy(arr).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255.-mean)/std
-        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
+        x=(torch.from_numpy(arr).permute(0,3,1,2).to(dev,dtype=torch.float32)/255.-mean)/std
+        with torch.inference_mode(),torch.autocast(dev.type,dtype=torch.float16):
             y=pathology.forward(name,models[name],x).float()
         assert y.shape==(len(arr),ENCODERS[name][0]) and torch.isfinite(y).all()
         return y.half().cpu().numpy()
+    embed.synchronize=(lambda:torch.cuda.synchronize(dev)) if dev.type=='cuda' else (lambda:None)
+    return embed
+
+def extract_locations(pool,slide,sel,embed,totals,hd_checked=False):
+    """One native decode per location -> L (m,d), H (m,4,d), Hd (m,4,d) float16 features per encoder."""
+    m=len(sel)
+    feats={nm:dict(L=np.zeros((m,d),np.float16),H=np.zeros((m,4,d),np.float16),Hd=np.zeros((m,4,d),np.float16)) for nm,(d,_) in ENCODERS.items()}
+    tasks=((slide,i,sel[i:i+BATCH]) for i in range(0,m,BATCH))
+    t_wait=time.monotonic();sizes=[]
+    for i,batch,native in buffered(pool,tasks):
+        totals['decode_wait']+=time.monotonic()-t_wait;sizes+=native;t_gpu=time.monotonic()
+        for nm,(d,interp) in ENCODERS.items():
+            L,H,Hd=batch[interp];b=len(L)
+            if not hd_checked:
+                assert np.array_equal(quadrants_of_L(L[0],interp),Hd[0]);hd_checked=True
+            feats[nm]['L'][i:i+b]=embed(nm,L)
+            feats[nm]['H'][i:i+b]=embed(nm,H.reshape(-1,224,224,3)).reshape(b,4,d)
+            feats[nm]['Hd'][i:i+b]=embed(nm,Hd.reshape(-1,224,224,3)).reshape(b,4,d)
+        embed.synchronize();totals['gpu']+=time.monotonic()-t_gpu;totals['locations']+=len(native);t_wait=time.monotonic()
+    return feats,sizes,hd_checked
+
+def cosine_to_stored(stored_file,idx,L):
+    stored=np.load(stored_file,mmap_mode='r')[idx].astype(np.float32);new=L.astype(np.float32)
+    return (stored*new).sum(1)/np.linalg.norm(stored,axis=1)/np.linalg.norm(new,axis=1)
+
+def geometry_record(patient,m,sizes,row):
+    w=np.array([s[0] for s in sizes]);h=np.array([s[1] for s in sizes])
+    mpp=float(row['mpp_x'])
+    return dict(patient_id=patient,locations=m,native_w_range=[int(w.min()),int(w.max())],native_h_range=[int(h.min()),int(h.max())],
+                mpp_x=mpp,mpp_y=float(row['mpp_y']),L_um_per_px=float(w.mean()*mpp/224),H_um_per_px=float(w.mean()/2*mpp/224))
+
+def finish(report,state,code,**extra):
+    """Record a non-PASS outcome in STAGE0_REPORT.json (a minimal report if the run stopped before writing one)."""
+    r=report if report is not None else dict(stage=0,code_sha256=sha(__file__))
+    r.update(state=state,exit_code=code,ended_unix=time.time(),**extra);save(OUT/'STAGE0_REPORT.json',r)
+    print(state,json.dumps({k:(v[:300] if isinstance(v,str) else v) for k,v in extra.items()}),flush=True);return code
+
+def main(argv=None):
+    import argparse
+    ap=argparse.ArgumentParser(description='High-res pilot stage 0 on lab-a6000 (4 locked cases, technical acceptance only).')
+    ap.add_argument('--device',default='cuda',help="torch device, e.g. cuda:0 (default 'cuda' = current device, as on the HPC)")
+    ap.add_argument('--workers',type=int,default=WORKERS,help='CZI decode processes (default %(default)s)')
+    ap.add_argument('--min-free-gib',type=float,default=MIN_FREE_GIB,help='shared GPU: exit 3 (retry later) instead of loading models when less is free (default %(default)s)')
+    ap.add_argument('--force',action='store_true',help='recompute although STAGE0_REPORT.json already records a PASS for this code')
+    args=ap.parse_args(argv)
+    assert not PFX or PFX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
+    assert ROOT.is_dir(),f'{ROOT} does not exist (HE_PATH_PREFIX={PFX!r})'
+    gate=TRANSFER/'verified_stage0'
+    if GATED and not gate.exists():
+        print(f'WAITING {gate} does not exist: stage-0 inputs not transferred and verified; refusing to start',flush=True);return WAITING
+    RUN_LOCK.parent.mkdir(parents=True,exist_ok=True);run_lock=RUN_LOCK.open('a')
+    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
+    except BlockingIOError:
+        run_lock.close();print(f'BUSY another stage-0 process holds {RUN_LOCK}; nothing done',flush=True);return EXIT_BUSY
+    signal.signal(signal.SIGTERM,_terminate)
+    ctx=dict(report=None,pool=None)
+    try:return _run(args,gate,ctx)
+    except IntegrityError as e:
+        close_pool(ctx['pool'],kill=True);ctx['pool']=None
+        return finish(ctx['report'],'FAIL',EXIT_FAIL,error=str(e))
+    except Exception as e:
+        close_pool(ctx['pool'],kill=True);ctx['pool']=None
+        if is_gpu_oom(e):  # shared card: another user's job took the memory; retry later
+            if ctx['report'] is None:print(json.dumps(dict(event='gpu_oom',device=args.device,error=str(e)[:300])),flush=True);return WAITING
+            return finish(ctx['report'],'waiting_gpu_out_of_memory',WAITING,error=str(e)[:2000])
+        tb=traceback.format_exc();print(tb,flush=True)
+        if ctx['report'] is None:return EXIT_ERROR
+        return finish(ctx['report'],'error',EXIT_ERROR,error=repr(e)[:2000],traceback=tb)
+    except BaseException:  # SIGTERM (SystemExit) or Ctrl-C: stop the decode workers too, leave an honest report
+        close_pool(ctx['pool'],kill=True);ctx['pool']=None
+        if ctx['report'] is not None:finish(ctx['report'],'interrupted',None)
+        raise
+    finally:
+        close_pool(ctx['pool']);run_lock.close()
+
+def _run(args,gate,ctx):
+    manifest=POOL/'patient_manifest.csv';got=sha(manifest)
+    want=json.loads((POOL/'LOCK.json').read_text()).get('patient_manifest_sha256')
+    if got!=want:raise IntegrityError(f'{manifest} sha256 {got} != {POOL/"LOCK.json"} patient_manifest_sha256 {want}')
+    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
+    lookup={r['patient_id']:r for r in rows};cases=select_cases(rows)
+    missing=missing_markers([f for p in cases for f in case_inputs(lookup[p])])
+    if missing:
+        print('WAITING unverified stage-0 inputs (no .verified marker): '+json.dumps(missing),flush=True);return WAITING
+    if not args.force and already_passed(cases,[Path(lookup[p]['slide_filename']).stem for p in cases]) is not None:
+        print(f'{PASS_STATE} already recorded for this code in {OUT/"STAGE0_REPORT.json"}; outputs re-hashed OK, nothing recomputed (--force to redo)',flush=True)
+        return EXIT_OK
+    import torch
+    torch.set_num_threads(2);torch.manual_seed(SEED)
+    require_gpu(args.device)
+    free=gpu_free_gib(args.device)
+    if free is not None and free<args.min_free_gib:
+        print(json.dumps(dict(event='gpu_busy',device=args.device,free_gib=round(free,1),need_gib=args.min_free_gib)),flush=True);return WAITING
+    t_start=time.monotonic()
+    report=ctx['report']=dict(state='running',stage=0,job_id=os.getenv('SLURM_JOB_ID'),scheduler_job=os.getenv('HE_SCHED_JOB'),code_sha256=sha(__file__),
+                cases=cases,locations_per_case=LOCATIONS,encoders={k:dict(dim=v[0],interpolation=v[1]) for k,v in ENCODERS.items()},checks={},timing={},
+                pool_lock_sha256=sha(POOL/'LOCK.json'),manifest_sha256=got,platform=platform_info(args.device),modules=check_import_layout(),
+                transfer_gate=str(gate) if GATED else None,workers=args.workers,gpu_free_gib_at_start=None if free is None else round(free,2),
+                started_unix=time.time(),inputs_verified={str(f):True for p in cases for f in case_inputs(lookup[p])} if GATED else None)
+    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'STAGE0_REPORT.json',report)
+    pathology=load_pathology()
+    models,infos=load_models(pathology,args.device)  # WeightMismatch -> FAIL (exit 1) in main()
+    for name in ENCODERS:report['encoders'][name].update(infos[name])
+    report['weights_reference']=dict(path=str(SMOKE),sha256=sha(SMOKE));save(OUT/'STAGE0_REPORT.json',report)
+    embed=embedder(pathology,models,args.device)
     totals=dict(locations=0,decode_wait=0.,gpu=0.)
     sims={n:[] for n in ENCODERS};geometry=[];hd_checked=False
-    ctx=multiprocessing.get_context('spawn')
-    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=ctx) as pool:
-        for patient in cases:
-            row=lookup[patient];stem=Path(row['slide_filename']).stem;dst=OUT/stem;dst.mkdir(parents=True,exist_ok=True)
-            assert sha(row['primary_path'])==row['primary_sha256']
-            with np.load(row['primary_path'],allow_pickle=False) as z:coords=z['coords'].copy()
-            n=int(row['primary_count']);assert len(coords)==n
-            idx=location_indices(n,patient);sel=coords[idx];m=len(idx)
-            np.save(dst/'location_indices.npy',idx);np.save(dst/'coords.npy',sel)
-            feats={nm:dict(L=np.zeros((m,d),np.float16),H=np.zeros((m,4,d),np.float16),Hd=np.zeros((m,4,d),np.float16)) for nm,(d,_) in ENCODERS.items()}
-            slide=str(ROOT/'data/he'/row['slide_filename']);tasks=((slide,i,sel[i:i+BATCH]) for i in range(0,m,BATCH))
-            t_wait=time.monotonic();sizes=[]
-            for i,batch,native in buffered(pool,tasks):
-                totals['decode_wait']+=time.monotonic()-t_wait;sizes+=native;t_gpu=time.monotonic()
-                for nm,(d,interp) in ENCODERS.items():
-                    L,H,Hd=batch[interp];b=len(L)
-                    if not hd_checked:
-                        assert np.array_equal(quadrants_of_L(L[0],interp),Hd[0]);hd_checked=True
-                    feats[nm]['L'][i:i+b]=embed(nm,L)
-                    feats[nm]['H'][i:i+b]=embed(nm,H.reshape(-1,224,224,3)).reshape(b,4,d)
-                    feats[nm]['Hd'][i:i+b]=embed(nm,Hd.reshape(-1,224,224,3)).reshape(b,4,d)
-                torch.cuda.synchronize();totals['gpu']+=time.monotonic()-t_gpu;totals['locations']+=len(native);t_wait=time.monotonic()
-            hashes={}
-            for nm,parts in feats.items():
-                for arm,a in parts.items():
-                    assert np.isfinite(a).all();f=dst/f'{nm}_{arm}.npy';np.save(f,a);hashes[f.name]=sha(f)
-                stored=np.load(STORED/stem/f'{nm}.npy',mmap_mode='r')[idx].astype(np.float32);new=parts['L'].astype(np.float32)
-                cos=(stored*new).sum(1)/np.linalg.norm(stored,axis=1)/np.linalg.norm(new,axis=1);sims[nm].append(cos)
-            w=np.array([s[0] for s in sizes]);h=np.array([s[1] for s in sizes])
-            mpp=float(row['mpp_x'])
-            geometry.append(dict(patient_id=patient,locations=m,native_w_range=[int(w.min()),int(w.max())],native_h_range=[int(h.min()),int(h.max())],
-                                 mpp_x=mpp,mpp_y=float(row['mpp_y']),L_um_per_px=float(w.mean()*mpp/224),H_um_per_px=float(w.mean()/2*mpp/224)))
-            save(dst/'complete.json',dict(state='complete',patient_id=patient,n_locations=m,files=hashes,code_sha256=report['code_sha256']))
-            report['visual_files']=report.get('visual_files',[])+visual(dst,patient,sel,slide,np.linspace(0,m-1,4,dtype=int).tolist())
+    pool=ctx['pool']=decode_pool(args.workers,report['modules'])
+    for patient in cases:
+        t_case=time.monotonic()
+        row=lookup[patient];stem=Path(row['slide_filename']).stem;dst=OUT/stem;dst.mkdir(parents=True,exist_ok=True)
+        assert sha(hpc(row['primary_path']))==row['primary_sha256']
+        with np.load(hpc(row['primary_path']),allow_pickle=False) as z:coords=z['coords'].copy()
+        n=int(row['primary_count']);assert len(coords)==n
+        idx=location_indices(n,patient);sel=coords[idx];m=len(idx)
+        np.save(dst/'location_indices.npy',idx);np.save(dst/'coords.npy',sel)
+        slide=str(ROOT/'data/he'/row['slide_filename'])
+        feats,sizes,hd_checked=extract_locations(pool,slide,sel,embed,totals,hd_checked)
+        hashes={}
+        for nm,parts in feats.items():
+            for arm,a in parts.items():
+                assert np.isfinite(a).all();f=dst/f'{nm}_{arm}.npy';np.save(f,a);hashes[f.name]=sha(f)
+            sims[nm].append(cosine_to_stored(STORED/stem/f'{nm}.npy',idx,parts['L']))
+        geometry.append(geometry_record(patient,m,sizes,row))
+        save(dst/'complete.json',dict(state='complete',patient_id=patient,n_locations=m,files=hashes,code_sha256=report['code_sha256']))
+        report['visual_files']=report.get('visual_files',[])+visual(dst,patient,sel,slide,np.linspace(0,m-1,4,dtype=int).tolist())
+        print(json.dumps(dict(event='case',case=stem,locations=m,seconds=round(time.monotonic()-t_case,1),
+                              L_min_cosine={nm:float(v[-1].min()) for nm,v in sims.items()})),flush=True)
+    close_pool(pool);ctx['pool']=None
     cos_all={nm:np.concatenate(v) for nm,v in sims.items()}
     report['checks']=dict(
         L_reproduces_stored={nm:dict(min_cosine=float(c.min()),mean_cosine=float(c.mean()),pass_=bool(c.min()>=.999)) for nm,c in cos_all.items()},
@@ -171,9 +406,10 @@
                           gpu_seconds_per_location=totals['gpu']/totals['locations'],
                           stage1_estimate_hours=stage1_locations/loc_s/3600,stage1_with_50pct_margin_hours=1.5*stage1_locations/loc_s/3600)
     ok=all(v['pass_'] for v in report['checks']['L_reproduces_stored'].values()) and report['checks']['subtiles_union_exact'] and hd_checked
-    report['state']='PASS_PENDING_HUMAN_VISUAL_REVIEW' if ok else 'FAIL'
+    report['state']=PASS_STATE if ok else 'FAIL'
+    report.update(exit_code=EXIT_OK if ok else EXIT_FAIL,ended_unix=time.time())
     save(OUT/'STAGE0_REPORT.json',report);print(report['state'],json.dumps(report['timing']),flush=True)
-    return 0 if ok else 1
+    return EXIT_OK if ok else EXIT_FAIL
 
 def quadrants_of_L(L,interp):
     """Independent recomputation of Hd from a finished L image."""
```

## 4. How to run on lab-a6000 (nothing has been started)
Deploy this directory (all files; `tests/` and `originals/` are inert) to `P/code/highres_a6000/` (the scheduler's
`highres_dir`). Then:
```bash
P=/data/workstation/he_ccrcc_isup_20261021; cd $P/code/highres_a6000
bash run_a6000.sh check        # static /work-literal + _path check on the files actually used (incl. the mirror full_patch_qc.py)
```
**Normal operation: start nothing here.** The scheduler launches `stage0_extract.py` (alone on GPU1, once
`P/transfer/verified_stage0` exists and its own weight pre-check passes) and, after stage 0 exits 0,
`stage1_extract.py --shard i --nshards 2 --device cuda` on whichever card is free (`CUDA_VISIBLE_DEVICES`, nice 10,
ionice, `CUDA_DEVICE_ORDER=PCI_BUS_ID`). Stage-0 waiting for `feat_uni` simply exits 3 and is relaunched by the
scheduler when a new `verified_*` marker appears.

Fallback (only while the scheduler is NOT running; the script refuses otherwise): start stage 0 first, alone.
The stage-1 lanes may be started together with it: they exit 3 without touching a GPU until the stage-0 report is a
PASS for the current code.
```bash
setsid nohup bash run_a6000.sh stage0 cuda:1     >> $P/logs/highres_stage0_a6000.log 2>&1 < /dev/null &
setsid nohup bash run_a6000.sh stage1 0 2 cuda:0 >> $P/logs/highres_stage1_shard0of2.log 2>&1 < /dev/null &
setsid nohup bash run_a6000.sh stage1 1 2 cuda:1 >> $P/logs/highres_stage1_shard1of2.log 2>&1 < /dev/null &
# pid files are written by the runners themselves: P/pids/highres_stage0.pid, highres_stage1_shard{0,1}of2.pid
kill -TERM "$(cat $P/pids/highres_stage1_shard0of2.pid)"     # stop one lane (job + decode workers)
```
Stop all fallback runners before starting the scheduler. Direct calls (debugging):
`HE_PATH_PREFIX=$P/hpcroot HE_PROJECT_DIR=$P CUDA_DEVICE_ORDER=PCI_BUS_ID $P/env/he310/bin/python -u stage0_extract.py --device cuda:1 [--workers 6 --min-free-gib 12 --force]`;
`... stage1_extract.py --shard 0 --nshards 2 --device cuda:0 [--workers 6 --wait 1800 --poll 120 --min-free-gib 12]`.

Where to look: `K/features/highres_pilot_v1_20261021_a6000/STAGE0_REPORT.json` (state, checks, timing, platform);
`K/features/highres_pilot_v1_20261021_stage1_a6000/shards/shard<i>of2.json` (`cases`, `waiting_for`, `stage0_gate`,
`errors`), `shards/shard<i>of2.runs.jsonl`, `RECIPE.json`, `QUARANTINE.json`, `<stem>/complete.json` / `error.json`.

### Interaction with the scheduler (a6000/scheduler, not part of this component)
* Scheduler stage-0 outcomes today: 0 done, 1 FAIL (terminal, holds stage 1), 3 waiting, anything else = retryable
  failure (2 retries, then `failed`, which also holds stage 1). New stage-0 codes map as intended for 0/1/3/2.
  **Recommended scheduler change (owner of a6000/scheduler):** treat stage-0 exit 5 (run lock held: another launcher)
  like stage-1 exit 5 (`busy`, waiting + alert) instead of a failure, and list exit 2 / 5 in its README table.
  With the scheduler as the only launcher, exit 5 should not occur.
* Stage 1 exit codes are unchanged in meaning for the scheduler (0/4 done, 3/5 waiting, 1 failure). The new
  `waiting_with_errors` (exit 3) keeps a shard relaunching on new data instead of failing it.

## 5. Local tests (CPU venv, fake worlds; `tests/run_all.py <scratch_dir>`)
Fakes: `tests/make_world.py` builds an HE_PATH_PREFIX mirror (manifest with HPC-style `/work/...` primary paths, pool
LOCK.json with the manifest sha256, coords, dummy CZIs + receipts, stored features from tiny deterministic encoders,
smoke summary, weight files, czifile stub, the ORIGINAL czi_pilot_v2.py in bib_phase1); `fake_full_patch_qc.py`
(synthetic position-consistent RGB reader installed as the mirror's full_patch_qc.py, so spawn workers use it through
the real import layout; `$FAKE_DECODE_SLEEP` simulates a hung read); `fake_validate.py`. The harness monkeypatches only
require_gpu, load_pathology, LOCATIONS=24, BATCH=8, the pins of the fake reader / fake lock, and the fault injections
listed in its docstring. `tests/test_runner.py` drives `run_a6000.sh` with a fake python (and a fake `flock` where the
OS has none). Run with `PYTHONDONTWRITEBYTECODE=1`; nothing is written into the component directory.
```
PASS verify_no_work_paths: runtime files clean | RESULT PASS files=5 violations=0
PASS verify_no_work_paths: flags the unpatched originals (6 /work literals + 3 unprefixed _path lookups) | RESULT FAIL files=4 violations=9
PASS verify_no_work_paths: _path lookups via assignment / comparison / .get() are flagged, hpc() and waivers are not |  unwrapped_work_literals=0 unwrapped_data_paths=3 comments=0
PASS prefix byte-identity (unset, empty, prefix, prefix/) + patched text minus prefix == original | PASS prefix=None: 12 paths + 3 sys.path insert lists compared; stage0 ROOT=/data/cluster/projects/korean_ccrcc_SBIAD2452 ...
PASS stage0 as a script: configuration crash (relative / missing HE_PATH_PREFIX) exits 2, never 1 (= FAIL) | {'relative/hpcroot': (2, 'AssertionError: HE_PATH_PREFIX must be an absolute path'), '/private/tmp/claude-501/-Users-x...
PASS transfer gate only with a prefix (HPC: no markers); OUT keeps the _a6000 suffix; run lock outside OUT | {None: 'False 0 highres_pilot_v1_20261021_a6000 True', '': 'False 0 highres_pilot_v1_20261021_a6000 True', '/x/hpcroot...
PASS import layout: patched czi_pilot_v2 (main+spawn worker), unchanged mirror full_patch_qc sha pinned 
PASS decode-pool initializer: workers re-check module path + sha256 (verified layout accepted, wrong sha breaks the pool) 
PASS import layout: bypassing the shim is detected | BYPASS_DETECTED Import layout differs from the A6000 design: {'czi_pilot_v2.py': PosixPath('/private/tmp/claude-501/-Use
PASS runner: stage1 exit 3,3,0 -> retried, stops with 0, pid file removed | rc=0 runs=3
PASS runner: job runs with nice 10, CUDA_DEVICE_ORDER=PCI_BUS_ID, --shard/--nshards/--device/--wait | pid=29898 nice=10 order=PCI_BUS_ID args=-u stage1_extract.py --shard 0 --nshards 2 --device cuda:0 --workers 6 --wait 1800 --...
PASS runner: stage0 crash (exit 2) retried MAX_ERROR_RETRIES=2 times, then stops with 2 | rc=2 runs=3
PASS runner: stage0 FAIL (exit 1) stops at once | rc=1 runs=2
PASS runner: stage1 error (exit 1) retried; a waiting exit 3 resets the consecutive-error count | rc=0 runs=6
PASS runner: lock held by another launcher (exit 5) stops at once | rc=5 runs=1
PASS runner: refuses (exit 2, nothing run) while the scheduler holds P/logs/scheduler.lock | d50091/scratchpad/hr_fix/final_run/runner/P/logs/scheduler.lock; it is the only launcher for high-res jobs while it runs
PASS runner: scheduler.lock present but not held -> runs 
PASS runner: pid file holds the runner's own pid; TERM stops the job and its children, exit 143, pid file removed | pidfile=30109 runner=30109 rc=143 worker_alive=False 0.3s
PASS runner tests ran to completion | 9 checks
PASS real lock: list_sha256 recomputes to the pinned constant | 264688b81e670a4b
PASS real lock: stage1_cases.txt (transfer order) == lock order | 120
PASS real lock: shards 0/2 + 1/2 disjoint, cover all 120, 6+6 cases from every 12-case transfer batch | [[6, 6], [6, 6], [6, 6]]
PASS partition is deterministic 
PASS stage0: refuses to start without transfer/verified_stage0 (exit 3) | 49de2d50091/scratchpad/hr_fix/final_run/w_stage0/project/transfer/verified_stage0 does not exist: stage-0 inputs not transferred and verified; refusing t...
PASS stage0: waits while per-case .verified markers are missing (exit 3, checked before any CUDA use) | WAITING unverified stage-0 inputs (no .verified marker): ["/private/tmp/claude-501/-Users-xlzhexue--claude-spares-spare-bMO...
PASS stage0: manifest sha256 != pool LOCK.json -> FAIL before extraction (exit 1) | /private/tmp/claude-501/-Users-xlzhexue--claude-spares-spare-bMO5HE/ef277c37-d9c5-4329-b9db-749de2d5
PASS stage0: weight sha256 != smoke summary -> FAIL before extraction (exit 1) | virchow2: weight /private/tmp/claude-501/-Users-xlzhexue--claude-spares-spare-bMO5HE/ef277c37-d9c5-4329-b9db-749de2d5009
PASS stage0: run lock held by another stage-0 process -> exit 5 (not 1), nothing written | cluster_user/projects/korean_ccrcc_SBIAD2452/features/highres_pilot_v1_20261021_a6000.lock; nothing done
PASS stage0: shared GPU with < --min-free-gib free -> exit 3, no model load, nothing written | [{'event': 'gpu_busy', 'device': 'cpu', 'free_gib': 1.5, 'need_gib': 12}]
PASS stage0: CUDA OOM while loading models -> exit 3, report state waiting_gpu_out_of_memory (not FAIL) | waiting_gpu_out_of_memory
PASS stage0: crash during extraction -> exit 2, report state error + traceback (not FAIL, not left as running) | ValueError('simulated crash during extraction')
PASS stage0: hung CZI decode -> DecodeTimeout, workers killed, exit 2 without hanging | 1.6s DecodeTimeout('no decoded batch within 1.0s (hung CZI read or stalled I/O)')
PASS stage0: SIGTERM -> decode workers killed, report state interrupted (not left as running), exit 143 | rc=143 workers=[30156, 30157] alive=[]
PASS stage0 (patched, CPU): PASS_PENDING_HUMAN_VISUAL_REVIEW (exit 0) | 1.2s cases=['korea:KR901', 'korea:KR902', 'korea:KR903', 'korea:KR904'] L={'uni': 1.0, 'virchow2': 1.0}
PASS stage0: per-case progress lines printed 
PASS stage0: platform (library versions) + modules + weight verification + manifest sha recorded | {'gpu': None, 'torch': '2.8.0', 'pillow': '11.3.0', 'numpy': '2.0.2', 'device': 'cpu'}
PASS stage0: PASS already recorded for this code -> exit 0, nothing recomputed or touched 
PASS stage0: an output no longer matching its complete.json -> recomputed (deterministic) -> PASS 
PASS stage0 original (HPC code, CPU substitutions) runs | tions_per_second_overall": 127.21875154948467, "gpu_seconds_per_location": 0.0021754162391304376, "stage1_estimate_hours": 0.006288381156521737, "stage1_with_50pct_margi...
PASS stage0 patched == HPC original: all npy/png bytes, complete.json, checks identical | 52 files compared; diffs=[]
PASS stage1 outputs == stage0 outputs for the overlapping case(s) (same imported functions) | 8 files identical, diffs=[]
PASS stage1: no stage-0 report -> exit 3 waiting_for_stage0; no GPU, no model load, nothing extracted | {'code': 3, 'reason': '/private/tmp/claude-501/-Users-xlzhexue--claude-spares-spare-bMO5HE/ef277c37-d9c5-4329-b9db-749de2d5...
PASS stage1: QUARANTINE.json lists the 4 release conditions (stage-0 PASS, visual 7.4, A800 parity, training check 7.6) 
PASS stage1: stage-0 report FAIL -> exit 1 blocked_by_stage0, nothing extracted | stage 0 FAILED (simulated: L parity 0.981); stage 1 must not start (protocol sec
PASS stage1: stage 0 still running -> exit 3 waiting_for_stage0 
PASS stage1 world: stage 0 PASS (its markers removed again afterwards) | 3395149347826103, "stage1_estimate_hours": 0.006313033339130436, "stage1_with_50pct_margin_hours": 0.009469550008695654}
PASS stage1: stage 0 PASS, no markers -> exit 3 (waiting), nothing extracted | {'event': 'exit', 'shard': 'shard0of1', 'state': 'waiting', 'code': 3, 'complete': 0, 'waiting': 2, 'fail': [], 'errors': [], 'stage0_gate': None}
PASS stage1: summary says what each case waits for | ['P/transfer/verified_stage0', 'K/data/he/HE_KR905.czi', 'K/features/nine_encoder_full_v1_20261010/HE_KR905/uni.npy', 'K/features/nine_encoder_full_v1_20261010/HE_KR905/virch...
PASS stage1: markers present but transfer/verified_stage0 missing (weights) -> exit 3 
PASS stage1: A ready, B not -> A extracted, exit 3 | 0.7s [{'event': 'case', 'case': 'HE_KR905', 'status': 'PASS', 'seconds': 0.2, 'L_min_cosine': {'uni': 0.9999999403953552, 'virchow2': 0.9999998807907104}}]
PASS stage1 A: location_indices = stage-0 seed rule, coords = primary[idx], shapes (n,d)/(n,4,d) float16, file sha256s | n=37 m=24 shapes={'uni_L': (24, 1024), 'uni_H': (24, 4, 1024), 'uni_Hd': (24, 4, 1024), 'virchow2_L': (24,...
PASS stage1 A: L vs stored cosine recorded (min/mean) and no label fields in complete.json | {'uni': 1.0, 'virchow2': 1.0}
PASS stage1 A: provenance: library versions in complete.json and recipe, platform, stage-0 report sha; RECIPE.json pinned; per-run history | {'torch': '2.8.0', 'pillow': '11.3.0', 'numpy': '2.0.2'}
PASS stage1: every decode worker imported the patched czi_pilot_v2 and the prefixed ROOT | 14 worker reader opens, pids=[30189, 30191, 30192, 30197, 30198]
PASS stage1: a bad case while others still wait for data -> exit 3 waiting_with_errors (lane keeps going), error recorded | {'HE_KR905': 'ERROR', 'HE_KR903': 'WAITING'}
PASS stage1 shard 1/2: assigned only B; B not ready -> exit 3 | ['HE_KR903']
PASS stage1 --wait 3 --poll 1: polls then exits 3 | 3.5s
PASS stage1 shard 1/2: B extracted -> exit 0 
PASS stage1 shard 0/2: A skipped after hash verification -> exit 0, files untouched 
PASS stage1 shard 0/1: everything complete -> exit 0, nothing recomputed 
PASS stage1: corrupted output file and nothing left to wait for -> exit 1 | {'HE_KR905': 'ERROR', 'HE_KR903': 'PASS'}
PASS stage1: restored file -> exit 0 
PASS stage1: shard lock held by another process -> exit 5 | BUSY another process holds shard0of1
PASS stage1: recipe change (LOCATIONS) -> RECIPE.json pin refuses to mix, exit 1, nothing touched 
PASS stage1: stage 0 passed under other stage-0 code -> exit 1 blocked_by_stage0 | stage0_extract.py changed since stage 0 passed (report 000000000000, current f7d43854f290)
PASS stage1: stage 0 passed under other library versions -> exit 1 blocked_by_stage0 | library versions differ from the stage-0 run [stage0, now]: {"pillow": ["0.0.1", "11.3.0"]
PASS stage1: re-extraction of A is bit-identical (deterministic) | 8 file hashes equal
PASS stage1: stored-L mismatch -> case FAIL (min cos < 0.999) recorded, exit 4 | {'uni': 1.0, 'virchow2': -0.013}
PASS stage1: weight file sha256 != smoke summary -> WeightMismatch, nothing extracted, exit 1 
PASS stage1: weight reference (smoke summary) changed -> refused (recipe pin / stage-0 gate), nothing extracted, exit 1 
PASS stage1: edited case lock -> list_sha256 assertion, exit 1 
PASS stage1: shared GPU with < --min-free-gib free -> no model load, exit 3 | [{'event': 'gpu_busy', 'device': 'cpu', 'free_gib': 1.5, 'need_gib': 12}]
PASS stage1: CUDA OOM while loading models -> exit 3 waiting (not 1), nothing extracted | {'event': 'exit', 'shard': 'shard0of1', 'state': 'waiting', 'code': 3, 'complete': 1, 'waiting': 1, 'fail': ['HE_KR903'], 'errors': [], '...
PASS stage1: CUDA OOM during a case -> case left incomplete, exit 3 (retry later) | {'HE_KR905': 'GPU_OOM_RETRY', 'HE_KR903': 'FAIL'}
PASS stage1: cuBLAS allocation failure (plain RuntimeError) -> treated as OOM, exit 3 | {'HE_KR905': 'GPU_OOM_RETRY', 'HE_KR903': 'FAIL'}
PASS stage1: retry after OOM completes A (B still FAIL) -> exit 4, error.json cleared | {'HE_KR905': 'PASS', 'HE_KR903': 'FAIL'}
PASS stage1: hung CZI decode -> DecodeTimeout recorded for the case, workers killed, no hang, exit 1 (nothing left to wait for) | 1.5s
PASS stage1: next run re-extracts A -> exit 4 (B FAIL) 
PASS stage1: SIGTERM -> decode workers killed, summary state interrupted, case left incomplete, exit 143 | rc=143 workers=[30259, 30260] alive=[]
PASS stage1: after SIGTERM the next run completes A -> exit 4 (B FAIL) 
PASS stage1: an outdated QUARANTINE.json is refreshed 
SUMMARY 81/81 passed
```

## 6. Known limitations
* Not executed on lab-a6000 (no GPU, CZI, czifile or real weights locally). The real `IndexedReader` / CZI decode,
  the real UNI / Virchow2 adapters (timm 1.0.20, torch 2.6.0+cu124, Pillow 11.1.0 on A6000) and CUDA fp16 numerics are
  untested here; stage 0's L-vs-stored check (>= 0.999) and the A6000-vs-A800 parity check (not part of this
  component, still to be written) are the guards. Local tests ran on Python 3.9 (A6000: 3.10); no 3.10-only syntax used.
* Stage 0 waits (exit 3) until `feat_uni` is verified, because it needs the 4 cases' stored `uni.npy` (section 0, B6).
* Partition is static (round-robin); no work stealing between GPUs. With CZIs arriving over 12-16 h the transfer, not
  the GPUs, is expected to be the bottleneck.
* `DECODE_TIMEOUT` (1800 s per batch) is a guess with a wide margin; a pathologically slow but healthy read would be
  reported as an error and retried.
* `run_a6000.sh` checks the scheduler lock with `flock -n -s` (a momentary shared lock). A scheduler started in that
  same instant would see its lock as held and refuse to start; rerun it.
* `RELEASE.json` is not enforced by this component; every downstream consumer must check it.
