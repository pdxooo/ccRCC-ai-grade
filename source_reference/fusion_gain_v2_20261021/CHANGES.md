# CHANGES: fusion_gain_v1_20261021 -> fusion_gain_v2_20261021

v2 is the GPU port of v1 for lab-a6000 (2x RTX A6000, torch 2.6.0+cu124, no Slurm). v1 never ran. The science is the
same as v1: members, arms, seeds, `seed_base`, splits, patch sampling (`patch_indices`, `branch_mask`, epoch order),
model heads, L2 normalisation, class weights, loss, AdamW, accumulation, clipping, selection by validation cross-entropy,
min 30, patience 20 and max 100 epochs, evaluation (all primary patches, exact global attention), primary endpoint and
the pre-registered P1-P3 with the paired bootstrap. Everything else is listed below. Unified diffs against
`../../fusion_gain_v1_20261021/` follow (regenerated after the round-2 review fixes).

## Deviations, by file

| File | Change | Effect on science |
|---|---|---|
| common.py | `ROOT = HE_PATH_PREFIX + '/data/cluster/projects/korean_ccrcc_SBIAD2452'`. With the prefix unset or empty, ROOT is exactly the original path. The prefix must be absolute; a trailing `/` is ignored. `qc_deps`, POOL, FEATURES, SPLITS and the new `DATA_READY` all derive from ROOT. `signature()` now also hashes `ready.py`. | none (paths only) |
| common.py | `CUBLAS_WORKSPACE_CONFIG` is forced to `:4096:8` (v1: `setdefault`, so a value exported in the shared account's shell was used silently); the inherited value is kept in `INHERITED_CUBLAS_WORKSPACE_CONFIG` and written to metadata.json. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced, so CUDA indices equal nvidia-smi indices. `OPENBLAS/OMP/MKL_NUM_THREADS` default to 4 (setdefault) before numpy is imported, so `ready.py` polls and direct launches do not start ~47 OpenBLAS threads against the account's shared nproc limit. | none (both workspace values are deterministic; now always the same one) |
| config.json | `version` changed to v2, `device` changed from `cpu` to `cuda`, and a `platform` note was added. Every other field is byte-identical (checked with a JSON comparison). | Runs on the GPU platform (see PROTOCOL Disclosures (c)) |
| train.py | `--device` (default `cuda`). The model and batch tensors live on that device. Batches go to the device as float16 and are then cast to float32, which gives the same values as v1's CPU `.float()`. Validation and test chunks are evaluated on the device, and probabilities are copied back with `.cpu()`. `seed_all` also seeds CUDA and turns off TF32 and cudnn benchmark. `use_deterministic_algorithms(True)` is unchanged. The checkpoint RNG state includes the CUDA generator of the training device, and resume restores it to the current device. `--threads` defaults to `SLURM_CPUS_PER_TASK` or 4. `F.cross_entropy` is kept: the first-round GPU code (`mil_first_round_v2_20261015/train.py:60,125`) ran this exact call on CUDA under `use_deterministic_algorithms(True)`; the smoke's unit checks execute it on the GPU before anything else. | GPU float arithmetic, and dropout masks come from the CUDA Philox generator. Platform effects only, shared by all arms |
| train.py | Formal runs (not `--smoke`) are gated in two steps. `formal_gate` (no CUDA context is created): device type must equal `config.device` (78); `ready.arm_status(arm)` ready (77); `SMOKE_PASS.json` belongs to this code signature and its part covering the arm (`single` for every `single:*` arm, `mdrop4` for mdrop4) is PASS for the same code and data root (78). Then `setup_device`, then `platform_gate`: the part's platform (`device_type`, GPU name, capability, torch, CUDA, cuDNN) equals this one (78). The DATA_READY path is `ROOT/...` through `common.DATA_READY`. | none (gates only) |
| train.py | `last.pt` stores `platform`; a resume on a different platform is refused with exit 78 (`REFUSED ... resume refused`). | none |
| train.py | Signals. Handlers are installed before the slow imports (when run as a script). First SIGTERM or SIGUSR1: pause at the end of the current epoch, or at once while preloading or before training starts; exit 75 with `paused.json` (now with `reason`, `signals`, `time`). Second SIGTERM or any SIGINT: stop at once (`StopNow`), exit 75; the run resumes bitwise from the last epoch's `last.pt` (all writes are tmp+rename). v1 installed the SIGUSR1/SIGTERM flag only after preloading, ignored it during the final evaluation, and SIGINT gave a traceback. | none (resume is bitwise identical; tested) |
| train.py | Exit 74 for transient GPU problems: CUDA unavailable/busy at start-up (the CUDA context is created in `setup_device`), `torch.cuda.OutOfMemoryError`, and RuntimeErrors carrying CUDA/cuBLAS/cuDNN allocation or busy messages. Integrity failures (fingerprint, receipt hash, DATA_READY sha) stay exit 1. | none |
| train.py | Preload logs `CACHE_BYTES` (expected) and `CACHE_LOADED` (bytes actually held, seconds, max RSS), and asserts they are equal. Before preloading, if `/proc/meminfo` MemAvailable is below cache + `--ram-margin-gib`, the run exits 76 without training; the default margin is 128 GiB (shared box; the earlier v2 draft used 8; v1 had only the Slurm allocation check). `--oom-score-adj` (default 500) raises the process's own `oom_score_adj`, so a host OOM kills this resumable job before other users' processes. Each process still keeps its own cache. | none |
| train.py | Provenance. `resources.json` (operational log, not hashed) records per epoch the device, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, GPU name and UUID (nvidia-smi format), host and pid, so every epoch is attributable to a physical card. `metadata.json` adds `invocation` (start, start epoch, card identity, cuBLAS workspace value used and inherited, RAM margin, oom_score_adj, scheduler job id). `complete.json` adds `physical_gpus` and `cross_card_resume`. | none |
| train.py | v1 bug fixed: after the epoch loop `history.json` is rewritten from the checkpoint state, so a crash between `save_pt(last.pt)` and `save_json(history.json)` in the final epoch can no longer hash a stale history into `complete.json`. `paused.json` is deleted when training resumes and when the run completes. A second process on the same run dir exits 79 (`LOCKED`). | none |
| train.py | `--smoke-shape formal` (smoke only): all fold train/val patients at the formal patch and chunk sizes, every validation patch, `--smoke-epochs` epochs, no test patients. `evaluate()` takes the patch limit directly instead of a smoke flag (same values). | none |
| smoke.py | Rewritten gate, same v1 unit checks (chunk equivalence, permutation invariance, gradients, dropped branch gets no gradient, deterministic non-empty masks, mean kept). Parts: `single` (single:phikon-v2, covers every single arm, as v1's only single-arm smoke did) and `mdrop4` (only when all four encoders are verified; otherwise `DEFERRED` in auto mode; `--part mdrop4` appends later). Each attempt lives in `smoke_runs/<arm>/attempt_NNN/` and is never deleted. An attempt adds to the unit checks every arm at its real input dimensions with synthetic features on the device (same seed bitwise; checkpoint/restore bitwise; chunked vs one-pass evaluation), then compares, at the reduced shape AND at the formal shape, an uninterrupted 2-epoch run with a 1-epoch run that is paused (75) and resumed in a fresh process: models, optimizer, all RNG states (incl. CUDA), selection, history and validation predictions must be bitwise identical. The resume legs run on `--resume-device`, by default the other visible GPU when two are visible (cross-card proof). TIMING.json comes from the formal-shape full leg (v1 timed one separate run). A failed check records the part as FAIL (revokes an earlier PASS of the same code); transient problems and stop signals record INCOMPLETE, which never replaces PASS or FAIL. One smoke at a time (`smoke.run.lock`, exit 5); signals are forwarded to the running train.py leg (SIGINT: stop at once) and smoke.py exits 75; each leg has a timeout (`--leg-timeout-seconds`, default 4 h; INCOMPLETE, exit 74). A CPU smoke is refused when config.device is `cuda` (exit 78). `--no-timing` was removed. Train legs are fresh subprocesses (v1 ran them in-process). | none |
| evaluate.py | Campaign check before pooling: every one of the 100 receipts must be COMPLETE and non-smoke, and hash `metadata.json` and `test_predictions.json`; `metadata.contract.code == signature()`, `contract.smoke` False, arm/seed/fold match, fingerprint matches the receipt; one platform, one split hash and one QC-pool hash across all 100 runs; SMOKE_PASS.json still PASS for every covering part under this signature and the same platform. RESULTS.json gets a `campaign` block (signature, platform, hashes, smoke attempts, physical GPUs, cross-card resumes) and a WARNING is printed if runs resumed across cards but no smoke part proved it. Pooling, metrics and bootstrap are unchanged. | none on a valid campaign; refuses mixed campaigns |
| PROTOCOL.md | Title is v2, with a note that v1 never ran and its CPU and Slurm sections are kept as history. New `Disclosures` section: (a) 183/300 validation-in-some-fold patients; (b) at least the 4th fusion attempt on the same Korean OOF; (c) A6000 GPU platform, features extracted on A800, all arms on the same platform. Section on running v2 on the A6000 (steps, smoke contents and states, exit codes, signals, RAM, deployment, campaign check). | documentation |
| ready.py (new) | `python ready.py <arm>\|all` exits 0 only if `<transfer>/verified_feat_<enc>` exists for every member AND all 300 `<enc>.npy` files and their `.verified` markers exist. `<transfer>` is `$HE_TRANSFER_DIR`, otherwise `dirname(HE_PATH_PREFIX)/transfer`. With no prefix (native HPC), only file existence is checked. Imports no torch: use it to poll. | none |
| run_arm.sh (new) | Foreground replacement for `run_arm.slurm`: folds 0-4 in turn. `run_arm.sh <arm> <seed> <gpu> [threads] [train.py options ...]` (extra options are passed through, e.g. `--ram-margin-gib`, `--max-runtime-seconds 82800`; without them train.py's default 79200 s applies, v1's Slurm script passed 82800). train.py runs in the background of the script and is waited for; USR1, TERM and INT are trapped and forwarded (USR1->USR1; first TERM->USR1 so a group-wide TERM stays a graceful pause; second TERM->INT; INT->INT); after a signal no further fold starts and the script exits with train.py's code (75). Exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the interpreter variable is `HE_PY` (was `PY`); prints a start banner (time, host, physical GPU, arm, seed, pid) and a line per fold. Not part of the code signature. | none |
| run_smoke.sh (new) | Foreground replacement for `run_smoke.slurm`; `exec`s smoke.py (which handles signals itself). Exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`, uses `HE_PY`, prints a start banner. `run_smoke.sh 0,1` is the recommended first run (cross-card resume). Not part of the code signature. | none |
| run_arm.slurm, run_smoke.slurm | Not carried over (no Slurm on lab-a6000). | none |
| tests/test_local.py (new) | Synthetic-mirror CPU test (see below). Not part of the code signature. | none |

## Round-2 review fixes

Blocking:

1. **A failed smoke re-run did not revoke an earlier PASS** (smoke.py / train.py formal_gate). Fixed: any failed check
   records `parts.<part>.state = FAIL` in SMOKE_PASS.json under its lock (formal_gate accepts only PASS, so formal runs
   and resumes of that part are refused with 78 until an attempt passes again). Earlier evidence is never deleted:
   each attempt has its own `smoke_runs/<arm>/attempt_NNN/` directory with an `attempt.json`; SMOKE_PASS.json keeps an
   `attempts` log. Transient errors and stop signals are recorded INCOMPLETE and never replace PASS or FAIL (they prove
   nothing either way). The local test reproduces the reviewer's scenario (injected failure after a PASS -> FAIL,
   formal run refused 78, attempt_001 intact).
2. **evaluate.py did not check that the 100 receipts form one campaign.** Fixed in `oof()`/`check_campaign()` as
   requested: hashed metadata.json, `contract.code == signature()`, `contract.smoke is False`, receipt
   `state == 'COMPLETE'`, one identical platform across all 100 runs, plus matching arm/seed/fold/fingerprint, one
   split and QC-pool hash, and the covering smoke parts still PASS under this signature and platform. Landed before any
   formal run, so it does not invalidate anything.
3. **run_arm.sh did not forward signals or wait for its child.** Fixed as suggested: train.py starts with `&`,
   `pid=$!`; USR1/TERM/INT are trapped and forwarded; the script loops on `wait $pid` until train.py has really exited
   (handling the interrupted-wait and already-collected-status races), takes the child's code, starts no further fold
   after a signal and exits with that code (75). Tested locally (bash 3.2; same semantics in CentOS7 bash 4.2): SIGUSR1
   and SIGTERM to the script's pid, and SIGTERM to its process group, all give a graceful pause, exit 75, no next fold,
   no orphan holding run.lock. (The scheduler launches train.py per fold directly, so this mainly protects manual use.)

Minor issues fixed: formal-shape determinism (formal-shape full vs resumed legs in every smoke attempt); cross-card
resume (default resume leg on the other visible card; per-attempt `cross_card_resume`; evaluate reports/warns);
per-epoch card provenance (GPU UUID, CUDA_VISIBLE_DEVICES); `CUDA_DEVICE_ORDER=PCI_BUS_ID` in code and scripts;
`CUBLAS_WORKSPACE_CONFIG` forced and recorded; stale `history.json` / `paused.json` / `SMOKE_FAIL_<part>.json`;
resume-on-other-platform now exits 78 as documented; GPU-free gates before CUDA initialisation; SMOKE_PASS.json
`parts_summary` and `state_note`; ModalityDropoutFusion exercised at real dimensions on the device in every part, so a
problem shows up in the `single` smoke rather than ~6 h later; `--max-runtime-seconds` difference documented and
passable; smoke-wide lock, signal forwarding and per-leg timeout; exit 74 for transient GPU failures; RAM margin
128 GiB, `oom_score_adj` 500, `"$@"` pass-through in run_arm.sh; signal handlers installed before imports, second
SIGTERM/SIGINT immediate; OpenBLAS thread default; smoke exit codes (2 is usage only, wrong platform 78, 4 only for a
TIMING write failure, 5 lock, 74/75); start banners and `HE_PY`.

Not changed in this component (by design or outside it):

- **Output location.** All outputs stay inside the code directory, because the scheduler reads
  `{fusion_dir}/runs/.../complete.json` and `{fusion_dir}/SMOKE_PASS.json`. Deployment rules are in PROTOCOL.md (deploy
  once, `rsync` without `--delete` and with the outputs excluded, make the six signature files read-only after the smoke).
- **Cross-card resume is recorded, not gated.** Both cards are the same platform (GPU name, capability, torch, CUDA,
  cuDNN). Refusing cross-card resumes would idle a card whenever the scheduler places a paused fold on the other one.
  Under the scheduler each smoke sees one card, so cross-card proof needs `run_smoke.sh 0,1` (or the scheduler giving
  smoke jobs both cards); evaluate.py warns if runs resumed across cards without that proof.
- **Scheduler-side items** (scheduler component): treat train.py 74 as retryable (it currently counts as a failure),
  treat smoke 74/75 as retryable and 5 as "another smoke running"; keep a global RAM budget that counts preloading jobs
  and stagger starts (the MemAvailable check is per process); poll readiness with `ready.py`; export thread variables
  for every launch; parallelism (each process needs ~1-1.5 GB of GPU memory, so more concurrent folds per card are
  possible within RAM/CPU limits).
- A crash in smoke.py before an attempt starts (e.g. unreadable config) still exits 1 without changing any part.
- CUDA itself is untested locally: deterministic kernels at formal shapes on the A6000, CUDA RNG checkpointing, the
  GPU UUID read-out (torch 2.6 has `uuid`, no PCI fields; mocked locally) and cross-card resume. The GPU smoke is the gate.

## Local test (CPU, synthetic data)

`python tests/test_local.py [--workdir DIR] [--keep]` builds a fake 300-patient mirror (manifest, BOM splits, LOCKs,
float16 features for the 4 members, complete.json, DATA_READY.json, transfer markers) in a temporary directory. It copies
the code byte for byte, except that config.json `device` becomes `cpu`, and drives the real CLIs with HE_PATH_PREFIX set.
Failures are injected only through `python -c` monkeypatches, never by editing the code. 18 checks: path prefix and
forced environment; ready.py exit codes; gates (78/77/78, CPU refused under the real config); smoke auto (single PASS,
reduced- and formal-shape bitwise resume, all arms at real dimensions, mdrop4 DEFERRED, attempt directories); FAIL
revokes PASS (formal run refused), evidence kept, transient attempts INCOMPLETE, re-pass restores; smoke lock (5) and
SIGTERM forwarding to the leg (75, no orphan); `--part mdrop4` appending; formal run uninterrupted vs paused+resumed
bitwise identical incl. test predictions, prediction structure, receipt hashes, history/paused.json hygiene,
provenance, ALREADY_COMPLETE; formal mdrop4 starts; train.py SIGTERM (pause) and SIGINT (immediate) both resume
bitwise identical, SIGTERM during start-up -> 75; RAM guard 76; lock 79; platform-mismatch resume 78; transient GPU
errors 74 vs integrity 1; GPU UUID parsing (mocked); run_arm.sh signal forwarding (pid and process group) then all 5
folds; evaluate.py one-campaign refusals (other code, mixed platform, smoke or non-COMPLETE receipt, wrong contract,
revoked smoke part) and the cross-card warning.

## Unified diffs against v1

### common.py

```diff
--- v1/common.py
+++ v2/common.py
@@ -1,13 +1,26 @@
 import os,sys,json,csv,hashlib
-os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
+# Forced (not setdefault): a value inherited from the shared account's shell must not silently change the cuBLAS workspace
+# between the smoke, pause and resume legs. The inherited value, if any, is recorded in metadata.json.
+INHERITED_CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
+os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
+# CUDA device indices = nvidia-smi indices (PCI bus order), so CUDA_VISIBLE_DEVICES picks the card the scheduler means.
+os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
+# The account's nproc limit is shared: never let numpy's OpenBLAS (qc_deps) start one thread per core on import.
+for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_k,'4')
 from pathlib import Path
-ROOT=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
+# HE_PATH_PREFIX (lab-a6000: /data/workstation/he_ccrcc_isup_20261021/hpcroot) is prepended to the HPC absolute path.
+# Unset or empty -> exactly the original HPC path. Every other data path derives from ROOT.
+HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
+PREFIX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')
+assert PREFIX=='' or PREFIX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
+ROOT=Path(PREFIX+HPC_ROOT)
 sys.path.insert(0,str(ROOT/'qc_deps'))
 import numpy as np
 HERE=Path(__file__).resolve().parent
 POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
 FEATURES=ROOT/'features/nine_encoder_full_v1_20261010'
 SPLITS=ROOT/'bib_phase1/first_round_v1_20261010'
+DATA_READY=ROOT/'bib_phase1/mil_first_round_v2_20261015/DATA_READY.json'
 DIMS={'resnet50':2048,'convnext_tiny':768,'vit_b_16':768,'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
 def sha(path):
     h=hashlib.sha256()
@@ -19,7 +32,7 @@
     tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
 def config():return json.loads((HERE/'config.json').read_text())
 def signature():
-    names=['common.py','train.py','smoke.py','evaluate.py','config.json']
+    names=['common.py','train.py','smoke.py','evaluate.py','ready.py','config.json']
     return hashlib.sha256(json.dumps({n:sha(HERE/n) for n in names},sort_keys=True).encode()).hexdigest()
 def read_csv(path):
     # Windows exports may carry a UTF-8 BOM; never change locked input bytes.
```

### config.json

```diff
--- v1/config.json
+++ v2/config.json
@@ -1,5 +1,5 @@
 {
-  "version": "fusion_gain_v1_20261021",
+  "version": "fusion_gain_v2_20261021",
   "target": "patient_level_ISUP_1_to_4",
   "members": ["phikon-v2", "uni", "gigapath", "uni2-h"],
   "member_rule": "Top-4 encoders by first-round mean VALIDATION macro-F1 (test folds not used for selection)",
@@ -29,7 +29,8 @@
   "selection": "validation_unweighted_cross_entropy",
   "class_weight": "training_n_over_4_times_class_count",
   "evaluation": "all_primary_patches_exact_global_attention",
-  "device": "cpu",
+  "device": "cuda",
+  "platform": "lab-a6000 RTX A6000, torch 2.6.0+cu124",
   "primary_endpoint": "pooled OOF AUC, high (ISUP 3-4) vs low (ISUP 1-2), score = P(G3)+P(G4)",
   "secondary_endpoints": ["pooled OOF quadratic weighted kappa (4-class)", "pooled OOF macro-F1 (4-class)"],
   "preregistered_comparisons": {
```

### train.py

```diff
--- v1/train.py
+++ v2/train.py
@@ -1,10 +1,29 @@
-"""Fusion-gain round: frozen-feature gated ABMIL on CPU. Same splits, sampling and head as the first round;
-selection by validation cross-entropy; 4 seeds; adds a modality-dropout mean-fusion arm."""
+"""Fusion-gain round v2: GPU port of fusion_gain_v1 (frozen-feature gated ABMIL). Same splits, sampling and head as the
+first round; selection by validation cross-entropy; 4 seeds; adds a modality-dropout mean-fusion arm.
+Exit codes: 0 complete (or already complete); 1 integrity or check failure (do not retry blindly); 2 usage error;
+74 transient GPU problem (CUDA unavailable/busy, out of GPU memory; resumable, retry later); 75 paused or stopped by a
+signal (resumable); 76 deferred: not enough available CPU RAM; 77 deferred: arm data not verified (formal only);
+78 refused: no passing smoke part, wrong platform, or a resume on another platform; 79 run dir locked by another process.
+Signals: the first SIGTERM or SIGUSR1 pauses at the next safe point (end of the current epoch; at once while the cache is
+preloading or before training starts) and exits 75. A second SIGTERM, or any SIGINT, stops at once and exits 75; the run
+then resumes from the last epoch checkpoint (last.pt), bitwise identically. SIGUSR1 never stops at once."""
+import signal
+class StopNow(BaseException):
+    """Raised by the signal handler for an immediate stop (second SIGTERM or any SIGINT)."""
+STOP=dict(requested=False,terms=0,signals=[],armed=False)
+def on_signal(signum,frame):
+    STOP['requested']=True;STOP['signals'].append(signal.Signals(signum).name)
+    if signum==signal.SIGTERM:STOP['terms']+=1
+    if STOP['armed'] and (signum==signal.SIGINT or STOP['terms']>=2):raise StopNow(signal.Signals(signum).name)
+def install_signal_handlers():
+    for s in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(s,on_signal)
+if __name__=='__main__':install_signal_handlers()  # before the slow imports below: a signal now only sets the flag
 from common import *
-import argparse,time,random,signal,fcntl
+import argparse,time,random,fcntl,socket,resource,traceback,datetime,re
 import torch
 from torch import nn
 import torch.nn.functional as F
+import ready as R
 
 class GatedMIL(nn.Module):
     def __init__(self,dimensions,hidden=256,attention_hidden=128,dropout=.25):
@@ -53,8 +72,25 @@
     assert spec['fusion']=='modality_dropout_mean'
     return ModalityDropoutFusion(dims,cfg['hidden'],cfg['attention_hidden'],cfg['dropout'],spec['branch_dim'])
 
+def max_rss_bytes():
+    r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;return int(r if sys.platform=='darwin' else r*1024)
+def mem_available():
+    try:
+        with open('/proc/meminfo') as f:
+            for line in f:
+                if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
+    except OSError:pass
+    return None
+def raise_oom_score(value):
+    """Make this (resumable) process the kernel OOM killer's preferred victim on the shared box. Raising needs no privilege."""
+    try:
+        with open('/proc/self/oom_score_adj','w') as f:f.write(str(int(value)))
+        with open('/proc/self/oom_score_adj') as f:return int(f.read())
+    except (OSError,ValueError):return None
+
 class BagStore:
-    def __init__(self,lookup,encoders):self.lookup=lookup;self.encoders=encoders;self.cache={}
+    """Per-process CPU-RAM cache of float16 features (no sharing between processes); batches are moved to `device`."""
+    def __init__(self,lookup,encoders,device='cpu'):self.lookup=lookup;self.encoders=encoders;self.device=torch.device(device);self.cache={}
     def path(self,patient):return FEATURES/Path(self.lookup[patient]['slide_filename']).stem
     def load(self,patient):
         if patient not in self.cache:
@@ -67,120 +103,235 @@
                 arrays.append(a)
             self.cache[patient]=arrays
         return self.cache[patient]
-    def preload(self,patients):
-        need=sum(int(self.lookup[p]['primary_count']) for p in patients)*sum(DIMS[x] for x in self.encoders)*2
+    def need(self,patients):return sum(int(self.lookup[p]['primary_count']) for p in patients)*sum(DIMS[x] for x in self.encoders)*2
+    def preload(self,patients,stop=lambda:False):
+        """Returns the bytes loaded, or None if stop() became true (the cache is then dropped)."""
+        need=self.need(patients)
         allocation=os.environ.get('SLURM_MEM_PER_NODE')
         if allocation and allocation.isdigit():assert need+8*1024**3<int(allocation)*1024**2,'Insufficient requested CPU RAM'
-        print('CACHE_BYTES',need,flush=True)
+        print('CACHE_BYTES',need,'GiB',round(need/1024**3,2),'patients',len(patients),'encoders',','.join(self.encoders),flush=True)
+        t=time.monotonic()
         for i,p in enumerate(patients):
+            if stop():self.cache.clear();print('CACHE_ABORTED after',i,'patients (stop requested)',flush=True);return None
             self.load(p)
             if (i+1)%25==0:print('CACHE_PATIENTS',i+1,'/',len(patients),flush=True)
+        loaded=sum(a.nbytes for p in patients for a in self.cache[p]);assert loaded==need
+        print('CACHE_LOADED bytes',loaded,'GiB',round(loaded/1024**3,2),'seconds',round(time.monotonic()-t,1),'max_rss_bytes',max_rss_bytes(),flush=True)
+        return loaded
     def tensors(self,patient,indices):
-        return [torch.from_numpy(np.ascontiguousarray(a[indices])).float() for a in self.load(patient)]
+        # float16 -> device -> float32 (exact conversion; identical values to v1's CPU .float()).
+        return [torch.from_numpy(np.ascontiguousarray(a[indices])).to(self.device).float() for a in self.load(patient)]
     def chunks(self,patient,chunk,limit=None):
         n=int(self.lookup[patient]['primary_count'])
         indices=patch_indices(n,limit,20261014,0,-1,patient)
         for i in range(0,len(indices),chunk):yield self.tensors(patient,indices[i:i+chunk])
 
+class TransientGPU(Exception):
+    """CUDA unavailable, busy or out of memory: not an integrity problem; the run can be retried later (exit 74)."""
+TRANSIENT_MARKERS=('out of memory','busy or unavailable','CUBLAS_STATUS_ALLOC_FAILED','CUBLAS_STATUS_NOT_INITIALIZED',
+                   'CUDNN_STATUS_ALLOC_FAILED','CUDNN_STATUS_NOT_INITIALIZED','no CUDA-capable device','CUDA driver initialization failed')
+def is_transient(e):
+    if isinstance(e,TransientGPU) or isinstance(e,getattr(torch.cuda,'OutOfMemoryError',())):return True
+    return isinstance(e,RuntimeError) and any(m in str(e) for m in TRANSIENT_MARKERS)
+def setup_device(name):
+    device=torch.device(name)
+    if device.type=='cuda':
+        try:
+            if not torch.cuda.is_available():raise TransientGPU('CUDA requested but not available')
+            if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
+            torch.cuda.set_device(device);torch.zeros(1,device=device)  # create the context now: a busy or full GPU fails here
+        except TransientGPU:raise
+        except RuntimeError as e:raise TransientGPU(f'CUDA initialisation failed on {device}: {e}') from e
+    return device
+def platform(device):
+    p=dict(device_type=device.type,torch=torch.__version__,cuda=torch.version.cuda)
+    if device.type=='cuda':p.update(gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),cudnn=torch.backends.cudnn.version())
+    return p
+def gpu_identity(device):
+    """Which physical card this process uses (platform() deliberately excludes it: both A6000s are one platform)."""
+    ident=dict(device=str(device),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'),host=socket.gethostname(),pid=os.getpid())
+    if device.type=='cuda':
+        pr=torch.cuda.get_device_properties(device);u=getattr(pr,'uuid',None);raw=getattr(u,'bytes',None);uuid=None
+        if raw is not None and len(raw)==16:h=bytes(bytearray(raw)).hex();uuid=f'GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'  # nvidia-smi format
+        elif u is not None and re.fullmatch(r'(GPU-)?[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(u)):uuid=str(u) if str(u).startswith('GPU-') else 'GPU-'+str(u)
+        ident.update(gpu_name=pr.name,gpu_uuid=uuid,pci_domain_id=getattr(pr,'pci_domain_id',None),pci_bus_id=getattr(pr,'pci_bus_id',None),pci_device_id=getattr(pr,'pci_device_id',None))
+    return ident
 def seed_all(seed,threads):
     random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
+    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
     torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
-def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
-def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
+    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
+def rng_state(device):
+    s=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
+    if device.type=='cuda':s['cuda']=torch.cuda.get_rng_state(device)
+    return s
+def restore_rng(s,device):
+    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
+    if device.type=='cuda':torch.cuda.set_rng_state(s['cuda'],device)
+    else:assert 'cuda' not in s,'checkpoint holds CUDA RNG state but device is not CUDA'
 def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
+def param_hash(model):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
 def save_pt(path,value):
     tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
-def evaluate(model,store,patients,cfg,role,fold,smoke=False):
+def evaluate(model,store,patients,cfg,role,fold,limit=None):
     model.eval();records=[]
     with torch.no_grad():
         for patient in patients:
-            logits=model.predict_chunks(store.chunks(patient,cfg['eval_chunk'],256 if smoke else None))
-            probs=logits.softmax(1)[0].numpy();assert np.isfinite(probs).all()
+            logits=model.predict_chunks(store.chunks(patient,cfg['eval_chunk'],limit))
+            probs=logits.softmax(1)[0].cpu().numpy();assert np.isfinite(probs).all()
             records.append(dict(patient_id=patient,fold=fold,role=role,ISUP_grade=int(store.lookup[patient]['ISUP_grade']),probabilities=probs.tolist()))
     return records,metrics([r['ISUP_grade']-1 for r in records],[r['probabilities'] for r in records])
 
 def roles_for(assign,fold):
     return {r:sorted(x['patient_id'] for x in assign if int(x['fold'])==fold and x['role']==r) for r in ('train','val','test')}
 
+def smoke_part(arm,cfg):
+    """Smoke part that must have passed before formal training of `arm`: 'single' covers every single:* arm, 'mdrop4' the fusion arm."""
+    return 'single' if cfg['arms'][arm]['fusion']=='none' else arm
+def formal_gate(arm,cfg,device_type):
+    """Checks that need no GPU. None if formal training of `arm` may start, else (exit_code, reason)."""
+    if device_type!=cfg['device']:return 78,f'device type {device_type} is not the configured platform {cfg["device"]}'
+    status=R.arm_status(arm,cfg)
+    if not status['ready']:return 77,'data not verified: '+json.dumps(status['missing'][:10])
+    path=HERE/'SMOKE_PASS.json'
+    if not path.exists():return 78,'SMOKE_PASS.json missing'
+    smoke=json.loads(path.read_text());part=smoke_part(arm,cfg);rec=smoke.get('parts',{}).get(part,{})
+    if smoke.get('state')!='PASS' or smoke.get('code_signature')!=signature():return 78,'SMOKE_PASS.json does not belong to this code signature'
+    if rec.get('state')!='PASS' or arm not in rec.get('covers',[]):return 78,f'smoke part {part} has not passed (state {rec.get("state")})'
+    if rec.get('code_signature')!=signature() or rec.get('root')!=str(ROOT):return 78,f'smoke part {part} ran with other code or another data root'
+    return None
+def platform_gate(arm,cfg,plat):
+    rec=json.loads((HERE/'SMOKE_PASS.json').read_text())['parts'][smoke_part(arm,cfg)]
+    if rec.get('platform')!=plat:return 78,f'smoke part {smoke_part(arm,cfg)} ran on {rec.get("platform")}, this is {plat}'
+    return None
+def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
+def pause(out,next_epoch,fid,reason):
+    save_json(out/'paused.json',dict(next_epoch=next_epoch,fingerprint=fid,reason=reason,signals=STOP['signals'],time=now()))
+    print('PAUSED',out,'next_epoch',next_epoch,reason,flush=True);return 75
+
 def run(args):
     cfg=config();lookup,assign=load_inputs();assert args.arm in cfg['arms'] and 0<=args.fold<5 and args.seed_index in cfg['seeds']
-    spec=cfg['arms'][args.arm];encoders=spec['members'];roles=roles_for(assign,args.fold)
+    spec=cfg['arms'][args.arm];encoders=spec['members'];roles=roles_for(assign,args.fold);eval_limit=None
     if args.smoke:
-        for role in ('train','val'):
-            roles[role]=[p for grade in range(1,5) for p in [x for x in roles[role] if int(lookup[x]['ISUP_grade'])==grade][:2]]
-            assert len(roles[role])==8
-        roles['test']=[];cfg=dict(cfg,max_epochs=args.smoke_epochs,min_epochs=args.smoke_epochs,train_patches=128,eval_chunk=128)
+        if args.smoke_shape=='reduced':
+            for role in ('train','val'):
+                roles[role]=[p for grade in range(1,5) for p in [x for x in roles[role] if int(lookup[x]['ISUP_grade'])==grade][:2]]
+                assert len(roles[role])==8
+            cfg=dict(cfg,max_epochs=args.smoke_epochs,min_epochs=args.smoke_epochs,train_patches=128,eval_chunk=128);eval_limit=256
+        else:  # formal shapes (all fold-0 train/val patients, 2048 train patches, eval chunk 4096, every val patch); epochs only
+            cfg=dict(cfg,max_epochs=args.smoke_epochs,min_epochs=args.smoke_epochs)
+        roles['test']=[]
     else:
-        smoke=json.loads((HERE/'SMOKE_PASS.json').read_text());assert smoke['state']=='PASS' and smoke['code_signature']==signature()
-        ready=json.loads((ROOT/'bib_phase1/mil_first_round_v2_20261015/DATA_READY.json').read_text())
+        refused=formal_gate(args.arm,cfg,torch.device(args.device).type)  # before any CUDA context is created
+        if refused:print('REFUSED',args.arm,'code',refused[0],refused[1],flush=True);return refused[0]
+    if STOP['requested']:print('STOPPED before start',STOP['signals'],flush=True);return 75
+    device=setup_device(args.device);plat=platform(device);ident=gpu_identity(device)
+    if not args.smoke:
+        refused=platform_gate(args.arm,cfg,plat)
+        if refused:print('REFUSED',args.arm,'code',refused[0],refused[1],flush=True);return refused[0]
+        ready=json.loads(DATA_READY.read_text())
         assert ready['state']=='PASS' and ready['patients']==300
         for p in lookup:assert sha(FEATURES/Path(lookup[p]['slide_filename']).stem/'complete.json')==ready['completion_sha256'][p]
     seed=arm_seed(cfg,args.seed_index,args.fold)
     runroot=Path(args.outroot) if args.outroot else HERE/('smoke_runs' if args.smoke else 'runs')
     out=runroot/args.arm.replace(':','_')/f'seed{args.seed_index}'/f'fold{args.fold}';out.mkdir(parents=True,exist_ok=True)
-    lockfile=(out/'run.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
+    lockfile=(out/'run.lock').open('w')
+    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
+    except BlockingIOError:print('LOCKED',out,flush=True);return 79
     fingerprint=dict(code=signature(),config=cfg,arm=args.arm,seed_index=args.seed_index,fold=args.fold,split_sha256=sha(SPLITS/'patient_splits.csv'),pool_sha256=sha(POOL/'LOCK.json'),smoke=args.smoke,patients=roles)
+    if args.smoke:fingerprint['smoke_shape']=args.smoke_shape
     fid=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
     if (out/'complete.json').exists():
         receipt=json.loads((out/'complete.json').read_text());assert receipt['fingerprint']==fid
         for name,digest in receipt['artifacts'].items():assert sha(out/name)==digest
         print('ALREADY_COMPLETE',out,flush=True);return 0
+    oom_adj=raise_oom_score(args.oom_score_adj) if args.oom_score_adj is not None else None
     seed_all(seed,args.threads)
-    model=make_model(args.arm,cfg)
+    model=make_model(args.arm,cfg).to(device)
     optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
     train_labels=np.array([int(lookup[p]['ISUP_grade'])-1 for p in roles['train']]);counts=np.bincount(train_labels,minlength=4)
-    assert (counts>0).all();weights=torch.tensor(len(train_labels)/(4*counts),dtype=torch.float32)
+    assert (counts>0).all();weights=torch.tensor(len(train_labels)/(4*counts),dtype=torch.float32,device=device)
     start_epoch=0;history=[];best_key=float('inf');best_model=None;best_epoch=-1;bad=0;finished_training=False
     if (out/'last.pt').exists():
         state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['fingerprint']==fid
+        if state['platform']!=plat:print('REFUSED','code',78,f'resume refused: run started on {state["platform"]}, this is {plat}',flush=True);return 78
         model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
-        start_epoch=state['next_epoch'];history=state['history'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch'];bad=state['bad'];finished_training=state['finished_training'];restore_rng(state['rng'])
-    init_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.parameters())).hexdigest()
-    store=BagStore(lookup,encoders);t0=time.monotonic();stop=[False]
-    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
-    if not finished_training:store.preload(roles['train']+roles['val'])
-    metadata=dict(fingerprint=fid,contract=fingerprint,class_counts=counts.tolist(),class_weights=weights.tolist(),trainable_parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,numpy=np.__version__,threads=args.threads,job_id=os.environ.get('SLURM_JOB_ID'),seed=seed)
+        start_epoch=state['next_epoch'];history=state['history'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch'];bad=state['bad'];finished_training=state['finished_training'];restore_rng(state['rng'],device)
+    init_hash=param_hash(model)
+    store=BagStore(lookup,encoders,device);t0=time.monotonic()
+    if not finished_training:
+        need=store.need(roles['train']+roles['val']);avail=mem_available()
+        print('MEM_AVAILABLE',avail,'NEED',need,'MARGIN_GIB',args.ram_margin_gib,flush=True)
+        if avail is not None and need+args.ram_margin_gib*1024**3>avail:print('INSUFFICIENT_RAM',out,flush=True);return 76
+        if store.preload(roles['train']+roles['val'],stop=lambda:STOP['requested']) is None:return pause(out,start_epoch,fid,'stop requested during preload')
+    metadata=dict(fingerprint=fid,contract=fingerprint,class_counts=counts.tolist(),class_weights=weights.cpu().tolist(),trainable_parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,numpy=np.__version__,threads=args.threads,job_id=os.environ.get('SLURM_JOB_ID'),seed=seed,
+                  device=str(device),platform=plat,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),host=socket.gethostname(),he_path_prefix=PREFIX,
+                  invocation=dict(started=now(),start_epoch=start_epoch,gpu=ident,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),inherited_cublas_workspace_config=INHERITED_CUBLAS_WORKSPACE_CONFIG,
+                                  ram_margin_gib=args.ram_margin_gib,oom_score_adj=oom_adj,scheduler_job=os.environ.get('HE_SCHED_JOB')))
     save_json(out/'metadata.json',metadata)
-    k=len(encoders);dropping=spec['fusion']=='modality_dropout_mean'
+    k=len(encoders);dropping=spec['fusion']=='modality_dropout_mean';resources=json.loads((out/'resources.json').read_text()) if (out/'resources.json').exists() else {}
     for epoch in range(start_epoch,cfg['max_epochs']):
         if finished_training:break
+        if STOP['requested']:return pause(out,epoch,fid,'stop requested before the epoch started')
+        if epoch==start_epoch:(out/'paused.json').unlink(missing_ok=True)  # training resumed: the pause note is history
+        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
         model.train();optimizer.zero_grad(set_to_none=True);epoch_start=time.monotonic();loss_sum=0.
         order=np.random.default_rng(stable_seed(seed,args.fold,epoch,'order')).permutation(len(roles['train']))
         for i,index in enumerate(order):
             patient=roles['train'][int(index)];n=int(lookup[patient]['primary_count']);y=int(lookup[patient]['ISUP_grade'])-1
             indices=patch_indices(n,cfg['train_patches'],seed,args.fold,epoch,patient)
             mask=branch_mask(k,spec['branch_drop'],seed,args.fold,epoch,patient) if dropping else None
-            logits=model(store.tensors(patient,indices),mask);loss=F.cross_entropy(logits,torch.tensor([y]),reduction='sum')*weights[y]
+            logits=model(store.tensors(patient,indices),mask);loss=F.cross_entropy(logits,torch.tensor([y],device=device),reduction='sum')*weights[y]
             assert bool(torch.isfinite(loss)),'Non-finite training loss'
             group_start=(i//cfg['accumulate_patients'])*cfg['accumulate_patients'];group_size=min(cfg['accumulate_patients'],len(order)-group_start)
             (loss/group_size).backward();loss_sum+=float(loss.detach())
             if (i+1)%cfg['accumulate_patients']==0 or i+1==len(order):
                 nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                 optimizer.step();optimizer.zero_grad(set_to_none=True)
-        records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,args.smoke)
+        records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,eval_limit)
         key=val['cross_entropy'];improved=key<best_key-1e-8
         if improved:best_key=key;best_epoch=epoch;best_model=cpu_state(model);bad=0
         else:bad+=1
         finished_training=epoch+1>=cfg['max_epochs'] or (epoch+1>=cfg['min_epochs'] and bad>=cfg['patience'])
         history.append(dict(epoch=epoch,training_loss=loss_sum/len(order),validation=val,seconds=time.monotonic()-epoch_start))
-        state=dict(fingerprint=fid,model=cpu_state(model),optimizer=optimizer.state_dict(),next_epoch=epoch+1,history=history,best_key=best_key,best_model=best_model,best_epoch=best_epoch,bad=bad,rng=rng_state(),finished_training=finished_training)
+        state=dict(fingerprint=fid,model=cpu_state(model),optimizer=optimizer.state_dict(),next_epoch=epoch+1,history=history,best_key=best_key,best_model=best_model,best_epoch=best_epoch,bad=bad,rng=rng_state(device),finished_training=finished_training,platform=plat)
         save_pt(out/'last.pt',state);save_json(out/'history.json',history)
-        print('EPOCH',args.arm,args.seed_index,args.fold,epoch,'train_loss',round(history[-1]['training_loss'],5),'val_ce',round(val['cross_entropy'],5),'val_f1',round(val['macro_f1'],5),'seconds',round(history[-1]['seconds'],2),flush=True)
-        if not finished_training and (stop[0] or time.monotonic()-t0>args.max_runtime_seconds or (args.stop_after_epochs and epoch+1>=args.stop_after_epochs)):
-            save_json(out/'paused.json',dict(next_epoch=epoch+1,fingerprint=fid));return 75
+        gpu_peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
+        resources[str(epoch)]=dict(seconds=history[-1]['seconds'],gpu_max_allocated_bytes=gpu_peak,gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,max_rss_bytes=max_rss_bytes(),finished=now(),**ident)
+        save_json(out/'resources.json',resources)  # operational log only (incl. which physical card ran the epoch); not an artifact and not hashed
+        print('EPOCH',args.arm,args.seed_index,args.fold,epoch,'train_loss',round(history[-1]['training_loss'],5),'val_ce',round(val['cross_entropy'],5),'val_f1',round(val['macro_f1'],5),'seconds',round(history[-1]['seconds'],2),'device',device,'gpu',ident.get('gpu_uuid'),'gpu_peak',gpu_peak,flush=True)
+        if not finished_training and (STOP['requested'] or time.monotonic()-t0>args.max_runtime_seconds or (args.stop_after_epochs and epoch+1>=args.stop_after_epochs)):
+            return pause(out,epoch+1,fid,'stop requested' if STOP['requested'] else 'max runtime or --stop-after-epochs')
     assert best_model is not None
-    final_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.parameters())).hexdigest()
+    save_json(out/'history.json',history)  # from the checkpoint state: a crash between last.pt and history.json cannot leave it stale
+    final_hash=param_hash(model)
     model.load_state_dict(best_model);save_pt(out/'best.pt',dict(model=best_model,epoch=best_epoch,metadata=metadata))
-    val_records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,args.smoke);save_json(out/'val_predictions.json',val_records)
+    val_records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,eval_limit);save_json(out/'val_predictions.json',val_records)
     test=None;names=['best.pt','last.pt','history.json','metadata.json','val_predictions.json']
     if not args.smoke:
-        store.cache.clear();test_records,test=evaluate(model,store,roles['test'],cfg,'test',args.fold,False);save_json(out/'test_predictions.json',test_records);names.append('test_predictions.json')
-    receipt=dict(state='COMPLETE',smoke=args.smoke,fingerprint=fid,arm=args.arm,seed_index=args.seed_index,fold=args.fold,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,parameters_changed_this_invocation=init_hash!=final_hash,seconds_this_invocation=time.monotonic()-t0,artifacts={n:sha(out/n) for n in names})
-    save_json(out/'complete.json',receipt);print('COMPLETE',args.arm,args.seed_index,args.fold,len(history),flush=True);return 0
+        store.cache.clear();test_records,test=evaluate(model,store,roles['test'],cfg,'test',args.fold,None);save_json(out/'test_predictions.json',test_records);names.append('test_predictions.json')
+    cards=sorted({str(r.get('gpu_uuid')) for r in resources.values() if isinstance(r,dict) and r.get('gpu_uuid')})
+    receipt=dict(state='COMPLETE',smoke=args.smoke,fingerprint=fid,arm=args.arm,seed_index=args.seed_index,fold=args.fold,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,parameters_changed_this_invocation=init_hash!=final_hash,seconds_this_invocation=time.monotonic()-t0,
+                 physical_gpus=cards,cross_card_resume=len(cards)>1,artifacts={n:sha(out/n) for n in names})
+    save_json(out/'complete.json',receipt);(out/'paused.json').unlink(missing_ok=True)
+    print('COMPLETE',args.arm,args.seed_index,args.fold,len(history),flush=True);return 0
 
 def parser():
     p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed-index',type=int,required=True);p.add_argument('--fold',type=int,required=True)
     p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-epochs',type=int,default=2);p.add_argument('--outroot');p.add_argument('--stop-after-epochs',type=int)
-    p.add_argument('--threads',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK','8')));p.add_argument('--max-runtime-seconds',type=float,default=22*3600)
+    p.add_argument('--smoke-shape',choices=['reduced','formal'],default='reduced',help='--smoke only: reduced (8+8 patients, 128 patches) or formal (fold train/val sets and formal patch/chunk sizes)')
+    p.add_argument('--device',default='cuda',help="torch device, e.g. cuda, cuda:1, cpu (formal runs must match config['device'])")
+    p.add_argument('--threads',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK','4')));p.add_argument('--max-runtime-seconds',type=float,default=22*3600)
+    p.add_argument('--ram-margin-gib',type=float,default=128.0,help='refuse to preload (exit 76) unless MemAvailable exceeds the cache by this much (shared box)')
+    p.add_argument('--oom-score-adj',type=int,default=500,help="raise /proc/self/oom_score_adj to this value (Linux; raising needs no privilege), so that on a host OOM this resumable job is killed before other users' processes")
     return p
-if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
+def main(argv=None):
+    install_signal_handlers();STOP['armed']=True
+    args=parser().parse_args(argv)
+    if STOP['requested']:print('STOPPED before start',STOP['signals'],flush=True);return 75
+    try:return run(args)
+    except StopNow as e:print('STOPPED_NOW signal',e.args[0],'(resumable from the last epoch checkpoint)',flush=True);return 75
+    except Exception as e:
+        if not is_transient(e):raise
+        traceback.print_exc();print('TRANSIENT_GPU_FAILURE',repr(e)[:500],'(exit 74; resumable, retry later)',flush=True);return 74
+if __name__=='__main__':raise SystemExit(main())
```

### smoke.py

```diff
--- v1/smoke.py
+++ v2/smoke.py
@@ -1,56 +1,258 @@
-"""Technical gate for fusion_gain_v1 (CPU). Writes SMOKE_PASS.json only if every check passes, then times one real epoch.
-Uses training/validation cases of fold 0 only; never opens test predictions."""
+"""Technical gate for fusion_gain_v2 (GPU port of v1). Uses training/validation cases of fold 0 only; never opens test
+predictions. Parts: 'single' = checks with single:phikon-v2 (covers every single:* arm: same code path);
+                    'mdrop4' = checks with mdrop4 (needs all four encoders verified).
+One attempt of a part runs in smoke_runs/<arm>/attempt_NNN/ (attempts are numbered and never deleted):
+  1. unit checks: toy shapes as in v1, plus every arm at its real input dimensions on the device with synthetic features
+     (same seed -> bitwise identical training; checkpoint/restore mid-run -> bitwise identical; chunked evaluation).
+  2. reduced shape (8+8 patients, 128 patches, 2 epochs): full run vs (1 epoch, exit 75, fresh-process resume on the
+     resume device); 3. formal shape (all fold-0 train/val patients, 2048 train patches, eval chunk 4096, every val patch,
+     2 epochs): the same comparison. Models, optimizer, RNG (incl. CUDA), selection, history and validation predictions
+     must be bitwise identical. TIMING.json is taken from the formal-shape full run.
+  The resume device is --resume-device, else the other visible GPU when two are visible (cross-card proof), else --device.
+SMOKE_PASS.json parts.<part>.state: PASS; FAIL (a check failed: revokes an earlier PASS of the same code, formal runs of
+the part are refused until a later attempt passes); INCOMPLETE (GPU busy/out of memory, not enough RAM, a leg timed out,
+or a stop signal: proves nothing, so it never replaces PASS or FAIL); DEFERRED (auto mode, mdrop4 data not verified yet);
+NOT_RUN. The top-level 'state' only says the file belongs to 'code_signature' (kept for the scheduler); see parts_summary.
+CLI: python smoke.py [--part auto|single|mdrop4] [--device cuda] [--resume-device DEV] [--threads 4] [--leg-timeout-seconds S]
+  auto (default): single, then mdrop4 if its data are verified, otherwise mdrop4 is recorded as DEFERRED.
+  --part mdrop4: run later; adds parts.mdrop4 (and TIMING mdrop4) without touching parts.single, provided the code
+                 signature is unchanged (a changed signature starts a fresh SMOKE_PASS.json).
+Only one smoke.py runs at a time (smoke.run.lock). SIGTERM/SIGUSR1/SIGINT stop the running train.py leg at once (it is
+sent SIGINT), the attempt is recorded INCOMPLETE and smoke.py exits 75.
+Exit: 0 requested parts passed (auto may defer mdrop4); 1 a check failed (part FAIL, SMOKE_FAIL_<part>.json); 2 usage;
+3 data for a requested part not verified; 4 parts passed but TIMING.json could not be written; 5 another smoke.py holds
+smoke.run.lock (nothing done); 74 transient (GPU busy/out of memory, RAM, leg timeout; part INCOMPLETE); 75 stopped by a
+signal (part INCOMPLETE); 78 device is not the configured platform."""
 from common import *
-import shutil,time,torch
+import argparse,subprocess,time,fcntl,datetime,traceback,copy,signal,torch
+from torch import nn
+import torch.nn.functional as F
 import train as T
+import ready as R
 
-def unit_checks():
-    T.seed_all(11,4)
+SMOKE_ARM={'single':'single:phikon-v2','mdrop4':'mdrop4'}
+STATES=('PASS','FAIL','INCOMPLETE','DEFERRED','NOT_RUN')
+TRANSIENT_CODES={74,76,79,-9}  # train.py: GPU busy/OOM, RAM, run dir locked; -9: SIGKILL (host OOM killer). Says nothing about the code.
+STATE_NOTE='Top-level state only says this file belongs to code_signature and root; formal runs are gated per part (parts.<part>.state must be PASS).'
+def covers(part,cfg):return [a for a in cfg['arms'] if T.smoke_part(a,cfg)==part]
+def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
+
+class Interrupted(Exception):
+    """A stop signal reached smoke.py: the attempt proves nothing either way."""
+class Transient(Exception):
+    """GPU busy or out of memory, not enough RAM, or a leg timed out."""
+SIG=dict(child=None,signals=[])
+def on_signal(signum,frame):
+    SIG['signals'].append(signal.Signals(signum).name);child=SIG['child']
+    if child is not None:
+        try:child.send_signal(signal.SIGINT)  # train.py: SIGINT = stop at once, exit 75 (resumable)
+        except OSError:pass
+
+def unit_checks(device,threads,cfg):
+    T.seed_all(11,threads)
     for model in (T.GatedMIL([7,11],16,8,0),T.ModalityDropoutFusion([7,11,5],16,8,0,16)):
-        model.eval();k=len(model.branches) if hasattr(model,'branches') else 2
-        dims=[7,11] if k==2 else [7,11,5];x=[torch.randn(19,d) for d in dims]
+        model.to(device).eval();k=len(model.branches) if hasattr(model,'branches') else 2
+        dims=[7,11] if k==2 else [7,11,5];x=[torch.randn(19,d).to(device) for d in dims]
         a=model(x);b=model.predict_chunks([[t[:5] for t in x],[t[5:12] for t in x],[t[12:] for t in x]])
         assert torch.allclose(a,b,atol=1e-6,rtol=1e-5),'chunk equivalence'
-        ix=torch.randperm(19);assert torch.allclose(a,model([t[ix] for t in x]),atol=1e-6,rtol=1e-5),'permutation invariance'
-        model.train();torch.nn.functional.cross_entropy(model(x),torch.tensor([2])).backward()
+        ix=torch.randperm(19).to(device);assert torch.allclose(a,model([t[ix] for t in x]),atol=1e-6,rtol=1e-5),'permutation invariance'
+        model.train();F.cross_entropy(model(x),torch.tensor([2],device=device)).backward()
         assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'gradients'
-    m=T.ModalityDropoutFusion([7,11,5],16,8,0,16);x=[torch.randn(9,d) for d in (7,11,5)]
-    torch.nn.functional.cross_entropy(m(x,[True,False,True]),torch.tensor([1])).backward()
+    m=T.ModalityDropoutFusion([7,11,5],16,8,0,16).to(device);x=[torch.randn(9,d).to(device) for d in (7,11,5)]
+    F.cross_entropy(m(x,[True,False,True]),torch.tensor([1],device=device)).backward()
     assert m.branches[1][0].weight.grad is None and m.branches[0][0].weight.grad.abs().sum()>0,'dropped branch gets no gradient'
     masks=[T.branch_mask(4,.25,1,0,e,f'p{i}') for e in range(20) for i in range(50)]
     assert all(any(mk) for mk in masks) and T.branch_mask(4,.25,1,0,3,'p1')==T.branch_mask(4,.25,1,0,3,'p1')
     kept=np.mean([sum(mk) for mk in masks]);assert 2.7<kept<3.3,kept
-    return dict(chunk_equivalence=True,permutation_invariance=True,gradients=True,dropped_branch_no_grad=True,mask_deterministic_nonempty=True,mean_branches_kept=float(kept))
+    real=real_dim_checks(device,threads,cfg)
+    return dict(chunk_equivalence=True,permutation_invariance=True,gradients=True,dropped_branch_no_grad=True,mask_deterministic_nonempty=True,mean_branches_kept=float(kept),real_dimensions=real,device=str(device))
 
+def real_dim_checks(device,threads,cfg):
+    """Every configured arm (mdrop4 included, whatever part runs) at its real input dimensions on `device`, with synthetic
+    features: the same seed gives bitwise identical parameters, a checkpoint/restore after step 2 gives bitwise identical
+    parameters, chunked evaluation (eval_chunk) is repeatable and matches one-pass evaluation."""
+    out={};chunk=cfg['eval_chunk']
+    for arm,spec in cfg['arms'].items():
+        dims=[DIMS[m] for m in spec['members']];dropping=spec['fusion']=='modality_dropout_mean'
+        g=torch.Generator().manual_seed(20261021)
+        bags=[[torch.randn(cfg['train_patches'],d,generator=g).half() for d in dims] for _ in range(3)]
+        big=[torch.randn(2*chunk+808,d,generator=g).half() for d in dims]
+        def train(steps,resume=None):
+            T.seed_all(7,threads);model=T.make_model(arm,cfg).to(device);first=0;snap=None
+            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
+            if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);T.restore_rng(resume['rng'],device);first=resume['step']
+            for step in range(first,steps):
+                model.train();x=[b.to(device).float() for b in bags[step%3]]
+                mask=T.branch_mask(len(dims),spec['branch_drop'],7,0,step,'synthetic') if dropping else None
+                loss=F.cross_entropy(model(x,mask),torch.tensor([step%4],device=device),reduction='sum')
+                assert bool(torch.isfinite(loss)),f'{arm}: non-finite loss'
+                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
+                if step+1==2:snap=dict(model=T.cpu_state(model),optimizer=copy.deepcopy(opt.state_dict()),rng=T.rng_state(device),step=2)
+            return model,snap
+        m1,snap=train(4);m2,_=train(4);m3,_=train(4,snap)
+        h=[T.param_hash(m) for m in (m1,m2,m3)]
+        assert h[0]==h[1],f'{arm}: same seed gave different parameters at real dimensions (non-deterministic kernels)'
+        assert h[0]==h[2],f'{arm}: checkpoint/restore after step 2 changed the parameters at real dimensions'
+        m1.eval();xb=[t.to(device).float() for t in big];n=len(big[0])
+        with torch.no_grad():one=m1(xb)
+        c1=m1.predict_chunks([[t[i:i+chunk] for t in xb] for i in range(0,n,chunk)]);c2=m1.predict_chunks([[t[i:i+chunk] for t in xb] for i in range(0,n,chunk)])
+        assert torch.equal(c1,c2),f'{arm}: chunked evaluation is not repeatable'
+        assert torch.allclose(one,c1,atol=1e-5,rtol=1e-4),f'{arm}: chunked vs one-pass evaluation differ'
+        out[arm]=dict(dims=dims,train_patches=cfg['train_patches'],eval_patches=n,steps=4,param_sha256=h[0],max_abs_chunk_vs_onepass=float((one-c1).abs().max()))
+        del m1,m2,m3,snap,xb
+    if device.type=='cuda':torch.cuda.empty_cache()
+    return out
+
 def params(path):return torch.load(path,map_location='cpu',weights_only=False)
-def same(a,b):return all(torch.equal(a[k],b[k]) for k in a)
+def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
+def equal(a,b):
+    """Recursive bitwise equality for checkpoint contents (tensors, numpy arrays, dicts, lists, tuples, scalars)."""
+    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
+    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
+    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
+    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
+    return a==b
+def strip(history):return [{k:v for k,v in h.items() if k!='seconds'} for h in history]
 
-def main():
-    report=dict(code_signature=signature(),unit=unit_checks(),runs={})
-    p=T.parser()
-    for arm in ('single:phikon-v2','mdrop4'):
-        root=HERE/'smoke_runs'/arm.replace(':','_')
-        if root.exists():shutil.rmtree(root)
-        full=p.parse_args(['--arm',arm,'--seed-index','0','--fold','0','--smoke','--outroot',str(root/'full')])
-        assert T.run(full)==0
-        part=p.parse_args(['--arm',arm,'--seed-index','0','--fold','0','--smoke','--outroot',str(root/'resume'),'--stop-after-epochs','1'])
-        assert T.run(part)==75
-        part.stop_after_epochs=None;assert T.run(part)==0
-        a=root/'full'/arm.replace(':','_')/'seed0/fold0';b=root/'resume'/arm.replace(':','_')/'seed0/fold0'
-        assert same(params(a/'best.pt')['model'],params(b/'best.pt')['model']),'resume changed best model'
-        assert same(params(a/'last.pt')['model'],params(b/'last.pt')['model']),'resume changed last model'
-        assert json.loads((a/'val_predictions.json').read_text())==json.loads((b/'val_predictions.json').read_text())
-        report['runs'][arm]=dict(resume_bitwise_identical=True)
-    report['state']='PASS';save_json(HERE/'SMOKE_PASS.json',report);print('SMOKE_PASS',flush=True)
-    timing={}
-    for arm in ('single:phikon-v2','single:uni2-h','mdrop4'):
-        root=HERE/'timing_runs'
-        if (root/arm.replace(':','_')).exists():shutil.rmtree(root/arm.replace(':','_'))
-        t=time.monotonic();args=p.parse_args(['--arm',arm,'--seed-index','0','--fold','0','--outroot',str(root),'--stop-after-epochs','2'])
-        assert T.run(args)==75
-        h=json.loads((root/arm.replace(':','_')/'seed0/fold0/history.json').read_text())
-        epoch=float(np.mean([e['seconds'] for e in h]));total=time.monotonic()-t
-        timing[arm]=dict(seconds_per_epoch=epoch,preload_and_overhead_seconds=total-sum(e['seconds'] for e in h))
-        print('TIMING',arm,json.dumps(timing[arm]),flush=True)
-    save_json(HERE/'TIMING.json',timing);print('TIMING_DONE',flush=True)
-if __name__=='__main__':main()
+def train_cmd(arm,outroot,device,threads,extra=()):
+    return [sys.executable,'-u',str(HERE/'train.py'),'--arm',arm,'--seed-index','0','--fold','0','--device',device,'--threads',str(threads),'--outroot',str(outroot),*extra]
+def call(cmd,expect,timeout):
+    """Run one train.py leg as a child; signals to smoke.py are forwarded to it (on_signal)."""
+    if SIG['signals']:raise Interrupted(f'stop signal {SIG["signals"]} before {" ".join(cmd[3:])}')
+    print('RUN',' '.join(cmd),flush=True);child=subprocess.Popen(cmd);SIG['child']=child
+    try:
+        if SIG['signals']:child.send_signal(signal.SIGINT)  # a signal arrived while the child was being created
+        try:code=child.wait(timeout=timeout)
+        except subprocess.TimeoutExpired:
+            child.send_signal(signal.SIGINT)
+            try:child.wait(timeout=300)
+            except subprocess.TimeoutExpired:child.kill();child.wait()
+            raise Transient(f'leg timed out after {timeout:.0f}s: {" ".join(cmd[3:])}')
+    finally:SIG['child']=None
+    if SIG['signals']:raise Interrupted(f'stop signal {SIG["signals"]} during {" ".join(cmd[3:])} (child exit {code})')
+    if code in TRANSIENT_CODES:raise Transient(f'{" ".join(cmd[3:])} returned {code} (GPU busy/OOM, RAM, lock or OOM kill)')
+    assert code==expect,f'{" ".join(cmd[3:])} returned {code}, expected {expect}'
+
+def leg(root,arm):return root/arm.replace(':','_')/'seed0'/'fold0'
+def compare(a,b,device,what):
+    """Bitwise comparison of an uninterrupted 2-epoch run (a) with a paused (1 epoch) and resumed run (b)."""
+    A=params(a/'last.pt');B=params(b/'last.pt')
+    assert same(params(a/'best.pt')['model'],params(b/'best.pt')['model']),f'{what}: resume changed best model'
+    assert same(A['model'],B['model']),f'{what}: resume changed last model'
+    assert equal(A['optimizer'],B['optimizer']),f'{what}: resume changed optimizer state'
+    assert equal(A['rng'],B['rng']),f'{what}: resume changed RNG state (incl. CUDA generator)'
+    assert A['next_epoch']==B['next_epoch']==2 and A['best_epoch']==B['best_epoch'] and A['best_key']==B['best_key'],f'{what}: resume changed selection'
+    assert strip(A['history'])==strip(B['history']),f'{what}: resume changed training/validation history'
+    assert json.loads((a/'val_predictions.json').read_text())==json.loads((b/'val_predictions.json').read_text()),f'{what}: resume changed predictions'
+    receipts={k:json.loads((r/'complete.json').read_text()) for k,r in (('full',a),('resume',b))}
+    assert all(x['state']=='COMPLETE' and x['smoke'] for x in receipts.values()) and receipts['full']['parameters_changed_this_invocation'],f'{what}: receipts'
+    assert not (a/'test_predictions.json').exists() and not (b/'test_predictions.json').exists(),'smoke must not touch test patients'
+    if device.type=='cuda':assert 'cuda' in A['rng'] and 'cuda' in B['rng'],'CUDA RNG state missing from checkpoint'
+    return {k:dict(fingerprint=v['fingerprint'],best_epoch=v['best_epoch'],epochs=v['epochs']) for k,v in receipts.items()}
+def epoch_cards(run):
+    res=json.loads((run/'resources.json').read_text());return [res[k].get('gpu_uuid') for k in sorted(res,key=int)]
+
+def real_part(part,args,device,resume_device,adir):
+    """Reduced-shape and formal-shape legs; returns (check record, timing record)."""
+    arm=SMOKE_ARM[part];to=args.leg_timeout_seconds
+    def legs(tag,extra):
+        full=adir/f'{tag}_full';res=adir/f'{tag}_resume'
+        t=time.monotonic();call(train_cmd(arm,full,args.device,args.threads,extra),0,to);wall=time.monotonic()-t
+        call(train_cmd(arm,res,args.device,args.threads,[*extra,'--stop-after-epochs','1']),75,to)
+        call(train_cmd(arm,res,resume_device,args.threads,extra),0,to)
+        return leg(full,arm),leg(res,arm),wall
+    a,b,_=legs('reduced',['--smoke']);reduced=compare(a,b,device,'reduced shape')
+    fa,fb,wall=legs('formal',['--smoke','--smoke-shape','formal']);formal=compare(fa,fb,device,'formal shape')
+    cards=epoch_cards(fb)  # [card of epoch 0 (first process), card of epoch 1 (resumed process)]
+    checks=dict(resume_bitwise_identical=True,formal_shape_bitwise_identical=True,
+                checked=['unit_toy','unit_real_dimensions_all_arms','best_model','last_model','optimizer','rng','selection','history','val_predictions','no_test_predictions','reduced_and_formal_shape'],
+                resume_cards=cards,cross_card_resume=len(cards)==2 and None not in cards and cards[0]!=cards[1],receipts=dict(reduced=reduced,formal=formal))
+    return checks,timing_from(fa,wall,args,device)
+def timing_from(run,wall,args,device):
+    h=json.loads((run/'history.json').read_text());res=json.loads((run/'resources.json').read_text())
+    peak=[r['gpu_max_allocated_bytes'] for r in res.values() if r['gpu_max_allocated_bytes'] is not None]
+    return dict(seconds_per_epoch=float(np.mean([e['seconds'] for e in h])),preload_and_overhead_seconds=wall-sum(e['seconds'] for e in h),gpu_max_allocated_bytes=max(peak) if peak else None,
+                gpu_max_reserved_bytes=max([r['gpu_max_reserved_bytes'] or 0 for r in res.values()]) or None,max_rss_bytes=max(r['max_rss_bytes'] for r in res.values()),
+                device=args.device,threads=args.threads,platform=T.platform(device),source='formal-shape full leg of the smoke (fold 0 train/val, 2 epochs; overhead includes preload and final validation)',finished=now())
+
+def locked_update(name,mutate):
+    """Read-modify-write of HERE/<name> under an exclusive lock."""
+    with (HERE/(name+'.lock')).open('w') as lk:
+        fcntl.flock(lk,fcntl.LOCK_EX);path=HERE/name
+        new=mutate(json.loads(path.read_text()) if path.exists() else None);save_json(path,new);return new
+def record_part(part,result,cfg):
+    """PASS and FAIL always replace the part's record (FAIL revokes a PASS); INCOMPLETE and DEFERRED never replace PASS or FAIL."""
+    sig=signature()
+    def mutate(cur):
+        if cur is None or cur.get('code_signature')!=sig or cur.get('root')!=str(ROOT):
+            cur=dict(state='PASS',state_note=STATE_NOTE,version=cfg['version'],code_signature=sig,root=str(ROOT),parts={p:dict(state='NOT_RUN',covers=covers(p,cfg)) for p in SMOKE_ARM},attempts=[])
+        old=cur['parts'][part]['state'];applied=result['state'] in ('PASS','FAIL') or old in ('NOT_RUN','DEFERRED','INCOMPLETE')
+        if applied:cur['parts'][part]=result
+        if 'attempt' in result:cur.setdefault('attempts',[]).append(dict(part=part,attempt=result['attempt'],state=result['state'],applied=applied,error=result.get('error'),evidence=result.get('evidence'),finished=result.get('finished')))
+        cur['parts_summary']={p:v['state'] for p,v in cur['parts'].items()};cur['state_note']=STATE_NOTE;cur['updated']=now()
+        assert cur['state']=='PASS' and all(v['state'] in STATES for v in cur['parts'].values())
+        return cur
+    new=locked_update('SMOKE_PASS.json',mutate)
+    if result['state']=='PASS':(HERE/f'SMOKE_FAIL_{part}.json').unlink(missing_ok=True)  # the failure stays in its attempt directory
+    return new
+def new_attempt(arm):
+    base=HERE/'smoke_runs'/arm.replace(':','_');base.mkdir(parents=True,exist_ok=True)
+    n=1+max([int(p.name[8:]) for p in base.glob('attempt_*') if p.name[8:].isdigit()],default=0)
+    d=base/f'attempt_{n:03d}';d.mkdir();return n,d
+def other_card(device):
+    if device.type!='cuda' or torch.cuda.device_count()<2:return None
+    return f'cuda:{1 if device.index==0 else 0}'
+
+def main(argv=None):
+    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument('--part',choices=['auto','single','mdrop4'],default='auto');p.add_argument('--device',default='cuda')
+    p.add_argument('--resume-device',help='device for the resume legs (default: the other visible GPU if two are visible, else --device)')
+    p.add_argument('--threads',type=int,default=4);p.add_argument('--leg-timeout-seconds',type=float,default=4*3600,help='per train.py leg; a timeout makes the attempt INCOMPLETE (exit 74)')
+    args=p.parse_args(argv)
+    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,on_signal)
+    lock=(HERE/'smoke.run.lock').open('w')
+    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
+    except BlockingIOError:print('SMOKE_LOCKED another smoke.py holds',HERE/'smoke.run.lock','- nothing done',flush=True);return 5
+    cfg=config()
+    if torch.device(args.device).type!=cfg['device']:print('REFUSED smoke on',args.device,'but the configured platform is',cfg['device'],flush=True);return 78
+    try:device=T.setup_device(args.device)
+    except Exception as e:
+        if not T.is_transient(e):raise
+        print('TRANSIENT_GPU_FAILURE',repr(e),flush=True);return 74
+    resume_device=args.resume_device or other_card(device) or args.device
+    parts=['single','mdrop4'] if args.part=='auto' else [args.part];timings={}
+    for part in parts:
+        if SIG['signals']:print('SMOKE_STOPPED by',SIG['signals'],'before part',part,flush=True);return 75
+        arm=SMOKE_ARM[part];status=R.arm_status(arm,cfg)
+        if not status['ready']:
+            if args.part=='auto' and part=='mdrop4':
+                record_part(part,dict(state='DEFERRED',covers=covers(part,cfg),reason='data not verified: '+json.dumps(status['missing'][:10]),recorded=now()),cfg)
+                timings[arm]='deferred';print('SMOKE_PART_DEFERRED',part,flush=True);continue
+            print('NOT_READY',part,json.dumps(status['missing'][:10]),flush=True);return 3
+        n,adir=new_attempt(arm);print('SMOKE_ATTEMPT',part,n,adir,'device',args.device,'resume_device',resume_device,flush=True)
+        base=dict(part=part,smoke_arm=arm,covers=covers(part,cfg),attempt=n,evidence=str(adir),device=args.device,resume_device=resume_device,platform=T.platform(device),
+                  gpu=T.gpu_identity(device),root=str(ROOT),he_path_prefix=PREFIX,code_signature=signature(),started=now())
+        try:
+            unit=unit_checks(device,args.threads,cfg);checks,timing=real_part(part,args,device,resume_device,adir)
+        except Exception as e:
+            kind='interrupted' if isinstance(e,Interrupted) else 'transient' if isinstance(e,Transient) or T.is_transient(e) else 'check_failed'
+            rec=dict(base,state='FAIL' if kind=='check_failed' else 'INCOMPLETE',kind=kind,error=repr(e),traceback=traceback.format_exc(),signals=SIG['signals'],finished=now())
+            save_json(adir/'attempt.json',rec)
+            if kind=='check_failed':save_json(HERE/f'SMOKE_FAIL_{part}.json',rec)
+            cur=record_part(part,rec,cfg)
+            print('SMOKE_PART_FAILED' if kind=='check_failed' else 'SMOKE_PART_INCOMPLETE',part,kind,repr(e),'-> parts.'+part,cur['parts'][part]['state'],flush=True)
+            return {'check_failed':1,'transient':74,'interrupted':75}[kind]
+        result=dict(base,state='PASS',**checks,unit=unit,data_status=R.arm_status(arm,cfg)['encoders'],finished=now())
+        save_json(adir/'attempt.json',result);record_part(part,result,cfg);timings[arm]=timing
+        print('SMOKE_PASS',part,'attempt',n,'cross_card_resume',checks['cross_card_resume'],flush=True);print('TIMING',arm,json.dumps(timing),flush=True)
+    if not timings:return 0
+    try:
+        def mutate(cur):
+            cur=cur if cur and cur.get('code_signature')==signature() else dict(code_signature=signature())
+            for arm,v in timings.items():
+                if v!='deferred' or not isinstance(cur.get(arm),dict):cur[arm]=v
+            return cur
+        locked_update('TIMING.json',mutate);print('TIMING_DONE',flush=True)
+    except Exception as e:print('TIMING_WRITE_FAILED',repr(e),flush=True);return 4
+    return 0
+if __name__=='__main__':raise SystemExit(main())
```

### evaluate.py

```diff
--- v1/evaluate.py
+++ v2/evaluate.py
@@ -1,24 +1,55 @@
-"""Pre-registered analysis for fusion_gain_v1: pooled OOF over 300 patients, paired patient bootstrap."""
+"""Pre-registered analysis for fusion_gain_v2: pooled OOF over 300 patients, paired patient bootstrap.
+Campaign check (it refuses to pool otherwise): each of the 100 runs (5 arms x 4 seeds x 5 folds) needs a COMPLETE, non-smoke
+receipt whose artifacts (incl. metadata.json and test_predictions.json) hash-verify; its hashed metadata.json must carry
+this code signature (contract.code == signature()), contract.smoke False, the run's own arm/seed/fold and the receipt
+fingerprint; all 100 runs must share one platform, one split hash and one QC-pool hash; and SMOKE_PASS.json must still
+record PASS, for this code signature and that same platform, for every smoke part covering an arm (a later FAIL revokes it).
+Runs resumed on the other card (receipt cross_card_resume) are listed in RESULTS.json campaign, with a WARNING if no smoke
+part proved cross-card bitwise resume."""
 from common import *
-def oof(arm,s):
+def smoke_part(arm,cfg):return 'single' if cfg['arms'][arm]['fusion']=='none' else arm  # same rule as train.smoke_part (no torch import here)
+def oof(arm,s,campaign):
     recs=[]
     for k in range(5):
         root=HERE/'runs'/arm.replace(':','_')/f'seed{s}'/f'fold{k}';done=json.loads((root/'complete.json').read_text())
-        for name,digest in done['artifacts'].items():assert sha(root/name)==digest
+        assert done.get('state')=='COMPLETE' and done.get('smoke') is False,f'{root}: receipt is not a COMPLETE formal run'
+        assert {'metadata.json','test_predictions.json'}<=set(done['artifacts']),f'{root}: receipt does not hash metadata.json and test_predictions.json'
+        for name,digest in done['artifacts'].items():assert sha(root/name)==digest,f'{root/name}: sha256 differs from the receipt'
+        md=json.loads((root/'metadata.json').read_text());c=md['contract']
+        assert md['fingerprint']==done['fingerprint'],f'{root}: metadata fingerprint differs from the receipt'
+        assert c['code']==signature(),f'{root}: trained with code {c["code"][:12]}, this code is {signature()[:12]} (not one campaign)'
+        assert c['smoke'] is False and (c['arm'],c['seed_index'],c['fold'])==(arm,s,k),f'{root}: contract is not the formal run {arm}/seed{s}/fold{k}'
+        for key,value in (('platform',md['platform']),('split_sha256',c['split_sha256']),('pool_sha256',c['pool_sha256'])):campaign[key].add(json.dumps(value,sort_keys=True))
+        campaign['physical_gpus'].update(done.get('physical_gpus') or [])
+        if done.get('cross_card_resume'):campaign['cross_card_resumes'].append(f'{arm}|seed{s}|fold{k}')
         recs+=json.loads((root/'test_predictions.json').read_text())
     recs.sort(key=lambda r:r['patient_id']);assert len(recs)==300 and len({r['patient_id'] for r in recs})==300
     return [r['patient_id'] for r in recs],np.array([r['ISUP_grade']-1 for r in recs]),np.array([r['probabilities'] for r in recs])
+def check_campaign(cfg,campaign):
+    for key in ('platform','split_sha256','pool_sha256'):assert len(campaign[key])==1,f'runs disagree on {key}: {sorted(campaign[key])} (not one campaign)'
+    platform=json.loads(next(iter(campaign['platform'])));sig=signature()
+    sp=json.loads((HERE/'SMOKE_PASS.json').read_text());assert sp.get('code_signature')==sig,'SMOKE_PASS.json belongs to other code'
+    smoke={}
+    for part in sorted({smoke_part(a,cfg) for a in cfg['arms']}):
+        rec=sp['parts'].get(part,{})
+        assert rec.get('state')=='PASS' and rec.get('code_signature')==sig,f'smoke part {part} is {rec.get("state")} for this code (a FAIL after the runs revokes them)'
+        assert rec.get('platform')==platform,f'smoke part {part} ran on {rec.get("platform")}, the runs on {platform}'
+        smoke[part]=dict(attempt=rec.get('attempt'),cross_card_resume=rec.get('cross_card_resume'),finished=rec.get('finished'))
+    return dict(runs=len(cfg['arms'])*len(cfg['seeds'])*5,code_signature=sig,platform=platform,split_sha256=json.loads(next(iter(campaign['split_sha256']))),
+                pool_sha256=json.loads(next(iter(campaign['pool_sha256']))),smoke_parts=smoke,physical_gpus=sorted(campaign['physical_gpus']),cross_card_resumes=campaign['cross_card_resumes'],
+                cross_card_resume_proven_by_smoke=all(v['cross_card_resume'] is True for v in smoke.values()))
 def auc(score,pos):
     p=score[pos];n=score[~pos];return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
 def scores(y,P):
     m=metrics(y,P);return dict(auc_high_low=auc(P[:,2:].sum(1),y>=2),qwk=m['quadratic_kappa'],macro_f1=m['macro_f1'],accuracy=m['accuracy'],confusion=m['confusion_matrix'])
 def main():
-    cfg=config();ids=None;P={};Y=None
+    cfg=config();ids=None;P={};Y=None;campaign=dict(platform=set(),split_sha256=set(),pool_sha256=set(),physical_gpus=set(),cross_card_resumes=[])
     for arm in cfg['arms']:
         for s in cfg['seeds']:
-            i,y,p=oof(arm,s)
+            i,y,p=oof(arm,s,campaign)
             if ids is None:ids,Y=i,y
             assert i==ids and (y==Y).all();P[(arm,s)]=p
+    camp=check_campaign(cfg,campaign)
     ref='single:'+cfg['reference_single'];singles=['single:'+m for m in cfg['members']]
     S=cfg['seeds'];avg=lambda keys:np.mean([P[k] for k in keys],0)
     sets={'ref_seed0':avg([(ref,0)]),'ref_4seeds':avg([(ref,s) for s in S]),
@@ -40,6 +71,10 @@
     out['preregistered']={'P1_encoder_vs_seed_diversity':boot('late4_seed0','ref_4seeds'),
                           'P2_full_late_fusion':boot('late4x4seeds','ref_4seeds'),
                           'P3_learned_fusion':boot('mdrop4_4seeds','ref_4seeds')}
+    out['campaign']=camp
     out['note']='Exploratory: Korean OOF folds were inspected in earlier rounds. Confirm on Scotland with the same locked comparisons.'
     save_json(HERE/'RESULTS.json',out);print(json.dumps(out['preregistered'],indent=1));print(json.dumps({k:{m:round(v[m],4) for m in ('auc_high_low','qwk','macro_f1','accuracy')} for k,v in out['per_set'].items()},indent=1))
+    print('CAMPAIGN',json.dumps({k:camp[k] for k in ('runs','code_signature','physical_gpus','cross_card_resumes','cross_card_resume_proven_by_smoke')}),flush=True)
+    if camp['cross_card_resumes'] and not camp['cross_card_resume_proven_by_smoke']:
+        print('WARNING',len(camp['cross_card_resumes']),'run(s) resumed on the other card, but no smoke part proved cross-card bitwise resume (run_smoke.sh 0,1):',camp['cross_card_resumes'],flush=True)
 if __name__=='__main__':main()
```

### PROTOCOL.md

```diff
--- v1/PROTOCOL.md
+++ v2/PROTOCOL.md
@@ -1,5 +1,13 @@
-# 融合增益试验 fusion_gain_v1_20261021（待批准）
+# 融合增益试验 fusion_gain_v2_20261021（待批准）
 
+> **v2 是 v1 的 GPU 移植版。** v1（fusion_gain_v1_20261021，CPU 版）从未在任何地方运行过。v2 的科学设计（成员、臂、种子、划分、抽样、模型头、选模规则、轮数、锁定的指标和比较）与 v1 完全相同，只改了运行平台（lab-a6000 的 RTX A6000 GPU）和数据路径前缀；逐项改动见 `CHANGES.md`。下文"为什么在 lxh_hd4 的 CPU 上跑"和"作业与耗时"两节是 v1 原文，保留不改，仅作历史记录；v2 的执行方式见文末"v2 在 A6000 上的执行"。
+
+## Disclosures（披露）
+
+- **(a) 成员选择用到了 OOF 测试集患者的标签。** 4 个成员是按第一轮各编码器的平均**验证集** Macro-F1 选的。五折里共有 **183/300** 名患者至少在一折中是验证病例；而每名患者都恰好在一折中是测试病例，所以这 183 名患者同时属于合并 OOF 测试集。也就是说，合并 OOF 测试集中这些患者的标签影响过成员选择。config.json 里 member_rule 写的"test folds not used for selection"只在"没有直接用测试折选"的意义上成立。
+- **(b) 这至少是在同一份韩国 OOF 上第四次尝试融合**（9 编码器拼接、概率子集搜索、固定 3 个编码器融合、本轮）。多次尝试本身会抬高最好结果的期望，因此本轮结果只能算**探索性**的，必须在苏格兰数据上用同样锁定的比较确认后才能作为结论。
+- **(c) 平台。** v2 在 lab-a6000 的 RTX A6000 GPU 上训练（torch 2.6.0+cu124，确定性算法、关闭 TF32）；特征是在 HPC 的 A800 上提取的，原样拷到 A6000，并逐个按 complete.json 中的 sha256 校验。**所有臂（4 个单模型和 mdrop4）都在同一平台上训练：** 正式训练要求设备类型等于 config 中的 `device`（cuda），并要求覆盖该臂的 smoke 在同一平台（GPU 型号、torch/CUDA/cuDNN 版本）上通过；断点续跑也只允许在同一平台上进行。GPU 与 CPU 的浮点结果不会逐位一致，v2 的数值不能与任何 CPU 运行混用。
+
 ## 为什么在 lxh_hd4 的 CPU 上跑
 
 - **A6000 不可行：** 数据都在 lxh_hd4 上，实测 lxh_hd4 → 本机约 0.43 MB/秒，本机 → A6000 约 0.15 MB/秒，传 60 GB 特征要好几天。
@@ -45,3 +53,29 @@
 | 技术验收 `run_smoke.slurm` | 1 个 CPU 作业：16 核、160 GB、时限 8 小时 | 单元检查；单模型和 mdrop4 的断点续跑必须逐位一致；再实测 3 种臂的每轮耗时。约 1—2 小时 |
 | 正式 `run_arm.slurm <臂> <种子>` | 20 个 CPU 作业（5 臂 × 4 种子），各 16 核、160 GB、时限 24 小时，验收通过后才启动（afterok） | 按 CPU 速度粗估：单模型每折约 20 分钟、每个作业约 2 小时；mdrop4 每折约 1 小时、每个作业约 5 小时。若 20 个作业能同时开始，**总共约半天**。以验收实测值为准，作业可续跑 |
 | 汇总 `evaluate.py` | 登录节点 | 几分钟 |
+
+## v2 在 A6000 上的执行
+
+lab-a6000 没有 Slurm，两个 `.slurm` 脚本由前台脚本 `run_smoke.sh` / `run_arm.sh` 代替，后台运行、日志和 pid 由调度器负责（调度器按折直接启动 `train.py`，不经过 `run_arm.sh`）。GPU 编号一律是 nvidia-smi 编号（代码强制 `CUDA_DEVICE_ORDER=PCI_BUS_ID`）。
+
+| 步骤 | 命令 | 前提 |
+|---|---|---|
+| 数据就绪 | `python ready.py <臂>` 返回 0（不加载 torch，适合轮询） | 该臂每个成员编码器都有 `transfer/verified_feat_<编码器>`，且 300 例的特征文件都有 `.verified` 标记 |
+| 技术验收（单模型部分） | 推荐 `run_smoke.sh 0,1`（续跑那一段放到另一张卡上，证明跨卡续跑逐位一致）；也可 `run_smoke.sh <GPU>` | Phikon-v2 已校验；见下方"技术验收的内容"；mdrop4 的数据没齐时记为 DEFERRED |
+| 技术验收（mdrop4 部分） | `run_smoke.sh <GPU>[,<GPU>] --part mdrop4` | 4 个编码器都已校验；追加到 SMOKE_PASS.json，不影响单模型部分 |
+| 正式训练 | `run_arm.sh <臂> <种子> <GPU> [线程] [train.py 选项]`（5 折依次跑，可续跑），或调度器按折启动 `train.py` | 该臂数据已校验，且覆盖该臂的 smoke 部分当前为 PASS（单模型臂看 single 部分，mdrop4 看 mdrop4 部分） |
+| 汇总 | `python evaluate.py` | 100 次训练（5 臂 × 4 种子 × 5 折）全部完成，且属于同一批次（见下） |
+
+**技术验收的内容。** 每次验收是 `smoke_runs/<臂>/attempt_NNN/` 下的一次"尝试"，编号递增、从不删除。内容：(1) 单元检查：v1 的玩具尺寸检查，外加 5 个臂各自的真实输入维度（合成特征）在 GPU 上的检查——同一种子两次训练逐位一致、第 2 步存档再恢复后逐位一致、分块评估可重复且与整块评估一致；(2) 缩小尺寸（8+8 例、128 块、2 轮）：一次跑完 vs 跑 1 轮暂停（退出 75）再在新进程里续跑；(3) **正式尺寸**（第 0 折全部训练/验证病例、每例 2048 块、评估分块 4096、验证集全部图块、2 轮）：同样的对比。模型、优化器、随机数状态（含 CUDA）、选模状态、历史和验证预测必须逐位一致。每轮耗时（TIMING.json）取自正式尺寸那次完整运行。
+
+**验收状态按部分记录。** `parts.<部分>.state`：PASS；FAIL（检查失败——**撤销**同一代码下之前的 PASS，覆盖该部分的正式训练和续跑一律拒绝，直到之后某次尝试重新通过）；INCOMPLETE（GPU 忙或显存不足、内存不足、超时或收到停止信号——什么都没证明，所以不会覆盖 PASS 或 FAIL）；DEFERRED；NOT_RUN。顶层的 `state` 只表示文件属于当前代码签名（保留给调度器），看 `parts_summary`。同一时间只能有一个 smoke.py（`smoke.run.lock`，否则退出 5）。
+
+**退出码。** train.py：0 完成；1 完整性/检查失败（不要盲目重试）；74 暂时性 GPU 问题（不可用、被占满、显存不足；可续跑，稍后重试）；75 暂停或被信号停止（可续跑）；76 可用内存不足；77 数据未校验；78 smoke 未通过、平台不符或在另一平台上续跑；79 运行目录被另一进程占用。smoke.py：0 通过；1 检查失败；3 数据未校验；4 已通过但 TIMING.json 写入失败；5 另一个 smoke 正在运行；74 暂时性问题；75 被信号停止；78 平台不符；2 用法错误。
+
+**信号。** train.py：第一次 SIGTERM 或 SIGUSR1 → 在本轮结束时暂停（预加载期间或开始训练前则立即暂停），退出 75；第二次 SIGTERM 或任何 SIGINT → 立即停止，退出 75，从上一轮的 last.pt 续跑，结果逐位一致。想立刻释放资源用 `kill -INT <pid>`（或 `kill -KILL`，同样可续跑，只丢掉当前这一轮）。`run_arm.sh` 把收到的信号（含发给整个进程组的）转给正在跑的 train.py 并等它退出，之后不再启动下一折，以 train.py 的退出码退出。smoke.py 收到信号时立即停止当前子进程，这次尝试记为 INCOMPLETE，退出 75。
+
+**内存。** 每个训练进程把本折训练集和验证集的特征读进自己的内存（Phikon-v2/UNI 约 9.2 GiB，GigaPath/UNI2-h 约 13.9 GiB，mdrop4 约 46 GiB），进程之间不共享数据。预加载前要求 MemAvailable ≥ 缓存 + 128 GiB（`--ram-margin-gib`，给共用这台机器的其他人留余量），否则退出 76，不训练；这个检查是每个进程各自做的，同时启动的进程可能一起通过，调度器需要错开启动并自己做全局内存预算。训练进程把自己的 `oom_score_adj` 调到 500：万一整机内存耗尽，内核先杀我们（可续跑）的进程。同一张卡上可以同时跑多个进程（每个进程显存约 1—1.5 GB）。
+
+**部署。** 所有输出（`runs/`、`smoke_runs/`、`SMOKE_PASS.json`、`SMOKE_FAIL_*.json`、`TIMING.json`、`RESULTS.json`、`*.lock`）都写在代码目录里（调度器按这个位置读）。部署只做一次，`rsync` 不要加 `--delete`，并排除这些输出；smoke 之后把 6 个签名文件（common.py、train.py、smoke.py、evaluate.py、ready.py、config.json）设为只读。改动其中任何一个字节都会使 SMOKE_PASS 失效（78）、使正在跑和已完成的训练都不能续跑或汇总（evaluate.py 只汇总同一代码签名的 100 次训练）。
+
+**一个批次。** evaluate.py 只在以下条件都满足时才汇总：100 份回执都是 COMPLETE 的正式训练，哈希全部核对；每份 metadata.json 的代码签名都等于当前代码、smoke 为 False、臂/种子/折与目录一致；100 次训练的平台、划分哈希、QC 池哈希各自只有一个值；SMOKE_PASS.json 里覆盖各臂的部分在当前代码下仍是 PASS，且平台相同。跨卡续跑过的训练会列在 RESULTS.json 的 `campaign` 里；如果没有任何 smoke 部分证明过跨卡续跑，会打印 WARNING。
```

ready.py, run_arm.sh, run_smoke.sh and tests/test_local.py are new files (no v1 counterpart); run_arm.slurm and run_smoke.slurm are not carried over.
