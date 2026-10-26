# Post hoc analyses (PLAN.md), Phikon-v2

Locked 2026-10-25T22:20:01+08:00; finished 2026-10-25T22:20:21+08:00.

## A. ISUP vs ISUP + AI grade (Harrell C)

- categorical ISUP. Scotland (Korean models frozen): 0.677 -> 0.623, dC -0.054 (-0.223 to +0.112). Korea (cross-fitted): 0.551 -> 0.727, dC +0.176 (+0.074 to +0.281).
- numeric ISUP. Scotland (Korean models frozen): 0.686 -> 0.631, dC -0.055 (-0.223 to +0.106). Korea (cross-fitted): 0.594 -> 0.735, dC +0.141 (+0.046 to +0.236).

## B. AI-grade HR adjusted for ISUP

- korea: categorical ISUP HR 4.27 (2.61-6.98), P = 7.2e-09, LR P = 1.8e-08; numeric ISUP HR 4.20 (2.57-6.87), P = 1.1e-08.
- scotland: categorical ISUP HR 3.83 (1.26-11.66), P = 0.018, LR P = 0.017; numeric ISUP HR 3.26 (1.06-10.06), P = 0.039.

## C. Scottish ISUP 1-2 split at the Korean threshold

- threshold 1.579: above 14/72, at or below 0/14, log-rank P = 0.12
