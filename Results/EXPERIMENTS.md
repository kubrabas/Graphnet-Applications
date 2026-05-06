# Experiment Log

All training and inference experiments for the P-ONE reconstruction pipeline.

Config files live in: `/project/def-nahee/kbas/graphnet/examples/08_pone/configs/`

---

## Classification (cascade vs track)

| exp    | config file             | notes | key result |
|--------|-------------------------|-------|------------|
| exp001 | classification/exp001.yml | baseline | |

---

## Track Reconstruction

| exp    | config file                   | arch     | notes | key result |
|--------|-------------------------------|----------|-------|------------|
| exp001 | track/exp001_separate.yml     | separate | baseline | |

---

## Cascade Reconstruction

| exp    | config file                     | arch     | notes | key result |
|--------|---------------------------------|----------|-------|------------|
| exp001 | cascade/exp001_separate.yml     | separate | baseline | |

---

## Inference (full pipeline)

| exp    | classif model | track model | cascade model | notes | key result |
|--------|---------------|-------------|---------------|-------|------------|
| exp001 | classif/exp001 | track/exp001 | cascade/exp001 | baseline end-to-end | |

---

> **arch** column: `separate` = zenith and azimuth trained as independent networks, `combined` = single network predicting both.
