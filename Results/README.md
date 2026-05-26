# Results Notes

This directory stores training, testing, plotting, and inference artifacts for
the P-ONE GraphNeT experiments.

## Classification Checkpoint Continuation

`classification/340StringMC/102_string_emax1e6/exp002` is the current
classification baseline for track-vs-cascade training. It was trained from
scratch and produced `best_model.pth`.

`classification/340StringMC/102_string_emax1e6/exp003` should be interpreted as
a low-learning-rate continuation / fine-tuning attempt starting from the
`exp002` best checkpoint:

```yaml
pretrained_weights: /project/def-nahee/kbas/Graphnet-Applications/Results/classification/340StringMC/102_string_emax1e6/exp002/best_model.pth
base_lr: 1.0e-6
peak_lr: 1.0e-4
```

The learning rate is intentionally lower than in `exp002` because the model is
not starting from random weights. The goal is to make small updates from the
previous best model rather than re-train aggressively from scratch.

Note: if the train/validation/test split has changed between `exp002` and
`exp003`, then `exp003` is not a strict resume of the same run. It is better
understood as a transfer/fine-tuning-style experiment that initializes from
`exp002` weights and adapts to the new split. This is acceptable, but results
should be compared with that distinction in mind.

If `pretrained_weights` is not set in a training config, the classification
training script starts from scratch. It does not automatically search the output
directory for old checkpoints.
