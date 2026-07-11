# Generic Geometry-Selection Model

This document describes the generic learnable geometry-selection idea used for
the P-ONE reconstruction experiments.

## Motivation

The usual reconstruction workflow trains a separate model for each detector
geometry. That is expensive when the goal is to compare many possible geometry
subsets.

The geometry-selection model instead starts from the full geometry and lets the
network learn which detector strings are most useful for reconstruction. The
base reconstruction model stays close to the normal DynEdge setup, but each
string receives a learnable gate. At the end of training, the gate scores can be
ranked to propose a compact detector geometry.

## High-Level Idea

Each event is represented as a graph:

- nodes are pulses/PMT hits;
- node features include `pmt_x`, `pmt_y`, `pmt_z`, `dom_time`, and `charge`;
- each node also carries a string identifier, currently `string_number`.

For every detector string, the model owns one learnable parameter:

```text
string_logits[string_id]
```

This is converted to a gate score:

```text
gate_score[string_id] = sigmoid(string_logits[string_id])
```

During the forward pass, each node looks up the gate score for its string:

```text
node_gate = gate_score[node.string_number]
```

The node contribution is then weighted by this gate. In the current setup, the
gate is applied to the node charge:

```text
charge_gated = charge * node_gate
```

The other node features remain unchanged. This makes the gate act like a soft
or hard detector-response selector without changing the spatial graph geometry
itself.

## String Budget

The model is encouraged to use only a fixed number of strings. For the current
experiments, the target budget is:

```text
70 active strings
```

The training loss is:

```text
total_loss = reconstruction_loss + budget_penalty
```

where:

```text
budget_penalty = lambda * (sum(gate_scores) - 70)^2
```

This means the model is not only optimizing reconstruction performance; it is
also learning a compact string subset.

## Gate Modes

The script supports two gate modes.

### Soft

In `soft` mode, each string gate is a continuous value between 0 and 1:

```text
gate = sigmoid(logit)
```

This is smooth and stable, but it does not force the forward pass to use exactly
70 strings at every step.

### Straight-Through Top-K

In `straight_through_topk` mode, the forward pass uses a hard top-k selection:

```text
top 70 strings -> gate 1
all others     -> gate 0
```

For backpropagation, gradients still flow through the soft sigmoid gates. This
keeps training differentiable while making the forward pass behave like an
actual 70-string geometry.

The current runs use:

```text
gate_mode: straight_through_topk
active_strings: 70
gate_on: charge
```

## What The Model Produces

Each trained model produces two kinds of results.

First, it produces the normal reconstruction artifacts:

```text
best_model.pth
validation_predictions.csv
validation_metrics_summary.csv
validation_*.png
training_history_by_epoch.csv
resources_and_time.csv
```

Second, it produces the geometry-selection artifact:

```text
learned_string_selection.csv
```

This file contains:

- `string_id`: detector string number;
- `gate_score`: learned score for that string;
- `selected_topk`: whether the string is in the selected top-70 set;
- `rank`: rank by gate score, where rank 1 is the most preferred string.

## Interpreting The Output

A single run gives one learned geometry candidate for one reconstruction
problem. For example:

```text
first_category / class0 / energy
```

That candidate is the top-70 set in:

```text
learned_string_selection.csv
```

The reconstruction quality for that candidate is summarized in:

```text
validation_metrics_summary.csv
```

For energy, useful metrics include:

```text
W_log10
bias_log10
mae_log10
rmse_log10
```

For zenith and azimuth, useful metrics include:

```text
W_deg
residual_deg_p50
kappa_p50
```

## Important Caveat

This method does not automatically produce one universal best geometry from a
single run. It produces one geometry candidate per routing class and target.

If the model is trained for:

```text
class0 energy
class0 zenith
class0 azimuth
class1 energy
class1 zenith
class1 azimuth
```

then there are six learned top-70 string rankings.

To get a single global geometry proposal, those rankings should be aggregated.
Possible aggregation rules include:

- choose strings that appear most often in the six top-70 sets;
- average `gate_score` across runs;
- average rank across runs;
- weight target-specific rankings by physics priority.

## Data Flow

Training does not read raw I3 files directly. It reads existing parquet files
through GraphNeT's `ParquetDataset`.

The current parquet files were produced from the existing
`EventPulseSeries_nonoise` pulsemap. The training script uses the parquet
`features` table.

The raw `PMT_Response_nonoise` pulsemap is not used directly by this training
script. To train on `PMT_Response_nonoise`, new parquet files would first need
to be produced from that pulsemap.

## Implementation

Main training script:

```text
/project/def-nahee/kbas/graphnet/examples/08_pone/train_scripts/train_geometry_selection.py
```

SLURM shell wrapper:

```text
/home/kbas/SlurmScripts/GraphNet/train_geometry_selection.sh
```

Example config:

```text
/project/def-nahee/kbas/graphnet/examples/08_pone/configs/reconstruction/exp003_geometry_selection.yml
```

The implementation is intentionally separate from the normal reconstruction
script so the standard training workflow remains unchanged.
