# Hyperparameter Tuning and Comparison Plan

## 1. Study goals

This study has two connected goals:

1. Select one routing method from the three classification approaches.
2. Compare the 102 string and 160 string geometries using that same routing method.

There are six initial end to end systems:
1. 102 string with `category1_isMuonCC`
2. 102 string with `category2_tauCC_others_muonCC`
3. 102 string with `category_3_contains_muon`
4. 160 string with `category1_isMuonCC`
5. 160 string with `category2_tauCC_others_muonCC`
6. 160 string with `category_3_contains_muon`

Each system contains a classifier and the route specific energy, zenith, and azimuth reconstruction models required by that routing definition.

## 2. Data rules

The canonical event split must remain fixed throughout the study.

1. Each physical event belongs to exactly one of train, validation, or test.
2. The same event remains in the same split for every geometry.
3. Classification uses the full geometry specific train and validation sets.
4. Reconstruction uses truth categorized views of the same canonical train and validation sets.
5. The test set is not truth categorized for inference. The trained classifier routes test events.
6. Hyperparameters, checkpoints, routing thresholds, and model choices must never be selected using test results.

## 3. Baseline comparison

The current baseline models use the same architecture and training hyperparameters for 102 string and 160 string. They provide the controlled comparison:

> How does geometry affect performance when the machine learning recipe is held fixed?

Baseline outputs must be preserved. Hyperparameter trials must be written to separate directories and must not overwrite baseline or final results.

## 4. Optimized comparison

The 102 string and 160 string systems will be tuned in separate Optuna studies. Their best hyperparameter values are allowed to differ because the geometries can respond differently to graph and optimization settings.

The comparison remains fair only if corresponding studies use:

1. The same hyperparameter search space.
2. The same number of trials.
3. The same pruning policy.
4. The same validation objective.
5. The same compute budget and resource class.
6. The same training and evaluation code.

The search space and trial budget must be locked before the first tuning submission. A result observed for one geometry must not be used to give the other geometry a more favorable search range or a larger budget.

This optimized comparison answers:

> What performance can each geometry reach after an equally budgeted optimization procedure?

Both the controlled baseline comparison and the separately optimized comparison will be reported.

## 5. Validation and test responsibilities

### Training set

The training set updates model parameters.

### Validation set

The validation set is used repeatedly for:

1. Early stopping.
2. Saving the best checkpoint.
3. Optuna objective evaluation.
4. Hyperparameter selection.
5. Routing threshold selection.
6. Selecting the final routing method.

### Test set

The test set is opened only after the tuning protocol, hyperparameters, checkpoints, thresholds, and routing method have been fixed. It is used for the final performance report and not for decision making.

## 6. Training duration

The baseline configuration uses:

```yaml
max_epochs: 30
early_stopping_patience: 5
```

Completed training histories must be audited before tuning. Reaching epoch 30 while validation loss is still improving indicates that the epoch ceiling may be too low.

The initial tuning proposal is:

```yaml
max_epochs: 60
early_stopping_patience: 8
```

This does not force every trial to run for 60 epochs. Early stopping and Optuna pruning may terminate unpromising trials earlier. The final values will be locked after the baseline history audit.

## 7. Optuna method

Optuna will use a TPE based search with a reproducible sampler seed.

1. Initial trials explore different parts of the search space.
2. Later trials use earlier validation results to sample promising regions more often.
3. Pruning may stop trials that are clearly underperforming.
4. Every trial records its complete configuration, objective, status, output path, and random seed.
5. Failed infrastructure jobs are recorded as failures and are not interpreted as poor model performance.

A trial is a new model training run. Parallel execution can reduce wall clock time, but it does not remove the total GPU cost of the trials.

## 8. Initial search priorities

The first search should remain small and interpretable. Initial priorities are:

1. Peak learning rate.
2. Number of graph neighbours.
3. Model capacity, only after the first search if needed.
4. Training loss weighting as a controlled ablation, not as an untracked default change.

Effective batch size should initially remain fixed at:

```text
batch_size 256 multiplied by accumulate_grad_batches 4 equals 1024
```

The baseline values must be included as a known trial. The exact numerical search ranges and trial count will be locked after reviewing all available baseline histories.

## 9. Training losses and tuning objectives

Training loss and the final scientific selection metric serve different purposes.

### Classification

The classifier is trained with its classification loss. Optuna initially uses a validation classification objective, with validation loss as the stable primary candidate and AUROC and class specific metrics recorded as diagnostics.

The final router is not selected only from classifier accuracy or AUROC. It is selected from the end to end validation reconstruction produced after routing.

### Reconstruction

Energy, zenith, and azimuth models retain their target appropriate training losses. Validation diagnostics must include:

1. Energy resolution and energy bias.
2. Opening angle resolution and directional bias diagnostics.
3. Performance by flavor and energy bin.

The exact scalar Optuna objective must be fixed before the first study. It must be computed only from validation data and used identically for corresponding 102 string and 160 string studies.

## 10. Event weights

The current baseline training uses equal loss contribution per event because training weights are disabled.

Three concepts must remain separate:

1. `final_weight` represents the target physics or flux distribution and is primarily an evaluation weight.
2. Classification class weights can compensate for class imbalance.
3. Reconstruction training balance weights can reduce domination by densely populated energy regions.

`final_weight` must not be inserted directly into the training loss without a dedicated validation study. Large weight variation can allow a small number of events to dominate optimization.

Any training balance weight must:

1. Be derived from training data only.
2. Be normalized and, if necessary, clipped.
3. Be evaluated as an explicit weighted versus unweighted ablation.
4. Be applied with the same rule across corresponding geometry studies.

Validation and final reports should show both unweighted metrics and `final_weight` weighted physics metrics where meaningful.

## 11. Routing threshold selection

Routing threshold optimization is not a new model training run.

After classifier and reconstruction training:

1. Store classifier scores for validation events.
2. Evaluate candidate routing thresholds.
3. Route each validation event to the appropriate trained reconstructor.
4. Measure end to end energy and directional reconstruction.
5. Select and lock the threshold using the predefined validation objective.
6. Apply the locked threshold unchanged to the test set.

The threshold that maximizes classification accuracy is not necessarily the threshold that gives the best end to end reconstruction.

## 12. Selecting one routing method

The final geometry comparison must use the same routing definition for 102 string and 160 string.

The three routing approaches will first be evaluated on validation data for both geometries. One routing method will then be selected globally using a predefined combined validation rule. The rule must consider both geometries and end to end reconstruction quality.

It is not valid to select one routing definition for 102 string and a different routing definition for 160 string in the primary geometry comparison, because routing and geometry effects would be mixed.

The learned model weights, routing thresholds, and optimal hyperparameters may differ between geometries. The routing label definition remains the same.

## 13. Final geometry comparison

The primary final comparison uses events that trigger in both geometries. This paired event subset ensures that both systems are evaluated on the same physical events.

The final report should contain:

1. The controlled same hyperparameter baseline comparison.
2. The separately optimized comparison with equal tuning budgets.
3. Performance on the common triggered test subset.
4. A separate acceptance oriented result on each geometry's full triggered test sample.
5. Unweighted and physics weighted metrics where appropriate.
6. Energy resolution, energy bias, opening angle, and flavor dependent results.

## 14. Reproducibility rules

1. Keep the canonical split manifest unchanged.
2. Record the code revision, source config, sampler seed, model seed, and Slurm job identifier for every trial.
3. Use deterministic trial names and isolated output directories.
4. Preserve all baseline results.
5. Do not silently retry a failed job with changed model settings.
6. Retries caused by infrastructure failure must reuse the same trial configuration.
7. Confirm the winning configuration with additional model seeds when compute permits.

## 15. Execution order

1. Complete and audit available baseline training histories.
2. Lock the epoch ceiling, patience, search space, objective, pruning rule, and trial budget.
3. Implement and validate the tuning infrastructure with a dry run and one small smoke trial.
4. Tune completed 160 string reconstruction systems.
5. Apply the same locked protocol to 102 string when its baselines finish.
6. Tune the six classifiers using equal budgets across geometries and routing methods.
7. Train or confirm the winning reconstruction configurations.
8. Optimize routing thresholds on validation data.
9. Select one global routing method using both geometries' validation results.
10. Run final inference on the untouched test set.
11. Compare 102 string and 160 string on common triggered events and report the acceptance oriented result separately.

## 16. Status on 2026-07-14

All 21 baseline reconstruction models for 160 string have completed and produced validation summaries.

Of these models:

1. Nineteen reached the 30 epoch ceiling.
2. Two stopped earlier through early stopping.
3. Several obtained their best validation loss at epoch 28 or 29.

This supports testing a higher epoch ceiling during tuning. The locked tuning protocol must still be finalized before submitting the first study.
