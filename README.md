# kaggle-predicting-molecular-properties

## Note on the evaluation metric

This is the grouped log mae -- i.e. the sum of the MAE scores for each of the 8
groups.

This is equivaluent to a sample-weighted MAE score,
where each sample is weighted by 1/n_samples_of_same_group.

If training a single model for all targets, we should use this
weighting scheme.

