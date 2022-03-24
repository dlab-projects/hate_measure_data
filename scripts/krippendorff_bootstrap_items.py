import datasets 
import numpy as np
import simpledorff
import pandas as pd
import pickle

from hate_measure.keys import items
from hate_measure.utils import recode_responses
from scipy.stats import bootstrap

recode = True
save_file = "krippendorff_items_recoded.pkl"
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
data = dataset['train'].to_pandas()
if recode:
    data = recode_responses(
        data,
        insult={1: 0, 2: 1, 3: 2, 4: 3},
        humiliate={1: 0, 2: 0, 3: 1, 4: 2},
        status={1: 0, 2: 0, 3: 1, 4: 1},
        dehumanize={1: 0, 2: 0, 3: 1, 4: 1},
        violence={1: 0, 2: 0, 3: 1, 4: 1},
        genocide={1: 0, 2: 0, 3: 1, 4: 1},
        attack_defend={1: 0, 2: 1, 3: 2, 4: 3},
        hatespeech={1: 0, 2: 1})
n_items = len(items)

def krippendorff_helper(arr1, arr2, arr3):
    """Helper function for bootstrapping Krippendorff's Alpha."""
    df = pd.DataFrame.from_dict(
        {'experiment_col': arr1,
         'annotator_col': arr2,
         'class_col': arr3})
    alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
        df,
        experiment_col='experiment_col',
        annotator_col='annotator_col',
        class_col='class_col')
    return alpha

confidence_low = np.zeros(n_items)
confidence_high = np.zeros(n_items)
standard_error = np.zeros(n_items)
alphas = np.zeros(n_items)


for idx, item in enumerate(items):
    chunk = data[['comment_id', 'annotator_id'] + [item]].values
    boot = bootstrap(
        (chunk[:, 0], chunk[:, 1], chunk[:, 2]),
        krippendorff_helper,
        vectorized=False,
        paired=True,
        method='basic',
        n_resamples=100)
    confidence_low[idx] = boot.confidence_interval.low
    confidence_high[idx] = boot.confidence_interval.high
    standard_error[idx] = boot.standard_error
    alphas[idx] = simpledorff.calculate_krippendorffs_alpha_for_df(
        chunk,
        experiment_col='comment_id',
        annotator_col='annotator_id',
        class_col=item)

results = {}
results['confidence_low'] = confidence_low
results['confidence_high'] = confidence_high
results['standard_error'] = standard_error
results['alphas'] = alphas

with open("krippendorff_items_recoded.pkl", "wb") as file:
    pickle.dump(results, file)