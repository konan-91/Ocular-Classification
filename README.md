# Ocular Classification

Detecting eye blinks in EEG recordings by turning short segments of signal into
topographic scalp maps and classifying them with a convolutional network.

Built for a cognitive science lab, where removing ocular artifacts was a manual
bottleneck: existing automated methods are not trusted on their own, so a trained
researcher has to visually inspect the data and decide whether an artifact is present.

![Example topoplot](https://github.com/user-attachments/assets/9d215fb3-ec54-4f41-ad3e-9a05e7dff231)

## The problem

EEG picks up far more than brain activity. Blinks, eye movements and muscle tension
produce large deflections that swamp the signals researchers actually want. This has to
be dealt with before any analysis can happen.

Automated removal exists, and independent component analysis is the standard approach,
but it is not reliable enough to run unsupervised in most labs. So the process is
semi manual: an algorithm proposes, a human checks. That human has to be trained, and
checking thousands of epochs is slow.

## The approach

A blink has a distinctive spatial signature. It is strong, frontal and symmetric across
the scalp, and it looks different from a saccade, which is lateralised. That is a
spatial pattern rather than a temporal one, which suggests treating it as an image
problem.

Each segment of EEG is averaged across a 700ms window and rendered as a topoplot, a
view of electrical activity across the scalp. Those images train a ResNet34 through
transfer learning.

Two design choices follow from this.

**The model never sees the EOG electrodes.** Those channels sit beside the eyes and
record the artifact almost directly, and the eye tracker trigger channels carry the
labels outright. Both are excluded, along with Fp1 and Fp2 which saturate on every
blink. The model has to work from the wider scalp topography, which is what would let
it run on recordings that have no EOG montage at all.

**The negative class is mostly saccades, not rest.** Separating a blink from resting data
is straightforward. Separating it from a large vertical eye movement is the harder case,
and the one where amplitude thresholding fails.

## Results

The model is scored on participants it never saw during training, and compared against
an ICA baseline on the same segments with a paired significance test.

Running `ocular benchmark` writes `artifacts/benchmark.json` and prints a comparison
table: accuracy, balanced accuracy and per-class recall for both the model and the ICA
baseline, alongside McNemar's test on their paired predictions and a confidence interval
on the accuracy difference.

## Evaluation design

Segments from one recording share a participant, an electrode montage and a session, and
one recording contributes hundreds of them. Splitting over segments would place the same
participant in both training and evaluation, and the resulting score would measure
recognition of known participants rather than generalisation to new ones.

Splits are drawn over recordings. No participant appears in more than one split, the
assignment is stored in the model checkpoint so evaluation cannot drift from training,
and `tests/test_splits.py` asserts the property directly.

Two related choices follow from the same concern. Class imbalance is handled by
oversampling the minority class during training rather than by discarding data, and every
class uses the same 700ms window so that segment duration cannot act as a cue.

## Running it

Requires Python 3.10 or newer.

```bash
git clone https://github.com/konan-91/Ocular-Classification.git
cd Ocular-Classification

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Download the dataset from [OSF](https://osf.io/2qgrd/) and unpack it into `data/raw/`,
so that the study folders sit directly inside.

```bash
# Render the recordings as topoplots and write the manifest.
# This is the slow step, since it draws one figure per segment.
ocular prepare

# Train. Add --device cpu if you have no GPU.
ocular train

# Score on the held out recordings.
ocular evaluate

# Compare against the ICA baseline, with McNemar's test.
ocular benchmark
```

Add `--limit-per-event 20` to `prepare` for a quick run that touches every stage without
rendering the whole dataset.

Each command has `--help`. Outputs land in `artifacts/`: the checkpoint, metrics as JSON,
confusion matrices, and the benchmark comparison.

## The ICA baseline

ICA identifies an ocular component rather than labelling segments, so the baseline needs
a defined scoring rule and threshold. Both are chosen to avoid handicapping it.

ICA is fit per recording on a 1 Hz highpassed copy, and the ocular component is
identified with MNE's `find_bads_eog`. Each segment is scored by that component's peak to
peak swing, since a blink is a large brief deflection and range separates it better than
variance.

Component amplitudes carry an arbitrary sign and scale that varies between recordings, so
scores are standardised within a recording using the median and MAD. A single raw cutoff
across participants would handicap the baseline for reasons unrelated to how well it
separates blinks.

The threshold is fitted on the validation recordings and applied unchanged to the test
recordings, which is the same information the model gets.

The baseline reads the EOG electrodes and the model does not. This asymmetry favours the
baseline and is retained, since the model is intended to run without an EOG montage.

## Why McNemar's test

Both methods are scored on the same segments, so their errors are paired and an ordinary
two sample test would be wrong. McNemar's test discards the segments the two methods agree
on and asks only whether the disagreements are lopsided. The exact binomial version is used
below 25 disagreements, where the chi squared approximation is unreliable.

The confidence interval on the accuracy difference is reported alongside the p value,
since it gives the size of the effect rather than only its presence.

## Layout

```
src/ocular/
  channels.py       which channels are scalp, trigger, EOG or auxiliary
  epochs.py         loading recordings and re-epoching around event onsets
  topoplots.py      rendering segments as images
  manifest.py       the dataset manifest, one row per image
  splits.py         participant level train, validation and test splits
  data.py           torch datasets, transforms and balanced sampling
  model.py          the ResNet classifier
  train.py          two phase transfer learning
  evaluate.py       inference and scoring
  ica_baseline.py   the ICA comparison method
  stats.py          McNemar's test and paired confidence intervals
  benchmark.py      the full comparison
  cli.py            command line entry points

notebooks/          narrative walkthroughs of each stage
tests/              runs on synthetic recordings, no dataset needed
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests build small synthetic recordings with the same structure as the real data, so
the whole pipeline including ICA and the significance test runs without the dataset
present. The leakage property has its own tests, since it is the thing most likely to
break silently.

## Limitations

The model classifies segments. It does not clean a recording. Wiring it into a
preprocessing pipeline, deciding what happens to a segment once flagged, and confirming
the cleaned data is still usable for analysis are separate pieces of work.

The dataset is cued: participants were asked to blink, so the blinks are deliberate and
well separated. Spontaneous blinks during a real task are smaller and land on top of
whatever else is happening, and this data cannot say how performance holds up there.

Every recording comes from one dataset and one broad montage family. Generalisation to a
different lab's setup is untested.

## Dataset

EEG data containing eye artifacts and resting brain activity, 45 recording sessions from
39 participants: <https://osf.io/2qgrd/>

Processing uses [MNE-Python](https://mne.tools/).
