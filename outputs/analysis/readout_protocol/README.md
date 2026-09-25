# Readout protocol, not label space (2026-09-03)

## Question

`a7_coco.py` reported that swapping the probe's label space from ImageNet 1000-way
single-label to COCO 80-way multi-hot fixes ImageNet probing's family-offset failure.
A full COCO2014 multi-label run under the *trained-probe* protocol
(`outputs/coco2014_multilabel_linear_probing/`, SGD + 13-LR sweep, live augmented
backbone, ~82k images) does not reproduce that gain — on the same 66 models it is worse
than ImageNet probing on whole-table rho and inverts in the strong-candidate band.

Three things differ between the two, not one: the readout (closed-form ridge vs SGD),
per-dimension feature standardization, and data scale + augmentation.

## Design

`p1_probe.py` holds data scale, augmentation, images, labels and split fixed — the same
2400/1000/1218 split of the same 4618 cached COCO val2017 patch-mean features and the
same 80-way multi-hot labels `a7_coco.py` used — and crosses the other two axes:

* ridge: SVD + 17-point alpha grid, selected on val mAP
* SGD: the ImageNet protocol's readout (13-point base-LR grid, `lr*batch/256`, cosine to
  zero, momentum 0.9, wd 0, `W~N(0,0.01)`, BCE), selected on val mAP over (LR, epoch)
* standardization: `(Z-mean)/std` per dimension, versus centering only

`p2_eval.py` scores all four against MLLM Avg under one-per-family, cross-family top-1
regret (the `c4_final.py` definition), the GT top-25% band, and the share of residual
variance explained by a per-family additive offset.

## Result (66 models, 13 families)

| readout | rho | one-per-family | cross-family regret | GT top 25% | family residual |
|---|---:|---:|---:|---:|---:|
| A ridge + standardized (= `mAP_obj80`) | 0.835 | 0.613 | 2.55 | **+0.750** | 0.784 |
| B ridge + centered only | 0.837 | 0.607 | 2.55 | +0.718 | 0.784 |
| C SGD + standardized | 0.775 | 0.541 | 3.44 | +0.664 | 0.771 |
| D SGD + centered only (`feature_normalization: False`) | 0.706 | 0.337 | 6.56 | +0.037 | 0.827 |
| E COCO2014 SGD, live backbone + aug, 82k | 0.581 | 0.404 | 4.55 | **-0.449** | 0.759 |
| F ImageNet probing (`probe_epoch1`) | 0.697 | 0.305 | 5.31 | -0.018 | 0.829 |

Controlled contrasts are A-D only; E and F differ in more than one way.

1. **Standardization is irrelevant to the ridge and decisive for SGD.** A vs B is a null
   (rho between the two scores is 0.999). C vs D moves one-per-family 0.541 -> 0.337,
   regret 3.44 -> 6.56, top-25% +0.664 -> +0.037. The ridge absorbs per-dimension scale
   through its spectrum; a single global learning rate does not.
2. **The optimizer alone is a small effect.** A vs C costs 0.613 -> 0.541 with score
   agreement 0.980.
3. **~~The label space is not what fixed anything.~~ Superseded by `p3_labels.py`; see
   below.** The D-versus-F comparison used to argue this is not controlled -- those two
   cells differ in dataset (COCO 2400 cached vs ImageNet 1.28M live) as well as in labels.
   The controlled label contrast is in the next section and it is non-zero.
4. **Standardization does not close the whole gap to E.** D still sits well above E in the
   top-25% band (+0.037 vs -0.449), and the two agree at only 0.655. The residual belongs
   to data scale and augmentation, which specifically damage the strong-candidate band.

## Consequence

The Readout Battery's numbers stand, but its explanation does not. The gain attributed to
multi-label label structure is a readout effect: a scale-adaptive, low-sample, closed-form
ridge on deterministic cached features. Any claim of the form "changing the label space
repairs probing" should be restated as "changing the readout repairs probing", and the
`mAP_dom` versus `mAP_obj80` control (both ridge) is not evidence for the label axis
against a trained probe.

## Reproduce

```bash
conda run --no-capture-output -n TokBench python outputs/analysis/readout_protocol/p1_probe.py
conda run --no-capture-output -n TokBench python outputs/analysis/readout_protocol/p2_eval.py
```

`p1_probe.py` takes ~16 s per encoder on one A100 (84 encoders, ~22 min) and writes
`readout_2x2.csv`. Pass encoder names as arguments to run a subset.


## Label axis, controlled (`p3_labels.py`, `p4_eval.py`)

The claim above that labels do nothing was drawn from an uncontrolled comparison. Running
`a7_coco.py`'s single-label control (`Ydom`, one-hot on the largest-area object) through all
four readout configurations gives the label contrast at fixed readout, on 79 models and 18
families.

| readout | labels | rho | one-per-family | regret | top 25% | family residual |
|---|---|---:|---:|---:|---:|---:|
| ridge + standardized | multi-hot | 0.838 | **0.618** | **2.56** | **+0.812** | 0.798 |
| ridge + standardized | one-hot | 0.762 | 0.515 | 5.20 | +0.651 | 0.755 |
| ridge + centered | multi-hot | 0.838 | 0.606 | 2.76 | +0.803 | 0.794 |
| ridge + centered | one-hot | 0.757 | 0.511 | 5.28 | +0.594 | 0.749 |
| SGD + standardized | multi-hot | 0.796 | 0.574 | 3.33 | +0.725 | 0.747 |
| SGD + standardized | one-hot | 0.689 | 0.481 | 5.54 | +0.364 | 0.677 |
| SGD + centered | multi-hot | 0.741 | 0.444 | 5.55 | +0.096 | 0.736 |
| SGD + centered | one-hot | 0.596 | 0.296 | 8.63 | +0.111 | 0.707 |
| ImageNet probing | one-hot | 0.731 | 0.361 | 5.00 | +0.216 | 0.806 |

**Both axes are real and roughly additive.**

* Label effect at fixed readout (multi-hot minus one-hot): one-per-family +0.103 / +0.094 /
  +0.094 / +0.148 and regret -2.64 / -2.52 / -2.21 / -3.08 across the four readouts. The
  sign is the same in every configuration, including the trained SGD probe, so the
  multi-label gain is **not** ridge-conditional. Only the top-25% band is inconsistent
  (-0.015 under SGD + centered).
* Readout effect at fixed labels (ridge+standardized minus SGD+centered): one-per-family
  +0.174 (multi-hot) and +0.219 (one-hot), regret -2.99 / -3.43.
* The readout axis is the larger of the two (~+0.20 versus ~+0.11 on one-per-family), and
  they compose: the worst cell (SGD + centered + one-hot, 0.296) to the best cell
  (ridge + standardized + multi-hot, 0.618) is +0.322.

So the Readout Battery's label choice does carry real signal; the earlier write-up
under-credited it because the only trained-probe evidence available then differed in
dataset as well. The honest statement is that **readout and label space are two separate,
additive contributions, with the readout the larger one** -- not that either alone explains
the repair.
