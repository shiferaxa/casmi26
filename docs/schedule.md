# Schedule to Dec 14 2026

Written Sep 18 2026 (Friday). Final submission deadline Dec 14 23:59 UTC, team merge deadline Dec 7. Twelve and a half weeks. Public LB 0.339, rank 11. The whole top 50 runs the same shared fingerprint checkpoints and candidate pool; the private board (267 molecules) will reshuffle that cluster. The plan is to be the team that owns a better fingerprint model, then to be disciplined about which two submissions get picked.

## Rules for every day

- np-examples first. No kernel run without an np-examples number, no submission without a local gain on class 2 or class 3. The training-structure holdout lied twice.
- One idea per submission. Five per day is a trap; noise on the public board is about 0.006.
- Every run gets a dated line in the log at the bottom of `plan.md` with the np-examples numbers and the LB score.
- Kaggle GPU quota on this account is 6 hours a week (checked Sep 18, refreshes Saturdays 00:00 UTC), TPU quota 20 hours. Inference kernels run on CPU. Training kernels checkpoint every epoch to `/kaggle/working` so a killed session loses at most one epoch. If 6 GPU hours a week proves too little, the options are a TPU port of the training script (20 free hours) or Colab Pro (about 10 dollars a month); that is a decision for the user.
- Commit at the end of every day that changed code.

## Week 1, Sep 18 to Sep 24: own fingerprint model, version 1

Goal: a model whose channel-4-alone MRR on np-examples matches or beats the public checkpoints, then an ensemble of both in the kernel.

- Day 1, Fri Sep 18. Commit (done). Measure the bar: channel 4 alone (fingerprint dot logits inside the mass window, no ranker) for the public checkpoints on np-examples, classes 2 and 3 (`scripts/eval_channel4.py`). Build the preprocessing kernel: train.parquet to padded peak tokens (128 per spectrum, m/z, intensity, precursor, adduct id, instrument family, collision energy, polarity), fingerprint targets from `data/cache/train_fp.npz` (upload as a private dataset), molecule-grouped split with np-examples fully held out plus 2 percent random structures for validation. Output becomes a Kaggle dataset the training kernels read.
- Day 2, Sat Sep 19. Training kernel v1 (queued to push itself when preprocessing finishes; it will only start once the quota refreshes Friday 20:00 EDT): the public FPNet architecture (d 512, 6 layers, 128 tokens), plain binary cross entropy, mixed precision, batch 512, cosine schedule, 5.4 hour session, checkpoint every epoch. Local checkpoint evaluator is `scripts/eval_channel4.py --models`.
- Day 3, Sun Sep 20. Pull the checkpoint. Evaluate channel 4 alone against the Day 1 bar. Plug it into the harness as an ensemble with the public checkpoints (average logits) and run the full pipeline on np-examples. If class 2 or 3 improves: kernel v7 and submit.
- Day 4, Mon Sep 21. Error analysis on np-examples misses: which bits are wrong, negative mode, multi-adduct molecules. Training v2: oversample timsTOF, add a merged-spectrum input like the public merged checkpoint, longer schedule. Launch.
- Day 5, Tue Sep 22. Ranker retrain now that channel 4 features changed: rows from the natural-product structures only (`dump_rank_features.py --overlap --overlap-classes 1,2,3`) with the new logits, `train_ranker.py`, np-examples check. Kernel v8 if better.
- Day 6, Wed Sep 23. Evaluate v2. Start a second seed of the best config for ensembling. Submit the best of the week.
- Day 7, Thu Sep 24. Weekly review: LB, GPU hours used, what moved np-examples. Decide week 2 scope. Commit.

## Week 2, Sep 25 to Oct 1: model version 2 and scoring

- Bigger or deeper model (d 768 or 8 layers) if v1 underfits; more epochs if it still improves.
- Contrastive spectrum-to-structure head (InfoNCE against the candidate fingerprint encoder) as a second score.
- Bernoulli log-likelihood scoring for channel 4 (bits dot logit p plus sum log(1 minus p)) as an extra ranker feature; the public dot product favours candidates with many bits set.
- Three-seed ensemble of the best config. Check kernel runtime stays under 6 hours with the extra inference.

## Week 3, Oct 2 to Oct 8: candidate recall

- Add LOTUS and NPAtlas to the pool, each measured separately on np-examples for dilution (ChEBI and LIPID MAPS hurt, so expect to keep only what helps).
- Molecular formula filter: predict or enumerate formulas from precursor mass and isotope-free constraints, drop candidates whose formula cannot match.
- Adduct consistency: a candidate must explain the neutral mass under every adduct the molecule was seen with (108 of 400 test molecules have more than one).

## Week 4, Oct 9 to Oct 15: the other channels

- Fragmentation channel is dead for molecules over 34 bonds (most of the test range). Replace with sampled bond breaks or a cap that keeps it informative.
- Analog channel tuning: window, number of analogs, similarity power, representative spectrum choice.
- LambdaMART ranker (LightGBM lambdarank) on natural-product rows with all new features, compared with the gradient boosting classifier.

## Weeks 5 and 6, Oct 16 to Oct 29: hard cases and robustness

- Negative mode and multi-adduct molecules as their own np-examples slices.
- Pseudo-labelling: confident test hits added to the library, measured on np-examples before use.
- Seed variance of the whole pipeline; anything inside the noise is dropped.

## Weeks 7 and 8, Oct 30 to Nov 12: consolidation and teaming

- A second, deliberately different pipeline variant (different model, different pool) for a final blend.
- Teaming decision. Look for a partner with a GPU or metabolomics background on the discussion board; merge deadline is Dec 7. A team roughly doubles the compute and halves the blind spots.

## Weeks 9 and 10, Nov 13 to Nov 26: final training

- Longest training runs of the best configs, full ensembles, everything logged with np-examples and LB.
- Start picking final submission candidates by agreement between np-examples and the public LB, not by LB alone.

## Week 11, Nov 27 to Dec 6: freeze

- Freeze features and models. Reproducibility run of the final kernel from a clean fork. Runtime margin under 6 hours of the 9 allowed.
- Dec 7: merge deadline passes; no more team changes.

## Week 12, Dec 7 to Dec 14: final picks

- Two final submissions: the best public LB run and the best np-examples run. They will differ; that is the point. Buffer days for Kaggle outages. Nothing new after Dec 11.
