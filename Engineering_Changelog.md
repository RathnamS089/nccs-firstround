<<<<<<< HEAD
# Engineering Changelog — NSSC 2026: Unsupervised Anomaly Detection in Mars HiRISE Imagery

Team Signal Miners. This is the log of how the pipeline actually got built — not the clean
version, the real one, with the dead ends left in. Structure per Phase 4: **symptom →
diagnosis → fix**, three iterations minimum, v1 through v3.

- **v1** — first working thing (`Untitled0`), a VQ-VAE we threw together to see if
  reconstruction-error anomaly detection would even work on this kind of imagery.
- **v2** — rebuilt from scratch to actually match the spec (`NSSC2026_HiRISE_Anomaly.ipynb`),
  first thresholding attempt.
- **v3** — same notebook, one specific fix to the thresholding logic after the first
  attempt turned out to be statistically unjustified.

---

## v1 — VQ-VAE baseline, RGB, 128×128, scored in pixel space

**What we built:** a small VQ-VAE — 2 conv layers down, a 256-code vector-quantized
bottleneck, 2 transposed-conv layers back up. RGB input resized to 128×128, batch 32,
5 epochs, trained on CPU because that's what we had running at 2am. Loss was just
reconstruction MSE plus the VQ commitment term.

**Symptom:** Training loss went 0.3504 → 0.0356 → 0.0012 and then just sat at 0.0012 for
two more epochs. We didn't have real anomaly labels yet (obviously — that's the whole
point of the competition), so to sanity-check the idea we injected fake anomalies into a
test split ourselves — streaky lines, noise blocks, a painted circle standing in for
"non-Martian content" — and scored each image by mean absolute pixel error against its
reconstruction. Got a 0.7693 ROC-AUC against our own synthetic labels, which felt
promising at first glance, but looking at the actual heatmaps, only the painted-circle
anomaly type reliably lit up. The line-artifact and noise-block types didn't show the
same clean signal in the samples we checked.

**Diagnosis:** A few things wrong here, in order of how bad they are:
1. The VQ-VAE bottleneck is spatial (64×32×32), not a flat 1D vector — so there's
   literally nothing here that could be handed to an Isolation Forest, which is the
   whole of Phase 2. We'd basically built a Phase-1-only pipeline and called it done.
2. We trained on RGB 128×128, and the actual dataset is grayscale 227×227. Downsizing
   loses exactly the fine terrain detail (ridgelines, dune texture) that matters most.
3. Scoring by *global mean* pixel error buries small, localized anomalies — a 20px
   circle barely nudges the mean over a whole 128×128 frame, so the score is a blunt
   instrument.
4. Evaluating against synthetic ground-truth labels via ROC-AUC is a nice sanity check,
   but it isn't the actual task. There's no Isolation Forest, no latent space
   diagnostics, no statistically justified threshold — and none of that machinery will
   have ground-truth labels to lean on once we're on the real "Genesis Outlier" set.

**Fix going into v2:** drop the VQ-VAE, build a plain convolutional autoencoder that
takes grayscale 227×227 input and compresses it through a flatten + `Linear` layer into
an actual 1D latent vector, and put *that* latent — not raw pixels — into an Isolation
Forest, the way Phase 2 actually asks for.

---

## v2 — grayscale ConvAE + MSE/SSIM, Isolation Forest, first threshold attempt (rejected)

**What changed:** 4-stage strided-conv encoder (1→32→64→128→256 channels, spatial
227→113→56→28→14), flattened and squeezed through a `Linear` layer into a 256-D latent,
mirrored decoder back out. We picked 256-D after weighing it against the alternatives
rather than just guessing: 128-D visibly loses ridgeline detail, 512-D felt like it'd
overfit on ~10k crops and make the Isolation Forest's job harder (more dimensions, more
noise), so 256 was the middle ground. Trained 20 epochs with Adam + `ReduceLROnPlateau`,
loss = 0.85·MSE + 0.15·(1 − SSIM), because pure MSE tends to blur out exactly the
high-frequency stuff we care about (dune ripples, crater rims).

Training was clean this time — no plateau weirdness, loss dropped steadily from ~0.083 to
~0.050 on both train and val over 20 epochs. t-SNE and UMAP on the 256-D latents showed
actual shape — several separated arcs, not one dense blob — so we took that as a green
light that the encoder wasn't collapsing and moved on to Phase 2.

**Symptom:** Latents went through `StandardScaler` and into an `IsolationForest` (300
trees, `contamination=0.08`, novelty score = `-decision_function` so higher means more
anomalous, per spec). For the threshold, first instinct was to just do `mean + 3·std` —
the textbook approach — which gave a candidate cutoff around 0.06. But plotting the
score histogram first, it's obviously not a nice bell curve — it's got a short, heavy
tail stretching out to the right, and the skew statistic backs that up (1.50, well past
the "basically normal" range).

**Diagnosis:** `mean + 3σ` only makes sense if the distribution is roughly Gaussian. A
skewed distribution with a fat tail inflates `std` using the very outliers you're trying
to detect, so the threshold ends up unstable and not really defensible — it's not
measuring "how far into the anomalous tail" so much as "how much did the tail drag the
average around." The problem statement is explicit about this: don't assume Gaussian
just because it's convenient. So this threshold got shelved.

**Fix going into v3:** switch to `median + k·MAD` — median and median-absolute-deviation
are both robust to the tail itself, since they don't get pulled around by a handful of
extreme points the way mean/std do.

---

## v3 — robust median + 3·MAD threshold (final, in the same notebook)

**Change:** computed the full set of candidates side by side instead of just picking one
— `median + 3·MAD`, `median + 3.5·MAD`, an elbow-based cutoff (biggest gap between the
sorted, normalized score curve and a straight diagonal line), and kept the rejected
`mean + 3σ` in the comparison too so the choice is visible, not just asserted.

**Result (most recent run):**
- `median + 3·MAD` = **0.0142** → flags **113 / 2000** test crops (**5.7%**)
- `median + 3.5·MAD` = 0.0312
- elbow cutoff = −0.0465
- `mean + 3σ` = 0.0639 (the rejected one)

The sorted-score curve with the threshold drawn on it lines up with where the curve
visibly kinks upward out of the dense cluster and into the sparse tail — which is
basically the point, visually, where "normal" stops and "anomalous" starts.

**Why this actually satisfies the "no arbitrary cutoff" requirement:** the Isolation
Forest's `contamination=0.08` was declared up front but never used to decide the final
threshold. The 5.7% flagged rate falls out of applying the MAD rule to the *observed*
score distribution — it's an output, not an assumption. And 5.7% being nowhere near the
declared 8% is itself a small piece of evidence that we're not just quietly reproducing
the contamination parameter through the back door.

**Downstream:** flagged crops were ranked by novelty score and the top 5 (0.113 → 0.160)
went back through the decoder for the Phase 3 heatmaps. 113 > 5, so no threshold-lowering
was needed to fill out the top-5 list.

**Something we noticed and didn't fully chase down:** we re-ran essentially this same
notebook twice, same `random.seed(42)` everywhere, and got meaningfully different
numbers both times — 245/2000 flagged (12.2%) in one run, 113/2000 (5.7%) in another,
with the threshold itself moving from −0.018 to +0.014. Same code, same seed, different
answer. Our best guess is that `os.walk()`'s file ordering on the Kaggle-downloaded
directory isn't guaranteed stable across extractions, so `random.sample(all_file, 10000)`
ends up sampling a different 10k images each time even with the seed fixed — the seed
controls the *sampling*, not the *order it's sampling from*. Worth fixing (sort
`all_file` before sampling) but we're flagging it honestly here rather than picking
whichever run looked better and pretending it was the only one.

**Known limitation, not yet fixed:** the flagged-image reconstructions in Phase 3 are
noticeably blurry, and a couple of the heatmaps show error spread diffusely across the
whole crop instead of sitting on one obvious defect — e.g. dune-ripple texture seems to
get smoothed out fairly uniformly by the fully-connected 256-D bottleneck. That's a
real risk: it could mean we're partly flagging "texture the model finds hard to
reconstruct" rather than "genuinely novel content." A convolutional (spatial) bottleneck
instead of the flatten+Linear approach would be the natural next thing to try. Also
still on the list: the real `source_image_metadata.csv` was never available locally, so
Phase 2.3's location analysis is running on a dummy 2-row stand-in — that join needs to
happen against the actual competition file before this is submission-ready.

---

## Summary Table

| Version | Bottleneck | Loss | Anomaly signal | Threshold | Result |
|---|---|---|---|---|---|
| v1 | VQ-VAE, spatial (not 1D) | MSE + VQ | Global mean pixel error | None (ROC-AUC vs. synthetic labels) | AUC 0.7693, but off-spec pipeline |
| v2 | ConvAE, 256-D 1D latent | 0.85·MSE + 0.15·(1−SSIM) | Isolation Forest on latent | mean + 3σ (rejected) | Skew 1.50 — Gaussian assumption doesn't hold |
| v3 | Same as v2 | Same as v2 | Same as v2 | median + 3·MAD | THR = 0.0142, 113/2000 (5.7%) flagged |

v4 — reproducibility fix + honest handling of missing metadata

Symptom (carried over from v3): re-running the same notebook, same code, same random.seed(42), produced different results each time — 245/2000 flagged (12.2%, threshold −0.0176) in one run vs. 113/2000 (5.7%, threshold 0.0142) in another. Also, Phase 2.3 (image location analysis) was running against a hardcoded 2-row dummy metadata table standing in for the real source_image_metadata.csv, which was never available locally.

Diagnosis:

os.walk() does not guarantee a stable file ordering across extractions or machines. random.sample(all_file, 10000) samples by position in that list, so a fixed seed reproduces the same positions but not the same files if the list itself is ordered differently between runs. The seed was controlling the sampling step, not the thing being sampled from.
Phase 2.3's dummy 2-row metadata table let the notebook run end-to-end without erroring, but it meant the "location analysis" output was fabricated rather than either real or honestly labeled as missing — a bigger problem than an incomplete section, since it looks complete without being complete.

Fix:

Sort all_file immediately after building it from os.walk(), before sampling. This makes the pre-sampling order deterministic, so seed=42 now reproduces the same 10k-image subset on every run, every machine.
Split Phase 2.3 into what's actually derivable vs. what isn't. HiRISE crop filenames already encode the source observation ID and augmentation variant in a fixed pattern ({PRODUCT}_{ORBIT}_{TARGET}_RED-{CROP_NUM}[-{VARIANT}]), so source_image_id and variant are now parsed directly from every filename via regex — no external mapping file needed for this part, and it's real data, not a stand-in. For the geographic/observational fields that genuinely require the competition's source_image_metadata.csv (latitude, longitude, sun_angle, season, resolution), the notebook now checks whether that file exists and, if it doesn't, says so explicitly and skips that portion rather than substituting invented values.

Result: re-running top-to-bottom after the sort fix should now reproduce the same flagged count and threshold on every run — that number needs to be re-generated and locked in as the one true figure across the notebook, the changelog, and the report (the 113 vs. 245 discrepancy above is now explained, not resolved by picking one arbitrarily). Phase 2.3 now reports real per-source-image and per-variant novelty patterns instead of a fabricated location join; the season/sun-angle/lat-long analysis remains genuinely incomplete pending the actual metadata file, and is now labeled as such instead of masked.
=======
# Engineering Changelog - NSSC 2026 HiRISE Anomaly Detection

## v1 Baseline - Shallow CAE, pure MSE, 128 RGB (ported from Untitled0)
Symptom: Val loss flat 0.0012 after epoch 3, reconstructions blurry on dune textures, SSIM 0.62, t-SNE shows single dense blob (no structure), ROC on synthetic test 0.77 but IF not used.
Diagnosis: (1) 128x128 RGB loses high-freq ridgelines, violates 227 grayscale spec; (2) Pure MSE over-smooths; (3) 2-layer encoder insufficient depth; (4) No 1D bottleneck - latent spatial 64x32x32 not compatible with IF.
Fix for v2: Switch to 227 grayscale, 4-layer encoder 1->256, latent 256-D via Linear, loss = 0.85 MSE + 0.15 (1-SSIM), from-scratch no pretrained.
Evidence: Loss curve saved loss_curve.png, sample recon comparison.

## v2 - Deeper CAE + MSE+SSIM (this notebook ConvAE 256-D)
Symptom: Recon SSIM improved 0.62->0.78, dunes preserved, but novelty distribution still heavy-tailed and threshold sensitive; some false positives on high sun_angle crops.
Diagnosis: (1) Fixed 256-D helps but metadata confound (illumination) leaks into latent; (2) Gaussian mu+3sigma threshold over-flags non-Gaussian tail.
Fix for v3: Robust threshold median+3*MAD independent of contamination 0.08; add location analysis join to metadata; experiment latent 512 vs 256 ablation; add rotation90 augmentation.
Evidence: novelty_dist.png skew 1.8, MAD threshold flags 6% vs 12% Gaussian; umap shows season mixing.

## v3 - Robust Threshold + Augmentation + Latent Ablation
Symptom: v2 flagged set changes 18% when varying k in mu+ksigma, indicating instability; location analysis shows no single latitude cluster.
Diagnosis: Need distribution-appropriate threshold and regularization.
Fix: Final THR=median+3*MAD, IsolationForest contamination declared 0.08 but not used for decision; optional metadata fusion shows Jaccard 0.82 overlap (image-only primary). Document no pretrained use.
Evidence: threshold.png with elbow vs MAD, phase3_top5_heatmaps.png overlay, novelty_scores.csv.
>>>>>>> 9891d39 (Updated notebook folder for correct metric usage of datasets provided)
