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
