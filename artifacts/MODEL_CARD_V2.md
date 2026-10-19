# Frozen RGB development models v2

Genuine trained weights, 1000-step CPU Adam/L1 runs, four NAIP training and four validation scenes. CNN seed17 and hybrid seed23 selected using validation only. Both are x2 RGB, width16; hybrid includes local residual CNN, window multi-head self-attention, channel/spatial attention and concatenation fusion. No shifted windows or official SwinIR reproduction claim.

One original held-out aerial scene (16 patches) was evaluated after protocol freeze at local commit 9109766. RGB CNN PSNR50.2614dB, hybrid50.2459dB, bicubic48.5183dB. This is a tiny synthetic Gaussian/area degradation study, not independent real cross-sensor validity. No hybrid superiority or novelty claim. Model output can invent details; no physical reflectance calibration. NAIP public-domain source, USDA/OpenSR attribution.

Checkpoints are inference-only, with optimizer state removed without changing learned weights. Source and prepared manifests preserve original bytes/checksums to rebuild frozen data across operating systems. Paths inside manifests may use legacy Windows separators; consumers must normalize them. Files contain no secrets or private user data.
