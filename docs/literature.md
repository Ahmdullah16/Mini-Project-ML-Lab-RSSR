# Literature verification log — 2026-10-18

This is a scoped primary-source review, not an exhaustive systematic review.
Published external performance is not reproduced here. DOI fields absent from
references.bib remain unverified; none are invented. Full per-paper losses,
training data, compute and evaluation extraction remains pending.

| Work | Verified source/status | Task and relevance | Local implementation status |
|---|---|---|---|
| SRCNN | Dong, Loy, He, Tang; ECCV 2014; https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html | Classical single-image SR; convolutional baseline; author page links MATLAB/Caffe code | Compact adaptation, not reproduction |
| EDSR | Lim, Son, Kim, Nah, Lee; CVPR Workshops 2017; https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html | Residual CNN SR; useful stronger CNN comparison | Compact residual model only |
| RCAN | Zhang, Li, Li, Wang, Zhong, Fu; 2018 preprint https://arxiv.org/abs/1807.02758 | Residual-in-residual and channel recalibration; establishes attention precedent | Channel attention teaching module only |
| SwinIR | Liang, Cao, Sun, Zhang, Van Gool, Timofte; ICCV Workshops 2021; https://arxiv.org/abs/2108.10257 | Window/shifted-window restoration; official https://github.com/JingyunLiang/SwinIR | Teaching adaptation plus executed official pretrained lightweight x2 control; see reports/official-swinir-validation.json and reports/final-heldout.json. Pretraining is unmatched. |
| HAT | Chen, Wang, Zhou, Qiao, Dong; CVPR 2023; https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html | Channel+window attention and cross-window interactions; combination already established; https://github.com/XPixelGroup/HAT | Not implemented |
| MambaIRv2 | Guo, Guo, Zha, Zhang, Li, Dai, Xia, Li; CVPR 2025; https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html | Attentive state-space restoration; strong recent candidate; https://github.com/csguoh/MambaIR | Deferred pending dependency/compute assessment |
| OpenSR-test | Official https://github.com/ESAOpenSR/opensr-test links IEEE document 10530998 | Remote-sensing consistency/synthesis/correctness, multispectral benchmark | Dataset subset used; diagnostics integration pending |
| DiffFuSR | Sarmad, Salberg, Kampffmeyer; arXiv v2 2025-11-28 reports TGRS acceptance; DOI 10.1109/TGRS.2025.3638896; https://arxiv.org/abs/2506.11764 | Diffusion RGB SR + multispectral fusion, degradation conditioning; overlap with sensor-consistency motivation | Not implemented |
| GeoSR-Bench | Li et al.; arXiv v2 2026-06-01, under review at TPAMI, not accepted according to source; https://arxiv.org/abs/2605.00310 | Downstream task utility, MODIS/Landsat and S2/NAIP; https://github.com/ai-spatial/GeoSR-Bench | Not downloaded or evaluated |

Natural-image SR results cannot establish reflectance fidelity or transfer across
sensors. OpenSR and GeoSR-Bench motivate evaluation beyond PSNR. HAT already combines
channel/window attention; DiffFuSR conditions restoration on degradation and handles
multispectral fusion. The proposed LR reliability gate needs a much closer overlap
review against confidence gating, uncertainty estimation and data-consistency work.
Current hypothesis is exploratory, not a novelty claim.

Next extraction pass: for each paper record precise dataset/bands/degradation,
losses, training steps, pretraining, parameters, hardware, evaluation crop and range,
code commit/license, limitations and applicability. Unknown fields stay pending.

October27 source update: the SEN2VENUS methodology and correction are now available through publisher-provided full texts. The dated [feasibility addendum](../reports/mp36-feasibility.md) records the reviewed sections and remaining version-specific limitations. This supersedes the source-access blocker without claiming paired evaluation.
