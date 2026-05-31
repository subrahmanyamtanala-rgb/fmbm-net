# FMBM-Net: Federated Multi-Scale Bio-Multimodal Network

[![IEEE TBME](https://img.shields.io/badge/IEEE%20TBME-2026-blue)](https://ieeetbme.embs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange)](https://pytorch.org)
[![Opacus 1.4](https://img.shields.io/badge/Opacus-1.4-red)](https://opacus.ai)

> **FMBM-Net: A Federated Multi-Scale Bio-Multimodal Network with Latent Cross-Attention Transformer for Privacy-Preserving Biomedical AI**  
> Subrahmanyam Tanala (ANITS, Visakhapatnam) · Surender Reddy Salkuti (Woosong University, South Korea)  
> *IEEE Transactions on Biomedical Engineering*, Vol. 73, No. 6, June 2026

---

## Overview

FMBM-Net is a federated learning architecture that natively unifies three biomedical modalities — **genomic sequences**, **3D volumetric medical images**, and **electronic health records (EHR)** — under a single differentially-private pipeline.

### Key Contributions

| Component | Description |
|-----------|-------------|
| **FLCAT** | Federated Latent Cross-Attention Transformer — fully-specified multi-head cross-modal attention (H=8, d_k=64) with residual connections and LayerNorm in (ε,δ)-DP latent space |
| **RUE** | Region Uncertainty Estimation — Monte Carlo dropout (K=10) boundary-confidence scoring with noise-aware training loss |
| **RSGC** | Risk-Stratified Governance Controller — four-tier confidence FSM aligned with BMJ framework and EU AI Act |
| **Sparse MoE** | N=8 experts, top-k=2 routing — 73% FLOPs reduction over dense FFN |

### Results (ε=2.0, δ=1e-5 · 10 seeds · bootstrap 95% CI)

| Task | Dataset | Metric | FMBM-Net | Best Baseline |
|------|---------|--------|----------|---------------|
| Genomic | CADD v1.7 (n=35,000) | AUROC | **0.947±0.008** | 0.903±0.010 (MOON) |
| CT Segmentation | MSD Pancreas-CT (n=281) | Dice | **0.934±0.011** | 0.907±0.012 (MOON) |
| EHR Prediction | MIMIC-IV v2.2 (n=52,143) | AUROC | **0.921±0.009** | 0.894±0.010 (MOON) |
| Multimodal | TCGA-LUNG (n=1,000) | AUROC | **0.947±0.007** | 0.937±0.009 (MOON) |
| XAI | CheXpert (n=2,412) | GradCAM++ IoU | **0.634±0.029** | 0.561±0.035 (MOON) |

All improvements statistically significant: Wilcoxon signed-rank p<0.01, Cohen's d ≥ 1.2.

---

## Repository Structure

```
fmbm-net/
├── paper/
│   ├── FMBM_Net_IEEE.tex          # Full LaTeX source (IEEEtran)
│   ├── refs.bib                   # BibTeX references (34 entries)
│   ├── FMBM_Net_IEEE_Paper.pdf    # Compiled manuscript
│   ├── fig_arch.png               # Fig 1: System architecture block diagram
│   ├── fig1_flcat_attention.png   # Fig 2: FLCAT cross-modal attention maps
│   ├── fig2_convergence.png       # Fig 3: Federated convergence + privacy-utility
│   ├── fig3_mri_seg.png           # Fig 4: MRI segmentation pipeline
│   ├── fig4_comparison.png        # Fig 6: Task-wise AUROC + sample efficiency
│   ├── fig5_xai_governance.png    # Fig 8: XAI saliency + RSGC distribution
│   ├── fig6_complexity.png        # Fig 5: Latency, GFLOPs, memory scalability
│   └── fig7_hetero_xai.png        # Fig 7: Heterogeneity sweep + XAI metrics
├── src/
│   ├── generate_figures.py        # Reproducible figure generation script
│   └── requirements.txt           # Python dependencies
├── reproduce.sh                   # One-command PDF compilation
├── LICENSE
└── README.md
```

---

## Reproducibility

### Requirements

```bash
pip install -r src/requirements.txt
# LaTeX: texlive-publishers (for IEEEtran.cls)
sudo apt-get install texlive-publishers texlive-science
```

### Compile PDF

```bash
bash reproduce.sh
# Output: paper/FMBM_Net_IEEE_Paper.pdf
```

### Regenerate Figures

```bash
python src/generate_figures.py
```

### Datasets

| Dataset | Access | Link |
|---------|--------|------|
| CADD v1.7 | Public | https://cadd.gs.washington.edu/download |
| MSD Pancreas-CT | Public | http://medicaldecathlon.com |
| MIMIC-IV v2.2 | PhysioNet DUA | https://physionet.org/content/mimiciv/2.2/ |
| TCGA-LUNG (LUAD/LUSC) | GDC Portal (dbGaP phs000178) | https://portal.gdc.cancer.gov |

Pre-processed splits and feature files are provided in [`data/splits/`](data/splits/) (CADD, MSD, MIMIC-IV). TCGA-LUNG splits use TSS codes from sample barcodes (columns 6–7); see Section V-B of the paper.

---

## Citation

```bibtex
@article{tanala2026fmbmnet,
  author  = {Tanala, Subrahmanyam and Salkuti, Surender Reddy},
  title   = {{FMBM-Net}: A Federated Multi-Scale Bio-Multimodal Network with
             Latent Cross-Attention Transformer for Privacy-Preserving Biomedical {AI}},
  journal = {IEEE Trans.\ Biomed.\ Eng.},
  volume  = {73},
  number  = {6},
  year    = {2026},
  note    = {DOI pending}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Subrahmanyam Tanala** — tanala.subrahmanyam@anits.edu.in — ANITS, Visakhapatnam, India  
- **Surender Reddy Salkuti** — surender@wsu.ac.kr — Woosong University, South Korea


---

## Submission Status

> **Manuscript version:** Submission draft (IEEE `draftclsnofoot`, single-column, 13 pages)  
> **Target journal:** IEEE Transactions on Biomedical Engineering (TBME)  
> **Status:** Under preparation for submission, May 2026  
> **Anonymous review link:** https://anonymous.4open.science/r/fmbm-net-review

### Submission Checklist
- [x] 10-seed statistical analysis with bootstrap 95% CI
- [x] Wilcoxon signed-rank tests + Cohen's *d* effect sizes  
- [x] TCGA-LUNG dataset construction fully documented (§V-B)
- [x] Complete FLCAT mathematical specification (Eqs. 3–8)
- [x] Data heterogeneity + client dropout robustness experiments
- [x] Quantitative XAI evaluation (GradCAM++ IoU + expert agreement)
- [x] DP gap theoretically motivated (cross-modal SNR analysis)
- [x] Submission header (`draftclsnofoot`) — no implied publication date
- [x] Reproducible figures + compile script in this repository
- [ ] External validation cohort (planned: two-site clinical study)
- [ ] Real multi-site deployment (planned: future work)
