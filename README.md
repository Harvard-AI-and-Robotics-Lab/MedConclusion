# MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts

<a href='https://arxiv.org/abs/2604.06505'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/datasets/harvardairobotics/MedConclusion'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MedConclusion_Dataset-blue'></a> <a href='https://huggingface.co/datasets/harvardairobotics/MedConclusion-Compact'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MedConclusion_Compact_Dataset-blue'></a>

## Datasets

The dataset is available on Hugging Face in two versions:
- [**MedConclusion (Full 5.7M Dataset)**](https://huggingface.co/datasets/harvardairobotics/MedConclusion)
- [**MedConclusion-Compact (Fast Prototyping Dataset)**](https://huggingface.co/datasets/harvardairobotics/MedConclusion-Compact)

## Evaluation Scripts

This repository contains the scripts used for generating and evaluating the conclusions:
- `generate.py`: Script to generate conclusions from abstract inputs using various LLMs via API.
- `evaluate.py`: Script to execute evaluation metrics (Rule-based scores like ROUGE/BLEU, Perplexity, and LLM-as-a-judge).
### Citation

If you find this work useful, please cite:

```bibtex
@article{li2026medconclusion,
  title={MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts},
  author={Li, Weiyue and Qian, Ruizhi and Li, Yi and Li, Yongce and Long, Yunfan and Cai, Jiahui and Luo, Yan and Wang, Mengyu},
  journal={arXiv preprint arXiv:2604.06505},
  year={2026}
}
```
