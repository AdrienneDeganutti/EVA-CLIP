<div align="center">

<h2><a href="https://arxiv.org/abs/2402.04252">EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters</a></h2>

[Quan Sun](https://github.com/Quan-Sun)<sup>1*</sup>, [Jinsheng Wang](https://github.com/Wolfwjs/)<sup>1*</sup>, [Qiying Yu](https://yqy2001.github.io)<sup>1,2*</sup>, [Yufeng Cui](https://scholar.google.com/citations?hl=en&user=5Ydha2EAAAAJ)<sup>1</sup>, [Fan Zhang](https://scholar.google.com/citations?user=VsJ39HMAAAAJ)<sup>1</sup>, [Xiaosong Zhang](https://zhangxiaosong18.github.io)<sup>1</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>1</sup>
 
<sup>1</sup> [BAAI](https://www.baai.ac.cn/english.html), <sup>2</sup> [THU](https://air.tsinghua.edu.cn) <br><sup>*</sup> equal contribution

</div>

This repository has been copied from https://github.com/baaivision/EVA
## BibTeX & Citation

```
@article{EVA-CLIP-18B,
  title={EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters}, 
  author={Quan Sun and Jinsheng Wang and Qiying Yu and Yufeng Cui and Fan Zhang and Xiaosong Zhang and Xinlong Wang},
  journal={arXiv preprint arXiv:2402.04252},
  year={2023}
}
```

This repository has been created as a personal archive for running EVA-CLIP-18B for feature extraction.

</div>

## Model Card

### EVA-CLIP-18B
| model name | image enc. init. ckpt | text enc. init. ckpt | total #params | training data  |  training batch size |  gpus for training | img. cls. avg. acc. | video cls. avg. acc. | retrieval MR | hf weight | pytorch weight |
|:-----|:-----|:-----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `EVA-CLIP-18B` | `EVA_18B_psz14` | `EVA02_CLIP_E_psz14_plus_s9B` | 18.1B | Merged-2B+ | 108K | 360 A100(40GB) | **80.7** | **75.0** | **87.8**| [🤗 HF](https://huggingface.co/BAAI/EVA-CLIP-18B) | [PT](https://huggingface.co/BAAI/EVA-CLIP-18B/resolve/main/EVA_CLIP_18B_psz14_s6B.fp16.pt) (`36.7GB`) |

</div>
