# Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors

<div align="center" margin-bottom="3em">
    <!-- <a href="https://arxiv.org/abs/2312.02010" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
  -->
<a href="https://lavi-lab.github.io/VG-LLM/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Website-VG_LLM-blue.svg" height="20" />
</a>

</div>
&nbsp

<div align="center" margin-bottom="3em">
<a target="_blank" href="https://github.com/zd11024">Duo Zheng<sup>*</sup></a>,
<a target="_blank" href="https://sega-hsj.github.io/">Shijia Huang<sup>*</sup></a>, 
<a target="_blank" href="https://github.com/lyy1994">Yanyang Li</a> and 
<a target="_blank" href="https://lwwangcse.github.io/">Liwei Wang<sup>&ddagger;</sup></a>

<sup>*</sup>Equal contribution.
<sup>&ddagger;</sup> Corresponding author.

<strong>
The Chinese University of Hong Kong<br>
</strong>
</div>
&nbsp;

Previous research has investigated the application of Multimodal Large Language
Models (MLLMs) in understanding 3D scenes by interpreting them as videos.
These approaches generally depend on comprehensive 3D data inputs, such as
point clouds or reconstructed Bird’s-Eye View (BEV) maps. In our research, we
advance this field by enhancing the capability of MLLMs to understand and reason
in 3D spaces directly from video data, without the need for additional 3D input.


## ✨Architecture Overview

VG-LLM integrates a 3D visual geometry encoder (based on [VGGT](https://github.com/facebookresearch/vggt)) with a conventional 2D visual encoder.
1.  Input video frames are processed by both encoders. The 2D encoder extracts semantic-aware visual features from individual images. The 3D visual geometry encoder processes the sequence to produce globally geometry-aware visual features, capturing inter-frame correspondences.
2.  Features from both encoders are fused at the patch level.
3.  These fused, geometry-augmented visual features, along with text embeddings of a question, are fed into an MLLM backbone ([Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)) to generate a response.

<p align="center">
    <img src="assets/model.png" width="90%"><br>
    <figcaption align="center">The architecture of our VG LLM.</figcaption>
</p>


## 🚀Main Results Highlights
* **3D Visual Grounding (ScanRefer):** Our model can directly predict the 3D oriented bounding box at the camera's coordinate system without any 3D data input, and obtains 34.1\% Acc@0.25. 
* **3D Dense Captioning (Scan2Cap):** Achieves competitive results (e.g., 74.1 CIDEr@0.5 on Scan2Cap) without explicit 3D scene data input.
* **3D Video Object Detection (curated from EmbodiedScan):** Shows significant recall improvement (e.g., +19.3 F1 for common classes in 6-frame setting) by better handling egocentric-allocentric transformations.
* **Spatial Reasoning (VSI-Bench):** Our 4B model achieves an average score of 46.1%, surpassing Gemini-1.5-Pro.
* **Generic Multimodal Benchmarks (CVBench, VideoMME, BLINK, TempCompass, NextQA)**: Enhancing spatial understanding incurs negligible loss on general multimodal performance.

<details>
<summary>Visualization results of VG LLM in 3D visual grounding tasks.</summary>
<p align="center">
    <img src="assets/vg.png" width="90%"><br>
    <figcaption align="center">Our model can identify the frame index in which the targer object appears in a video stream, as well as its oriented 3D bounding box in the current frame. In this illustration, we show the video, the model's predicted oriented 3D bounding boxes (highlighted in green), and the ground truth 3D oriented bounding boxes (highlighted in blue). As shown in the figure, our model can effectively identify spatial relationships such as "far away," "opposite," and "next to" based on the video input.</figcaption>
</p>
</details>

<details>
<summary>Visualization results of VG LLM in 3D video object detection.</summary>
<p align="center">
    <img src="assets/det.png" width="90%"><br>
    <figcaption align="center">Our model can identify all objects througtout a video and output their oriented 3D bounding boxes in the unified coordinate system. As shown in the figure, our model can effectively detect objects of different granularities, including sink, bin, telephone, etc., and output their bounding boxes in a unified coordinate system.</figcaption>
</p>
</details>

## 📋Todo List

- [ ] Release the model weights.
- [ ] Release the inference demo.
- [ ] Release the evaluation code.
- [ ] Release the preprocessing data and scripts.
- [ ] Release the training scripts for VG LLM.


## ⚙️Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lavi-lab/VG-LLM](https://github.com/lavi-lab/VG-LLM)
    cd VG-LLM
    ```

2.  **Create a Conda environment and install dependencies:**
    We recommend using Python 3.10.
    ```bash
    conda create -n vgllm python=3.10
    conda activate vgllm
    pip install -e .
    ```


## 📊Datasets

VG-LLM is trained and evaluated on a variety of datasets:

* **3D Scene Understanding:**
    * **3D Visual Grounding:** [ScanRefer](https://github.com/daveredrum/ScanRefer), with 24 uniformly sampling frames per scene.
    * **3D Dense Captioning:** [Scan2Cap](https://github.com/daveredrum/Scan2Cap), using Mask3D-detected object proposals extracted from [LEO](https://github.com/embodied-generalist/embodied-generalist). We uniformly sample 16 frames for each scene.
    * **3D Video Object Detection:** Curated from [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan), with consecutive frames sampled at 1 FPS.
* **Spatial Reasoning Instruction Tuning:**
    * [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M): We used a subset of ~234K samples (3% of original). Data prep follows official codebase, navigation type discarded.
    * [LLaVA-Video-178K (LLaVA-Hound split)](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): We used a subset of ~63K samples (25% of original). Frames sampled at 2 FPS, 4-8 frames total.
    * Evaluation Benchmarks: We adopt [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME), [TempCompass](https://huggingface.co/datasets/lmms-lab/TempCompass), [NextQA](https://huggingface.co/datasets/lmms-lab/NExTQA) for evaluation.


## Training

We train two models separately for 3D scene understanding and spatial reasoning tasks.

* **Hardware:** Experiments were conducted on 8x H100 80G GPUs.
* **Settings:** Trained for one epoch, Adam optimizer, batch size 16, warmup ratio 0.03, learning rate 5e-6.
* **Frozen Components:** The MLLM’s visual encoder, the 3D geometry encoder, and the multimodal connector are frozen.
* **Training duration:** ~8 hours for 3D scene understanding, ~12 hours for spatial reasoning.

<!-- The training scripts will be released soon. -->



## Acknowledgements


* This work is built upon excellent previous research, including [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [VGGT](https://github.com/facebookresearch/vggt), [SPAR-7M](https://github.com/fudan-zvg/spar), [LLaVA-Video-178K](https://github.com/LLaVA-VL/LLaVA-NeXT), and various 3D datasets like [ScanNet](https://github.com/ScanNet/ScanNet), [ScanRefer](https://github.com/daveredrum/ScanRefer), [Scan2Cap](https://github.com/daveredrum/Scan2Cap), [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan).
* We thank the developers of [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their evaluation framework.
