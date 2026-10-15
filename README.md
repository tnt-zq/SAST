# Semantic-Aware Guided Image Style Transfer with Pretrained Diffusion Models

The proposed framework performs training-free image style transfer based on a pretrained Stable Diffusion model. It integrates Semantic-Aware Attention Fusion (SAAF), Residual Style-Aware Latent Initialization (RSLI), and LAB color alignment to balance style fidelity, content preservation, and color consistency.

## 1. Environment
Install the dependencies using the provided requirements file:

```bash
conda create -n semantic-style python=3.10 -y
conda activate semantic-style

pip install -r requirements.txt
```
## 2.Download Pretrained Weights

### Stable Diffusion v1.5

Download the Stable Diffusion v1.5 weights from the [Stable Diffusion v1.5 repository on Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

### Segment Anything Model (SAM)

Download the SAM ViT-B checkpoint from the [official Segment Anything repository](https://github.com/facebookresearch/segment-anything).

Checkpoint: `sam_vit_b_01ec64.pth`

Place the checkpoint at:

```text
models/sam_vit_b_01ec64.pth
```

### CLIP

Download CLIP ViT-L/14 from [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14).

Place the model files in:

```text
models/clip/clip-vit-large-patch14/
```

Alternatively, configure the model-loading code to download the weights automatically.

## Run Inference

For running the proposed method, use:

```bash
python run.py --cnt <content_img_dir> --sty <style_img_dir> 
```
