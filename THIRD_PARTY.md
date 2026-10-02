# Component sources

The ESD and UCE baseline engines are the DiCM experiment implementations. Their prompts, update scope, seeds, and optimization settings are recorded in each engine and its output `protocol.json`.

- ESD: [paper](https://arxiv.org/abs/2303.07345), [authors' project](https://erasing.baulab.info/).
- UCE: [paper](https://arxiv.org/abs/2308.14761), [authors' project](https://unified.baulab.info/).
- Diffusion pipelines: [Hugging Face Diffusers](https://github.com/huggingface/diffusers).
- SD1.5: [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- SDXL: [model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- CLIP: [OpenAI repository](https://github.com/openai/CLIP), [checkpoint](https://huggingface.co/openai/clip-vit-large-patch14).
- DINOv2: [Meta repository](https://github.com/facebookresearch/dinov2), [checkpoint](https://huggingface.co/facebook/dinov2-large).
- Object detector: [torchvision](https://github.com/pytorch/vision), Faster R-CNN ResNet50 FPN V2 default weights.
- Mixed-domain detector: [NudeNet](https://github.com/notAI-tech/NudeNet).
- Caption annotations: [COCO](https://cocodataset.org/).

Model weights and datasets are downloaded from their providers and retain their respective licenses. The repository's MIT license covers its code.
