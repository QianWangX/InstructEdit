# InstructEdit Implementation

This is the implementation of **InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions**.<br> 
This code base is modified based on the repo [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). 

### Set up environment
Please set the environment variable manually as follows if you want to build a local GPU environment:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
python -m pip install -e GroundingDINO
```

Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

Install Tag2Text:

```bash
git submodule update --init --recursive
cd Tag2Text && pip install -r requirements.txt
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

More details can be found in [install segment anything](https://github.com/facebookresearch/segment-anything#installation) and [install GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install) and [install OSX](https://github.com/IDEA-Research/OSX)

After setting up the environment, please also specify the openai key in `chatgpt.py`.

### Playground
We provide a notebook (`grounded_sam_instructedit_demo.ipynb`), a python script (`grounded_sam_instructedit_demo.py`) and a gradio app (`gradio_intructedit.py`) for you to play around with.
