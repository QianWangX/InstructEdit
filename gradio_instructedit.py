import argparse
import os
import copy
import gradio as gr

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from stable_diffusion_masked_diffedit import StableDiffusionMaskedDiffeditPipeline
from diffusers import DDIMScheduler

# chatgpt
from chatgpt import call_chatgpt

# blip2
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)
    
def adjust_encoding_ratio(instructions, feedbacks, cur_encoding_ratio):
    full_instructions = []
    for instruction in instructions:
        full_instruction = '{}. The current encoding ratio is {}.'.format(instruction, cur_encoding_ratio)
        full_instructions.append(full_instruction)
        
    encoding_ratio = call_chatgpt(full_instructions, feedbacks=feedbacks)
    
    return encoding_ratio


def maskedit(image_path, user_instructions, is_blip2_description, encoding_ratio, mask_mode):


    # cfg
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
    grounded_checkpoint = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'  # change the path of the model
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    # image_path = args.input_image
    # user_instructions = args.user_instructions
    # is_blip2_description = args.is_blip2_description
    box_threshold = 0.3
    text_threshold = 0.25
    # encoding_ratio = args.encoding_ratio
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load image
    image_pil, image = load_image(image_path)
    
    if is_blip2_description:
        blip_processor = AutoProcessor.from_pretrained('Salesforce/blip2-flan-t5-xl')
        blip_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)
        
        blip_prompt_1 = "Is this a photo, a painting, a drawing, or other kind of arts?"
        blip_inputs = blip_processor(image_pil, text=blip_prompt_1, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**blip_inputs, max_new_tokens=20)
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        blip_prompt_2 = "{} of".format(generated_text.capitalize())
        blip_inputs = blip_processor(image_pil, text=blip_prompt_2, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**blip_inputs, max_new_tokens=20)
        description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        description = "{} {}.".format(blip_prompt_2, description)
        descriptions = [description]    
        print('Blip2 description: {}'.format(description))
    else:
        descriptions = None
    instructions = [user_instructions]
    det_prompt, prompt_inversion, prompt, _ = call_chatgpt(instructions, descriptions=descriptions)
    print('Segmentation prompt: {}'.format(det_prompt))
    print('Inverted prompt: {}'.format(prompt_inversion))
    print('Editing prompt: {}'.format(prompt))


    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)


    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    # masks: [1, 1, 512, 512]

    # Masked diffedit pipeline
    if mask_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
        
    mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)

    scheduler =DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                        set_alpha_to_one=False)
    pipe = StableDiffusionMaskedDiffeditPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler)
    pipe.to(device)

    image_pil = image_pil.resize((512, 512))
    mask_pil = mask_pil.resize((512, 512))
    image = pipe(prompt, image=image_pil, mask_image=mask_pil, prompt_inversion=prompt_inversion, encoding_ratio=encoding_ratio).images[0]
    image = image.resize(size)
    
    return image


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""# Instruct Diffedit: to edit an image based on a user instruction. 
                No mask or prompting is needed.""")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='filepath', label='Input image', value='./example_input/cat_dog.jpg')
            user_instructions = gr.Textbox(lines=3, label='User instructions', value='Change the cat to a fox')
            is_blip2_description = gr.Checkbox(label='Use Blip2 description')
            encoding_ratio = gr.Slider(label='Encoding ratio', minimum=0.0, maximum=1.0, step=0.05, value=0.5)
            mask_mode = gr.Dropdown(label='Mask mode', choices=['merge', 'max'], value='max')
        
        with gr.Column():    
            output_image = gr.Image(type='pil', label='Output image')           
            edit_btn = gr.Button('Edit')
            edit_btn.click(fn=maskedit, inputs=[image, user_instructions, is_blip2_description, encoding_ratio, mask_mode], outputs=[output_image], api_name='maskedit')
            
demo.queue().launch(share=True)
