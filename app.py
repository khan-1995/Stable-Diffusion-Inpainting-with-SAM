import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor , sam_model_registry

sam_model_checkpoint = "weights/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_model_checkpoint)

sam.to(device)

predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    # torch_dtype = torch.float16,
    )

pipe = pipe.to(torch_device=device) # type: ignore

select_pixels = []
with gr.Blocks() as demo :
    with gr.Row():
        input_image = gr.Image(label="Input")
        mask_image = gr.Image(label="Mask")
        output_image = gr.Image(label="Ouput")

    with gr.Blocks():
        text = gr.Textbox(lines=1,label="Prompt")    

    with gr.Row():
        submit = gr.Button("submit")

    def inpaint_img(img,mask,prompt):
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        img = img.resize((512,512))
        mask = mask.resize((512,512))

        output = pipe(prompt=prompt,image=img,mask_image=mask).images[0] # type: ignore
        return output
    
    def generate_mask(input_image,event : gr.SelectData):
        select_pixels.append(event.index)
        predictor.set_image(input_image)

        input_points = np.array(select_pixels)
        input_label = np.ones(input_points.shape[0])
        mask, _,_ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=False
        )

        mask = Image.fromarray(mask[0,:,:])

        return mask

    input_image.select(generate_mask,[input_image],[mask_image])

    submit.click(inpaint_img,inputs=[input_image,mask_image,text],outputs=[output_image])

# print("Hello world")

if __name__== "__main__":
    demo.launch()
