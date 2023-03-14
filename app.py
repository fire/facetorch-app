import os
import json
import argparse
import operator
import gradio as gr
import torchvision
from typing import Tuple, Dict
from facetorch import FaceAnalyzer
from facetorch.datastruct import ImageData
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity

parser = argparse.ArgumentParser(description="App")
parser.add_argument(
    "--path-conf",
    type=str,
    default="config.merged.yml",
    help="Path to the config file",
)

args = parser.parse_args()

cfg = OmegaConf.load(args.path_conf)
analyzer = FaceAnalyzer(cfg.analyzer)

def inference(path_image: str) -> Tuple:
    response = analyzer.run(
        path_image=path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=None,
    )
    
    pil_image = torchvision.transforms.functional.to_pil_image(response.img)
    
    au_dict_str = str({face.indx: face.preds["au"].other["multi"] for face in response.faces})
    response_str = str(response)
        
    os.remove(path_image)
    
    out_tuple = (pil_image, au_dict_str, response_str)
    return out_tuple


title = "Face Analysis"
description = "Demo of facetorch, a face analysis Python library that implements open-source pre-trained neural networks for face detection, representation learning, verification, expression recognition, action unit detection, deepfake detection, and 3D alignment. Try selecting one of the example images or upload your own. Feel free to duplicate this space and run it faster on a GPU instance. This work would not be possible without the researchers and engineers who trained the models (sources and credits can be found in the facetorch repository)."
article = "<p style='text-align: center'><a href='https://github.com/tomas-gajarsky/facetorch' style='text-align:center' target='_blank'>facetorch GitHub repository</a></p>"

demo=gr.Interface(
    inference,
    [gr.Image(label="Input", type="filepath")],
    [gr.Image(type="pil", label="Face Detection and 3D Landmarks"),
     gr.Textbox(label="Facial Action Unit Detection"),
     gr.Textbox(label="Response")],
    title=title,
    description=description,
    article=article,
    examples=[["./test5.jpg"], ["./test.jpg"], ["./test4.jpg"], ["./test8.jpg"], ["./test6.jpg"], ["./test3.jpg"], ["./test10.jpg"]],
)
demo.queue(concurrency_count=1, api_open=False)
demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
