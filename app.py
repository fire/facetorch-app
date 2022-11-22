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

def gen_sim_dict_str(response: ImageData, pred_name: str = "verify", index: int = 0)-> str:     
    if len(response.faces) > 0:
        base_emb = response.faces[index].preds[pred_name].logits
        sim_dict = {face.indx: cosine_similarity(base_emb, face.preds[pred_name].logits, dim=0).item() for face in response.faces}
        sim_dict_sort = dict(sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
        sim_dict_sort_str = str(sim_dict_sort)
    else:
        sim_dict_sort_str = ""
        
    return sim_dict_sort_str


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
    
    fer_dict_str = str({face.indx: face.preds["fer"].label for face in response.faces})
    deepfake_dict_str = str({face.indx: face.preds["deepfake"].label for face in response.faces})
    response_str = str(response)
    
    sim_dict_str_embed = gen_sim_dict_str(response, pred_name="embed", index=0)
    sim_dict_str_verify = gen_sim_dict_str(response, pred_name="verify", index=0)
    
    os.remove(path_image)
    
    out_tuple = (pil_image, fer_dict_str, deepfake_dict_str, sim_dict_str_embed, sim_dict_str_verify, response_str)
    return out_tuple


title = "Face Analysis"
description = "Demo of facetorch, a face analysis Python library that implements open-source pre-trained neural networks for face detection, representation learning, verification, expression recognition, deepfake detection, and 3D alignment. Try selecting one of the example images or upload your own. This work would not be possible without the researchers and engineers who trained the models (sources and credits can be found in the facetorch repository)."
article = "<p style='text-align: center'><a href='https://github.com/tomas-gajarsky/facetorch' style='text-align:center' target='_blank'>facetorch GitHub repository</a></p>"

demo=gr.Interface(
    inference,
    [gr.Image(label="Input", type="filepath")],
    [gr.Image(type="pil", label="Face Detection and 3D Landmarks"),
     gr.Textbox(label="Facial Expression Recognition"),
     gr.Textbox(label="DeepFake Detection"),
     gr.Textbox(label="Cosine similarity of Face Representation Embeddings"),
     gr.Textbox(label="Cosine similarity of Face Verification Embeddings"),
     gr.Textbox(label="Response")],
    title=title,
    description=description,
    article=article,
    examples=[["./test5.jpg"], ["./test.jpg"], ["./test4.jpg"], ["./test8.jpg"], ["./test6.jpg"], ["./test3.jpg"], ["./test10.jpg"]],
)
demo.queue(concurrency_count=3, api_open=False)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
