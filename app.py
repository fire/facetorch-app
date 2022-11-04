import json
import operator
import gradio as gr
import torchvision
from typing import Tuple, Dict
from facetorch import FaceAnalyzer
from facetorch.datastruct import ImageData
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity


cfg = OmegaConf.load("config.merged.yml")
analyzer = FaceAnalyzer(cfg.analyzer)

def get_sim_dict_str(response: ImageData, pred_name: str = "verify", index: int = 0)-> str:
    base_emb = response.faces[index].preds[pred_name].logits
    sim_dict = {face.indx: cosine_similarity(base_emb, face.preds[pred_name].logits, dim=0).item() for face in response.faces}
    sim_dict_sort = dict(sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
    sim_dict_sort_str = str(sim_dict_sort)
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
    
    sim_dict_str_embed = get_sim_dict_str(response, pred_name="embed", index=0)
    sim_dict_str_verify = get_sim_dict_str(response, pred_name="verify", index=0)
    
    out_tuple = (pil_image, fer_dict_str, deepfake_dict_str, sim_dict_str_embed, sim_dict_str_verify, response_str)
    return out_tuple


title = "facetorch"
description = "Demo of facetorch, a Python library that can detect faces and analyze facial features using deep neural networks. The goal is to gather open-sourced face analysis models from the community and optimize them for performance using TorchScript. Try selecting one of the example images or upload your own."
article = "<p style='text-align: center'><a href='https://github.com/tomas-gajarsky/facetorch' target='_blank'>facetorch GitHub repository</a></p>"

demo=gr.Interface(
    inference,
    [gr.inputs.Image(label="Input", type="filepath")],
    [gr.outputs.Image(type="pil", label="Output"),
     gr.outputs.Textbox(label="Facial Expression Recognition"),
     gr.outputs.Textbox(label="DeepFake Detection"),
     gr.outputs.Textbox(label="Cosine similarity on Face Representation Embeddings"),
     gr.outputs.Textbox(label="Cosine similarity on Face Verification Embeddings"),
     gr.outputs.Textbox(label="Response")],
    title=title,
    description=description,
    article=article,
    examples=[["./test5.jpg"], ["./test.jpg"], ["./test4.jpg"], ["./test2.jpg"], ["./test8.jpg"], ["./test6.jpg"], ["./test3.jpg"]],
)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
