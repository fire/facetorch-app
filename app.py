import json
import gradio as gr
import torchvision
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf


cfg = OmegaConf.load("config.merged.yml")
analyzer = FaceAnalyzer(cfg.analyzer)


def inference(path_image):
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
    
    base_emb = response.faces[0].preds["verify"].logits
    sim_dict = {face.indx: cosine_similarity(base_emb, face.preds["verify"].logits, dim=0).item() for face in response.faces}
    sim_dict_sort = dict(sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True))
    sim_dict_sort_str = str(sim_dict_sort)
    
    out_tuple = (pil_image, fer_dict_str, deepfake_dict_str, sim_dict_sort_str, response_str)
    return out_tuple


title = "facetorch"
description = "Demo of facetorch, a Python library that can detect faces and analyze facial features using deep neural networks. The goal is to gather open-sourced face analysis models from the community and optimize them for performance using TorchScrip. Try selecting one of the example images or upload your own."
article = "<p style='text-align: center'><a href='https://github.com/tomas-gajarsky/facetorch' target='_blank'>facetorch GitHub repository</a></p>"

demo=gr.Interface(
    inference,
    [gr.inputs.Image(label="Input", type="filepath")],
    [gr.outputs.Image(type="pil", label="Output"),
     gr.outputs.Textbox(label="Facial Expression Recognition"),
     gr.outputs.Textbox(label="DeepFake Detection"),
     gr.outputs.Textbox(label="Cosine similarity on Face Verification Embeddings"),
     gr.outputs.Textbox(label="Response")],
    title=title,
    description=description,
    article=article,
    examples=[["./test.jpg"], ["./test2.jpg"], ["./test3.jpg"], ["./test4.jpg"]],
)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
