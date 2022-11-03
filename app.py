import gradio as gr
import torchvision
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf


cfg = OmegaConf.load("config.merged.yaml")
analyzer = FaceAnalyzer(cfg.analyzer)


def inference(path_image):
    response = analyzer.run(
        path_image=path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )
    pil_image = torchvision.transforms.functional.to_pil_image(response.img)
    return pil_image


def main():
    title = "facetorch"
    description = "Demo for facetorch, a Python library that can detect faces and analyze facial features using deep neural networks. The goal is to gather open sourced face analysis models from the community and optimize them for performance using TorchScrip. Try selecting one of the example images or upload your own."
    article = "<p style='text-align: center'><a href='https://github.com/tomas-gajarsky/facetorch' target='_blank'>Github Repo</a></p>"

    gr.Interface(
        inference,
        [gr.inputs.Image(label="Input")],
        gr.outputs.Image(type="pil", label="Output"),
        title=title,
        description=description,
        article=article,
        examples=[["./data/input/test.jpg"], ["./data/input/test2.jpg"]],
    ).launch(debug=True)


if __name__ == "__main__":
    main()
