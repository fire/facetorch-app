---
title: facetorch - Face Analysis
emoji: 🥹
colorFrom: red
colorTo: black
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
task_categories:
- face-detection
- face-representation
- face-verification
- facial-expression-recognition
- deepfake-detection
- face-alignment
- 3D-face-alignment

---


# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-42.png "facetorch logo") facetorch
![build](https://github.com/tomas-gajarsky/facetorch/actions/workflows/build.yml/badge.svg?branch=main)
![lint](https://github.com/tomas-gajarsky/facetorch/actions/workflows/lint.yml/badge.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/facetorch)](https://anaconda.org/conda-forge/facetorch)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


[Documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html), [Docker Hub](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch) [(GPU)](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch-gpu)

Facetorch is a Python library that can detect faces and analyze facial features using deep neural networks. The goal is to gather open-sourced face analysis models from the community, optimize them for performance using TorchScript and combine them to create a face analysis tool that one can:

1. configure using [Hydra](https://hydra.cc/docs/intro/) (OmegaConf)
2. reproduce with [conda-lock](https://github.com/conda-incubator/conda-lock) and [Docker](https://docs.docker.com/get-docker/)
3. accelerate on CPU and GPU with [TorchScript](https://pytorch.org/docs/stable/jit.html)
4. extend by uploading a model file to Google Drive and adding a config YAML file to the repository

Please, use the library responsibly with caution and follow the 
[ethics guidelines for Trustworthy AI from European Commission](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html). 
The models are not perfect and may be biased.