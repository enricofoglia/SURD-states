# Decomposition of causality by states and interaction type

_A Python repository for decomposing synergistic, unique, and redundant causalities by states._

## Introduction
We introduce a state-aware causal inference method that quantifies causality in terms of information gain about
future states. The effectiveness of the proposed approach stems from two key features: its ability
to characterize causal influence as a function of the system state, and its capacity to distinguish
both redundant and synergistic effects among variables. The formulation is non-intrusive and requires only pairs of past and future events, facilitating its application in both computational and experimental investigations. The method also identifies the amount of causality that remains unaccounted for due to unobserved variables. The approach can be used to detect causal relationships in systems with multiple variables, dependencies at different time lags, and instantaneous links.

<img width="4499" height="1031" alt="fig_method_states_v1" src="https://github.com/user-attachments/assets/6b04b6e7-78f0-4751-bd65-e6347ae48a6c" />

## System requirements

The method is designed to operate efficiently on standard computing systems. However, the computational demands increase with the complexity of the probability density functions being estimated. To ensure optimal performance, we recommend a minimum of 16 GB of RAM and a quad-core processor with a clock speed of at least 3.3 GHz per core. The performance metrics provided in this repository are based on tests conducted on macOS with an ARM64 architecture and 16 GB of RAM, and on Linux systems running Red Hat version 8.8-0.8. These configurations have demonstrated sufficient performance for the operations utilized by SURD. Users should consider equivalent or superior specifications to achieve similar performance.

## Getting started

After cloning the repository, you can set up the environment needed to run the scripts successfully by following the instructions below. The python environment is managed using [uv](https://github.com/astral-sh/uv.git), which is a convenient python package and project management tool built in rust. To get started, make sure to have uv installed on your machine, then navigate to the `examples` directory and simply run:

```bash
uv run E01_benchmark_source.py
```

uv will take care of installing all necessary packages and use the correct python version. The first time the code is run, a `.venv` directory is created, and uv will automatically use it as a virtual environment whenever it is invoked. 
The examples are formatted using [marimo](https://docs.marimo.io/), a modern python notebook format: other than being run as stand alone python packages, the examples can be opened in a browser as if they were jupyter notebooks running:

```bash
uv run marimo edit E01_benchmark_source.py
```

## Citation

If you use our method in your research or software, please cite the following paper:

```bibtex
@misc{states2025,
title={Observational causality by states and interaction type for scientific discovery},
author={Mart{\'\i}nez-S{\'a}nchez, {\'A}lvaro and Lozano-Dur{\'a}n, Adri{\'a}n},
archivePrefix={arXiv},
primaryClass={physics.data-an},
eprint={2505.10878},
year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
