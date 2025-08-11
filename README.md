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

After cloning the repository, you can set up the environment needed to run the scripts successfully by following the instructions below. You can create an environment using `conda` with all the required packages by running:
```sh
conda env create -f environment.yml
```
This command creates a new conda environment and installs the packages as specified in the `environment.yml` file in about 50 seconds. After installing the dependencies, make sure to activate the newly created conda environment with:
```sh
conda activate surd
```

## Citation

If you use our method in your research or software, please cite the following paper:

```bibtex
@misc{states2025,
author={Mart{\'\i}nez-S{\'a}nchez, {\'A}lvaro and Lozano-Dur{\'a}n, Adri{\'a}n},
title={Observational causality by states and interaction type for scientific discovery},
archivePrefix={arXiv},
primaryClass={physics.data-an},
eprint={2505.10878},
year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
