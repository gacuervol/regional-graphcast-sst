<p align="center">
  <img src="https://img.shields.io/badge/RESEARCH%20PRODUCT-EXPERIMENTAL-5e81ac?style=for-the-badge" alt="Header Banner">
</p>

<h1 align="center">Deep learning weather models for subregional ocean forecasting: A case study on the Canary current upwelling system</h1>

<p align="center">
  <strong>
    <a href="https://www.linkedin.com/in/giovanny-alejandro-cuervo-londo%C3%B1o-79a057158/">Giovanny A. Cuervo-Londoño</a>, 
    <a href="https://www.linkedin.com/in/javier-s%C3%A1nchez-p%C3%A9rez-22142513/">Javier Sánchez</a>, 
    <a href="https://www.linkedin.com/in/%C3%A1ngel-rodr%C3%ADguez-santana-50525936/">Ángel Rodríguez-Santana</a>.
  </strong>
  <br>
  <small>
    Instituto Universitario de Investigación en Acuicultura Sostenible y Ecosistemas Marinos (ECOAQUA), 
    Instituto Universitario de Cibernética, Empresas y Sociedad (IUCES), 
    Universidad de Las Palmas de Gran Canaria, 35017, Spain.
  </small>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" alt="Divider">
</p>

<p align="center">
  <a href="#academic-anchor">Academic Anchor</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#evolution-roadmap">Roadmap</a> •
  <a href="#governance">Governance</a>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.ocemod.2026.102782"><img src="https://img.shields.io/badge/DOI-10.1016/j.ocemod.2026.102782-5e81ac?style=flat-square&logo=doi&logoColor=white" alt="DOI"></a>
  <a href="https://arxiv.org/abs/2505.24429"><img src="https://img.shields.io/badge/arXiv-2505.24429-b48ead?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <!-- <img src="https://img.shields.io/github/v/release/usuario/repo?style=flat-square&logo=git" alt="Version"> -->
  <!-- <a href="https://github.com/user/repo/actions"><img src="https://img.shields.io/badge/build-passing-a3be8c?style=flat-square&logo=githubactions&logoColor=white" alt="Build"></a> -->
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-4c566a?style=flat-square" alt="License"></a>
</p>

## 📢 Latest Updates
- **June 6, 2025**: Preprint released on [arxiv link](https://arxiv.org/abs/2505.24429).
- **June 18, 2026**: Ocean Modelling paper released on [Elsevier link](https://doi.org/10.1016/j.ocemod.2026.102782). 🔥🔥
- **June 30, 2026**: Initial public codebase release. *Note: This represents my early research code; it is functional but lacks production-grade structure and testing. Refactoring is recommended for further integration.*


## 📖 Ecosystem Paradigm: Paper as a Blueprint, Code as a Tool

This repository is **not a static code dump**. While it contains the exact methodology to replicate our peer-reviewed findings on the Canary Current Upwelling System (CCUS), the codebase has evolved into an active, maintained library designed for high-resolution oceanographic forecasting, regional benchmarking, and operational integration.

* **The Paper (The Theory):** Defines the mathematical foundations for adapting global Graph Neural Networks (GraphCast) to sub-regional oceanography, formalizing the loss function masks and multi-scale graph mesh adjustments required to mitigate spatial discontinuities and coastal noise.
* **The Codebase (The Product):** Features optimized JAX/Flax implementations, post-publication artifact filtering, and robust support for L4 satellite-based datasets, transitioning from theoretical validation to a scalable operational tool.

---
<a name="academic-anchor"></a>
## ⚓ Academic Anchor & Citation

If you use this tool, framework, or the optimized pre-trained weights in your research or institutional pipelines, please cite both the foundational paper and the software ecosystem:

```bibtex
@article{Cuervo-Londono2026,
  title = {Deep learning weather models for subregional ocean forecasting: A case study on the Canary current upwelling system},
  journal = {Ocean Modelling},
  volume = {203},
  pages = {102782},
  year = {2026},
  issn = {1463-5003},
  doi = {https://doi.org/10.1016/j.ocemod.2026.102782},
  author = {Giovanny A. Cuervo-Londoño and Javier Sánchez and Ángel Rodríguez-Santana},
}
@software{regional_graphcast_sst,
  author    = {Cuervo-Londo{\~n}o, Giovanny A.},
  title     = {Regional-GraphCast-SST},
  version   = {v1.0.0},
  year      = {2026},
  url       = {https://github.com/gacuervol/regional-graphcast-sst}
}

```
<a name="key-features"></a>
## ✨ Key Features (Living Software Product)

<!-- Unlike the original script deployment, version `1.0+` includes: -->

* **Regionalized Graph Architecture:** Square curvilinear mesh adaptation to handle sub-regional spatial discontinuities, effectively replacing global icosahedral structures.
* **Operational Scalability:** JAX-accelerated inference capable of generating 20-day SST forecasts in <3 minutes on mid-range workstation GPUs (Quadro RTX 4000).
* **Artifact Mitigation:** Post-hoc mesh coherence filters that suppress triangular interpolation noise common in multi-scale GNN decoders without sacrificing the precision of upwelling fronts.

---
<a name="installation"></a>
## 📦 Installation

### Dependencies

```bash
conda env create -f environment.yml

```

---
<a name="quick-start"></a>
## 🚀 Quick Start

Train the regional engine specifically for the Moroccan subregion (CCUS):

```bash
nohup ~/repo/training.sh &
```

<a name="reproducibility"></a>
## 🧪 Reproducibility Archive (The Frozen Snapshot)

To ensure strict compliance with scientific integrity, the exact environment, initialization seeds, and source code used to generate the tables and figures within the journal paper are completely preserved.

To replicate the paper's exact figures:

**Checkout the demo notebook:**
```bash
demo.ipynb
```
*Note: While the results generated via this snapshot align with the paper and include additional data, they lack the detailed context, explanations, and discussions thoroughly developed in the original article.*

---
<a name="evolution-roadmap"></a>
## 🗺️ Evolution & Roadmap

Since the initial paper acceptance, the software has moved forward. Here is our current development trajectory:

* [x] **Phase 1 (Publication):** Adaptation of GraphCast to sub-regional SST datasets (CCUS).
* [x] **Phase 2 (Productization):** Implementation of spatially-weighted loss functions and in-situ validation against EasyCORA drifter datasets.
* [ ] **Phase 3 (Active Scaling):** Integration of multi-variable forecasts (Salinity, Currents) to support physics-informed training strategies.
* [ ] **Phase 4 (Enterprise Integration):** Native ONNX export for real-time edge processing on coastal buoy arrays.

---
<a name="governance"></a>
## 🤝 Contribution & Governance

As a living tool utilized by academic laboratories and oceanographic organizations, we welcome community extensions, performance patches, and bug reports.

* **Corporate/Institutional Support:** If your organization requires custom integration or commercial-adjacent modifications, please contact the maintainers at `giovanny.cuervo@ulpgc.es`.
* **Issue Tracking:** For algorithm bugs or unexpected artifact behaviors, please open an issue with a Minimal Reproducible Example (MRE) using the provided issue templates.

---

## 📄 License

* **Software Core:** Licensed under the [MIT License](https://opensource.org/licenses/MIT).
* **Pre-trained Weights:** Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
