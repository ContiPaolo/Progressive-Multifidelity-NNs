<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![DOI][doi-shield]][doi-url]

# Progressive multi-fidelity learning with neural networks for physical system prediction
Source code of the paper [Progressive multi-fidelity learning with neural networks sfor physical system predictions](https://www.sciencedirect.com/science/article/pii/S0045782526001544) by Conti, Guo, Frangi and Manzoni (2025).

<img width="1307" height="615" alt="image" src="https://github.com/user-attachments/assets/73a0cf9f-f789-4318-bbc0-dd9b78a03b19" />

## Abstract
Highly accurate datasets from numerical or physical experiments are often expensive and time-consuming to acquire, posing a significant challenge for applications that require precise evaluations, potentially across multiple scenarios and in real-time. Even building sufficiently accurate surrogate models can be extremely challenging with limited high-fidelity data. Conversely, less expensive, low-fidelity data can be computed more easily and encompass a broader range of scenarios. By leveraging multi-fidelity information, prediction capabilities of surrogates can be improved. However, in practical situations, data may be different in types, come from sources of different modalities, and not be concurrently available, further complicating the modeling process. To address these challenges, we introduce a progressive multi-fidelity surrogate model. This model can sequentially incorporate diverse data types using tailored encoders. Multi-fidelity regression from the encoded inputs to the target quantities of interest is then performed using neural networks. Input information progressively flows from lower to higher fidelity levels through two sets of connections: concatenations among all the encoded inputs, and additive connections among the final outputs. This dual connection system enables the model to exploit correlations among different datasets while ensuring that each level makes an additive correction to the previous level without altering it. This approach prevents performance degradation as new input data are integrated into the model and automatically adapts predictions based on the available inputs. We demonstrate the effectiveness of the approach on numerical benchmarks and a real-world case study, showing that it reliably integrates multi-modal data and provides accurate predictions, maintaining performance when generalizing across time and parameter variations.

## Test cases available:
- *Reaction–diffusion* (Section 3). A parametric, spatio-temporalsystem where high-fidelity simulations of spiral wave dynamics are reconstructed from coarse, noisy low-fidelity simulations with parametric bias.
- *Navier–Stokes* (Section 4). A computational fluid dynamics benchmark leveraging hierarchical low-fidelity inputs (drag and lift coefficients, outlet sensors, and partial-domain snapshots) to reconstruct unsteady flow behavior.
- *Air pollution* (Section 5). A real-world case using sensor data that combine temperature, humidity, and co-pollutant measurements from low-cost devices to estimate expensive benzene concentrations, despite missing or unreliable low-fidelity signals.

## Datasets:
*Reaction-diffusion* and *Navier-Stokes* datasets are available on [Zenodo](https://doi.org/10.5281/zenodo.17379475). 

After downloading, place the files in the following directories `examples/reactiondiffusion/data/` and `examples/navierstokes/data/`, respectively.

*Air pollution* dataset is available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality).

> :warning: **Airpollution example will be made available soon**

## Installation:

#### 1. Clone the repository
```bash
git clone https://github.com/ContiPaolo/Progressive-Multifidelity-NNs
cd Progressive-Multifidelity-NNs
```

#### 2. Create virtual environment
```
python -m venv .venv
```
Activate it:

  * **Linux/macOS:**
  ```
  source .venv/bin/activate
  ```
  
  * **Windows (cmd):**
  ```
  .venv\Scripts\activate
  ```
  
  * **Windows (PowerShell):**
  ```
  .venv\Scripts\Activate.ps1
  ```

#### 3. Install required packages
```
pip install -r requirements.txt
```
 **Additionally for macOS:**: ```pip install -r requirements-macos.txt```

#### 4. Install package in editable mode

```
pip install -e .
```


[arxiv-shield]: https://img.shields.io/badge/arXiv-2405.20905-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2510.13762
[doi-shield]: https://zenodo.org/badge/DOI/10.5281/zenodo.17379475.svg
[doi-url]: https://doi.org/10.5281/zenodo.17379475

