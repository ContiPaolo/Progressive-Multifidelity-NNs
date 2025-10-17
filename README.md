<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]

# Progressive multi-fidelity neural networks
Source code of the paper [Progressive multi-fidelity learning for physical system predictions](https://arxiv.org/abs/2510.13762) by Conti, Guo, Frangi and Manzoni (2025).

<img width="1307" height="615" alt="image" src="https://github.com/user-attachments/assets/73a0cf9f-f789-4318-bbc0-dd9b78a03b19" />

### Abstract
Highly accurate datasets from numerical or physical experiments are often expensive and time-consuming to acquire, posing a significant challenge for applications that require precise evaluations, potentially across multiple scenarios and in real-time. Even building sufficiently accurate surrogate models can be extremely challenging with limited high-fidelity data. Conversely, less expensive, low-fidelity data can be computed more easily and encompass a broader range of scenarios. By leveraging multi-fidelity information, prediction capabilities of surrogates can be improved. However, in practical situations, data may be different in types, come from sources of different modalities, and not be concurrently available, further complicating the modeling process. To address these challenges, we introduce a progressive multi-fidelity surrogate model. This model can sequentially incorporate diverse data types using tailored encoders. Multi-fidelity regression from the encoded inputs to the target quantities of interest is then performed using neural networks. Input information progressively flows from lower to higher fidelity levels through two sets of connections: concatenations among all the encoded inputs, and additive connections among the final outputs. This dual connection system enables the model to exploit correlations among different datasets while ensuring that each level makes an additive correction to the previous level without altering it. This approach prevents performance degradation as new input data are integrated into the model and automatically adapts predictions based on the available inputs. We demonstrate the effectiveness of the approach on numerical benchmarks and a real-world case study, showing that it reliably integrates multi-modal data and provides accurate predictions, maintaining performance when generalizing across time and parameter variations.

### Test cases available:
- *Reaction–diffusion PDE problem*. A parametric, spatio-temporalsystem where high-fidelity simulations of spiral wave dynamics are reconstructed from coarse, noisy low-fidelity simulations with parametric bias.
- *Navier–Stokes PDE problem*. A computational fluid dynamics benchmark leveraging hierarchical low-fidelity inputs (drag and lift coefficients, outlet sensors, and partial-domain snapshots) to reconstruct unsteady flow behavior.
- *Air pollution monitoring*. A real-world case using sensor data that combine temperature, humidity, and co-pollutant measurements from low-cost devices to estimate expensive benzene concentrations, despite missing or unreliable low-fidelity signals.

> :warning: **Datasets and airpollution example will be made available in the upcoming days**

### Installation

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

* **Linux/macOS:** ```source .venv/bin/activat```

* **Windows (cmd):** ```.venv\Scripts\activate```

* **Windows (PowerShell):** ```.venv\Scripts\Activate.ps1```

#### 3. Install the package in editable mode 
```
pip install -e . 
```

#### 4. Install additional requirements

```
pip install -r requirements.txt
```


[arxiv-shield]: https://img.shields.io/badge/arXiv-2405.20905-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2510.13762
