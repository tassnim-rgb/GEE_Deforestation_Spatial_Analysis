# GEE_Deforestation_Spatial_Analysis

## Overview

This notebook pulls **real Landsat-derived forest data directly from Google Earth Engine** no pre-downloaded CSVs, and no Kaggle datasets! and combines it with deep learning classification and rigorous spatial mathematics to analyse Amazon deforestation from 2001 to 2023.

The analysis covers three stages:

1. **Temporal analysis** : annual deforestation statistics for Mato Grosso, Pará, and Rondônia extracted live from the Hansen Global Forest Change dataset at 30 m resolution.
2. **Deep learning classification** : three models trained on 4-channel Landsat chips (Red + NIR + SWIR1 + treecover%) to segment forest vs. non-forest.
3. **Spatial fragmentation mathematics** : connected component analysis, power law patch-size distributions, Shape Index, Euler characteristic topology, and a Comprehensive Fragmentation Index computed directly on GEE-derived masks.

---

### Notebook Structure

| Section | Description |
|---------|-------------|
| **0** | Install & authenticate to Google Earth Engine |
| **1** | Load Hansen GFC v1.12, extract annual loss statistics, interactive geemap visualisation, chip extraction |
| **2** | Data pipeline : normalisation, augmentation, DataLoader setup |
| **3** | Shared training utilities (train loop, evaluation, plotting) |
| **4** | Custom 4-channel CNN |
| **5** | ResNet-18 adapted for 4-channel Landsat input |
| **6** | ResNet-18 + Focal Loss + Mixup + SGDR (best model) |
| **7** | Model comparison : accuracy, IoU, confusion matrix |
| **8** | Spatial mathematics : power law, Shape Index, Euler χ, CFI |
| **9** | Results, discussion, conservation implications |
| **10** | Final summary dashboard |

---

## Key Figures

| Figure | What it shows |
|--------|---------------|
| Fig. 1 | Annual deforestation (bar) + remaining forest cover (line), 2001–2023, 3 states |
| Fig. 2 | Extracted 64×64 GEE chips across Red / NIR / SWIR1 bands |
| Fig. 3 | Model comparison accuracy, IoU, confusion matrix |
| Fig. 4 | Power law patch-size CCDF with log-log regression |
| Fig. 5 | Shape Index distribution + edge–area fractal scaling |
| Fig. 6 | Euler characteristic distribution + χ–LCC ecological quadrants |
| Fig. 7 | Mato Grosso forest cover trajectory + rate of change |
| Fig. 8 | Comprehensive Fragmentation Index distribution + cover–CFI scatter |
| Fig. 9 | Full summary dashboard |

---

## Mathematical Core

### Power Law (Patch Size Distribution)
$$P(A \geq a) \sim a^{-\alpha}$$
Exponent α̂ fitted by log-log OLS on the CCDF. α < 2 → scale-free fragmentation.

### Shape Index
$$SI_i = \frac{P_i}{2\sqrt{\pi A_i}} \geq 1$$
SI = 1 for a perfect circle; higher values indicate irregular, edge-exposed patches.

### Euler Characteristic
$$\chi = C - H$$
where C = connected forest components, H = enclosed holes (interior clearings).  
χ < 0 → Swiss-cheese landscape; χ = 1 → single intact patch.

### Comprehensive Fragmentation Index

---

## Deep Learning Models

| Model | Architecture | Loss | Key feature |
|-------|-------------|------|-------------|
| Custom CNN | 4-layer conv, AdaptiveAvgPool | BCE + pos_weight | Native 4-channel support |
| ResNet-18 | ImageNet pretrained, 4ch adapted | BCE + pos_weight | Transfer learning, differential LR |
| ResNet-18 Enhanced | Full fine-tune, deeper head | Focal (α=0.75, γ=2) | Mixup augmentation + SGDR warm restarts |

All models ingest **4-channel chips**: Red, NIR, SWIR1, and treecover%. NIR is the primary discriminant for vegetation; SWIR1 separates moist from dry biomass information unavailable from RGB alone.

---

## Prerequisites

### 1. Google Earth Engine Access (free)
Register at: https://code.earthengine.google.com/register

### 2. Google Cloud Project
- Create a project at https://console.cloud.google.com
- Enable the Earth Engine API
- Note your **Project ID**

### 3. Set Your Project ID in the Notebook
In Section 0, replace:
```python
GEE_PROJECT = "your-gee-project-id"
```

---

## Running on Google Colab

1. Click the **Open in Colab** badge above (or upload the `.ipynb` manually).
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run Section 0, a browser tab will open for GEE authentication.
4. Run all cells in order.

> **Estimated runtime:** ~45–90 minutes on a T4 GPU (chip extraction from GEE is the bottleneck).

---

## Data Source

**Hansen Global Forest Change v1.12**  
University of Maryland — Hansen, Potapov, Moore, Hancher et al.  
GEE Dataset: `UMD/hansen/global_forest_change_2024_v1_12`  
Coverage: Global, 2000–2024, 30 m/pixel (Landsat-derived)

Study Region: Brazilian Amazon deforestation arc
- **Mato Grosso** — [-61.0, -18.0, -51.0, -7.0]
- **Pará** — [-56.0, -12.0, -46.0, -2.0]
- **Rondônia** — [-66.0, -13.5, -59.0, -7.5]

Sample region for chip extraction: Central Pará [-56.5, -12.5, -54.5, -10.5]

---

## Dependencies

```
earthengine-api
geemap
torch >= 2.0
torchvision
numpy
matplotlib
seaborn
scipy
scikit-image
scikit-learn
Pillow
rasterio
```

All installed automatically in Section 0.

---

## Key Insight

> **Forest cover fraction and fragmentation are not interchangeable.**

A landscape at 60% cover with CFI=0.8 is ecologically more threatened than one at 50% cover with CFI=0.2 — because the former's patches are small, irregular, and topologically disconnected. Spatial mathematics reveals what aggregate deforestation statistics cannot.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this notebook in research, please cite the Hansen Global Forest Change dataset:

> Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853.
