# Comprehensive Clustering Project: From Foundations to Frontier

This repository contains a series of Google Colab notebooks exploring the full spectrum of clustering techniques. The project moves from fundamental algorithms implemented from scratch to state-of-the-art methods utilizing Large Language Model (LLM) embeddings and multi-modal foundation models like ImageBind.

## 📂 Project Structure

The project is divided into three logical phases: Classical Foundations, Professional Efficiency & Outliers, and the Deep Learning Frontier.

---

### Phase 1: Classical Foundations

#### Part A: K-Means Clustering from Scratch
**File:** `kmeans_scratch.ipynb`
* **Description:** Implementation of the K-Means algorithm using only fundamental Python libraries like NumPy to understand the iterative "Assign and Update" logic.
* **Key Concepts:** Euclidean distance computation, random centroid initialization, convergence criteria, and Silhouette analysis.
* **Libraries Used:** `numpy`, `matplotlib`, `scikit-learn` (for evaluation only).

#### Part B: Hierarchical Clustering
**File:** `hierarchical_clustering.ipynb`
* **Description:** Exploring agglomerative clustering to build a "family tree" of data relationships.
* **Key Concepts:** Dendrogram visualization, linkage methods (Ward, Single, Complete), and distance metrics.
* **Libraries Used:** `scipy`, `scikit-learn`.

#### Part C: Gaussian Mixture Models (GMM)
**File:** `gaussian_mixture_models.ipynb`
* **Description:** Moving from "hard" clustering to "soft" probabilistic clustering.
* **Key Concepts:** Expectation-Maximization (EM) algorithm, handling elliptical cluster shapes, and model selection via BIC/AIC.
* **Libraries Used:** `scikit-learn`, `seaborn`.

---

### Phase 2: Professional Efficiency & Outliers

#### Part D: DBSCAN Clustering with PyCaret
**File:** `dbscan_pycaret.ipynb`
* **Description:** Utilizing a low-code library to identify density-based clusters of arbitrary shapes.
* **Key Concepts:** Density reachability, automatic noise/outlier detection, and automated ML workflows.
* **Libraries Used:** `pycaret`, `pandas`.

#### Part E: Anomaly Detection with PyOD
**File:** `anomaly_detection_pyod.ipynb`
* **Description:** Shifting focus from grouping commonalities to identifying the "misfits" in a dataset.
* **Key Concepts:** Isolation Forest algorithm, contamination rates, and anomaly score visualization.
* **Libraries Used:** `pyod`, `matplotlib`.

---

### Phase 3: The Deep Learning Frontier

#### Part F: Time-Series Clustering
**File:** `timeseries_clustering.ipynb`
* **Description:** Grouping temporal data where the sequence and trend are the primary features.
* **Key Concepts:** Dynamic Time Warping (DTW), time-series scaling, and Barycenter trend visualization.
* **Libraries Used:** `tslearn`, `matplotlib`.

#### Part G: Document Clustering with LLM Embeddings
**File:** `llm_document_clustering.ipynb`
* **Description:** Using a pre-trained Transformer model to cluster text based on semantic meaning rather than keyword matching.
* **Key Concepts:** High-dimensional text embeddings, dimensionality reduction (PCA), and semantic similarity.
* **Libraries Used:** `sentence-transformers`, `scikit-learn`.

#### Part H: Image Clustering with ImageBind
**File:** `image_clustering_imagebind.ipynb`
* **Description:** Utilizing Meta AI’s ImageBind model to categorize visual content using deep vision transformer embeddings.
* **Key Concepts:** Vision Transformer (ViT) features, zero-shot categorization, and semantic visual grouping.
* **Libraries Used:** `torch`, `imagebind`, `torchvision`.

#### Part I: Audio Clustering with ImageBind
**File:** `audio_clustering_imagebind.ipynb`
* **Description:** Applying the same foundation model to categorize complex sound patterns into meaningful groups.
* **Key Concepts:** Audio spectrogram processing, acoustic fingerprints, and cross-modal embedding consistency.
* **Libraries Used:** `torch`, `imagebind`, `torchaudio`.

---

## 📊 Clustering Quality Metrics

Each notebook includes rigorous evaluation to ensure cluster validity:
* **Silhouette Score:** Measures how similar a point is to its own cluster compared to others (Range: -1 to 1).
* **Inertia (SSE):** Used in K-Means to measure the tightness of clusters.
* **BIC/AIC:** Used for Gaussian Mixture Models to balance model fit with complexity.
* **Dendrograms:** Visual verification of hierarchical separation.
* **PCA/t-SNE:** Dimensionality reduction used to visualize high-dimensional clusters in 2D/3D space.

## 💾 Datasets

This project utilizes a mix of synthetic and real-world data:
* **Synthetic:** `make_blobs` and `make_moons` for algorithm benchmarking.
* **Time-Series:** The "Trace" dataset for signal pattern recognition.
* **Text:** Curated document lists for semantic testing.
* **Multi-modal:** Official ImageBind assets including dog/car images and corresponding WAV audio files.

## 🧠 Key Learning Outcomes

1. **Algorithmic Depth:** Gained a deep understanding of the mathematical foundations of clustering by implementing K-Means from scratch.
2. **Method Versatility:** Learned to select the right algorithm (Density-based, Probabilistic, or Hierarchical) based on the specific shape and nature of the data.
3. **Advanced Representations:** Mastered the use of LLM and Foundation Model embeddings to cluster complex, unstructured data like text, images, and audio.
4. **Professional Evaluation:** Developed the ability to quantitatively assess clustering quality using industry-standard metrics.

## 🔗 References

* [Scikit-Learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
* [PyCaret Clustering Tutorial](https://pycaret.gitbook.io/docs/get-started/tutorials)
* [Sentence-Transformers (SBERT)](https://www.sbert.net/)
* [Meta AI ImageBind Repository](https://github.com/facebookresearch/ImageBind)
* [Python Data Science Handbook (Jake VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/)
