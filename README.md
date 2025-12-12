# 📑 Assignment Overview

This assignment explores clustering techniques ranging from fundamental algorithms implemented from scratch to state-of-the-art methods using pre-trained models and embeddings. Each notebook includes proper documentation, visualizations, and clustering quality metrics.

---

## 📂 Assignment Structure

### Part E: Multi-Modal Semantic Clustering
**File:** `imagebind_multimodal_clustering.ipynb`

* **Implementation of ImageBind Foundation Model**: Leveraging Meta AI’s `imagebind_huge` architecture to extract high-dimensional embeddings across different modalities.
* **Cross-Modality Alignment**: Mapping `.jpg` visual data and `.wav` audio data into a singular, shared neighborhood.
* **K-Means Integration**: Applying unsupervised clustering on joint embeddings to group distinct media types by semantic concept rather than file format.
* **Advanced Environment Patching**: Implementing custom loaders to bypass `torchcodec` dependencies and resolving `torchvision` versioning conflicts.

**Key Concepts:**
* Joint Embedding Spaces (Visual-Audio Alignment)
* Surgical Tensor Reshaping for 1D Spec-Transformers
* Semantic vs. Feature-based Clustering
* Zero-shot Modality Transfer

---

## 📊 Clustering Quality Metrics

To evaluate the performance of the semantic alignment, the following metrics are utilized within the notebook:

* **Semantic Consistency**: Verification that different modalities of the same concept (e.g., Dog Image + Dog Audio) are assigned to the same Cluster ID.
* **Cosine Similarity Analysis**: Measuring the angular distance between vectors in the joint space to quantify how "related" the model perceives the audio and image to be.
* **Inertia (Within-Cluster Sum of Squares)**: Assessing the compactness of the semantic clusters formed in the 1024-dimensional space.

---

## 🗄️ Datasets

This implementation utilizes a specialized multi-modal sample set:
* **Vision Data**: High-resolution imagery representing distinct classes (e.g., Dog and Car).
* **Audio Data**: Waveform files (.wav) sampled at 16kHz, capturing environmental sounds corresponding to the visual classes (e.g., barking, engine idling).
* **Preprocessing**: All data is normalized to 16kHz mono audio and 224x224 patches for vision to meet the ImageBind encoder requirements.

---

## 💡 Key Learning Outcomes

* **Modality Agnosticism**: Understanding how a single model can process diverse data types without requiring separate, specialized heads for each task.
* **Handling Library Deprecation**: Gained experience in "Monkey-Patching" and overriding internal library loaders to maintain compatibility with evolving PyTorch ecosystems.
* **Vector Space Intuition**: Visualizing how high-dimensional vectors can represent abstract "concepts" (like "Dogginess") rather than just raw pixel or frequency data.

---

## 🔗 References

* **Meta AI Research**: [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.06755)
* **GitHub Repository**: [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)
* **Scikit-Learn Documentation**: [K-Means Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html#k-means)
