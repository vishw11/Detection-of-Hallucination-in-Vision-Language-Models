# Detection of Hallucination in Vision-Language Models

A research project focused on analyzing and reducing hallucination in Vision-Language Models (VLMs) using BLIP, attention visualization, CLIP similarity, Grad-CAM, and contrastive decoding.

---

## Overview

Vision-Language Models combine image understanding with natural language reasoning and are widely used in:

- Visual Question Answering (VQA)  
- Image Captioning  
- Multimodal Search  
- Human-AI Interaction  

Despite strong performance, these models may generate **hallucinations** — outputs not grounded in the input image.

Example:

- Image contains only a strawberry  
- Question: *What is the color of banana?*  
- Model Output: **yellow**

This project investigates such behavior and explores methods to detect and reduce hallucination.

---

## Objectives

- Understand Transformer and Vision Transformer architecture  
- Study hallucination behavior in BLIP model  
- Detect unsupported predictions using interpretability tools  
- Evaluate contrastive decoding for hallucination mitigation  
- Fine-tune BLIP on structured datasets for systematic testing  

---

## Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- BLIP  
- OpenCV  
- Matplotlib  
- NumPy  
- CUDA GPU  

---

## Methods Used

### 1. Attention Visualization

Analyzed attention between CLS token and image patches from the final transformer layer to verify whether predictions focus on relevant image regions.

### 2. CLIP Similarity

Compared image-text semantic similarity scores to check whether generated answers align with image content.

### 3. Grad-CAM

Applied gradient-based patch activation visualization to inspect important image regions influencing prediction.

### 4. Contrastive Decoding

Re-ranked top candidate answers using:

```text
Score(a) = -log P(a|q,I) + α log P(a|q,Id)
