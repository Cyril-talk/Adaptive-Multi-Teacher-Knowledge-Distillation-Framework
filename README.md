# Adaptive-Multi-Teacher-Knowledge-Distillation-Framework
Adaptive Multi-Teacher Knowledge Distillation Framework with Foundation Models for Medical Image Analysis
# Adaptive-Multi-Teacher-Knowledge-Distillation-Framework

A multi-teacher knowledge distillation framework for medical image segmentation, classification, and recognition tasks.

## ⚠️ Project Status

> **Note**: This repository is currently under active development. The code structure is being continuously refined and reorganized for better clarity and maintainability. We appreciate your patience and welcome any feedback or contributions.

### Current Status
- ✅ Core functionality implemented
- ✅ Multi-teacher distillation framework working
- ✅ Support for 6 medical datasets
- 🔄 Code structure optimization in progress
- 🔄 Documentation being improved
- 📋 Comprehensive refactoring planned

### Upcoming Improvements
- [ ] Unified code structure
- [ ] Modular architecture redesign
- [ ] Enhanced documentation
- [ ] Code style standardization
- [ ] Unit tests and CI/CD pipeline
- [ ] Performance benchmark

---

## 🎯 Overview

This project implements a novel multi-teacher knowledge distillation framework for medical image analysis. The framework leverages multiple pre-trained foundation models (MedSAM, USFM, RETFound, BioMedParse) to distill knowledge into a lightweight student model, achieving competitive performance across various medical imaging tasks.

### Key Innovations
- **Dynamic Teacher Gating**: Automatically adjusts teacher contributions based on task relevance
- **Cross-Task Knowledge Transfer**: Enables knowledge sharing across different medical imaging domains
- **Heterogeneous Distillation**: Handles different teacher architectures and output formats
- **Adaptive Loss Weighting**: Dynamically balances multiple distillation objectives

---

## ✨ Features

- 🏥 **Multi-Task Support**: Segmentation, classification, and recognition
- 👨‍🏫 **Multiple Teacher Models**: MedSAM, USFM, RETFound_MAE, BioMedParse
- 🎓 **Advanced Distillation**: Feature-level, output-level, and contrastive distillation
- 🔄 **Dynamic Gating**: Automatic teacher weight adjustment
- 📊 **Comprehensive Metrics**: Task-specific evaluation metrics


Dataset download path：
- BUSI：https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
Kvasir-seg：https://datasets.simula.no/kvasir-seg/
COVID：https://aistudio.baidu.com/datasetdetail/127908
MSD-heart：https://aistudio.baidu.com/datasetdetail/23911
APTOS2019：https://www.kaggle.com/datasets/mariaherrerot/aptos2019/data
ISIC2017：https://aistudio.baidu.com/datasetdetail/65747

Weight file download path:
USFM：https://github.com/openmedlab/USFM
MedSAM：https://huggingface.co/wanglab/medsam-vit-base
RETFound：https://huggingface.co/RETFound/RETFound
BiomedParse：https://huggingface.co/microsoft/BiomedParse



