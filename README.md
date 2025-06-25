Multimodal Hateful Meme Detection using Knowledge Distillation and Hierarchical Vision Transformer Framework
This project presents a multimodal deep learning framework that detects hateful content in memes by fusing visual and textual features. It introduces an efficient pipeline that leverages Knowledge Distillation for textual feature compression and a Hierarchical Vision Transformer (HVT) for visual understanding, enabling accurate and scalable hate speech detection in complex multimodal inputs.

Textual Branch – Knowledge Distillation
Uses a distilled BERT model to extract semantic-rich text embeddings efficiently.

Knowledge distillation compresses a large teacher model into a smaller, faster student model without significant performance loss.

Efficient for real-time or edge deployment (e.g., mobile devices).

🖼️ Visual Branch – Hierarchical Vision Transformer (HVT)
Employs a Hierarchical Vision Transformer for multi-scale visual understanding.

Captures both local and global dependencies using a window-based attention mechanism.

Reduces computational complexity through sliding window attention and hierarchical token merging.

🔄 Cross-Modal Fusion
Extracted text and image features are concatenated to form a fused representation F.

F is passed through a softmax classifier for final label prediction.

📚** Datasets Used**
MMHS150K
Twitter-based multimodal hate speech dataset with 150K samples.

Hateful Memes Challenge (HMC)
Curated by Facebook AI to emphasize the need for multimodal reasoning.

MultiOFF
A small-scale dataset of offensive memes with multimodal annotations.

Training Overview
Uses standard supervised learning with cross-entropy loss.

Employs adaptive learning rate decay, early stopping, and batch-based training.

Visual features processed through HVT layers and pooled before fusion.

Textual features extracted via a distilled transformer with tokenization and embedding layers.

**Experimental Setup**
xperimental Setup
The experiments were conducted across two different hardware configurations to ensure robustness, performance benchmarking, and scalability of the proposed model.

🔧 Primary Setup: NVIDIA TITAN RTX GPUs
For large-scale training (e.g., MMHS150K and Hateful Memes Challenge), experiments were conducted using a high-performance deep learning workstation equipped with:

Component	Details
GPU	2 × NVIDIA TITAN RTX (24 GB each, operated in parallel)
Total VRAM	48 GB
Framework	PyTorch with CUDA & cuDNN
Usage	Parallel GPU training for multimodal feature extraction and fusion

This configuration was used for accelerated training of transformer-based models with large batch sizes and high-resolution image inputs.

🖥️ Secondary Setup: AMD ROCm-enabled Local System
For development, testing, and evaluation, experiments were also performed on an AMD-based system with the following specifications:

Component	Details
CPU	AMD Ryzen 7 5700X3D
GPU	AMD Radeon RX 6800 XT (ROCm-enabled, Ubuntu)
RAM	32 GB DDR4 @ 3200 MHz
Storage (ROM)	8 TB (4 TB HDD + 4 TB SSD [3 TB NVMe + 1 TB SATA])
Motherboard	Gigabyte B550M DS3H AC
Case	NZXT H510 Flow
Cooling	Cooler Master Hyper 212 + 2 ARGB/eSports fans
Power Supply	Reactor Core 750W PSU

This setup was primarily used for debugging, model inference, and validation tasks under ROCm-supported environments with PyTorch.

Both configurations were tested with the same software stack and configurations to ensure consistency in results and cross-hardware reproducibility.
All experiments for training and evaluating the multimodal hateful meme detection models were conducted on a high-performance local workstation with the following specifications:

