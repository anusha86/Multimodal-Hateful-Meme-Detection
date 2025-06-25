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

📚 Datasets Used
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



