{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "57157507",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "\n",
    "import torch\n",
    "from torch.nn.utils.rnn import pad_sequence\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "from PIL import Image\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision.models as models\n",
    "from datetime import datetime\n",
    "import torch.optim.lr_scheduler as scheduler\n",
    "import json\n",
    "import torchvision.transforms as transforms\n",
    "import random\n",
    "\n",
    "from transformers import DistilBertForSequenceClassification\n",
    "from transformers import DistilBertTokenizerFast\n",
    "from .swin_transformer import SwinTransformer\n",
"from .swin_transformer_v2 import SwinTransformerV2\n",
"from .swin_transformer_moe import SwinTransformerMoE\n",
"from .swin_mlp import SwinMLP\n",
"from .simmim import build_simmim\n",

    "\n",
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "634cc2a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "experiment_name = '010 - (main experiment) DistilBERT and HVT'\n",
    "\n",
    "output_savepath = '/home/anusha/outputs/'"
   ]
  },