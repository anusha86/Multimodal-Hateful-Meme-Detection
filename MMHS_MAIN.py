  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3d1eb324",
   "metadata": {},
   "outputs": [],
   "source": [
    "class AnushaDataset(Dataset):\n",
    "    \n",
    "    def __init__(self, rootpath, jsonpath, keypath, allcaptionvectorspath, transform = None, freq_threshold = 5):\n",
    "        \n",
    "        self.rootpath = rootpath\n",
    "        \n",
    "        f = open(jsonpath)\n",
    "        self.alldata = json.load(f)\n",
    "        \n",
    "        self.keys = torch.load(keypath)\n",
    "        random.shuffle(self.keys)\n",
    "        \n",
    "#         self.allcaptionvectors = torch.load(allcaptionvectorspath)\n",
    "        print('Files loaded successfully')\n",
    "        \n",
    "        self.transform = transform\n",
    "        \n",
    "        \n",
    "        \n",
    "#         self.vocab = Vocabulary(freq_threshold, itospath, stoipath)\n",
    "#         print('Vocabulary called and built')\n",
    "#         self.vocab.build_vocabulary(self.allcaptions)\n",
    "#         print('Vocabulary built')\n",
    "        \n",
    "    def __len__(self):\n",
    "        return 140000\n",
    "#         return len(self.alldata)\n",
    "    \n",
    "    \n",
    "    def __getitem__(self, index):\n",
    "        \n",
    "        key = self.keys[index]\n",
    "        \n",
    "        imgname = key + '.jpg'\n",
    "        \n",
    "        img = Image.open(self.rootpath + imgname)\n",
    "        \n",
    "        cap = self.alldata[key]['tweet_text']\n",
    "    \n",
    "        label = self.alldata[key]['labels'][0]\n",
    "        \n",
    "        if(label != 0):\n",
    "            label = 1\n",
    "\n",
    "        \n",
    "        if(self.transform):\n",
    "            img = self.transform(img)\n",
    "        \n",
    "        return (img, cap, label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "80e41035",
   "metadata": {},
   "outputs": [],
   "source": [
    "transformation = transforms.Compose([transforms.Resize((256,256)), \n",
    "                                     transforms.ToTensor()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0d48a09b",
   "metadata": {},
   "outputs": [],
   "source": [
    "alldatapath = '/home/anusha/mmhs/archive/MMHS150K_GT.json'\n",
    "allcaptionvectorspath = '/home/anusha/mmhs/archive/allcaptionvectors.pt'\n",
    "keypath = '/home/anusha/mmhs/archive/validkeys.pt'\n",
    "imgpath = '/home/anusha/mmhs/archive/img_resized/'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8c47d4c0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files loaded successfully\n"
     ]
    }
   ],
   "source": [
    "dataset = AnushaDataset(imgpath, alldatapath,keypath,  allcaptionvectorspath, transform = transformation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "48755c8c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "140000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ebd8c4fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Coming for the red room let’s go cunt https://t.co/80yEePyiPs'"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset[5666][1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4600ac81",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_loader(dataset, batch_size = 4, shuffle = True):\n",
    "    \n",
    "#     pad_idx = dataset.vocab.stoi[\"<PAD>\"]\n",
    "    pad_idx = 0\n",
    "    \n",
    "    loader = DataLoader(dataset = dataset,\n",
    "                       batch_size = batch_size, \n",
    "                       shuffle=shuffle)\n",
    "    \n",
    "    return loader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f657ca4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "loader = get_loader(dataset, batch_size = 32)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c365c505",
   "metadata": {},
   "source": [
    "# max sequence length for this dataset is 77"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ca575e6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'torch.utils.data.dataset.Subset'>\n"
     ]
    }
   ],
   "source": [
    "trainset, validationset, testset = torch.utils.data.random_split(dataset, [100000, \n",
    "                                                                           20000, \n",
    "                                                                           20000])\n",
    "\n",
    "# trainset, validationset, testset = torch.utils.data.random_split(dataset, [8000, \n",
    "#                                                                            1000, \n",
    "#                                                                            1000])\n",
    "\n",
    "# trainset, validationset, testset = torch.utils.data.random_split(dataset, [40000, \n",
    "#                                                                            5000, \n",
    "#                                                                            5000])\n",
    "print(type(trainset))\n",
    "\n",
    "trainloader = get_loader(trainset, batch_size = 16)\n",
    "validationloader = get_loader(validationset, batch_size = 16)\n",
    "testloader = get_loader(testset, batch_size = 16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7c4fbad0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6250\n",
      "1250\n",
      "1250\n"
     ]
    }
   ],
   "source": [
    "num_train_batches = len(trainloader)\n",
    "num_validation_batches = len(validationloader)\n",
    "num_test_batches = len(testloader)\n",
    "\n",
    "print(num_train_batches)\n",
    "print(num_validation_batches)\n",
    "print(num_test_batches)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b8109251",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Device used for training is  cuda:0\n"
     ]
    }
   ],
   "source": [
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(\"Device used for training is \",device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "dc6f5c44",
   "metadata": {},
   "outputs": [],
   "source": [
    "model2 = HVT50_32x4d(pretrained=True)"
   ]
  },
 class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "73e1d845",
   "metadata": {},
   "outputs": [],
   "source": [
    "class HVT(nn.Module):\n",
    "    \n",
    "    def __init__(self, model2):\n",
    "        \n",
    "        super().__init__()\n",
    "        \n",
    "        self.conv1 = model2.conv1\n",
    "        self.bn1 = model2.bn1\n",
    "        self.act1 = model2.act1\n",
    "        self.maxpool = model2.maxpool\n",
    "        self.layer1 = model2.layer1\n",
    "        self.layer2 = model2.layer2\n",
    "        self.layer3 = model2.layer3\n",
    "        self.layer4 = model2.layer4\n",
    "        self.global_pool = model2.global_pool\n",
    "        self.fc = nn.Linear(2048,16)\n",
    "        \n",
    "    def forward(self, x):       \n",
    "        \n",
    "        x = self.conv1(x)\n",
    "        x = self.bn1(x)\n",
    "        x = self.act1(x)\n",
    "        x = self.maxpool(x)\n",
    "        \n",
    "        x = self.layer1(x) \n",
    "        x = self.att1(x)\n",
    "#         print('1: ', x.shape)\n",
    "\n",
    "        x = self.layer2(x)\n",
    "        x = self.att2(x)\n",
    "        \n",
    "#         print('2: ', x.shape)\n",
    "        \n",
    "        x = self.layer3(x)\n",
    "        x = self.att3(x)\n",
    "        \n",
    "#         print('3: ', x.shape)\n",
    "        \n",
    "        x = self.layer4(x)\n",
    "        x = self.att4(x)\n",
    "        \n",
    "#         print('4: ', x.shape)\n",
    "        \n",
    "\n",
    "\n",
    "        x = self.global_pool(x)\n",
    "        x = self.fc(x)\n",
    "\n",
    "        \n",
    "        return x       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a11ad09b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class FinalModel(nn.Module):\n",
    "    \n",
    "    def __init__(self, pretrainedCNN = False):\n",
    "        \n",
    "        super(FinalModel, self).__init__()\n",
    "        \n",
    "        # text model - DistilBert producing final features of size 16\n",
    "#         self.textmodel = DistilBertForSequenceClassification.from_pretrained(\"distilbert-base-uncased\")\n",
    "#         self.textmodel.classifier = nn.Linear(768,16)\n",
    "        \n",
    "        # HVT model for image branch\n",
    "        self.HVT = HVT(model2)\n",
    "        \n",
    "        self.fc1 = nn.Linear(16, 2)\n",
    "        \n",
    "    def forward(self, img, inputs, att):\n",
    "        \n",
    "#         print(inputs)\n",
    "#         print('------------------------')\n",
    "#         print(att)\n",
    "#         print('-------------------------')\n",
    "#         print('length of op : ', len(att))\n",
    "        \n",
    "        out1 = self.HVT(img)\n",
    "#         out2 = self.textmodel(input_ids = inputs,\n",
    "#                              op_mask = att)\n",
    "        \n",
    "#         print('HVT output shape : ', out1.shape)\n",
    "#         print('distilbert output shape : ', out2.logits.shape)\n",
    "        \n",
    "#         out3 = torch.cat((out1, out2.logits), 1)\n",
    "        \n",
    "#         print('Shape after concat : ', out3.shape)\n",
    "        \n",
    "        output = self.fc1(out1)\n",
    "\n",
    "        \n",
    "        return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "23b88656",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_size = 1\n",
    "sequence_length = 77\n",
    "num_layers = 2\n",
    "hidden_size = 64\n",
    "num_classes = 32\n",
    "lr = 0.01\n",
    "batch_size = 16\n",
    "epochs = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fafd82cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = FinalModel(pretrainedCNN = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "5297b06f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = model.to(device)\n",
    "model = nn.DataParallel(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c92868fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# aa = torch.rand(4, 3, 256,256).to(device)\n",
    "# bb = torch.rand(4, 77, 1).to(device)\n",
    "\n",
    "# abcd = model(aa, bb)\n",
    "\n",
    "# print('final output shape : ', abcd.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "7b7497db",
   "metadata": {},
   "outputs": [],
   "source": [
    "# img = torch.rand(16, 3, 256, 256)\n",
    "# cap = torch.rand(16, 80, 1)\n",
    "\n",
    "# out = model(img, cap)\n",
    "\n",
    "# print(out.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05a92925",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "348e6936",
   "metadata": {},
   "outputs": [],
   "source": [
    "def resizeCaptions(captions):\n",
    "    \n",
    "    new_captions = {}\n",
    "    \n",
    "    sequence_length = 100\n",
    "    \n",
    "    input_ids = captions['input_ids']\n",
    "    op_mask = captions['op_mask']\n",
    "    \n",
    "    new_inputs = []\n",
    "    new_att = []\n",
    "    \n",
    "    for i in range(len(input_ids)):       \n",
    "        \n",
    "        count = sequence_length - len(input_ids[i])\n",
    "        \n",
    "        i1 = input_ids[i]\n",
    "        a1 = op_mask[i]\n",
    "        \n",
    "#         print('old length : ', len(i1))\n",
    "        \n",
    "        append_zeros = [0] * count\n",
    "        append_ones = [1] * count\n",
    "        \n",
    "        i1 = i1 + append_zeros\n",
    "        a1 = a1 + append_ones\n",
    "        \n",
    "        new_inputs.append(i1)\n",
    "        new_att.append(a1)\n",
    "    \n",
    "    new_captions['input_ids'] = new_inputs\n",
    "    new_captions['op_mask'] = new_att\n",
    "   \n",
    "    return new_captions\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d986fc6d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# max = 0 \n",
    "# fast_tokenizer = DistilBertTokenizerFast.from_pretrained(\"distilbert-base-uncased\")\n",
    "# for i, data in enumerate(trainloader):\n",
    "    \n",
    "#     _, cap, _ = data\n",
    "# #     print(type(cap))\n",
    "#     cap = fast_tokenizer(list(cap))\n",
    "# #     print(cap)\n",
    "# #     print(len(cap['input_ids']))\n",
    "#     cap = resizeCaptions(cap)\n",
    "#     print(len(cap['input_ids']))\n",
    "#     print('----------')\n",
    "#     break\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "0bd695a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "lossFunction = nn.CrossEntropyLoss()\n",
    "# optimizer = optim.Adam(model.parameters(), lr=0.1)\n",
    "optimizer = optim.SGD(model.parameters(), lr=0.001)\n",
    "decayLR = scheduler.StepLR(optimizer, step_size=1, gamma=0.5)\n",
    "sig = nn.Sigmoid()\n",
    "softmax = nn.Softmax(dim=1)\n",
    "\n",
    "fast_tokenizer = DistilBertTokenizerFast.from_pretrained(\"distilbert-base-uncased\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "d69a1c3f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1, Batch 10, Loss 0.64224425 \n",
      "Epoch 1, Batch 20, Loss 0.64836451 \n",
      "Epoch 1, Batch 30, Loss 0.65039027 \n",
      "Epoch 1, Batch 40, Loss 0.62774490 \n",
      "Epoch 1, Batch 50, Loss 0.62291050 \n",
      "Epoch 1, Batch 60, Loss 0.65822278 \n",
      "Epoch 1, Batch 70, Loss 0.65275683 \n",
      "Epoch 1, Batch 80, Loss 0.64150244 \n",
      "Epoch 1, Batch 90, Loss 0.62177199 \n",
      "Epoch 1, Batch 100, Loss 0.64150430 \n",
      "Epoch 1, Batch 110, Loss 0.63525578 \n",
      "Epoch 1, Batch 120, Loss 0.62634476 \n",
      "Epoch 1, Batch 130, Loss 0.60285869 \n",
      "Epoch 1, Batch 140, Loss 0.60779629 \n",
      "Epoch 1, Batch 150, Loss 0.63965335 \n",
      "Epoch 1, Batch 160, Loss 0.62533965 \n",
      "Epoch 1, Batch 170, Loss 0.60029997 \n",
      "Epoch 1, Batch 180, Loss 0.63615797 \n",
      "Epoch 1, Batch 190, Loss 0.63292221 \n",
      "Epoch 1, Batch 200, Loss 0.64939809 \n",
      "Epoch 1, Batch 210, Loss 0.63143033 \n",
      "Epoch 1, Batch 220, Loss 0.62029364 \n",
      "Epoch 1, Batch 230, Loss 0.58648555 \n",
      "Epoch 1, Batch 240, Loss 0.63435921 \n",
      "Epoch 1, Batch 250, Loss 0.61780897 \n",
      "Epoch 1, Batch 260, Loss 0.57746214 \n",
      "Epoch 1, Batch 270, Loss 0.62100384 \n",
      "Epoch 1, Batch 280, Loss 0.65069646 \n",
      "Epoch 1, Batch 290, Loss 0.61061488 \n",
      "Epoch 1, Batch 300, Loss 0.63605536 \n",
      "Epoch 1, Batch 310, Loss 0.59890153 \n",
      "Epoch 1, Batch 320, Loss 0.60668693 \n",
      "Epoch 1, Batch 330, Loss 0.63924298 \n",
      "Epoch 1, Batch 340, Loss 0.65606309 \n",
      "Epoch 1, Batch 350, Loss 0.61781108 \n",
      "Epoch 1, Batch 360, Loss 0.60779980 \n",
      "Epoch 1, Batch 370, Loss 0.62666631 \n",
      "Epoch 1, Batch 380, Loss 0.62535838 \n",
      "Epoch 1, Batch 390, Loss 0.61460423 \n",
      "Epoch 1, Batch 400, Loss 0.61863273 \n",
      "Epoch 1, Batch 410, Loss 0.58509731 \n",
      "Epoch 1, Batch 420, Loss 0.59801727 \n",
      "Epoch 1, Batch 430, Loss 0.64638109 \n",
      "Epoch 1, Batch 440, Loss 0.58316070 \n",
      "Epoch 1, Batch 450, Loss 0.64881452 \n",
      "Epoch 1, Batch 460, Loss 0.60252408 \n",
      "Epoch 1, Batch 470, Loss 0.62356882 \n",
      "Epoch 1, Batch 480, Loss 0.66246065 \n",
      "Epoch 1, Batch 490, Loss 0.63979910 \n",
      "Epoch 1, Batch 500, Loss 0.61506463 \n",
      "Epoch 1, Batch 510, Loss 0.60840944 \n",
      "Epoch 1, Batch 520, Loss 0.65582355 \n",
      "Epoch 1, Batch 530, Loss 0.60875090 \n",
      "Epoch 1, Batch 540, Loss 0.62366875 \n",
      "Epoch 1, Batch 550, Loss 0.58879535 \n",
      "Epoch 1, Batch 560, Loss 0.61720745 \n",
      "Epoch 1, Batch 570, Loss 0.61250412 \n",
      "Epoch 1, Batch 580, Loss 0.60516359 \n",
      "Epoch 1, Batch 590, Loss 0.65879372 \n",
      "Epoch 1, Batch 600, Loss 0.64210726 \n",
      "Epoch 1, Batch 610, Loss 0.56537057 \n",
      "Epoch 1, Batch 620, Loss 0.58969552 \n",
      "Epoch 1, Batch 630, Loss 0.61367891 \n",
      "Epoch 1, Batch 640, Loss 0.62216423 \n",
      "Epoch 1, Batch 650, Loss 0.69307046 \n",
      "Epoch 1, Batch 660, Loss 0.63262557 \n",
      "Epoch 1, Batch 670, Loss 0.65093942 \n",
      "Epoch 1, Batch 680, Loss 0.60679398 \n",
      "Epoch 1, Batch 690, Loss 0.62450203 \n",
      "Epoch 1, Batch 700, Loss 0.64818223 \n",
      "Epoch 1, Batch 710, Loss 0.58744994 \n",
      "Epoch 1, Batch 720, Loss 0.61687028 \n",
      "Epoch 1, Batch 730, Loss 0.58466989 \n",
      "Epoch 1, Batch 740, Loss 0.61637093 \n",
      "Epoch 1, Batch 750, Loss 0.61298032 \n",
      "Epoch 1, Batch 760, Loss 0.62502903 \n",
      "Epoch 1, Batch 770, Loss 0.62941744 \n",
      "Epoch 1, Batch 780, Loss 0.59503945 \n",
      "Epoch 1, Batch 790, Loss 0.62582828 \n",
      "Epoch 1, Batch 800, Loss 0.62733026 \n",
      "Epoch 1, Batch 810, Loss 0.59720387 \n",
      "Epoch 1, Batch 820, Loss 0.65637900 \n",
      "Epoch 1, Batch 830, Loss 0.61863188 \n",
      "Epoch 1, Batch 840, Loss 0.57939661 \n",
      "Epoch 1, Batch 850, Loss 0.61303431 \n",
      "Epoch 1, Batch 860, Loss 0.63068828 \n",
      "Epoch 1, Batch 870, Loss 0.65484797 \n",
      "Epoch 1, Batch 880, Loss 0.60999551 \n",
      "Epoch 1, Batch 890, Loss 0.64817772 \n",
      "Epoch 1, Batch 900, Loss 0.63025907 \n",
      "Epoch 1, Batch 910, Loss 0.60431930 \n",
      "Epoch 1, Batch 920, Loss 0.61309030 \n",
      "Epoch 1, Batch 930, Loss 0.60754587 \n",
      "Epoch 1, Batch 940, Loss 0.59252051 \n",
      "Epoch 1, Batch 950, Loss 0.61997226 \n",
      "Epoch 1, Batch 960, Loss 0.60474982 \n",
      "Epoch 1, Batch 970, Loss 0.63564361 \n",
      "Epoch 1, Batch 980, Loss 0.61992128 \n",
      "Epoch 1, Batch 990, Loss 0.61974132 \n",
      "Epoch 1, Batch 1000, Loss 0.59946370 \n",
      "Epoch 1, Batch 1010, Loss 0.59742975 \n",
      "Epoch 1, Batch 1020, Loss 0.59483103 \n",
      "Epoch 1, Batch 1030, Loss 0.60401825 \n",
      "Epoch 1, Batch 1040, Loss 0.60047227 \n",
      "Epoch 1, Batch 1050, Loss 0.65035813 \n",
      "Epoch 1, Batch 1060, Loss 0.66982109 \n",
      "Epoch 1, Batch 1070, Loss 0.57760843 \n",
      "Epoch 1, Batch 1080, Loss 0.62526687 \n",
      "Epoch 1, Batch 1090, Loss 0.65039239 \n",
      "Epoch 1, Batch 1100, Loss 0.58969347 \n",
      "Epoch 1, Batch 1110, Loss 0.63957795 \n",
      "Epoch 1, Batch 1120, Loss 0.56694818 \n",
      "Epoch 1, Batch 1130, Loss 0.60979613 \n",
      "Epoch 1, Batch 1140, Loss 0.63560926 \n",
      "Epoch 1, Batch 1150, Loss 0.64387458 \n",
      "Epoch 1, Batch 1160, Loss 0.61322796 \n",
      "Epoch 1, Batch 1170, Loss 0.62278977 \n",
      "Epoch 1, Batch 1180, Loss 0.61582333 \n",
      "Epoch 1, Batch 1190, Loss 0.62453127 \n",
      "Epoch 1, Batch 1200, Loss 0.58486817 \n",
      "Epoch 1, Batch 1210, Loss 0.58569506 \n",
      "Epoch 1, Batch 1220, Loss 0.58280777 \n",
      "Epoch 1, Batch 1230, Loss 0.57971703 \n",
      "Epoch 1, Batch 1240, Loss 0.61092492 \n",
      "Epoch 1, Batch 1250, Loss 0.59021370 \n",
      "Epoch 1, Batch 1260, Loss 0.56964045 \n",
      "Epoch 1, Batch 1270, Loss 0.62644219 \n",
      "Epoch 1, Batch 1280, Loss 0.62143925 \n",
      "Epoch 1, Batch 1290, Loss 0.63832364 \n",
      "Epoch 1, Batch 1300, Loss 0.59339111 \n",
      "Epoch 1, Batch 1310, Loss 0.64555517 \n",
      "Epoch 1, Batch 1320, Loss 0.62510898 \n",
      "Epoch 1, Batch 1330, Loss 0.62991126 \n",
      "Epoch 1, Batch 1340, Loss 0.60971872 \n",
      "Epoch 1, Batch 1350, Loss 0.60075102 \n",
      "Epoch 1, Batch 1360, Loss 0.64001841 \n",
      "Epoch 1, Batch 1370, Loss 0.56997527 \n",
      "Epoch 1, Batch 1380, Loss 0.60549176 \n",
      "Epoch 1, Batch 1390, Loss 0.61095608 \n",
      "Epoch 1, Batch 1400, Loss 0.61808454 \n",
      "Epoch 1, Batch 1410, Loss 0.64630296 \n",
      "Epoch 1, Batch 1420, Loss 0.63647801 \n",
      "Epoch 1, Batch 1430, Loss 0.59803412 \n",
      "Epoch 1, Batch 1440, Loss 0.58910671 \n",
      "Epoch 1, Batch 1450, Loss 0.59682611 \n",
      "Epoch 1, Batch 1460, Loss 0.59897859 \n",
      "Epoch 1, Batch 1470, Loss 0.57413424 \n",
      "Epoch 1, Batch 1480, Loss 0.58789804 \n",
      "Epoch 1, Batch 1490, Loss 0.57753466 \n",
      "Epoch 1, Batch 1500, Loss 0.62904069 \n",
      "Epoch 1, Batch 1510, Loss 0.61457497 \n",
      "Epoch 1, Batch 1520, Loss 0.60321266 \n",
      "Epoch 1, Batch 1530, Loss 0.61152227 \n",
      "Epoch 1, Batch 1540, Loss 0.62606035 \n",
      "Epoch 1, Batch 1550, Loss 0.57874086 \n",
      "Epoch 1, Batch 1560, Loss 0.64492888 \n",
      "Epoch 1, Batch 1570, Loss 0.62645204 \n",
      "Epoch 1, Batch 1580, Loss 0.57380115 \n",
      "Epoch 1, Batch 1590, Loss 0.61671170 \n",
      "Epoch 1, Batch 1600, Loss 0.63835596 \n",
      "Epoch 1, Batch 1610, Loss 0.61425482 \n",
      "Epoch 1, Batch 1620, Loss 0.59032352 \n",
      "Epoch 1, Batch 1630, Loss 0.66247905 \n",
      "Epoch 1, Batch 1640, Loss 0.62029898 \n",
      "Epoch 1, Batch 1650, Loss 0.59716981 \n",
      "Epoch 1, Batch 1660, Loss 0.56822620 \n",
      "Epoch 1, Batch 1670, Loss 0.53776450 \n",
      "Epoch 1, Batch 1680, Loss 0.62359561 \n",
      "Epoch 1, Batch 1690, Loss 0.57873678 \n",
      "Epoch 1, Batch 1700, Loss 0.69467044 \n",
      "Epoch 1, Batch 1710, Loss 0.62143227 \n",
      "Epoch 1, Batch 1720, Loss 0.62524717 \n",
      "Epoch 1, Batch 1730, Loss 0.59789270 \n",
      "Epoch 1, Batch 1740, Loss 0.56718906 \n",
      "Epoch 1, Batch 1750, Loss 0.59211854 \n",
      "Epoch 1, Batch 1760, Loss 0.59370788 \n",
      "Epoch 1, Batch 1770, Loss 0.59213739 \n",
      "Epoch 1, Batch 1780, Loss 0.58605904 \n",
      "Epoch 1, Batch 1790, Loss 0.64261880 \n",
      "Epoch 1, Batch 1800, Loss 0.66838384 \n",
      "Epoch 1, Batch 1810, Loss 0.59148408 \n",
      "Epoch 1, Batch 1820, Loss 0.63035017 \n",
      "Epoch 1, Batch 1830, Loss 0.66570655 \n",
      "Epoch 1, Batch 1840, Loss 0.64779543 \n",
      "Epoch 1, Batch 1850, Loss 0.60135836 \n",
      "Epoch 1, Batch 1860, Loss 0.65041003 \n",
      "Epoch 1, Batch 1870, Loss 0.61575888 \n",
      "Epoch 1, Batch 1880, Loss 0.65086513 \n",
      "Epoch 1, Batch 1890, Loss 0.63801365 \n",
      "Epoch 1, Batch 1900, Loss 0.64304692 \n",
      "Epoch 1, Batch 1910, Loss 0.63885739 \n",
      "Epoch 1, Batch 1920, Loss 0.58602751 \n",
      "Epoch 1, Batch 1930, Loss 0.67828875 \n",
      "Epoch 1, Batch 1940, Loss 0.60803401 \n",
      "Epoch 1, Batch 1950, Loss 0.61006164 \n",
      "Epoch 1, Batch 1960, Loss 0.61411783 \n",
      "Epoch 1, Batch 1970, Loss 0.65064511 \n",
      "Epoch 1, Batch 1980, Loss 0.63757446 \n",
      "Epoch 1, Batch 1990, Loss 0.62092387 \n",
      "Epoch 1, Batch 2000, Loss 0.63616206 \n",
      "Epoch 1, Batch 2010, Loss 0.60620115 \n",
      "Epoch 1, Batch 2020, Loss 0.61705270 \n",
      "Epoch 1, Batch 2030, Loss 0.68125330 \n",
      "Epoch 1, Batch 2040, Loss 0.60162493 \n",
      "Epoch 1, Batch 2050, Loss 0.57938155 \n",
      "Epoch 1, Batch 2060, Loss 0.52670391 \n",
      "Epoch 1, Batch 2070, Loss 0.65050634 \n",
      "Epoch 1, Batch 2080, Loss 0.62273019 \n",
      "Epoch 1, Batch 2090, Loss 0.63741255 \n",
      "Epoch 1, Batch 2100, Loss 0.62683101 \n",
      "Epoch 1, Batch 2110, Loss 0.66062621 \n",
      "Epoch 1, Batch 2120, Loss 0.62807537 \n",
      "Epoch 1, Batch 2130, Loss 0.65078835 \n",
      "Epoch 1, Batch 2140, Loss 0.60392601 \n",
      "Epoch 1, Batch 2150, Loss 0.65047103 \n",
      "Epoch 1, Batch 2160, Loss 0.58767500 \n",
      "Epoch 1, Batch 2170, Loss 0.63500836 \n",
      "Epoch 1, Batch 2180, Loss 0.59201358 \n",
      "Epoch 1, Batch 2190, Loss 0.63332130 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1, Batch 2200, Loss 0.54982919 \n",
      "Epoch 1, Batch 2210, Loss 0.63824033 \n",
      "Epoch 1, Batch 2220, Loss 0.59109630 \n",
      "Epoch 1, Batch 2230, Loss 0.59791929 \n",
      "Epoch 1, Batch 2240, Loss 0.63411282 \n",
      "Epoch 1, Batch 2250, Loss 0.64980086 \n",
      "Epoch 1, Batch 2260, Loss 0.68977098 \n",
      "Epoch 1, Batch 2270, Loss 0.60859207 \n",
      "Epoch 1, Batch 2280, Loss 0.64009657 \n",
      "Epoch 1, Batch 2290, Loss 0.60021194 \n",
      "Epoch 1, Batch 2300, Loss 0.60243666 \n",
      "Epoch 1, Batch 2310, Loss 0.61239148 \n",
      "Epoch 1, Batch 2320, Loss 0.67750339 \n",
      "Epoch 1, Batch 2330, Loss 0.63908034 \n",
      "Epoch 1, Batch 2340, Loss 0.56598020 \n",
      "Epoch 1, Batch 2350, Loss 0.54303707 \n",
      "Epoch 1, Batch 2360, Loss 0.61179126 \n",
      "Epoch 1, Batch 2370, Loss 0.59703642 \n",
      "Epoch 1, Batch 2380, Loss 0.63391497 \n",
      "Epoch 1, Batch 2390, Loss 0.63800979 \n",
      "Epoch 1, Batch 2400, Loss 0.62866371 \n",
      "Epoch 1, Batch 2410, Loss 0.63891471 \n",
      "Epoch 1, Batch 2420, Loss 0.59842714 \n",
      "Epoch 1, Batch 2430, Loss 0.59928231 \n",
      "Epoch 1, Batch 2440, Loss 0.59963068 \n",
      "Epoch 1, Batch 2450, Loss 0.57206851 \n",
      "Epoch 1, Batch 2460, Loss 0.64003971 \n",
      "Epoch 1, Batch 2470, Loss 0.65725091 \n",
      "Epoch 1, Batch 2480, Loss 0.64153214 \n",
      "Epoch 1, Batch 2490, Loss 0.62721283 \n",
      "Epoch 1, Batch 2500, Loss 0.66156401 \n",
      "Epoch 1, Batch 2510, Loss 0.58924396 \n",
      "Epoch 1, Batch 2520, Loss 0.58424171 \n",
      "Epoch 1, Batch 2530, Loss 0.60093381 \n",
      "Epoch 1, Batch 2540, Loss 0.62496992 \n",
      "Epoch 1, Batch 2550, Loss 0.60457477 \n",
      "Epoch 1, Batch 2560, Loss 0.66259305 \n",
      "Epoch 1, Batch 2570, Loss 0.62384221 \n",
      "Epoch 1, Batch 2580, Loss 0.64423165 \n",
      "Epoch 1, Batch 2590, Loss 0.64878705 \n",
      "Epoch 1, Batch 2600, Loss 0.61329175 \n",
      "Epoch 1, Batch 2610, Loss 0.63730429 \n",
      "Epoch 1, Batch 2620, Loss 0.57879492 \n",
      "Epoch 1, Batch 2630, Loss 0.61907905 \n",
      "Epoch 1, Batch 2640, Loss 0.64963296 \n",
      "Epoch 1, Batch 2650, Loss 0.61844528 \n",
      "Epoch 1, Batch 2660, Loss 0.58755400 \n",
      "Epoch 1, Batch 2670, Loss 0.57773905 \n",
      "Epoch 1, Batch 2680, Loss 0.64108307 \n",
      "Epoch 1, Batch 2690, Loss 0.59511643 \n",
      "Epoch 1, Batch 2700, Loss 0.64179726 \n",
      "Epoch 1, Batch 2710, Loss 0.58199359 \n",
      "Epoch 1, Batch 2720, Loss 0.61328398 \n",
      "Epoch 1, Batch 2730, Loss 0.65275543 \n",
      "Epoch 1, Batch 2740, Loss 0.67285162 \n",
      "Epoch 1, Batch 2750, Loss 0.57743087 \n",
      "Epoch 1, Batch 2760, Loss 0.65995982 \n",
      "Epoch 1, Batch 2770, Loss 0.58342451 \n",
      "Epoch 1, Batch 2780, Loss 0.63175736 \n",
      "Epoch 1, Batch 2790, Loss 0.60913388 \n",
      "Epoch 1, Batch 2800, Loss 0.61115737 \n",
      "Epoch 1, Batch 2810, Loss 0.62514677 \n",
      "Epoch 1, Batch 2820, Loss 0.58715691 \n",
      "Epoch 1, Batch 2830, Loss 0.59887911 \n",
      "Epoch 1, Batch 2840, Loss 0.60660138 \n",
      "Epoch 1, Batch 2850, Loss 0.61467066 \n",
      "Epoch 1, Batch 2860, Loss 0.59320524 \n",
      "Epoch 1, Batch 2870, Loss 0.65065976 \n",
      "Epoch 1, Batch 2880, Loss 0.63555991 \n",
      "Epoch 1, Batch 2890, Loss 0.59471804 \n",
      "Epoch 1, Batch 2900, Loss 0.60459170 \n",
      "Epoch 1, Batch 2910, Loss 0.65007967 \n",
      "Epoch 1, Batch 2920, Loss 0.65091339 \n",
      "Epoch 1, Batch 2930, Loss 0.60276698 \n",
      "Epoch 1, Batch 2940, Loss 0.62836513 \n",
      "Epoch 1, Batch 2950, Loss 0.61047881 \n",
      "Epoch 1, Batch 2960, Loss 0.63030703 \n",
      "Epoch 1, Batch 2970, Loss 0.58790520 \n",
      "Epoch 1, Batch 2980, Loss 0.57964318 \n",
      "Epoch 1, Batch 2990, Loss 0.62519357 \n",
      "Epoch 1, Batch 3000, Loss 0.63200862 \n",
      "Epoch 1, Batch 3010, Loss 0.60165801 \n",
      "Epoch 1, Batch 3020, Loss 0.57852117 \n",
      "Epoch 1, Batch 3030, Loss 0.60948194 \n",
      "Epoch 1, Batch 3040, Loss 0.58518986 \n",
      "Epoch 1, Batch 3050, Loss 0.57688183 \n",
      "Epoch 1, Batch 3060, Loss 0.60261315 \n",
      "Epoch 1, Batch 3070, Loss 0.59515377 \n",
      "Epoch 1, Batch 3080, Loss 0.53662022 \n",
      "Epoch 1, Batch 3090, Loss 0.57650432 \n",
      "Epoch 1, Batch 3100, Loss 0.59432433 \n",
      "Epoch 1, Batch 3110, Loss 0.65818413 \n",
      "Epoch 1, Batch 3120, Loss 0.63605417 \n",
      "Epoch 1, Batch 3130, Loss 0.60679165 \n",
      "Epoch 1, Batch 3140, Loss 0.59669385 \n",
      "Epoch 1, Batch 3150, Loss 0.56049876 \n",
      "Epoch 1, Batch 3160, Loss 0.63506348 \n",
      "Epoch 1, Batch 3170, Loss 0.63737441 \n",
      "Epoch 1, Batch 3180, Loss 0.59554496 \n",
      "Epoch 1, Batch 3190, Loss 0.62019485 \n",
      "Epoch 1, Batch 3200, Loss 0.62963311 \n",
      "Epoch 1, Batch 3210, Loss 0.59965577 \n",
      "Epoch 1, Batch 3220, Loss 0.66191548 \n",
      "Epoch 1, Batch 3230, Loss 0.61086591 \n",
      "Epoch 1, Batch 3240, Loss 0.60499946 \n",
      "Epoch 1, Batch 3250, Loss 0.61030657 \n",
      "Epoch 1, Batch 3260, Loss 0.58842015 \n",
      "Epoch 1, Batch 3270, Loss 0.61753841 \n",
      "Epoch 1, Batch 3280, Loss 0.60824560 \n",
      "Epoch 1, Batch 3290, Loss 0.63101262 \n",
      "Epoch 1, Batch 3300, Loss 0.66899190 \n",
      "Epoch 1, Batch 3310, Loss 0.59071409 \n",
      "Epoch 1, Batch 3320, Loss 0.61204242 \n",
      "Epoch 1, Batch 3330, Loss 0.61025456 \n",
      "Epoch 1, Batch 3340, Loss 0.59048218 \n",
      "Epoch 1, Batch 3350, Loss 0.60203336 \n",
      "Epoch 1, Batch 3360, Loss 0.60604042 \n",
      "Epoch 1, Batch 3370, Loss 0.56631145 \n",
      "Epoch 1, Batch 3380, Loss 0.60343312 \n",
      "Epoch 1, Batch 3390, Loss 0.59842871 \n",
      "Epoch 1, Batch 3400, Loss 0.66420588 \n",
      "Epoch 1, Batch 3410, Loss 0.62507676 \n",
      "Epoch 1, Batch 3420, Loss 0.59431401 \n",
      "Epoch 1, Batch 3430, Loss 0.64439825 \n",
      "Epoch 1, Batch 3440, Loss 0.62018588 \n",
      "Epoch 1, Batch 3450, Loss 0.65195445 \n",
      "Epoch 1, Batch 3460, Loss 0.61590293 \n",
      "Epoch 1, Batch 3470, Loss 0.64190304 \n",
      "Epoch 1, Batch 3480, Loss 0.56897838 \n",
      "Epoch 1, Batch 3490, Loss 0.60337833 \n",
      "Epoch 1, Batch 3500, Loss 0.55766658 \n",
      "Epoch 1, Batch 3510, Loss 0.60269147 \n",
      "Epoch 1, Batch 3520, Loss 0.60514850 \n",
      "Epoch 1, Batch 3530, Loss 0.58824454 \n",
      "Epoch 1, Batch 3540, Loss 0.59431146 \n",
      "Epoch 1, Batch 3550, Loss 0.62559735 \n",
      "Epoch 1, Batch 3560, Loss 0.61209616 \n",
      "Epoch 1, Batch 3570, Loss 0.58626266 \n",
      "Epoch 1, Batch 3580, Loss 0.57937356 \n",
      "Epoch 1, Batch 3590, Loss 0.59652831 \n",
      "Epoch 1, Batch 3600, Loss 0.61549751 \n",
      "Epoch 1, Batch 3610, Loss 0.61017438 \n",
      "Epoch 1, Batch 3620, Loss 0.60207570 \n",
      "Epoch 1, Batch 3630, Loss 0.59848596 \n",
      "Epoch 1, Batch 3640, Loss 0.64952300 \n",
      "Epoch 1, Batch 3650, Loss 0.59681042 \n",
      "Epoch 1, Batch 3660, Loss 0.62161516 \n",
      "Epoch 1, Batch 3670, Loss 0.57196992 \n",
      "Epoch 1, Batch 3680, Loss 0.58436647 \n",
      "Epoch 1, Batch 3690, Loss 0.61206861 \n",
      "Epoch 1, Batch 3700, Loss 0.58425334 \n",
      "Epoch 1, Batch 3710, Loss 0.64276770 \n",
      "Epoch 1, Batch 3720, Loss 0.61204249 \n",
      "Epoch 1, Batch 3730, Loss 0.58162320 \n",
      "Epoch 1, Batch 3740, Loss 0.58440506 \n",
      "Epoch 1, Batch 3750, Loss 0.61075355 \n",
      "Epoch 1, Batch 3760, Loss 0.59827581 \n",
      "Epoch 1, Batch 3770, Loss 0.63668576 \n",
      "Epoch 1, Batch 3780, Loss 0.53110327 \n",
      "Epoch 1, Batch 3790, Loss 0.64960954 \n",
      "Epoch 1, Batch 3800, Loss 0.61121146 \n",
      "Epoch 1, Batch 3810, Loss 0.57087759 \n",
      "Epoch 1, Batch 3820, Loss 0.63562924 \n",
      "Epoch 1, Batch 3830, Loss 0.60613922 \n",
      "Epoch 1, Batch 3840, Loss 0.67370218 \n",
      "Epoch 1, Batch 3850, Loss 0.64225952 \n",
      "Epoch 1, Batch 3860, Loss 0.62867028 \n",
      "Epoch 1, Batch 3870, Loss 0.59477886 \n",
      "Epoch 1, Batch 3880, Loss 0.65443581 \n",
      "Epoch 1, Batch 3890, Loss 0.60426887 \n",
      "Epoch 1, Batch 3900, Loss 0.58540835 \n",
      "Epoch 1, Batch 3910, Loss 0.57487242 \n",
      "Epoch 1, Batch 3920, Loss 0.62337171 \n",
      "Epoch 1, Batch 3930, Loss 0.63101076 \n",
      "Epoch 1, Batch 3940, Loss 0.58066187 \n",
      "Epoch 1, Batch 3950, Loss 0.65250900 \n",
      "Epoch 1, Batch 3960, Loss 0.58899322 \n",
      "Epoch 1, Batch 3970, Loss 0.54647169 \n",
      "Epoch 1, Batch 3980, Loss 0.59214828 \n",
      "Epoch 1, Batch 3990, Loss 0.60341103 \n",
      "Epoch 1, Batch 4000, Loss 0.63353856 \n",
      "Epoch 1, Batch 4010, Loss 0.62330907 \n",
      "Epoch 1, Batch 4020, Loss 0.61297903 \n",
      "Epoch 1, Batch 4030, Loss 0.64838443 \n",
      "Epoch 1, Batch 4040, Loss 0.61603910 \n",
      "Epoch 1, Batch 4050, Loss 0.62315608 \n",
      "Epoch 1, Batch 4060, Loss 0.62191051 \n",
      "Epoch 1, Batch 4070, Loss 0.60704791 \n",
      "Epoch 1, Batch 4080, Loss 0.65757470 \n",
      "Epoch 1, Batch 4090, Loss 0.60863235 \n",
      "Epoch 1, Batch 4100, Loss 0.58163844 \n",
      "Epoch 1, Batch 4110, Loss 0.59063854 \n",
      "Epoch 1, Batch 4120, Loss 0.60528313 \n",
      "Epoch 1, Batch 4130, Loss 0.58598971 \n",
      "Epoch 1, Batch 4140, Loss 0.56367663 \n",
      "Epoch 1, Batch 4150, Loss 0.60706911 \n",
      "Epoch 1, Batch 4160, Loss 0.60585558 \n",
      "Epoch 1, Batch 4170, Loss 0.64870957 \n",
      "Epoch 1, Batch 4180, Loss 0.54351918 \n",
      "Epoch 1, Batch 4190, Loss 0.55451959 \n",
      "Epoch 1, Batch 4200, Loss 0.66006390 \n",
      "Epoch 1, Batch 4210, Loss 0.59632466 \n",
      "Epoch 1, Batch 4220, Loss 0.67278014 \n",
      "Epoch 1, Batch 4230, Loss 0.59809917 \n",
      "Epoch 1, Batch 4240, Loss 0.66337870 \n",
      "Epoch 1, Batch 4250, Loss 0.65037221 \n",
      "Epoch 1, Batch 4260, Loss 0.57663246 \n",
      "Epoch 1, Batch 4270, Loss 0.62297527 \n",
      "Epoch 1, Batch 4280, Loss 0.62833112 \n",
      "Epoch 1, Batch 4290, Loss 0.58268699 \n",
      "Epoch 1, Batch 4300, Loss 0.59426103 \n",
      "Epoch 1, Batch 4310, Loss 0.56979599 \n",
      "Epoch 1, Batch 4320, Loss 0.60049269 \n",
      "Epoch 1, Batch 4330, Loss 0.59761243 \n",
      "Epoch 1, Batch 4340, Loss 0.56856146 \n",
      "Epoch 1, Batch 4350, Loss 0.59748822 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1, Batch 4360, Loss 0.60248836 \n",
      "Epoch 1, Batch 4370, Loss 0.60971016 \n",
      "Epoch 1, Batch 4380, Loss 0.65187132 \n",
      "Epoch 1, Batch 4390, Loss 0.59664699 \n",
      "Epoch 1, Batch 4400, Loss 0.63146396 \n",
      "Epoch 1, Batch 4410, Loss 0.66452782 \n",
      "Epoch 1, Batch 4420, Loss 0.63943095 \n",
      "Epoch 1, Batch 4430, Loss 0.63465079 \n",
      "Epoch 1, Batch 4440, Loss 0.59810226 \n",
      "Epoch 1, Batch 4450, Loss 0.57400634 \n",
      "Epoch 1, Batch 4460, Loss 0.61971514 \n",
      "Epoch 1, Batch 4470, Loss 0.66124590 \n",
      "Epoch 1, Batch 4480, Loss 0.60826603 \n",
      "Epoch 1, Batch 4490, Loss 0.61319262 \n",
      "Epoch 1, Batch 4500, Loss 0.56555977 \n",
      "Epoch 1, Batch 4510, Loss 0.62181080 \n",
      "Epoch 1, Batch 4520, Loss 0.63433613 \n",
      "Epoch 1, Batch 4530, Loss 0.65439582 \n",
      "Epoch 1, Batch 4540, Loss 0.64061729 \n",
      "Epoch 1, Batch 4550, Loss 0.60254397 \n",
      "Epoch 1, Batch 4560, Loss 0.61369382 \n",
      "Epoch 1, Batch 4570, Loss 0.56131551 \n",
      "Epoch 1, Batch 4580, Loss 0.61824968 \n",
      "Epoch 1, Batch 4590, Loss 0.60993657 \n",
      "Epoch 1, Batch 4600, Loss 0.63904563 \n",
      "Epoch 1, Batch 4610, Loss 0.59108267 \n",
      "Epoch 1, Batch 4620, Loss 0.58131711 \n",
      "Epoch 1, Batch 4630, Loss 0.61702278 \n",
      "Epoch 1, Batch 4640, Loss 0.59837393 \n",
      "Epoch 1, Batch 4650, Loss 0.61036052 \n",
      "Epoch 1, Batch 4660, Loss 0.58386880 \n",
      "Epoch 1, Batch 4670, Loss 0.57696139 \n",
      "Epoch 1, Batch 4680, Loss 0.61842760 \n",
      "Epoch 1, Batch 4690, Loss 0.56247628 \n",
      "Epoch 1, Batch 4700, Loss 0.64040321 \n",
      "Epoch 1, Batch 4710, Loss 0.61776475 \n",
      "Epoch 1, Batch 4720, Loss 0.58192960 \n",
      "Epoch 1, Batch 4730, Loss 0.57771634 \n",
      "Epoch 1, Batch 4740, Loss 0.67626516 \n",
      "Epoch 1, Batch 4750, Loss 0.56083315 \n",
      "Epoch 1, Batch 4760, Loss 0.59506375 \n",
      "Epoch 1, Batch 4770, Loss 0.61877443 \n",
      "Epoch 1, Batch 4780, Loss 0.66489631 \n",
      "Epoch 1, Batch 4790, Loss 0.57032809 \n",
      "Epoch 1, Batch 4800, Loss 0.61124058 \n",
      "Epoch 1, Batch 4810, Loss 0.61347049 \n",
      "Epoch 1, Batch 4820, Loss 0.64589987 \n",
      "Epoch 1, Batch 4830, Loss 0.65692844 \n",
      "Epoch 1, Batch 4840, Loss 0.62550628 \n",
      "Epoch 1, Batch 4850, Loss 0.64152170 \n",
      "Epoch 1, Batch 4860, Loss 0.54524717 \n",
      "Epoch 1, Batch 4870, Loss 0.64158776 \n",
      "Epoch 1, Batch 4880, Loss 0.60087807 \n",
      "Epoch 1, Batch 4890, Loss 0.57136460 \n",
      "Epoch 1, Batch 4900, Loss 0.61004497 \n",
      "Epoch 1, Batch 4910, Loss 0.58405326 \n",
      "Epoch 1, Batch 4920, Loss 0.63029261 \n",
      "Epoch 1, Batch 4930, Loss 0.58444591 \n",
      "Epoch 1, Batch 4940, Loss 0.63427767 \n",
      "Epoch 1, Batch 4950, Loss 0.62067668 \n",
      "Epoch 1, Batch 4960, Loss 0.57576375 \n",
      "Epoch 1, Batch 4970, Loss 0.61683733 \n",
      "Epoch 1, Batch 4980, Loss 0.61923443 \n",
      "Epoch 1, Batch 4990, Loss 0.62312700 \n",
      "Epoch 1, Batch 5000, Loss 0.58540253 \n",
      "Epoch 1, Batch 5010, Loss 0.63469221 \n",
      "Epoch 1, Batch 5020, Loss 0.59106133 \n",
      "Epoch 1, Batch 5030, Loss 0.65002037 \n",
      "Epoch 1, Batch 5040, Loss 0.58747327 \n",
      "Epoch 1, Batch 5050, Loss 0.59400910 \n",
      "Epoch 1, Batch 5060, Loss 0.59167233 \n",
      "Epoch 1, Batch 5070, Loss 0.60994719 \n",
      "Epoch 1, Batch 5080, Loss 0.64792030 \n",
      "Epoch 1, Batch 5090, Loss 0.61419214 \n",
      "Epoch 1, Batch 5100, Loss 0.60752505 \n",
      "Epoch 1, Batch 5110, Loss 0.67483462 \n",
      "Epoch 1, Batch 5120, Loss 0.59612722 \n",
      "Epoch 1, Batch 5130, Loss 0.61894742 \n",
      "Epoch 1, Batch 5140, Loss 0.64490263 \n",
      "Epoch 1, Batch 5150, Loss 0.67709940 \n",
      "Epoch 1, Batch 5160, Loss 0.56546440 \n",
      "Epoch 1, Batch 5170, Loss 0.62540084 \n",
      "Epoch 1, Batch 5180, Loss 0.61662820 \n",
      "Epoch 1, Batch 5190, Loss 0.57530219 \n",
      "Epoch 1, Batch 5200, Loss 0.59995527 \n",
      "Epoch 1, Batch 5210, Loss 0.58098200 \n",
      "Epoch 1, Batch 5220, Loss 0.56673549 \n",
      "Epoch 1, Batch 5230, Loss 0.59066787 \n",
      "Epoch 1, Batch 5240, Loss 0.65584891 \n",
      "Epoch 1, Batch 5250, Loss 0.66459333 \n",
      "Epoch 1, Batch 5260, Loss 0.62117073 \n",
      "Epoch 1, Batch 5270, Loss 0.64946537 \n",
      "Epoch 1, Batch 5280, Loss 0.64311122 \n",
      "Epoch 1, Batch 5290, Loss 0.68974434 \n",
      "Epoch 1, Batch 5300, Loss 0.59327270 \n",
      "Epoch 1, Batch 5310, Loss 0.63363937 \n",
      "Epoch 1, Batch 5320, Loss 0.60960478 \n",
      "Epoch 1, Batch 5330, Loss 0.55067671 \n",
      "Epoch 1, Batch 5340, Loss 0.57677092 \n",
      "Epoch 1, Batch 5350, Loss 0.60189044 \n",
      "Epoch 1, Batch 5360, Loss 0.59495606 \n",
      "Epoch 1, Batch 5370, Loss 0.63134037 \n",
      "Epoch 1, Batch 5380, Loss 0.66029541 \n",
      "Epoch 1, Batch 5390, Loss 0.60184470 \n",
      "Epoch 1, Batch 5400, Loss 0.64145957 \n",
      "Epoch 1, Batch 5410, Loss 0.55126864 \n",
      "Epoch 1, Batch 5420, Loss 0.61713021 \n",
      "Epoch 1, Batch 5430, Loss 0.62887830 \n",
      "Epoch 1, Batch 5440, Loss 0.60062085 \n",
      "Epoch 1, Batch 5450, Loss 0.63852839 \n",
      "Epoch 1, Batch 5460, Loss 0.60860512 \n",
      "Epoch 1, Batch 5470, Loss 0.60690351 \n",
      "Epoch 1, Batch 5480, Loss 0.58091316 \n",
      "Epoch 1, Batch 5490, Loss 0.59840189 \n",
      "Epoch 1, Batch 5500, Loss 0.62096163 \n",
      "Epoch 1, Batch 5510, Loss 0.64108549 \n",
      "Epoch 1, Batch 5520, Loss 0.57560214 \n",
      "Epoch 1, Batch 5530, Loss 0.60188137 \n",
      "Epoch 1, Batch 5540, Loss 0.58974913 \n",
      "Epoch 1, Batch 5550, Loss 0.58277364 \n",
      "Epoch 1, Batch 5560, Loss 0.63833852 \n",
      "Epoch 1, Batch 5570, Loss 0.54767424 \n",
      "Epoch 1, Batch 5580, Loss 0.59926443 \n",
      "Epoch 1, Batch 5590, Loss 0.61836979 \n",
      "Epoch 1, Batch 5600, Loss 0.59784601 \n",
      "Epoch 1, Batch 5610, Loss 0.60541281 \n",
      "Epoch 1, Batch 5620, Loss 0.61944663 \n",
      "Epoch 1, Batch 5630, Loss 0.53364826 \n",
      "Epoch 1, Batch 5640, Loss 0.57421665 \n",
      "Epoch 1, Batch 5650, Loss 0.59368699 \n",
      "Epoch 1, Batch 5660, Loss 0.59465913 \n",
      "Epoch 1, Batch 5670, Loss 0.59343490 \n",
      "Epoch 1, Batch 5680, Loss 0.61635081 \n",
      "Epoch 1, Batch 5690, Loss 0.60105629 \n",
      "Epoch 1, Batch 5700, Loss 0.60142878 \n",
      "Epoch 1, Batch 5710, Loss 0.59776738 \n",
      "Epoch 1, Batch 5720, Loss 0.54697775 \n",
      "Epoch 1, Batch 5730, Loss 0.60743324 \n",
      "Epoch 1, Batch 5740, Loss 0.63077070 \n",
      "Epoch 1, Batch 5750, Loss 0.64417062 \n",
      "Epoch 1, Batch 5760, Loss 0.56946138 \n",
      "Epoch 1, Batch 5770, Loss 0.58189982 \n",
      "Epoch 1, Batch 5780, Loss 0.62662610 \n",
      "Epoch 1, Batch 5790, Loss 0.55889434 \n",
      "Epoch 1, Batch 5800, Loss 0.60311992 \n",
      "Epoch 1, Batch 5810, Loss 0.62279855 \n",
      "Epoch 1, Batch 5820, Loss 0.62410648 \n",
      "Epoch 1, Batch 5830, Loss 0.65081577 \n",
      "Epoch 1, Batch 5840, Loss 0.64306831 \n",
      "Epoch 1, Batch 5850, Loss 0.63602903 \n",
      "Epoch 1, Batch 5860, Loss 0.55321651 \n",
      "Epoch 1, Batch 5870, Loss 0.59830425 \n",
      "Epoch 1, Batch 5880, Loss 0.56389017 \n",
      "Epoch 1, Batch 5890, Loss 0.59236668 \n",
      "Epoch 1, Batch 5900, Loss 0.61327731 \n",
      "Epoch 1, Batch 5910, Loss 0.58766606 \n",
      "Epoch 1, Batch 5920, Loss 0.62441334 \n",
      "Epoch 1, Batch 5930, Loss 0.64327034 \n",
      "Epoch 1, Batch 5940, Loss 0.58300665 \n",
      "Epoch 1, Batch 5950, Loss 0.60995805 \n",
      "Epoch 1, Batch 5960, Loss 0.59290367 \n",
      "Epoch 1, Batch 5970, Loss 0.57838503 \n",
      "Epoch 1, Batch 5980, Loss 0.61327461 \n",
      "Epoch 1, Batch 5990, Loss 0.61066418 \n",
      "Epoch 1, Batch 6000, Loss 0.59828985 \n",
      "Epoch 1, Batch 6010, Loss 0.63915603 \n",
      "Epoch 1, Batch 6020, Loss 0.61171350 \n",
      "Epoch 1, Batch 6030, Loss 0.59432798 \n",
      "Epoch 1, Batch 6040, Loss 0.62050549 \n",
      "Epoch 1, Batch 6050, Loss 0.61996519 \n",
      "Epoch 1, Batch 6060, Loss 0.69022661 \n",
      "Epoch 1, Batch 6070, Loss 0.66165119 \n",
      "Epoch 1, Batch 6080, Loss 0.59883576 \n",
      "Epoch 1, Batch 6090, Loss 0.56949721 \n",
      "Epoch 1, Batch 6100, Loss 0.57322978 \n",
      "Epoch 1, Batch 6110, Loss 0.59651594 \n",
      "Epoch 1, Batch 6120, Loss 0.61209282 \n",
      "Epoch 1, Batch 6130, Loss 0.60847189 \n",
      "Epoch 1, Batch 6140, Loss 0.61798248 \n",
      "Epoch 1, Batch 6150, Loss 0.61552630 \n",
      "Epoch 1, Batch 6160, Loss 0.58425516 \n",
      "Epoch 1, Batch 6170, Loss 0.59944897 \n",
      "Epoch 1, Batch 6180, Loss 0.60238806 \n",
      "Epoch 1, Batch 6190, Loss 0.61034489 \n",
      "Epoch 1, Batch 6200, Loss 0.62125959 \n",
      "Epoch 1, Batch 6210, Loss 0.59823397 \n",
      "Epoch 1, Batch 6220, Loss 0.60360662 \n",
      "Epoch 1, Batch 6230, Loss 0.67501193 \n",
      "Epoch 1, Batch 6240, Loss 0.54385905 \n",
      "Epoch 1, Batch 6250, Loss 0.65539339 \n",
      "------------\n",
      "Validating\n",
      "------------\n",
      "Validation loss :  0.683649730682373\n",
      "Validation loss :  0.6258529454469681\n",
      "Validation loss :  0.6473041981458664\n",
      "Validation loss :  0.6054190516471862\n",
      "Validation loss :  0.7209984004497528\n",
      "Validation loss :  0.5941847950220108\n",
      "Validation loss :  0.6244047433137894\n",
      "Validation loss :  0.6248847246170044\n",
      "Validation loss :  0.5887470006942749\n",
      "Validation loss :  0.6299384534358978\n",
      "Validation loss :  0.5973999738693238\n",
      "Validation loss :  0.6160289973020554\n",
      "Validation loss :  0.6646403193473815\n",
      "Validation loss :  0.5556186586618423\n",
      "Validation loss :  0.6268046259880066\n",
      "Validation loss :  0.5883816689252853\n",
      "Validation loss :  0.6186613023281098\n",
      "Validation loss :  0.6367329299449921\n",
      "Validation loss :  0.6230440616607666\n",
      "Validation loss :  0.5858548939228058\n",
      "Validation loss :  0.6315957307815552\n",
      "Validation loss :  0.6049544155597687\n",
      "Validation loss :  0.610661256313324\n",
      "Validation loss :  0.5985853344202041\n",
      "Validation loss :  0.6437429368495942\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation loss :  0.6020740866661072\n",
      "Validation loss :  0.6004333317279815\n",
      "Validation loss :  0.6621137917041778\n",
      "Validation loss :  0.5752637028694153\n",
      "Validation loss :  0.650105482339859\n",
      "Validation loss :  0.5842912197113037\n",
      "Validation loss :  0.6469241350889205\n",
      "Validation loss :  0.5842399746179581\n",
      "Validation loss :  0.6050743401050568\n",
      "Validation loss :  0.591272509098053\n",
      "Validation loss :  0.6026142239570618\n",
      "Validation loss :  0.6345649033784866\n",
      "Validation loss :  0.58714859187603\n",
      "Validation loss :  0.5980906009674072\n",
      "Validation loss :  0.5649757951498031\n",
      "Validation loss :  0.6250353932380677\n",
      "Validation loss :  0.6474300861358643\n",
      "Validation loss :  0.6175581008195877\n",
      "Validation loss :  0.5881327331066132\n",
      "Validation loss :  0.6613548576831818\n",
      "Validation loss :  0.5648038506507873\n",
      "Validation loss :  0.6144133687019349\n",
      "Validation loss :  0.6436223268508912\n",
      "Validation loss :  0.5895591706037522\n",
      "Validation loss :  0.5701066493988037\n",
      "Validation loss :  0.5989602744579315\n",
      "Validation loss :  0.6048187196254731\n",
      "Validation loss :  0.6070051848888397\n",
      "Validation loss :  0.5988852918148041\n",
      "Validation loss :  0.6128726780414582\n",
      "Validation loss :  0.6099204510450363\n",
      "Validation loss :  0.5837803691625595\n",
      "Validation loss :  0.6791754484176635\n",
      "Validation loss :  0.6421085864305496\n",
      "Validation loss :  0.602013349533081\n",
      "Validation loss :  0.5975840091705322\n",
      "Validation loss :  0.6286517798900604\n",
      "Validation loss :  0.5920246005058288\n",
      "Validation loss :  0.5957961499691009\n",
      "Validation loss :  0.6107203185558319\n",
      "Validation loss :  0.5952759712934494\n",
      "Validation loss :  0.6178149878978729\n",
      "Validation loss :  0.6521562874317169\n",
      "Validation loss :  0.5940170586109161\n",
      "Validation loss :  0.6053444892168045\n",
      "Validation loss :  0.6334855318069458\n",
      "Validation loss :  0.5412043809890748\n",
      "Validation loss :  0.6739311397075654\n",
      "Validation loss :  0.600643339753151\n",
      "Validation loss :  0.6465200424194336\n",
      "Validation loss :  0.6464494585990905\n",
      "Validation loss :  0.589591783285141\n",
      "Validation loss :  0.6697683870792389\n",
      "Validation loss :  0.5490397870540619\n",
      "Validation loss :  0.6332999348640442\n",
      "Validation loss :  0.6032924950122833\n",
      "Validation loss :  0.5713375031948089\n",
      "Validation loss :  0.5850139349699021\n",
      "Validation loss :  0.6661512076854705\n",
      "Validation loss :  0.6141322046518326\n",
      "Validation loss :  0.5621453583240509\n",
      "Validation loss :  0.6358585625886917\n",
      "Validation loss :  0.6093493700027466\n",
      "Validation loss :  0.6510034620761871\n",
      "Validation loss :  0.6302728563547134\n",
      "Validation loss :  0.5889652460813523\n",
      "Validation loss :  0.5919883221387863\n",
      "Validation loss :  0.6323517620563507\n",
      "Validation loss :  0.6133285850286484\n",
      "Validation loss :  0.6130997538566589\n",
      "Validation loss :  0.5681726694107055\n",
      "Validation loss :  0.5975197643041611\n",
      "Validation loss :  0.6453526705503464\n",
      "Validation loss :  0.6182646572589874\n",
      "Validation loss :  0.6279183089733124\n",
      "Validation loss :  0.6039554715156555\n",
      "Validation loss :  0.5988161206245423\n",
      "Validation loss :  0.6108377486467361\n",
      "Validation loss :  0.5941187798976898\n",
      "Validation loss :  0.6318174570798873\n",
      "Validation loss :  0.5670550644397736\n",
      "Validation loss :  0.569586780667305\n",
      "Validation loss :  0.639609283208847\n",
      "Validation loss :  0.6183977842330932\n",
      "Validation loss :  0.5900931745767594\n",
      "Validation loss :  0.5847356617450714\n",
      "Validation loss :  0.6044777870178223\n",
      "Validation loss :  0.6031804144382477\n",
      "Validation loss :  0.6091582655906678\n",
      "Validation loss :  0.6316048979759217\n",
      "Validation loss :  0.6517788469791412\n",
      "Validation loss :  0.5992384105920792\n",
      "Validation loss :  0.6000545382499695\n",
      "Validation loss :  0.6483575314283371\n",
      "Validation loss :  0.5876780420541763\n",
      "Validation loss :  0.6199049323797226\n",
      "Validation loss :  0.6063931584358215\n",
      "Validation loss :  0.6109918534755707\n",
      "Validation loss :  0.6721520006656647\n",
      "Validation loss :  0.618703407049179\n",
      "---------\n",
      "Correct :  13922\n",
      "Total :  20000\n",
      "Final Validation accuracy :  0.6961\n",
      "---------\n",
      "Epoch time :  4090.715755\n",
      "---------------------------------\n",
      "Previous Learning Rate :  [0.0005]\n",
      "------------------------\n"
     ]
    }
   ],
   "source": [
    "epochs = 1\n",
    "\n",
    "losses=[]\n",
    "vacc = []\n",
    "vlosses = []\n",
    "\n",
    "for j in range(epochs):\n",
    "    \n",
    "    epoch_start = datetime.now()\n",
    "    \n",
    "    add_loss = 0.0\n",
    "    run_loss2 = 0\n",
    "    \n",
    "    for i,data in enumerate(trainloader):        \n",
    "        \n",
    "        image, caption, label = data \n",
    "        \n",
    "        caption = list(caption)\n",
    "        caption = fast_tokenizer(caption)\n",
    "        caption = resizeCaptions(caption)\n",
    "        \n",
    "        inputs = caption['input_ids']\n",
    "        op_mask = caption['op_mask']\n",
    "    \n",
    "        image = image.to(device)\n",
    "        inputs = torch.tensor(inputs).to(device)\n",
    "        op_mask = torch.tensor(op_mask).to(device)\n",
    "        label = label.to(device)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        output = model(image, inputs, op_mask)        \n",
    " \n",
    "        loss = lossFunction(output, label)\n",
    "        \n",
    "        add_loss += loss.item()\n",
    "        run_loss2 += loss.item()\n",
    "        \n",
    "        loss.backward()\n",
    "        \n",
    "        optimizer.step()\n",
    "        \n",
    "        if(i % 10 == 9):\n",
    "            print('Epoch %d, Batch %d, Loss %.8f ' % (j+1, i+1, add_loss / 10))\n",
    "            add_loss = 0.0    \n",
    "    \n",
    "    losses.append(run_loss2 / num_train_batches)\n",
    "    \n",
    "    print('------------')\n",
    "    print('Validating')\n",
    "    print('------------')\n",
    "    \n",
    "    total = 0\n",
    "    correct = 0\n",
    "    vrun_loss=0\n",
    "    \n",
    "    model.eval()\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        \n",
    "        add_vloss = 0.0\n",
    "        \n",
    "        for k, vdata in enumerate(validationloader):\n",
    "            \n",
    "            image, caption, label = vdata \n",
    "        \n",
    "            caption = list(caption)\n",
    "            caption = fast_tokenizer(caption)\n",
    "            caption = resizeCaptions(caption)\n",
    "\n",
    "            inputs = caption['input_ids']\n",
    "            op_mask = caption['op_mask']\n",
    "\n",
    "            image = image.to(device)\n",
    "            inputs = torch.tensor(inputs).to(device)\n",
    "            op_mask = torch.tensor(op_mask).to(device)\n",
    "            val_label = label.to(device)\n",
    "            \n",
    "            val_output = model(image, inputs, op_mask)\n",
    "            \n",
    "            vloss = lossFunction(val_output, val_label)\n",
    "            \n",
    "            add_vloss += vloss.item()\n",
    "            vrun_loss += vloss.item()\n",
    "            \n",
    "            if(k%10 == 9):\n",
    "                print('Validation loss : ', add_vloss / 10)\n",
    "                add_vloss = 0.0\n",
    "            \n",
    "            class_probability, class_prediction = torch.max(val_output, 1)\n",
    "            \n",
    "            total += len(val_label)\n",
    "            \n",
    "            correct += (class_prediction == val_label).sum().item()\n",
    "            \n",
    "        val_accuracy = correct / total\n",
    "        \n",
    "        vlosses.append(vrun_loss / num_validation_batches)\n",
    "        vacc.append(val_accuracy)\n",
    "        print('---------')\n",
    "        print('Correct : ', correct)\n",
    "        print('Total : ', total)\n",
    "        print('Final Validation accuracy : ', val_accuracy)\n",
    "        print('---------')\n",
    "        epoch_end = datetime.now()\n",
    "        print('Epoch time : ', (epoch_end - epoch_start).total_seconds())\n",
    "        print('---------------------------------')\n",
    "        \n",
    "    model.train()\n",
    "    decayLR.step()\n",
    "    \n",
    "    print('Previous Learning Rate : ', decayLR.get_last_lr())\n",
    "#     aa1, aa2 = model.module.getAlpha()\n",
    "#     alpha1List.append(aa1)\n",
    "#     alpha2List.append(aa2)\n",
    "\n",
    "    print('------------------------')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "4c44ce73",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correct :  15000\n",
      "Total :  20000\n",
      "Test accuracy is  0.75\n"
     ]
    }
   ],
   "source": [
    "correct = 0\n",
    "total = 0\n",
    "\n",
    "all_test_labels = torch.tensor([]).to(device)\n",
    "all_predicted_test_labels = torch.tensor([]).to(device)\n",
    "all_predicted_test_probabilities = torch.tensor([]).to(device)\n",
    "all_predicted_fake_probabilities = torch.tensor([]).to(device)\n",
    "\n",
    "with torch.no_grad():\n",
    "    \n",
    "    for i, data in enumerate(testloader):\n",
    "        \n",
    "        image, caption, label = vdata \n",
    "        \n",
    "        caption = list(caption)\n",
    "        caption = fast_tokenizer(caption)\n",
    "        caption = resizeCaptions(caption)\n",
    "\n",
    "        inputs = caption['input_ids']\n",
    "        op_mask = caption['op_mask']\n",
    "\n",
    "        image = image.to(device)\n",
    "        inputs = torch.tensor(inputs).to(device)\n",
    "        op_mask = torch.tensor(op_mask).to(device)\n",
    "        test_label = label.to(device)\n",
    "        \n",
    "        all_test_labels = torch.cat([all_test_labels, test_label])\n",
    "        \n",
    "        test_output = model(image, inputs, op_mask)\n",
    "        \n",
    "        test_output2 = softmax(test_output)\n",
    "        \n",
    "        test_output3, _ = torch.max(test_output2, dim=1)\n",
    "\n",
    "        \n",
    "#         print('Output on Test Batch')\n",
    "#         print(test_output.shape)\n",
    "#         print('------------------------')\n",
    "        \n",
    "        loss = lossFunction(test_output, test_label)\n",
    "        \n",
    "#         print('Loss value : ', loss.item())\n",
    "#         print('Acutal Labels : ', test_label)\n",
    "        \n",
    "        \n",
    "        class_probability, class_prediction = torch.max(test_output, 1)\n",
    "        \n",
    "#         print('Predicted Label : ', class_prediction)\n",
    "#         print('-----------------')\n",
    "        \n",
    "        all_predicted_test_labels = torch.cat([all_predicted_test_labels, class_prediction])\n",
    "        \n",
    "        all_predicted_test_probabilities = torch.cat([all_predicted_test_probabilities, test_output3])\n",
    "        \n",
    "        all_predicted_fake_probabilities = torch.cat([all_predicted_fake_probabilities, test_output2[:, 1]])\n",
    "        \n",
    "        total += len(test_label)\n",
    "        \n",
    "        correct += (class_prediction == test_label).sum().item()\n",
    "        \n",
    "    final_test_accuracy = correct/total\n",
    "    \n",
    "    print('Correct : ', correct)\n",
    "    print('Total : ', total)\n",
    "    print('Test accuracy is ', final_test_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "3236e46a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_8484/29604202.py:4: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.\n",
      "  plt.style.use('seaborn-bright')\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnYAAAHVCAYAAAB8NLYkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABJE0lEQVR4nO3deXiNd/7/8ddJSCSRxJpFGkJRoUQby4QpOtKGqqJaqQliKVONvWbCpda2tKVtqgxlCLpRBjW1N7SjqHUoFWuJLbFUk4iScHL//ujP+fZUkJycSNx9Pq7rXM353J/7c78/507qdd3bsRiGYQgAAAD3PZfiLgAAAADOQbADAAAwCYIdAACASRDsAAAATIJgBwAAYBIEOwAAAJMg2AEAAJgEwQ4AAMAkCHYAAAAmQbADAAAwiRIR7KZPn66QkBCVKVNGTZs21fbt22/bt1WrVrJYLLe82rVrZ+tjGIbGjBmjwMBAeXh4KDIyUkeOHLkXUwEAACg2xR7sFi1apGHDhmns2LHavXu3wsLCFBUVpfPnz+fZf+nSpUpNTbW99u/fL1dXVz3//PO2Pm+//bamTp2qmTNnatu2bfLy8lJUVJSuXbt2r6YFAABwz1kMwzCKs4CmTZuqcePGmjZtmiQpNzdXwcHBGjhwoEaMGHHX9RMSEjRmzBilpqbKy8tLhmGoSpUqeuWVVzR8+HBJUkZGhvz9/TVv3jy98MILdx0zNzdXZ8+elbe3tywWS+EmCAAAUAiGYejy5cuqUqWKXFzufEyu1D2qKU85OTnatWuXRo4caWtzcXFRZGSktm7dmq8x5syZoxdeeEFeXl6SpOPHjystLU2RkZG2Pr6+vmratKm2bt2aZ7DLzs5Wdna27f2ZM2dUt25dR6cFAADgdKdOndIDDzxwxz7FGuwuXrwoq9Uqf39/u3Z/f38dPHjwrutv375d+/fv15w5c2xtaWlptjF+P+bNZb83adIkjR8//pb2U6dOycfH5651AAAAFJXMzEwFBwfL29v7rn2LNdgV1pw5c1S/fn01adKkUOOMHDlSw4YNs72/+QH6+PgQ7AAAQImQn8vDivXmiUqVKsnV1VXnzp2zaz937pwCAgLuuO6VK1e0cOFC9enTx6795noFGdPd3d0W4ghzAADgflWswc7NzU3h4eFKSkqyteXm5iopKUkRERF3XHfx4sXKzs5Wt27d7NqrV6+ugIAAuzEzMzO1bdu2u44JAABwPyv2U7HDhg1TbGysGjVqpCZNmighIUFXrlxRr169JEk9evRQUFCQJk2aZLfenDlz1LFjR1WsWNGu3WKxaMiQIXr99ddVq1YtVa9eXaNHj1aVKlXUsWPHezUtAICTWa1WXb9+vbjLAJyudOnScnV1dcpYxR7soqOjdeHCBY0ZM0ZpaWlq2LCh1qxZY7v54eTJk7fc2nvo0CF9++23WrduXZ5j/uMf/9CVK1fUr18/paen689//rPWrFmjMmXKFPl8AADOZRiG0tLSlJ6eXtylAEWmXLlyCggIKPRj1or9OXYlUWZmpnx9fZWRkcH1dgBQzFJTU5Weni4/Pz95enryfFGYimEY+uWXX3T+/HmVK1dOgYGBt/QpSC4p9iN2AADcjtVqtYW63196A5iFh4eHJOn8+fPy8/Mr1GnZYv9KMQAAbufmNXWenp7FXAlQtG7+jhf2OlKCHQCgxOP0K8zOWb/jBDsAAACTINgBAACYBMEOAID7REhIiBISEvLd/+uvv5bFYuFRMX8gBDsAAJzMYrHc8TVu3DiHxt2xY4f69euX7/7NmjVTamqqfH19HdpefhEgSw4edwIAgJOlpqbafl60aJHGjBmjQ4cO2drKli1r+9kwDFmtVpUqdfd/kitXrlygOtzc3O763eswF47YAQDgZAEBAbaXr6+vLBaL7f3Bgwfl7e2t1atXKzw8XO7u7vr222917NgxdejQQf7+/ipbtqwaN26sr776ym7c35+KtVgs+te//qVOnTrJ09NTtWrV0ooVK2zLf38kbd68eSpXrpzWrl2r0NBQlS1bVm3atLELojdu3NCgQYNUrlw5VaxYUfHx8YqNjS3U13L+/PPP6tGjh8qXLy9PT0+1bdtWR44csS1PSUlR+/btVb58eXl5ealevXpatWqVbd2YmBhVrlxZHh4eqlWrlhITEx2uxew4YgcAuO80aiSlpd377QYESDt3OmesESNGaMqUKapRo4bKly+vU6dO6amnntIbb7whd3d3LViwQO3bt9ehQ4dUtWrV244zfvx4vf3225o8ebI++OADxcTEKCUlRRUqVMiz/y+//KIpU6boo48+kouLi7p166bhw4frk08+kSS99dZb+uSTT5SYmKjQ0FC9//77Wr58uR5//HGH59qzZ08dOXJEK1askI+Pj+Lj4/XUU0/pwIEDKl26tOLi4pSTk6P//ve/8vLy0oEDB2xHNUePHq0DBw5o9erVqlSpko4ePaqrV686XIvZEewAAPedtDTpzJnirqJwJkyYoCeeeML2vkKFCgoLC7O9f+2117Rs2TKtWLFCAwYMuO04PXv2VNeuXSVJEydO1NSpU7V9+3a1adMmz/7Xr1/XzJkz9eCDD0qSBgwYoAkTJtiWf/DBBxo5cqQ6deokSZo2bZrt6Jkjbga6zZs3q1mzZpKkTz75RMHBwVq+fLmef/55nTx5Up07d1b9+vUlSTVq1LCtf/LkST3yyCNq1KiRpF+PWuL2CHYAgPtOcV025szt3gwqN2VlZWncuHFauXKlUlNTdePGDV29elUnT5684zgNGjSw/ezl5SUfHx+dP3/+tv09PT1toU6SAgMDbf0zMjJ07tw5NWnSxLbc1dVV4eHhys3NLdD8bkpOTlapUqXUtGlTW1vFihX10EMPKTk5WZI0aNAg9e/fX+vWrVNkZKQ6d+5sm1f//v3VuXNn7d69W08++aQ6duxoC4i4FcEOAHDfcdbp0OLk5eVl93748OFav369pkyZopo1a8rDw0PPPfeccnJy7jhO6dKl7d5bLJY7hrC8+huGUcDqnevFF19UVFSUVq5cqXXr1mnSpEl65513NHDgQLVt21YpKSlatWqV1q9fr9atWysuLk5Tpkwp1ppLKm6eAACgBNi8ebN69uypTp06qX79+goICNCJEyfuaQ2+vr7y9/fXjh07bG1Wq1W7d+92eMzQ0FDduHFD27Zts7X99NNPOnTokOrWrWtrCw4O1ksvvaSlS5fqlVde0ezZs23LKleurNjYWH388cdKSEjQrFmzHK7H7DhiBwBACVCrVi0tXbpU7du3l8Vi0ejRox0+/VkYAwcO1KRJk1SzZk3VqVNHH3zwgX7++ed8fZfpvn375O3tbXtvsVgUFhamDh06qG/fvvrwww/l7e2tESNGKCgoSB06dJAkDRkyRG3btlXt2rX1888/a+PGjQoNDZUkjRkzRuHh4apXr56ys7P15Zdf2pbhVgQ7AABKgHfffVe9e/dWs2bNVKlSJcXHxyszM/Oe1xEfH6+0tDT16NFDrq6u6tevn6KiouTq6nrXdVu0aGH33tXVVTdu3FBiYqIGDx6sp59+Wjk5OWrRooVWrVplOy1stVoVFxen06dPy8fHR23atNF7770n6ddn8Y0cOVInTpyQh4eHHnvsMS1cuND5EzcJi1HcJ9ZLoMzMTPn6+iojI0M+Pj7FXQ4A/GFdu3ZNx48fV/Xq1VWmTJniLucPKTc3V6GhoerSpYtee+214i7HtO70u16QXMIROwAAYJOSkqJ169apZcuWys7O1rRp03T8+HH99a9/Le7SkA/cPAEAAGxcXFw0b948NW7cWM2bN9e+ffv01VdfcV3bfYIjdgAAwCY4OFibN28u7jLgII7YAQAAmATBDgAAwCQIdgAAACZBsAMAADAJgh0AAIBJEOwAAABMgmAHAEAJ1apVKw0ZMsT2PiQkRAkJCXdcx2KxaPny5YXetrPGwb1FsAMAwMnat2+vNm3a5Lls06ZNslgs+v777ws87o4dO9SvX7/Clmdn3Lhxatiw4S3tqampatu2rVO39Xvz5s1TuXLlinQbfzQEOwAAnKxPnz5av369Tp8+fcuyxMRENWrUSA0aNCjwuJUrV5anp6czSryrgIAAubu735NtwXkIdgAAONnTTz+typUra968eXbtWVlZWrx4sfr06aOffvpJXbt2VVBQkDw9PVW/fn199tlndxz396dijxw5ohYtWqhMmTKqW7eu1q9ff8s68fHxql27tjw9PVWjRg2NHj1a169fl/TrEbPx48dr7969slgsslgstpp/fyp23759+stf/iIPDw9VrFhR/fr1U1ZWlm15z5491bFjR02ZMkWBgYGqWLGi4uLibNtyxMmTJ9WhQweVLVtWPj4+6tKli86dO2dbvnfvXj3++OPy9vaWj4+PwsPDtXPnTkm/fudt+/btVb58eXl5ealevXpatWqVw7XcL/hKMQDA/adRIykt7d5vNyBA+v/B4U5KlSqlHj16aN68eRo1apQsFoskafHixbJareratauysrIUHh6u+Ph4+fj4aOXKlerevbsefPBBNWnS5K7byM3N1bPPPit/f39t27ZNGRkZdtfj3eTt7a158+apSpUq2rdvn/r27Stvb2/94x//UHR0tPbv3681a9boq6++kiT5+vreMsaVK1cUFRWliIgI7dixQ+fPn9eLL76oAQMG2IXXjRs3KjAwUBs3btTRo0cVHR2thg0bqm/fvnedT17zuxnqvvnmG924cUNxcXGKjo7W119/LUmKiYnRI488ohkzZsjV1VV79uxR6dKlJUlxcXHKycnRf//7X3l5eenAgQMqW7Zsgeu47xi4RUZGhiHJyMjIKO5SAOAP7erVq8aBAweMq1ev2i8ICjIM6d6/goLyXXtycrIhydi4caOt7bHHHjO6det223XatWtnvPLKK7b3LVu2NAYPHmx7X61aNeO9994zDMMw1q5da5QqVco4c+aMbfnq1asNScayZctuu43Jkycb4eHhtvdjx441wsLCbun323FmzZpllC9f3sjKyrItX7lypeHi4mKkpaUZhmEYsbGxRrVq1YwbN27Y+jz//PNGdHT0bWtJTEw0fH1981y2bt06w9XV1Th58qSt7YcffjAkGdu3bzcMwzC8vb2NefPm5bl+/fr1jXHjxt122yXNbX/XjYLlEo7YAQDuPwEBJX67derUUbNmzTR37ly1atVKR48e1aZNmzRhwgRJktVq1cSJE/X555/rzJkzysnJUXZ2dr6voUtOTlZwcLCqVKlia4uIiLil36JFizR16lQdO3ZMWVlZunHjhnx8fPI9j5vbCgsLk5eXl62tefPmys3N1aFDh+Tv7y9JqlevnlxdXW19AgMDtW/fvgJt67fbDA4OVnBwsK2tbt26KleunJKTk9W4cWMNGzZML774oj766CNFRkbq+eef14MPPihJGjRokPr3769169YpMjJSnTt3dui6xvsN19gBAO4/O3dKp0/f+1c+TsP+Vp8+ffTvf/9bly9fVmJioh588EG1bNlSkjR58mS9//77io+P18aNG7Vnzx5FRUUpJyfHaR/T1q1bFRMTo6eeekpffvml/ve//2nUqFFO3cZv3TwNepPFYlFubm6RbEv69Y7eH374Qe3atdOGDRtUt25dLVu2TJL04osv6scff1T37t21b98+NWrUSB988EGR1VJSEOwAACgiXbp0kYuLiz799FMtWLBAvXv3tl1vt3nzZnXo0EHdunVTWFiYatSoocOHD+d77NDQUJ06dUqpqam2tu+++86uz5YtW1StWjWNGjVKjRo1Uq1atZSSkmLXx83NTVar9a7b2rt3r65cuWJr27x5s1xcXPTQQw/lu+aCuDm/U6dO2doOHDig9PR01a1b19ZWu3ZtDR06VOvWrdOzzz6rxMRE27Lg4GC99NJLWrp0qV555RXNnj27SGotSQh2AAAUkbJlyyo6OlojR45UamqqevbsaVtWq1Yt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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.style.use('seaborn-bright')\n",
    "\n",
    "fig = plt.figure()\n",
    "\n",
    "# fig.set_size_inches(15.5, 10.5)\n",
    "\n",
    "ax = plt.axes()\n",
    "\n",
    "x_values = range(epochs)\n",
    "\n",
    "losses2 = torch.tensor(losses).cpu()\n",
    "vlosses2 = torch.tensor(vlosses).cpu()\n",
    "vacc2 = torch.tensor(vacc).cpu()\n",
    "\n",
    "ax.plot(x_values, losses2, color='blue',  linewidth=2, label='Training Loss' )\n",
    "ax.plot(x_values, vlosses2, color='red',  linewidth=2, label='Validation Loss')\n",
    "ax.plot(x_values, vacc2, color='green',  linewidth=2, label='Validation Accuracy')\n",
    "\n",
    "\n",
    "plt.legend()\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.ylabel(\"Loss/Accuracy\")\n",
    "plt.tight_layout()\n",
    "\n",
    "plotsavepath = output_savepath + experiment_name + ' training curve values.csv'\n",
    "\n",
    "pd.DataFrame({'epochs': x_values, 'train loss':losses, \n",
    "              'validation loss': vlosses,\n",
    "             'validation accuracy': vacc}).to_csv(plotsavepath, index=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "546a0d46",
   "metadata": {},
   "source": [
    "# confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "6c45c4d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[15000     0]\n",
      " [ 5000     0]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "confusionMatrix = confusion_matrix(all_test_labels.cpu(), all_predicted_test_labels.cpu(), labels = range(2))\n",
    "\n",
    "print(confusionMatrix)\n",
    "\n",
    "confusionmatrixpath = output_savepath + experiment_name + '-confusionmatrix.pt'\n",
    "\n",
    "confusion_dictionary = {0:confusionMatrix}\n",
    "\n",
    "torch.save(confusion_dictionary, confusionmatrixpath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "96e3338b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "# confusionMatrixPath = '/home/ankit/code/k/figures/'\n",
    "\n",
    "# plot_name = 'confusion' + '_' + feature1 + '_' + feature2 + '_' + feature3 + '_' + str(int(a1*100)) + '_' + str(int(a2*100)) + '_' + str(int(a3*100)) + '.png'\n",
    "\n",
    "# sns.set(rc={'figure.figsize':(13.7,10.27)})\n",
    "\n",
    "ax = sns.heatmap(confusionMatrix, annot = True, \n",
    "                 xticklabels=['Non Hate', 'Hate'], \n",
    "                 yticklabels=['Non Hate', 'Hate'],\n",
    "                 fmt='d')\n",
    "\n",
    "fig = ax.get_figure()\n",
    "\n",
    "# fig.savefig(confusionMatrixPath + plot_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "54bebbd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "mcc2 = metrics.matthews_corrcoef(all_test_labels.cpu(), all_predicted_test_labels.cpu())\n",
    "print(mcc2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6305e158",
   "metadata": {},
   "source": [
    "# precision recall f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f574dd25",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ankit/anaconda3/envs/pytorch/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1334: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(0.0, 0.0, 0.0)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import precision_recall_fscore_support\n",
    "\n",
    "precision, recall, f1, _ = precision_recall_fscore_support(all_test_labels.cpu(), \n",
    "                                                           all_predicted_test_labels.cpu(), \n",
    "                                                           labels = range(2),\n",
    "                                                           average = 'binary')\n",
    "\n",
    "precision, recall, f1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c7b26760",
   "metadata": {},
   "source": [
    "# AUC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "35653d05",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3958333333333333\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "auc_score = metrics.roc_auc_score(all_test_labels.cpu(), \n",
    "                                  all_predicted_fake_probabilities.cpu(), \n",
    "                                  \n",
    "                                  )\n",
    "\n",
    "print(auc_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "54e24b86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'accuracy': 0.75,\n",
       " 'precision': 0.0,\n",
       " 'recall': 0.0,\n",
       " 'f1': 0.0,\n",
       " 'auc': 0.3958333333333333,\n",
       " 'mcc': 0.0}"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics_dictionary = {}\n",
    "\n",
    "metrics_dictionary['accuracy'] = final_test_accuracy\n",
    "metrics_dictionary['precision'] = precision\n",
    "metrics_dictionary['recall'] = recall\n",
    "metrics_dictionary['f1'] = f1\n",
    "metrics_dictionary['auc'] = auc_score\n",
    "metrics_dictionary['mcc'] = mcc2\n",
    "\n",
    "savefullpath = output_savepath + experiment_name + '-result-metrics.pt'\n",
    "\n",
    "torch.save(metrics_dictionary, savefullpath)\n",
    "\n",
    "metrics_dictionary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f67f2354",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.   0.   0.   0.25 0.25 0.5  0.5  0.75 0.75 1.  ]\n",
      "[0.         0.08333333 0.25       0.25       0.41666667 0.41666667\n",
      " 0.75       0.75       1.         1.        ]\n",
      "10\n"
     ]
    }
   ],
   "source": [
    "fpr2, tpr2, threshold = metrics.roc_curve(all_test_labels.cpu(), \n",
    "                                  all_predicted_fake_probabilities.cpu(), pos_label = 1)\n",
    "\n",
    "print(tpr2)\n",
    "print((fpr2))\n",
    "print(len(threshold))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "cb20704a",
   "metadata": {},
   "outputs": [],
   "source": [
    "tprfprsavepath = output_savepath + experiment_name + ' AUC Values.csv'\n",
    "\n",
    "pd.DataFrame({'False Positive Rate': fpr2, 'True Positive Rate':tpr2, 'Threshold': threshold}).to_csv(tprfprsavepath, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "77c7426b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fb4d049c5b0>]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGdCAYAAADAAnMpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAeVElEQVR4nO3dcWzU9f3H8Vdb6BVjWzAd11Juq+AUFaTYSleQGJfORg2OPxYbMbQjilOZcTSbUoFWRSkzwki02ogy/UMHYpQYaeqwkxikC7HQxE3AYNEW5A4aZ68r2kLv8/vDH+cqLfRbaN/e8XwkF+OXz/fu3Y8N9/Tbu2uCc84JAADASKL1AAAA4MJGjAAAAFPECAAAMEWMAAAAU8QIAAAwRYwAAABTxAgAADBFjAAAAFOjrAcYjEgkoi+//FKpqalKSEiwHgcAAAyCc06dnZ2aMGGCEhMHvv4REzHy5ZdfKhAIWI8BAACGoK2tTRMnThzwz2MiRlJTUyV998WkpaUZTwMAAAYjHA4rEAhEn8cHEhMxcupHM2lpacQIAAAx5mwvseAFrAAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAlOcY+eCDDzR37lxNmDBBCQkJ2rJly1nP2b59u6699lr5fD5ddtllevnll4cwKgAAiEeeY6Srq0vTp09XTU3NoNYfPHhQt956q2688UY1NzfrD3/4g+6++269++67nocFAADxx/Pvprn55pt18803D3p9bW2tLr30Uq1Zs0aSdOWVV2rHjh36y1/+ouLiYq8PDwAA4sywv2aksbFRRUVFfY4VFxersbFxwHO6u7sVDof73AAAwPmXny9NnPjdP60Me4wEg0H5/f4+x/x+v8LhsL755pt+z6murlZ6enr0FggEhntMAAAuSMGgdPjwd/+08qN8N01FRYU6Ojqit7a2NuuRAADAMPH8mhGvMjMzFQqF+hwLhUJKS0vTmDFj+j3H5/PJ5/MN92gAAOBHYNivjBQWFqqhoaHPsW3btqmwsHC4HxoAAMQAzzHy3//+V83NzWpubpb03Vt3m5ub1draKum7H7GUlpZG1997771qaWnRQw89pH379um5557T66+/riVLlpyfrwAAAMQ0zzHy0UcfacaMGZoxY4Ykqby8XDNmzFBlZaUk6ciRI9EwkaRLL71UW7du1bZt2zR9+nStWbNGL774Im/rBQAAkqQE55yzHuJswuGw0tPT1dHRobS0NOtxAACIGxMnfvdumuxs6dCh83vfg33+/lG+mwYAAFw4iBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaGFCM1NTXKyclRSkqKCgoKtGvXrjOuX7duna644gqNGTNGgUBAS5Ys0bfffjukgQEAQHzxHCObNm1SeXm5qqqqtHv3bk2fPl3FxcU6evRov+tfe+01LV26VFVVVdq7d69eeuklbdq0SY888sg5Dw8AAGKf5xhZu3atFi1apIULF+qqq65SbW2tLrroIm3YsKHf9Tt37tTs2bM1f/585eTk6KabbtIdd9xx1qspAADgwuApRnp6etTU1KSioqLv7yAxUUVFRWpsbOz3nFmzZqmpqSkaHy0tLaqrq9Mtt9wy4ON0d3crHA73uQEAgPg0ysvi9vZ29fb2yu/39znu9/u1b9++fs+ZP3++2tvbdf3118s5p5MnT+ree+89449pqqur9dhjj3kZDQAAxKhhfzfN9u3btWrVKj333HPavXu33nzzTW3dulUrV64c8JyKigp1dHREb21tbcM9JgAAMOLpykhGRoaSkpIUCoX6HA+FQsrMzOz3nBUrVmjBggW6++67JUnTpk1TV1eX7rnnHi1btkyJiaf3kM/nk8/n8zIaAACIUZ6ujCQnJysvL08NDQ3RY5FIRA0NDSosLOz3nOPHj58WHElJSZIk55zXeQEAQJzxdGVEksrLy1VWVqb8/HzNnDlT69atU1dXlxYuXChJKi0tVXZ2tqqrqyVJc+fO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8rDOHIuGss8rVqzQggULdPfdd0uSpk2bpq6uLt1zzz1atmyZEhP5/+vzYaDnwrS0tGG5KiLF8ZWR5ORk5eXlqaGhIXosEomooaFBhYWF/Z5TWFjYZ70kbdu2bcD1GNo+S9JTTz2llStXqr6+Xvn5+SMxaszzutdTpkzRxx9/rObm5ujttttu04033qjm5mYFAoGRHD9mDOV7evbs2Tpw4EA09iTp008/VVZWFiEygKHs8/Hjx08LjlMB6Pg1a+eNyXPhsL009kdg48aNzufzuZdfftl98skn7p577nFjx451wWDQOefcggUL3NKlS6PrP/zwQzdq1Cj39NNPu71797qqqire2jsIXvd59erVLjk52b3xxhvuyJEj0VtnZ6fVlxAzvO71D/FumsHxus+tra0uNTXV/f73v3f79+9377zzjhs/frx74oknrL6EmOB1n6uqqlxqaqr729/+5lpaWtzf//53N3nyZHf77bdbfQkxobOz0+3Zs8ft2bPHSXJr1651e/bscV988YVzzrmlS5e6BQsWRNefemvvn/70J7d3715XU1PDW3vP1TPPPON++tOfuuTkZDdz5kz3z3/+M/pnN9xwgysrK+uz/vXXX3eXX365S05OdldffbXbunXrCE8cm7zs889+9jMn6bRbVVXVyA8eg7x+T/8vYmTwvO7zzp07XUFBgfP5fG7SpEnuySefdCdPnhzhqWOPl30+ceKEe/TRR93kyZNdSkqKCwQC7v7773f/+c9/Rn7wGPL+++/3+3fuqb0tKytzN9xww2nn5ObmuuTkZDdp0iT317/+dVhnTHCOa1sAAMBO3L5mBAAAxAZiBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJj6PzRBhxxCZIdBAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "ax = plt.axes()\n",
    "\n",
    "ax.plot(fpr2, tpr2, color='blue',  linewidth=2 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "febec389",
   "metadata": {},
   "source": [
    "# save model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "1c2da906",
   "metadata": {},
   "outputs": [],
   "source": [
    "modelsavepath = output_savepath + experiment_name + ' saved model.pt'\n",
    "\n",
    "torch.save(model.state_dict(), modelsavepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fade6c41",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ba271c7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
