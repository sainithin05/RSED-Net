import torch
import torch.nn as nn

# ==============================
# CBAM ATTENTION MODULE
# ==============================

class CBAM(nn.Module):
    # ==============================
# CBAM ATTENTION MODULE
# ==============================

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):

        super(CBAM, self).__init__()

        # =================================
        # CHANNEL ATTENTION
        # =================================

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(

            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),

            nn.ReLU(),

            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        # =================================
        # SPATIAL ATTENTION
        # =================================

        self.spatial = nn.Sequential(

            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),

            nn.Sigmoid()
        )

    def forward(self, x):

        # =================================
        # CHANNEL ATTENTION
        # =================================

        avg_out = self.mlp(self.avg_pool(x))

        max_out = self.mlp(self.max_pool(x))

        channel_attention = self.sigmoid_channel(avg_out + max_out)

        x = x * channel_attention

        # =================================
        # SPATIAL ATTENTION
        # =================================

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_input = torch.cat([avg_out, max_out], dim=1)

        spatial_attention = self.spatial(spatial_input)

        x = x * spatial_attention

        return x
    
# ==============================
# RESIDUAL BLOCK
# ==============================

class ResidualBlock(nn.Module):
    # ==============================
# RESIDUAL BLOCK
# ==============================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(channels),

            nn.ReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        out = self.block(x)

        out = out + residual

        out = self.relu(out)

        return out

# ==============================
# RETINEX DECOMPOSITION
# ==============================

class RetinexDecomposition(nn.Module):
    # ==============================
# RETINEX DECOMPOSITION (UPGRADED)
# ==============================

class RetinexDecomposition(nn.Module):

    def __init__(self):
        super(RetinexDecomposition, self).__init__()

        # --------------------------------
        # REFLECTANCE BRANCH
        # --------------------------------

        self.reflectance = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),

            nn.Sigmoid()
        )

        # --------------------------------
        # ILLUMINATION BRANCH
        # --------------------------------

        self.illumination = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            ResidualBlock(32),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):

        # --------------------------------
        # DECOMPOSITION
        # --------------------------------

        R = self.reflectance(x)

        L = self.illumination(x)

        return R, L

# ==============================
# TRANSFORMER BLOCK
# ==============================

class TransformerBlock(nn.Module):
    # ==============================
# SIMPLE TRANSFORMER BLOCK
# ==============================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout=0.1
    ):

        super(TransformerBlock, self).__init__()

        # =================================
        # LAYER NORMALIZATION
        # =================================

        self.norm1 = nn.LayerNorm(embed_dim)

        # =================================
        # MULTIHEAD ATTENTION
        # =================================

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # =================================
        # SECOND NORMALIZATION
        # =================================

        self.norm2 = nn.LayerNorm(embed_dim)

        # =================================
        # FEED FORWARD NETWORK
        # =================================

        self.ffn = nn.Sequential(

            nn.Linear(embed_dim, ff_dim),

            nn.ReLU(),

            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # =================================
        # FLATTEN SPATIAL DIMENSIONS
        # =================================

        x = x.flatten(2).transpose(1, 2)

        # Shape:
        # B × (H*W) × C

        # =================================
        # SELF ATTENTION
        # =================================

        attn_input = self.norm1(x)

        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input
        )

        x = x + attn_output

        # =================================
        # FEED FORWARD
        # =================================

        ff_input = self.norm2(x)

        ff_output = self.ffn(ff_input)

        x = x + ff_output

        # =================================
        # RESHAPE BACK
        # =================================

        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        return x

# ==============================
# SHARED ENCODER
# ==============================

class SharedEncoder(nn.Module):
    # ==============================
# DEEP SHARED ENCODER
# WITH CBAM ATTENTION
# + TRANSFORMER BOTTLENECK
# ==============================

class SharedEncoder(nn.Module):

    def __init__(self):

        super(SharedEncoder, self).__init__()

        # =================================
        # STAGE 1
        # =================================

        self.initial = nn.Sequential(

            nn.Conv2d(
                6,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU()
        )

        self.res1 = ResidualBlock(64)

        # =================================
        # CBAM ATTENTION
        # =================================

        self.cbam1 = CBAM(64)

        self.pool1 = nn.MaxPool2d(2)

        # =================================
        # STAGE 2
        # =================================

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU()
        )

        self.res2 = ResidualBlock(128)

        # =================================
        # CBAM ATTENTION
        # =================================

        self.cbam2 = CBAM(128)

        self.pool2 = nn.MaxPool2d(2)

        # =================================
        # STAGE 3
        # =================================

        self.conv3 = nn.Sequential(

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU()
        )

        self.res3 = ResidualBlock(256)

        # =================================
        # CBAM ATTENTION
        # =================================

        self.cbam3 = CBAM(256)

        self.pool3 = nn.MaxPool2d(2)

        # =================================
        # TRANSFORMER BOTTLENECK
        # =================================

        self.transformer = TransformerBlock(

            embed_dim=256,

            num_heads=4
        )

    def forward(self, x):

        # =================================
        # STAGE 1
        # =================================

        x1 = self.initial(x)

        x1 = self.res1(x1)

        # =================================
        # CBAM
        # =================================

        x1 = self.cbam1(x1)

        p1 = self.pool1(x1)

        # =================================
        # STAGE 2
        # =================================

        x2 = self.conv2(p1)

        x2 = self.res2(x2)

        # =================================
        # CBAM
        # =================================

        x2 = self.cbam2(x2)

        p2 = self.pool2(x2)

        # =================================
        # STAGE 3
        # =================================

        x3 = self.conv3(p2)

        x3 = self.res3(x3)

        # =================================
        # CBAM
        # =================================

        x3 = self.cbam3(x3)

        # =================================
        # BOTTLENECK
        # =================================

        bottleneck = self.pool3(x3)

        # =================================
        # TRANSFORMER
        # =================================

        bottleneck = self.transformer(
            bottleneck
        )

        return x1, x2, x3, bottleneck

# ==============================
# REFLECTANCE DECODER
# ==============================

class ReflectanceDecoder(nn.Module):
    # ==============================
# LIGHTWEIGHT REFLECTANCE DECODER
# 192 → 96 → 48 VERSION
# ==============================

class ReflectanceDecoder(nn.Module):

    def __init__(self):

        super(ReflectanceDecoder, self).__init__()

        # =================================
        # UPSAMPLE 1
        # 256 -> 192
        # =================================

        self.up1 = nn.ConvTranspose2d(

            256,

            192,

            kernel_size=2,

            stride=2
        )

        self.conv1 = nn.Sequential(

            nn.Conv2d(

                448,   # 192 + 256

                192,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # UPSAMPLE 2
        # 192 -> 96
        # =================================

        self.up2 = nn.ConvTranspose2d(

            192,

            96,

            kernel_size=2,

            stride=2
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(

                224,   # 96 + 128

                96,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # UPSAMPLE 3
        # 96 -> 48
        # =================================

        self.up3 = nn.ConvTranspose2d(

            96,

            48,

            kernel_size=2,

            stride=2
        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(

                112,   # 48 + 64

                48,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # FINAL RGB OUTPUT
        # =================================

        self.final = nn.Conv2d(

            48,

            3,

            kernel_size=1
        )

    def forward(self, bottleneck, x3, x2, x1):

        # =================================
        # UPSAMPLE 1
        # =================================

        d1 = self.up1(bottleneck)

        d1 = torch.cat([d1, x3], dim=1)

        d1 = self.conv1(d1)

        # =================================
        # UPSAMPLE 2
        # =================================

        d2 = self.up2(d1)

        d2 = torch.cat([d2, x2], dim=1)

        d2 = self.conv2(d2)

        # =================================
        # UPSAMPLE 3
        # =================================

        d3 = self.up3(d2)

        d3 = torch.cat([d3, x1], dim=1)

        d3 = self.conv3(d3)

        # =================================
        # OUTPUT
        # =================================

        out = torch.sigmoid(
            self.final(d3)
        )

        return out

# ==============================
# ILLUMINATION DECODER
# ==============================

class IlluminationDecoder(nn.Module):
    # ==============================
# LIGHTWEIGHT ILLUMINATION DECODER
# 192 → 96 → 48 VERSION
# ==============================

class IlluminationDecoder(nn.Module):

    def __init__(self):

        super(IlluminationDecoder, self).__init__()

        # =================================
        # UPSAMPLE 1
        # 256 -> 192
        # =================================

        self.up1 = nn.ConvTranspose2d(

            256,

            192,

            kernel_size=2,

            stride=2
        )

        self.conv1 = nn.Sequential(

            nn.Conv2d(

                448,   # 192 + 256

                192,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # UPSAMPLE 2
        # 192 -> 96
        # =================================

        self.up2 = nn.ConvTranspose2d(

            192,

            96,

            kernel_size=2,

            stride=2
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(

                224,   # 96 + 128

                96,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # UPSAMPLE 3
        # 96 -> 48
        # =================================

        self.up3 = nn.ConvTranspose2d(

            96,

            48,

            kernel_size=2,

            stride=2
        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(

                112,   # 48 + 64

                48,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =================================
        # FINAL RGB OUTPUT
        # =================================

        self.final = nn.Conv2d(

            48,

            3,

            kernel_size=1
        )

    def forward(self, bottleneck, x3, x2, x1):

        # =================================
        # UPSAMPLE 1
        # =================================

        d1 = self.up1(bottleneck)

        d1 = torch.cat([d1, x3], dim=1)

        d1 = self.conv1(d1)

        # =================================
        # UPSAMPLE 2
        # =================================

        d2 = self.up2(d1)

        d2 = torch.cat([d2, x2], dim=1)

        d2 = self.conv2(d2)

        # =================================
        # UPSAMPLE 3
        # =================================

        d3 = self.up3(d2)

        d3 = torch.cat([d3, x1], dim=1)

        d3 = self.conv3(d3)

        # =================================
        # OUTPUT
        # =================================

        out = torch.sigmoid(
            self.final(d3)
        )

        return out

# ==============================
# RSED-NET
# ==============================

class RetinexNet(nn.Module):

    def __init__(self):
        super(RetinexNet, self).__init__()

        self.retinex = RetinexDecomposition()

        self.encoder = SharedEncoder()

        self.reflectance_decoder = ReflectanceDecoder()

        self.illumination_decoder = IlluminationDecoder()

    def forward(self, x):

        R, L = self.retinex(x)

        combined = torch.cat([R, L], dim=1)

        x1, x2, x3, bottleneck = self.encoder(combined)

        R_enhanced = self.reflectance_decoder(
            bottleneck,
            x3,
            x2,
            x1
        )

        L_enhanced = self.illumination_decoder(
            bottleneck,
            x3,
            x2,
            x1
        )

        enhanced = R_enhanced * L_enhanced

        enhanced = enhanced + x

        enhanced = torch.clamp(
            enhanced,
            0,
            1
        )

        return enhanced


if __name__ == "__main__":

    model = RetinexNet()

    x = torch.randn(1, 3, 256, 256)

    enhanced = model(x)



