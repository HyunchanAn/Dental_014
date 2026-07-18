import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    하악골 및 주요 ROI에 연산을 집중하게 만드는 Spatial Attention 모듈
    의사가 필름을 판독할 때 뼈의 경계선(하악골 하연 등)에 주목하는 방식을 모사합니다.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel pooling (Avg + Max)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.conv1(x_cat)
        return x * self.sigmoid(attn_map)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=(256, 512)):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(img_size, int):
            self.num_patches = (img_size // patch_size) ** 2
        else:
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Spatial Attention pre-processing
        self.spatial_attention = SpatialAttention()
        
        # Shallow feature extraction + Patch creation
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class OsteoMAENet(nn.Module):
    """
    차세대 아키텍처: Masked Autoencoder (MAE) + ViT 구조
    Head 1: Masked Patch Reconstruction (해면골/피질골의 본질적 기하학 학습)
    Head 2: Osteoporosis Classification (C1, C2, C3) & SupCon Embedding
    """
    def __init__(self, in_channels=3, img_size=(256, 512), patch_size=16, emb_size=768, 
                 depth=12, num_heads=12, num_classes=3, mask_ratio=0.75):
        super(OsteoMAENet, self).__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        if isinstance(img_size, int):
            self.num_patches = (img_size // patch_size) ** 2
        else:
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # 1. Patch Embedding + Spatial Attention
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        # Position embedding & CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_size))
        
        # 2. Transformer Encoder (ViT Backbone)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=emb_size*4, 
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_size)
        
        # 3. Head 1: MAE Decoder (Reconstruction)
        decoder_emb_size = 512
        self.decoder_embed = nn.Linear(emb_size, decoder_emb_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_size))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_emb_size))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_emb_size, nhead=8, dim_feedforward=decoder_emb_size*4, 
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)
        self.decoder_norm = nn.LayerNorm(decoder_emb_size)
        
        # Reconstruct pixel patches (patch_size * patch_size * in_channels)
        self.decoder_pred = nn.Linear(decoder_emb_size, patch_size**2 * in_channels)
        
        # 4. Head 2: Classification / SupCon Projection
        # CLS 토큰을 사용해 최종 클래스 예측 및 조영 학습용 임베딩 반환
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.supcon_head = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """ 무작위로 패치를 마스킹하여(예: 75%) 안 보이는 패치를 생성 """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1) # (N, L)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_kept, ids_restore, ids_keep

    def forward_encoder(self, x, mask=True):
        x = self.patch_embed(x) # (B, N, E)
        x = x + self.pos_embed[:, 1:, :] # add pos embed w/o cls token
        
        ids_restore = None
        if mask:
            x, ids_restore, _ = self.random_masking(x)
            
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x)
        x = self.norm(x)
        return x, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        # 마스킹된 부분에 mask token 끼워넣기
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # CLS 토큰과 다시 병합
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        # 패치 차원으로 복원
        x = self.decoder_pred(x[:, 1:, :]) 
        return x

    def forward(self, x, mask=True):
        latent, ids_restore = self.forward_encoder(x, mask=mask)
        
        # Head 1: MAE Reconstruction
        pred_pixel = None
        if mask and ids_restore is not None:
            pred_pixel = self.forward_decoder(latent, ids_restore)
            
        # Head 2: Classification & SupCon Embeddings (Use CLS token)
        cls_feat = latent[:, 0]
        class_logits = self.classifier(cls_feat)
        supcon_embeds = self.supcon_head(cls_feat)
        supcon_embeds = nn.functional.normalize(supcon_embeds, dim=1) # L2 normalize for SupCon
        
        return pred_pixel, class_logits, supcon_embeds
