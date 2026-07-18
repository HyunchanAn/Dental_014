import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalFocalLoss(nn.Module):
    """
    Focal Loss가 결합된 서수 회귀 손실 (Ordinal Focal Loss)
    모드 붕괴(Mode Collapse) 방지를 위해 역빈도(Class Weights)와 Focal 계수를 함께 적용합니다.
    """
    def __init__(self, num_classes=3, gamma=2.0, alpha=None):
        super(OrdinalFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha  # e.g. torch.tensor([3.29, 1.0, 2.15])
        self.register_buffer('distance_matrix', self._create_distance_matrix(num_classes))

    def _create_distance_matrix(self, num_classes):
        mat = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                mat[i, j] = float(abs(i - j))
        return mat

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 1. Focal Term 계산: (1 - p_t)^gamma
        p_t = (probs * target_one_hot).sum(dim=1)
        focal_term = (1.0 - p_t) ** self.gamma
        
        # 2. Alpha (Class Weights) 적용
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
            
        # 3. Ordinal Penalties 적용
        penalties = torch.matmul(target_one_hot, self.distance_matrix)
        log_probs = torch.log(probs + 1e-7)
        
        # 4. 최종 손실: 가중치(alpha_t) * 초점(focal_term) * CE_Ordinal
        ce_ordinal = -torch.sum((target_one_hot + 0.5 * penalties) * log_probs, dim=1)
        loss = alpha_t * focal_term * ce_ordinal
        
        return loss.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    같은 클래스의 임베딩은 당기고, 다른 클래스의 임베딩은 밀어내는 손실.
    C2(골감소증)와 C3(골다공증)처럼 텍스처 경계가 모호한 클래스 간의 분별력을 크게 향상시킵니다.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        # features: [batch_size, embed_dim] (L2 normalized)
        device = features.device
        batch_size = features.shape[0]

        # 라벨 마스크 (같은 클래스인지 여부)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 코사인 유사도 행렬 계산 (features are L2 normalized, so dot product is cosine sim)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # 안정적인 Softmax 계산을 위해 최대값 빼기
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 자기 자신과의 유사도는 마스킹 (1.0)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Log-softmax 계산
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # 양성 샘플(Positive samples)에 대한 평균 log-prob
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs) # division by zero 방지
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

class OsteoCompositeLoss(nn.Module):
    """
    새로운 아키텍처를 위한 종합 손실 함수
    1. MAE Reconstruction Loss (MSE) - 패치 복원용
    2. Ordinal Regression Loss - C1, C2, C3의 순차적 심각도 고려
    3. Supervised Contrastive Loss - 클래스 간 거리 확보
    """
    def __init__(self, recon_weight=1.0, ordinal_weight=1.0, supcon_weight=0.5):
        super(OsteoCompositeLoss, self).__init__()
        self.recon_weight = recon_weight
        self.ordinal_weight = ordinal_weight
        self.supcon_weight = supcon_weight
        
        self.mse_loss = nn.MSELoss()
        # 데이터 불균형 역빈도(Inverse Frequency) 가중치 (C1: 34, C2: 112, C3: 52 -> Normalize to C2=1.0)
        class_weights = torch.tensor([3.29, 1.0, 2.15])
        self.ordinal_loss = OrdinalFocalLoss(num_classes=3, gamma=2.0, alpha=class_weights)
        self.supcon_loss = SupConLoss()

    def forward(self, pred_pixel, target_pixel, class_logits, supcon_embeds, targets):
        # 1. MAE Reconstruction Loss
        # pred_pixel: (B, N_masked, Patch_Dim)
        # target_pixel: (B, N_masked, Patch_Dim)
        loss_recon = torch.tensor(0.0).to(targets.device)
        if pred_pixel is not None and target_pixel is not None:
            loss_recon = self.mse_loss(pred_pixel, target_pixel)
            
        # 2. Ordinal Classification Loss
        loss_ordinal = self.ordinal_loss(class_logits, targets)
        
        # 3. Supervised Contrastive Loss
        loss_supcon = self.supcon_loss(supcon_embeds, targets)
        
        # Total Loss
        total_loss = (self.recon_weight * loss_recon) + \
                     (self.ordinal_weight * loss_ordinal) + \
                     (self.supcon_weight * loss_supcon)
                     
        return total_loss, loss_recon, loss_ordinal, loss_supcon
