import torch

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ==============================

# TESTING

# ==============================

model.eval()

psnr_metric = PeakSignalNoiseRatio(
data_range=1.0
).to(device)

ssim_metric = StructuralSimilarityIndexMeasure(
data_range=1.0
).to(device)

total_psnr = 0
total_ssim = 0

with torch.no_grad():

```
for low_img, high_img in test_loader:

    low_img = low_img.to(device)

    high_img = high_img.to(device)

    enhanced, _, _ = model(low_img)

    enhanced = torch.clamp(
        enhanced,
        0,
        1
    )

    psnr_value = psnr_metric(
        enhanced,
        high_img
    ).item()

    ssim_value = ssim_metric(
        enhanced,
        high_img
    ).item()

    total_psnr += psnr_value

    total_ssim += ssim_value

avg_psnr = total_psnr / len(test_loader)

avg_ssim = total_ssim / len(test_loader)

print(f"Average PSNR : {avg_psnr:.4f}")

print(f"Average SSIM : {avg_ssim:.4f}")
