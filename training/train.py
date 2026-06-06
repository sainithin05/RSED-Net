import os
import torch

# =====================================

# TRAINING CONFIGURATION

# =====================================

num_epochs = 100
checkpoint_path = "latest_checkpoint.pth"

# =====================================

# RESUME TRAINING

# =====================================

start_epoch = 0

if os.path.exists(checkpoint_path):


checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1

print(
    f"Resuming Training From Epoch {start_epoch}"
)


# =====================================

# TRAINING LOOP

# =====================================

for epoch in range(start_epoch, num_epochs):

```
model.train()

total_loss = 0

for low_img, high_img in train_loader:

    low_img = low_img.to(device)

    high_img = high_img.to(device)

    optimizer.zero_grad()

    enhanced, R_out, L_out = model(
        low_img
    )

    enhanced = torch.clamp(
        enhanced,
        0,
        1
    )

    loss = combined_loss(
        enhanced,
        high_img
    )

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

scheduler.step()

avg_loss = total_loss / len(train_loader)

current_lr = optimizer.param_groups[0]["lr"]

print(
    f"Epoch [{epoch+1}/{num_epochs}] "
    f"Loss: {avg_loss:.4f} "
    f"LR: {current_lr:.8f}"
)

# =================================
# SAVE CHECKPOINT EVERY 10 EPOCHS
# =================================

if (epoch + 1) % 10 == 0:

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
            model.state_dict(),
            "optimizer_state_dict":
            optimizer.state_dict(),
            "scheduler_state_dict":
            scheduler.state_dict(),
        },
        checkpoint_path,
    )

    print(
        f"Checkpoint Saved At Epoch {epoch+1}"
    )


# =====================================

# SAVE FINAL MODEL

# =====================================

torch.save(
model.state_dict(),
"Rsednet.pth"
)

print("\nTraining Completed")

print(
"Final Model Saved Successfully"
)
