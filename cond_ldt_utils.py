import matplotlib.pyplot as plt

def latent_space_check(train_z):
    # --- sanity checks ---
    # 1) Shape & dtype
    print("train_mus.shape:", train_z.shape)
    print("dtype:", train_z.dtype)

    # 2) Global mean & std (should be ≈0, ≈1 if your encoder outputs are normalized)
    all_vals = train_z.view(-1)
    print("global mean:", all_vals.mean().item())
    print("global std: ", all_vals.std().item())

    # 3) Per‑dimension stats
    per_dim_mean = train_z.mean(dim=(0, 1))  # → [D]
    per_dim_std = train_z.std(dim=(0, 1))  # → [D]
    for i in range(min(100, train_z.size(-1))):
        print(f"dim {i:2d}: mean={per_dim_mean[i]:6.3f}, std={per_dim_std[i]:6.3f}")

    # 4) Histogram (optional, but very informative)
    plt.hist(all_vals.numpy(), bins=100)
    plt.title("Histogram of μ values")
    plt.xlabel("μ value")
    plt.ylabel("count")
    plt.show()
