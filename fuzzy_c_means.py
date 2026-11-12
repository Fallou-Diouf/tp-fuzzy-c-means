import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

# Création du dossier de sortie pour les figures du rapport LaTeX
os.makedirs("figures", exist_ok=True)

# =========================================================
# 1. CŒUR DE L'ALGORITHME FUZZY C-MEANS (INSTRUMENTÉ)
# =========================================================

def update_centers(X, U, m):
    """Calcule la nouvelle position des centroïdes (C x d)."""
    U_m = U ** m
    numerator = U_m.T @ X
    denominator = U_m.sum(axis=0)[:, None]
    return numerator / denominator

def compute_distances(X, centers):
    """Calcule la matrice des distances euclidiennes (N x C)."""
    return np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

def update_memberships(d, m):
    """Mise à jour de la matrice d'appartenance U (N x C)."""
    power = 2.0 / (m - 1.0)
    zero_mask = (d == 0)
    has_zero = zero_mask.any(axis=1)

    # Évite l'avertissement de division par zéro avant application du masque
    with np.errstate(divide='ignore'):
        weights = d ** -power
    U_new = weights / weights.sum(axis=1, keepdims=True)

    if has_zero.any():
        U_new[has_zero] = zero_mask[has_zero] / zero_mask[has_zero].sum(axis=1, keepdims=True)

    return U_new

def compute_objective(U, d, m):
    """Calcule la fonction objectif J_m."""
    return np.sum((U ** m) * (d ** 2))

def partition_coefficient(U):
    """Calcule le Partition Coefficient (PC) dans [1/C, 1]."""
    N = U.shape[0]
    return np.sum(U ** 2) / N

def partition_entropy(U):
    """Calcule la Partition Entropy (PE) dans [0, ln(C)]."""
    N = U.shape[0]
    eps = 1e-12  # Évite log(0)
    return -np.sum(U * np.log(U + eps)) / N

def fuzzy_c_means(X, C=2, m=2.0, max_iter=150, tol=1e-5, seed=42):
    """
    Algorithme FCM complet avec suivi de convergence et métriques.
    Retourne : U, centers, history_Jm, elapsed_time
    """
    np.random.seed(seed)  # Reproductibilité scientifique des expériences
    N = X.shape[0]
    
    U = np.random.rand(N, C)
    U = U / U.sum(axis=1, keepdims=True)
    
    history_Jm = []
    start_time = time.perf_counter()

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = update_centers(X, U, m)
        d = compute_distances(X, centers)
        
        # Enregistrement de J_m avant mise à jour de U
        Jm = compute_objective(U, d, m)
        history_Jm.append(Jm)
        
        U = update_memberships(d, m)
        
        if np.abs(U - U_old).max() < tol:
            # Enregistrer la valeur finale de J_m après convergence
            d_final = compute_distances(X, update_centers(X, U, m))
            history_Jm.append(compute_objective(U, d_final, m))
            break

    elapsed_time = time.perf_counter() - start_time

    # Tri des clusters par luminance moyenne croissante (sombre -> clair)
    luminance = centers.mean(axis=1)
    order = np.argsort(luminance)
    centers = centers[order]
    U = U[:, order]

    return U, centers, history_Jm, elapsed_time


# =========================================================
# 2. CHARGEMENT DES DONNÉES
# =========================================================
# Remplacez par vos propres chemins d'images (ex: cameraman.png et peppers.png)
path_gray = "image.png" 
img_gray_pil = Image.open(path_gray).convert("L").resize((256, 256))
img_gray = np.array(img_gray_pil, dtype=np.float32) / 255.0
H, W = img_gray.shape
X_gray = img_gray.reshape(-1, 1)


# =========================================================
# FIGURE 1 : RÉSULTAT DE BASE (C=2, m=2) + CONVERGENCE Jm
# =========================================================
U_base, centers_base, hist_Jm, t_base = fuzzy_c_means(X_gray, C=2, m=2.0)
pc_base = partition_coefficient(U_base)
pe_base = partition_entropy(U_base)

print(f"--- EXPÉRIENCE 1 (Base C=2, m=2) ---")
print(f"Itérations : {len(hist_Jm)-1} | Temps : {t_base*1000:.2f} ms")
print(f"Centres : v0={centers_base[0,0]:.3f}, v1={centers_base[1,0]:.3f}")
print(f"Métriques : PC = {pc_base:.4f} | PE = {pe_base:.4f}\n")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
custom_cmap = ListedColormap(['#d62728', '#1f77b4'])

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("1. Image originale")
axes[0].axis("off")

im1 = axes[1].imshow(U_base[:, 0].reshape(H, W), cmap="Reds", vmin=0, vmax=1)
axes[1].set_title(f"2. Heatmap Sombre (C0)\nv0 = {centers_base[0, 0]:.2f}")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].axis("off")

im2 = axes[2].imshow(U_base[:, 1].reshape(H, W), cmap="Blues", vmin=0, vmax=1)
axes[2].set_title(f"3. Heatmap Clair (C1)\nv1 = {centers_base[1, 0]:.2f}")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
axes[2].axis("off")

seg_base = np.argmax(U_base, axis=1).reshape(H, W)
axes[3].imshow(seg_base, cmap=custom_cmap)
axes[3].set_title(f"4. Segmentation Nette\nPC = {pc_base:.3f} | PE = {pe_base:.3f}")
axes[3].axis("off")

axes[4].plot(range(len(hist_Jm)), hist_Jm, marker='o', markersize=4, color='#2ca02c', linewidth=2)
axes[4].set_title(f"5. Convergence de Jm\n({len(hist_Jm)-1} itér. en {t_base*1000:.1f} ms)")
axes[4].set_xlabel("Itérations")
axes[4].set_ylabel("Fonction objectif Jm")
axes[4].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/fig1_base_convergence.png", dpi=300)
plt.close()


# =========================================================
# FIGURE 2 : SENSIBILITÉ AU PARAMÈTRE DE FLOU (m)
# =========================================================
m_values = [1.1, 2.0, 4.0]
fig, axes = plt.subplots(2, len(m_values), figsize=(14, 8))

print(f"--- EXPÉRIENCE 2 (Sensibilité à m) ---")
for idx, m_val in enumerate(m_values):
    U_m, c_m, hist_m, _ = fuzzy_c_means(X_gray, C=2, m=m_val)
    pc_m = partition_coefficient(U_m)
    pe_m = partition_entropy(U_m)
    print(f"m = {m_val:.1f} -> Itér: {len(hist_m)-1:2d} | PC = {pc_m:.4f} | PE = {pe_m:.4f}")

    # Ligne 1 : Heatmap du cluster sombre (C0)
    im = axes[0, idx].imshow(U_m[:, 0].reshape(H, W), cmap="viridis", vmin=0, vmax=1)
    axes[0, idx].set_title(f"Heatmap C0 (m = {m_val})\nPC = {pc_m:.3f} | PE = {pe_m:.3f}")
    plt.colorbar(im, ax=axes[0, idx], fraction=0.046, pad=0.04)
    axes[0, idx].axis("off")

    # Ligne 2 : Carte d'incertitude (Entropie locale par pixel)
    entropy_map = -np.sum(U_m * np.log2(U_m + 1e-12), axis=1).reshape(H, W)
    im_ent = axes[1, idx].imshow(entropy_map, cmap="magma", vmin=0, vmax=1)
    axes[1, idx].set_title(f"Carte d'incertitude (m = {m_val})\n(1 = incertitude maximale)")
    plt.colorbar(im_ent, ax=axes[1, idx], fraction=0.046, pad=0.04)
    axes[1, idx].axis("off")

plt.tight_layout()
plt.savefig("figures/fig2_parametre_m.png", dpi=300)
plt.close()
print()


# =========================================================
# FIGURE 3 : EXTENSION MULTI-CLUSTERS (C=3) & COULEUR (RGB)
# =========================================================
# Si vous n'avez pas d'image couleur dédiée, on crée une version RGB synthétique
# ou vous pouvez charger votre propre image couleur (ex: Image.open("peppers.png"))
try:
    img_color_pil = Image.open("image_couleur.png").convert("RGB").resize((256, 256))
    img_color = np.array(img_color_pil, dtype=np.float32) / 255.0
except FileNotFoundError:
    # Fallback propre : utilise l'image d'origine si elle était en RGB avant .convert("L")
    img_color_pil = Image.open(path_gray).convert("RGB").resize((256, 256))
    img_color = np.array(img_color_pil, dtype=np.float32) / 255.0

X_color = img_color.reshape(-1, 3)
U_c3, centers_c3, hist_c3, t_c3 = fuzzy_c_means(X_color, C=3, m=2.0)
pc_c3 = partition_coefficient(U_c3)

print(f"--- EXPÉRIENCE 3 (Multi-clusters C=3 en RGB) ---")
print(f"Itérations : {len(hist_c3)-1} | Temps : {t_c3*1000:.2f} ms | PC = {pc_c3:.4f}\n")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img_color)
axes[0].set_title("1. Image RGB (C=3)")
axes[0].axis("off")

cmaps_c3 = ["Reds", "Greens", "Blues"]
for c_idx in range(3):
    im = axes[c_idx + 1].imshow(U_c3[:, c_idx].reshape(H, W), cmap=cmaps_c3[c_idx], vmin=0, vmax=1)
    axes[c_idx + 1].set_title(f"{c_idx+2}. Heatmap Cluster {c_idx}")
    plt.colorbar(im, ax=axes[c_idx + 1], fraction=0.046, pad=0.04)
    axes[c_idx + 1].axis("off")

# Reconstruction de l'image segmentée avec les vraies couleurs des centroïdes !
labels_c3 = np.argmax(U_c3, axis=1)
segmented_rgb = centers_c3[labels_c3].reshape(H, W, 3)
axes[4].imshow(np.clip(segmented_rgb, 0, 1))
axes[4].set_title(f"5. Quantification Couleur\n(PC = {pc_c3:.3f})")
axes[4].axis("off")

plt.tight_layout()
plt.savefig("figures/fig3_multicluster_rgb.png", dpi=300)
plt.close()


# =========================================================
# FIGURE 4 : ROBUSTESSE AU BRUIT (LIMITE DU FCM STANDARD)
# =========================================================
np.random.seed(42)
noise = np.random.normal(0, 0.12, img_gray.shape)
img_noisy = np.clip(img_gray + noise, 0, 1)
X_noisy = img_noisy.reshape(-1, 1)

U_noisy, centers_noisy, _, _ = fuzzy_c_means(X_noisy, C=2, m=2.0)
pc_noisy = partition_coefficient(U_noisy)
seg_noisy = np.argmax(U_noisy, axis=1).reshape(H, W)

# Calcul du taux de pixels qui ont changé de classe à cause du bruit
pixel_error_rate = np.mean(seg_base != seg_noisy) * 100

print(f"--- EXPÉRIENCE 4 (Robustesse au bruit gaussien sigma=0.12) ---")
print(f"PC sans bruit : {pc_base:.4f} -> PC avec bruit : {pc_noisy:.4f}")
print(f"Taux de pixels mal classés par rapport à la référence : {pixel_error_rate:.2f}%\n")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_noisy, cmap="gray")
axes[0].set_title("1. Image bruitée (σ = 0.12)")
axes[0].axis("off")

im_n = axes[1].imshow(U_noisy[:, 0].reshape(H, W), cmap="Reds", vmin=0, vmax=1)
axes[1].set_title(f"2. Heatmap C0 bruitée\nPC chute à {pc_noisy:.3f}")
plt.colorbar(im_n, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].axis("off")

axes[2].imshow(seg_noisy, cmap=custom_cmap)
axes[2].set_title("3. Segmentation Nette\n(Effet 'poivre et sel')")
axes[2].axis("off")

diff_map = (seg_base != seg_noisy).astype(float)
axes[3].imshow(diff_map, cmap="gray")
axes[3].set_title(f"4. Carte des erreurs\n({pixel_error_rate:.1f}% de pixels altérés)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("figures/fig4_robustesse_bruit.png", dpi=300)
plt.close()

print("Toutes les figures ont été générées avec succès dans le dossier 'figures/'.")