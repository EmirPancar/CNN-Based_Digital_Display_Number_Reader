"""
Dosya Adı: train_Digits.py

Rakam Sınıflandırıcı Eğitim Scripti — v2 (boş bölme destekli)
===============================================================
collect_digits.py ile toplanan verilerle CNN eğitir.
0-9 rakamlarına ek olarak "bos" sınıfını da öğrenir.

Klasör yapısı:
  data/digits/0/   …  data/digits/9/   → rakam örnekleri
  data/digits/bos/                     → boş bölme örnekleri

Kurulum:
  pip install torch torchvision scikit-learn matplotlib

Kullanım:
  python train_digits.py

Çıktı:
  digit_model.pth  (lcd_reader.py bu dosyayı okur)
"""

import os
import numpy as np
import cv2
from pathlib import Path

DATA_DIR   = "data/digits"
MODEL_FILE = "digit_model.pth"
BLANK_LABEL = "bos"

# ── Kütüphane kontrolü ─────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import pickle
    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False


# ================================================================
#  Sınıf listesi: 0-9 + "bos"
#  Etiket → indeks eşlemesi burada merkezi olarak yönetilir.
# ================================================================
def build_class_list(counts):
    """
    Verisi olan sınıfları döndürür.
    Rakamlar önce (sıralı), ardından "bos" gelir.
    Örn: [0, 1, 3, 5, 7, 8, 9, "bos"]
    """
    classes = sorted([d for d in range(10) if counts.get(d, 0) > 0])
    if counts.get(BLANK_LABEL, 0) > 0:
        classes.append(BLANK_LABEL)   # "bos" en sona
    return classes


# ================================================================
#  Veri yükleme
# ================================================================
def load_data():
    X, y_raw = [], []

    # 0-9 rakamları
    for digit in range(10):
        folder = Path(DATA_DIR) / str(digit)
        if not folder.exists():
            continue
        for fp in folder.glob("*.png"):
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (32, 32))
            X.append(img)
            y_raw.append(digit)           # int etiket

    # Boş bölme
    bos_folder = Path(DATA_DIR) / BLANK_LABEL
    if bos_folder.exists():
        for fp in bos_folder.glob("*.png"):
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (32, 32))
            X.append(img)
            y_raw.append(BLANK_LABEL)     # string etiket

    X = np.array(X, dtype=np.float32) / 255.0
    return X, y_raw   # y_raw: int veya "bos" karışık liste


def check_data():
    counts = {}
    for d in range(10):
        folder = Path(DATA_DIR) / str(d)
        counts[d] = len(list(folder.glob("*.png"))) if folder.exists() else 0

    bos_folder = Path(DATA_DIR) / BLANK_LABEL
    counts[BLANK_LABEL] = len(list(bos_folder.glob("*.png"))) if bos_folder.exists() else 0

    print("\n  Mevcut veri:")
    for d in range(10):
        bar    = "█" * min(counts[d], 40)
        status = "OK" if counts[d] >= 20 else "AZ" if counts[d] > 0 else "YOK"
        print(f"  {d}  : {bar} ({counts[d]:3d}) [{status}]")

    bar    = "█" * min(counts[BLANK_LABEL], 40)
    status = "OK" if counts[BLANK_LABEL] >= 20 else "AZ" if counts[BLANK_LABEL] > 0 else "YOK"
    print(f"  bos: {bar} ({counts[BLANK_LABEL]:3d}) [{status}]")

    total            = sum(counts.values())
    classes_with_data = sum(1 for v in counts.values() if v > 0)
    return total, classes_with_data, counts


# ================================================================
#  PyTorch CNN
# ================================================================
class DigitCNN(nn.Module):
    def __init__(self, n_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DigitDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X).unsqueeze(1)   # (N,1,32,32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.aug = transforms.Compose([
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self.aug(x)
        return x, self.y[idx]


def train_pytorch(X, y_raw, counts):
    print("\n  [PyTorch CNN] Eğitim başlıyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Cihaz: {device}")

    # Sınıf listesi ve etiket→indeks eşlemesi
    classes   = build_class_list(counts)
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    n_classes = len(classes)

    print(f"  Siniflar ({n_classes}): {classes}")

    # y_raw (int veya "bos") → indeks dizisine çevir
    y_mapped = np.array([label_map[yi] for yi in y_raw], dtype=np.int64)

    # Train / val ayırımı
    idx   = np.random.permutation(len(X))
    split = int(len(idx) * 0.85)
    tr_idx, val_idx = idx[:split], idx[split:]

    tr_ds  = DigitDataset(X[tr_idx],  y_mapped[tr_idx],  augment=True)
    val_ds = DigitDataset(X[val_idx], y_mapped[val_idx], augment=False)
    tr_dl  = DataLoader(tr_ds,  batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model     = DigitCNN(n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state   = None
    EPOCHS       = 40

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == EPOCHS:
            tr_acc = tr_correct / max(tr_total, 1)
            star   = " *" if val_acc == best_val_acc else ""
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={tr_loss/tr_total:.4f}  "
                  f"train={tr_acc*100:.1f}%  "
                  f"val={val_acc*100:.1f}%{star}")

    # Kaydet
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "classes":     classes,      # örn. [0,1,2,...,9,"bos"]
        "label_map":   label_map,
        "n_classes":   n_classes,
        "backend":     "pytorch",
    }, MODEL_FILE)
    print(f"\n  [OK] Model kaydedildi: {MODEL_FILE}")
    print(f"  Siniflar: {classes}")
    print(f"  En iyi val dogrulugu: {best_val_acc*100:.1f}%")
    return best_val_acc


# ================================================================
#  Scikit-learn SVM (PyTorch yoksa fallback)
# ================================================================
def train_sklearn(X, y_raw, counts):
    print("\n  [Scikit-learn SVM] Eğitim başlıyor...")

    classes   = build_class_list(counts)
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y_mapped  = np.array([label_map[yi] for yi in y_raw], dtype=np.int64)

    X_flat = X.reshape(len(X), -1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, class_weight="balanced")),
    ])

    cv_n    = min(5, len(X_flat) // 10 + 1)
    scores  = cross_val_score(model, X_flat, y_mapped, cv=cv_n)
    print(f"  Çapraz dogrulama: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    model.fit(X_flat, y_mapped)

    save_path = MODEL_FILE.replace(".pth", "_sklearn.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "model":   model,
            "classes": classes,
            "backend": "sklearn",
        }, f)

    print(f"  [OK] Model kaydedildi: {save_path}")
    print(f"  Siniflar: {classes}")
    return scores.mean()


# ================================================================
#  Ana program
# ================================================================
def main():
    print()
    print("=" * 60)
    print("  RAKAM SINIFLANDIRICI EĞİTİMİ — v2 (bos destekli)")
    print("=" * 60)

    total, n_classes, counts = check_data()

    if total == 0:
        print("\n  [HATA] Hiç veri bulunamadı!")
        print("  Önce collect_digits.py ile örnek toplayın.")
        return

    if total < 50:
        print(f"\n  [UYARI] Toplam {total} örnek az.")
        answer = input("  Devam edilsin mi? (e/h): ").strip().lower()
        if answer != "e":
            return

    bos_count = counts.get(BLANK_LABEL, 0)
    if bos_count == 0:
        print("\n  [UYARI] Boş bölme örneği yok!")
        print("  collect_digits.py'de [BOŞLUK] tuşuyla boş bölme örnekleri toplayın.")
        print("  Boş bölme olmadan 4/5 rakam ayrımı doğru çalışmaz.")
        answer = input("  Yine de devam edilsin mi? (e/h): ").strip().lower()
        if answer != "e":
            return
    elif bos_count < 20:
        print(f"\n  [UYARI] Boş bölme örneği az ({bos_count}).")
        print("  Daha fazla boş bölme örneği önerilir (en az 30).")

    print(f"\n  Toplam {total} görüntü, {n_classes} sınıf")

    X, y_raw = load_data()
    print(f"  Yüklendi: X={X.shape}, toplam={len(y_raw)} etiket")

    if USE_TORCH:
        acc     = train_pytorch(X, y_raw, counts)
        backend = "pytorch"
    elif USE_SKLEARN:
        print("\n  PyTorch bulunamadı, Scikit-learn SVM kullanılıyor.")
        acc     = train_sklearn(X, y_raw, counts)
        backend = "sklearn"
    else:
        print("\n  [HATA] Ne PyTorch ne de Scikit-learn kurulu!")
        print("    pip install torch torchvision")
        print("    pip install scikit-learn")
        return

    print()
    print("=" * 60)
    print(f"  EĞİTİM TAMAMLANDI — Dogruluk: {acc*100:.1f}%  [{backend}]")
    print("=" * 60)
    print()
    print("  Sonraki adım: lcd_reader.py'yi çalıştırın.")
    if acc < 0.90:
        print()
        print("  İpucu: Doğruluk düşükse daha fazla örnek toplayın.")
        print("  Özellikle sorunlu sınıflar: 0,1,6,8,9 ve bos.")


if __name__ == "__main__":
    main()