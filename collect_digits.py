"""
Rakam Örnek Toplayıcı — Bölme Bazlı v3
========================================
Her kayıt işleminde hangi bölmenin (slot) kaydedileceği sırayla ilerler:
  1. bölme → 2. bölme → 3. bölme → 4. bölme → 5. bölme → 1. bölme → ...

Tuşlar:
  [0-9]   → aktif bölmedeki rakamı o etiketle kaydet, sonraki bölmeye geç
  [BOŞLUK]→ aktif bölmeyi BOŞ olarak kaydet, sonraki bölmeye geç
  [D]     → son kaydı sil ve bir önceki bölmeye geri dön
  [,]     → bir önceki bölmeye git (kaydetmeden)
  [.]     → bir sonraki bölmeye geç (kaydetmeden)
  [Q]     → çıkış

Boş bölme örnekleri: data/digits/bos/*.png
Rakam örnekleri   : data/digits/0/*.png  …  data/digits/9/*.png
"""

import cv2
import numpy as np
import json
import os
import time

CONFIG_FILE = "lcd_config.json"
DATA_DIR    = "data/digits"
BLANK_LABEL = "bos"

for d in range(10):
    os.makedirs(f"{DATA_DIR}/{d}", exist_ok=True)
os.makedirs(f"{DATA_DIR}/{BLANK_LABEL}", exist_ok=True)


def count_samples():
    counts = {}
    for d in range(10):
        path = f"{DATA_DIR}/{d}"
        counts[d] = len([f for f in os.listdir(path) if f.endswith(".png")])
    bos_path = f"{DATA_DIR}/{BLANK_LABEL}"
    counts[BLANK_LABEL] = len([f for f in os.listdir(bos_path) if f.endswith(".png")])
    return counts


def preprocess_digit_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_CUBIC)
    return resized


def main():
    if not os.path.exists(CONFIG_FILE):
        print("[HATA] lcd_config.json bulunamadı.")
        print("  Önce lcd_reader.py ile kalibrasyon yapın.")
        return

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    if not all(k in config for k in ("x", "y", "w", "h")):
        print("[HATA] ROI koordinatları eksik.")
        return

    n_digits = config.get("n_digits", 5)
    x, y, w, h = config["x"], config["y"], config["w"], config["h"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print("=" * 60)
    print("  RAKAM ÖRNEK TOPLAYICI — Bölme Bazlı v3")
    print("=" * 60)
    print(f"  ROI: x={x} y={y} w={w} h={h}  |  {n_digits} bölme")
    print()
    print("  [0-9]    Rakamı kaydet → sonraki bölme")
    print("  [BOŞLUK] Bölmeyi BOŞ olarak kaydet → sonraki bölme")
    print("  [D]      Son kaydı sil, önceki bölmeye dön")
    print("  [,] [.]  Bölme değiştir (kaydetmeden)")
    print("  [Q]      Çıkış")
    print()

    current_slot = 0
    last_saved   = []
    counts       = count_samples()
    step         = w // n_digits

    SLOT_COLORS = [
        (0,   220,  80),
        (80,  180, 255),
        (255, 160,   0),
        (220,  80, 220),
        (0,   220, 220),
    ]

    def slot_color(i):
        return SLOT_COLORS[i % len(SLOT_COLORS)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display  = frame.copy()
        roi_full = frame[y:y+h, x:x+w]

        # Tüm bölme çerçeveleri
        for i in range(n_digits):
            dx        = x + i * step
            color     = slot_color(i)
            thickness = 3 if i == current_slot else 1
            cv2.rectangle(display, (dx, y), (dx + step, y + h), color, thickness)
            cv2.putText(display, str(i + 1),
                        (dx + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Aktif bölme etiketi
        active_dx  = x + current_slot * step
        active_col = slot_color(current_slot)
        label_txt  = f">> BOLME {current_slot + 1} / {n_digits} <<"
        (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lx = max(0, min(active_dx + step // 2 - lw // 2,
                        display.shape[1] - lw - 4))
        cv2.rectangle(display, (lx - 4, y - lh - 18), (lx + lw + 4, y - 2),
                      (0, 0, 0), -1)
        cv2.putText(display, label_txt, (lx, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, active_col, 2)

        # Aktif bölme önizlemesi (sağ üst)
        slot_preview = roi_full[:, current_slot * step:(current_slot + 1) * step]
        if slot_preview.size > 0:
            ph = 80
            pw = int(slot_preview.shape[1] * ph / slot_preview.shape[0])
            preview  = cv2.resize(slot_preview, (pw, ph))
            px_start = display.shape[1] - pw - 10
            display[10:10+ph, px_start:px_start+pw] = preview
            cv2.rectangle(display,
                          (px_start - 2, 8), (px_start + pw + 2, 10 + ph + 2),
                          active_col, 2)
            cv2.putText(display, f"Bolme {current_slot+1} onizleme",
                        (px_start, 10 + ph + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, active_col, 1)

        # Alt panel
        H, W = display.shape[:2]
        cv2.rectangle(display, (0, H - 125), (W, H), (20, 20, 20), -1)

        bar_line = "  ".join([f"{d}:{counts[d]}" for d in range(10)])
        bar_line += f"  bos:{counts.get(BLANK_LABEL, 0)}"
        cv2.putText(display, "Ornek sayisi:", (10, H - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(display, bar_line, (10, H - 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 230, 130), 1)

        eksik = [str(d) for d in range(10) if counts[d] < 30]
        if counts.get(BLANK_LABEL, 0) < 30:
            eksik.append("bos")
        if eksik:
            msg = f"Hedef: 30+ ornek  |  Eksik: {', '.join(eksik)}"
            cv2.putText(display, msg, (10, H - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 180, 255), 1)
        else:
            cv2.putText(display,
                        "Tum siniflar icin yeterli ornek var! [Q] cikis",
                        (10, H - 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (80, 220, 80), 1)

        hint = ("[0-9] kaydet  [BOSLUK] bos kaydet  "
                "[D] sil+geri  [,] onceki  [.] sonraki  [Q] cikis")
        cv2.putText(display, hint, (10, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        cv2.imshow("Ornek Toplayici", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue

        # ── Q: çıkış ──────────────────────────────────────
        if key in (ord("q"), ord("Q")):
            break

        # ── 0-9 veya BOŞLUK: kaydet ───────────────────────
        elif key in range(ord("0"), ord("9") + 1) or key == ord(" "):

            is_blank   = (key == ord(" "))
            label      = BLANK_LABEL if is_blank else chr(key)
            anim_text  = "BOS KAYDEDILDI" if is_blank else f"'{label}' KAYDEDILDI"
            anim_color = (80, 80, 255)    if is_blank else (0, 255, 255)

            slot_img  = roi_full[:, current_slot * step:(current_slot + 1) * step].copy()
            digit_img = preprocess_digit_roi(slot_img)

            ts        = int(time.time() * 1000)
            folder    = f"{DATA_DIR}/{label}"
            os.makedirs(folder, exist_ok=True)
            save_path = f"{folder}/bolme{current_slot+1}_{label}_{ts}.png"
            cv2.imwrite(save_path, digit_img)

            if is_blank:
                counts[BLANK_LABEL] = counts.get(BLANK_LABEL, 0) + 1
                total_label = counts[BLANK_LABEL]
            else:
                counts[int(label)] += 1
                total_label = counts[int(label)]

            last_saved.append((save_path, label))
            print(f"  [+] Bolme {current_slot+1}/{n_digits}  "
                  f"etiket='{label}'  (toplam: {total_label})  → {save_path}")

            # Onay animasyonu
            cv2.rectangle(display,
                          (active_dx, y), (active_dx + step, y + h),
                          anim_color, 4)
            cv2.putText(display, anim_text,
                        (active_dx, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, anim_color, 2)
            cv2.imshow("Ornek Toplayici", display)
            cv2.waitKey(250)

            current_slot = (current_slot + 1) % n_digits

        # ── D: son kaydı sil ──────────────────────────────
        elif key in (ord("d"), ord("D")):
            if last_saved:
                path, label = last_saved.pop()
                if os.path.exists(path):
                    os.remove(path)
                    if label == BLANK_LABEL:
                        counts[BLANK_LABEL] = max(0, counts.get(BLANK_LABEL, 0) - 1)
                    else:
                        counts[int(label)] = max(0, counts[int(label)] - 1)
                    print(f"  [-] Silindi: {path}")
                current_slot = (current_slot - 1) % n_digits
                print(f"  [←] Bolme {current_slot+1}'e donuldu")
            else:
                print("  [!] Silinecek kayıt yok.")

        # ── , : önceki bölme ──────────────────────────────
        elif key in (ord(","), ord("<"), 81):
            current_slot = (current_slot - 1) % n_digits
            print(f"  [←] Bolme {current_slot+1}")

        # ── . : sonraki bölme ─────────────────────────────
        elif key in (ord("."), ord(">"), 83):
            current_slot = (current_slot + 1) % n_digits
            print(f"  [→] Bolme {current_slot+1}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("  TOPLAMA TAMAMLANDI")
    print("=" * 60)
    counts = count_samples()
    for d in range(10):
        bar = "█" * min(counts[d], 50)
        print(f"  {d}  : {bar} ({counts[d]} örnek)")
    bar = "█" * min(counts.get(BLANK_LABEL, 0), 50)
    print(f"  bos: {bar} ({counts.get(BLANK_LABEL, 0)} örnek)")

    total = sum(v for v in counts.values())
    print(f"\n  Toplam: {total} görüntü")
    if total >= 100:
        print("  Eğitime hazır! Şimdi şunu çalıştırın:")
        print("    python train_digits.py")
    else:
        print("  Daha fazla örnek tolamanız önerilir (toplam 100+).")


if __name__ == "__main__":
    main()