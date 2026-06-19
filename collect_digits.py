"""
Dosya Adı: collect_digits.py

Rakam Örnek Toplayıcı — Bölme Bazlı v4 (Manuel Ayarlanabilir Bölmeler)
======================================================================
Her kayıt işleminde hangi bölmenin (slot) kaydedileceği sırayla ilerler.

Tuşlar:
  [0-9]   → aktif bölmedeki rakamı o etiketle kaydet, sonraki bölmeye geç
  [BOŞLUK]→ aktif bölmeyi BOŞ olarak kaydet, sonraki bölmeye geç
  [D]     → son kaydı sil ve bir önceki bölmeye geri dön
  [,]     → bir önceki bölmeye git (kaydetmeden)
  [.]     → bir sonraki bölmeye geç (kaydetmeden)
  
  -- BÖLME AYARI --
  [F]     → Aktif bölmeyi Sola kaydır
  [H]     → Aktif bölmeyi Sağa kaydır
  [T]     → Aktif bölmeyi Genişlet
  [G]     → Aktif bölmeyi Daralt

  [Q]     → çıkış
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


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


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

    # Eğer json içinde özel slot listesi yoksa veya eksikse, eşit olarak oluştur ve kaydet
    if "slots" not in config or len(config["slots"]) != n_digits:
        step = w // n_digits
        config["slots"] = [{"x": i * step, "w": step} for i in range(n_digits)]
        save_config(config)

    cap = cv2.VideoCapture(1) # Kamera indexiniz 1 ise 1 kalmalı
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print("=" * 60)
    print("  RAKAM ÖRNEK TOPLAYICI — Bölme Bazlı v4")
    print("=" * 60)
    print(f"  ROI: x={x} y={y} w={w} h={h}  |  {n_digits} bölme")
    print("  [F] Sola Kaydir  |  [H] Saga Kaydir")
    print("  [T] Genislet     |  [G] Daralt")
    print()

    current_slot = 0
    last_saved   = []
    counts       = count_samples()

    SLOT_COLORS = [
        (0,   220,  80),
        (80,  180, 255),
        (255, 160,   0),
        (220,  80, 220),
        (0,   220, 220),
    ]

    def slot_color(i): return SLOT_COLORS[i % len(SLOT_COLORS)]

    while True:
        ret, frame = cap.read()
        if not ret: break

        display  = frame.copy()
        roi_full = frame[y:y+h, x:x+w]

        # Tüm bölme çerçevelerini çiz
        for i in range(n_digits):
            slot_x = config["slots"][i]["x"]
            slot_w = config["slots"][i]["w"]
            dx     = x + slot_x
            
            color     = slot_color(i)
            thickness = 3 if i == current_slot else 1
            cv2.rectangle(display, (dx, y), (dx + slot_w, y + h), color, thickness)
            cv2.putText(display, str(i + 1), (dx + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Aktif bölme referansları
        active_sx  = config["slots"][current_slot]["x"]
        active_sw  = config["slots"][current_slot]["w"]
        active_dx  = x + active_sx
        active_col = slot_color(current_slot)
        
        # Etiket
        label_txt  = f">> BOLME {current_slot + 1} / {n_digits} <<"
        (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lx = max(0, min(active_dx + active_sw // 2 - lw // 2, display.shape[1] - lw - 4))
        cv2.rectangle(display, (lx - 4, y - lh - 18), (lx + lw + 4, y - 2), (0, 0, 0), -1)
        cv2.putText(display, label_txt, (lx, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, active_col, 2)

        # Aktif bölme önizlemesi (sağ üst) - Crash önlemek için sınır kontrolü
        crop_start = max(0, min(active_sx, w - 1))
        crop_end   = max(crop_start + 1, min(active_sx + active_sw, w))
        slot_preview = roi_full[:, crop_start:crop_end]
        
        if slot_preview.size > 0:
            ph = 80
            pw = int(slot_preview.shape[1] * ph / slot_preview.shape[0])
            preview  = cv2.resize(slot_preview, (pw, ph))
            px_start = display.shape[1] - pw - 10
            display[10:10+ph, px_start:px_start+pw] = preview
            cv2.rectangle(display, (px_start - 2, 8), (px_start + pw + 2, 10 + ph + 2), active_col, 2)

        # Alt panel
        H, W = display.shape[:2]
        cv2.rectangle(display, (0, H - 140), (W, H), (20, 20, 20), -1)

        bar_line = "  ".join([f"{d}:{counts[d]}" for d in range(10)]) + f"  bos:{counts.get(BLANK_LABEL, 0)}"
        cv2.putText(display, "Ornek sayisi:", (10, H - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(display, bar_line, (10, H - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 230, 130), 1)

        hint1 = "[0-9] Kaydet   [BOSLUK] Bos Kaydet   [D] Sil+Geri   [,] Onceki   [.] Sonraki   [Q] Cikis"
        hint2 = "Bölme Ayarı: [F] Sola Kaydır   [H] Sağa Kaydır   [T] Genişlet   [G] Daralt"
        cv2.putText(display, hint1, (10, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        cv2.putText(display, hint2, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1)

        cv2.imshow("Ornek Toplayici", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 255: continue

        # Çıkış
        if key in (ord("q"), ord("Q")): break
        
        # ── MANUEL BÖLME AYARLARI ──
        elif key in (ord("f"), ord("F")):
            config["slots"][current_slot]["x"] -= 2
            save_config(config)
        elif key in (ord("h"), ord("H")):
            config["slots"][current_slot]["x"] += 2
            save_config(config)
        elif key in (ord("t"), ord("T")):
            config["slots"][current_slot]["w"] += 2
            save_config(config)
        elif key in (ord("g"), ord("G")):
            config["slots"][current_slot]["w"] = max(5, config["slots"][current_slot]["w"] - 2)
            save_config(config)

        # Kayıt (Rakam veya Boşluk)
        elif key in range(ord("0"), ord("9") + 1) or key == ord(" "):
            is_blank   = (key == ord(" "))
            label      = BLANK_LABEL if is_blank else chr(key)
            anim_text  = "BOS KAYDEDILDI" if is_blank else f"'{label}' KAYDEDILDI"
            anim_color = (80, 80, 255) if is_blank else (0, 255, 255)

            slot_img  = roi_full[:, crop_start:crop_end].copy()
            digit_img = preprocess_digit_roi(slot_img)

            ts        = int(time.time() * 1000)
            folder    = f"{DATA_DIR}/{label}"
            os.makedirs(folder, exist_ok=True)
            save_path = f"{folder}/bolme{current_slot+1}_{label}_{ts}.png"
            cv2.imwrite(save_path, digit_img)

            if is_blank:
                counts[BLANK_LABEL] = counts.get(BLANK_LABEL, 0) + 1
            else:
                counts[int(label)] += 1

            last_saved.append((save_path, label))
            
            # Onay animasyonu
            cv2.rectangle(display, (active_dx, y), (active_dx + active_sw, y + h), anim_color, 4)
            cv2.putText(display, anim_text, (active_dx, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, anim_color, 2)
            cv2.imshow("Ornek Toplayici", display)
            cv2.waitKey(250)

            current_slot = (current_slot + 1) % n_digits

        # Silme ve Yönlendirme
        elif key in (ord("d"), ord("D")):
            if last_saved:
                path, label = last_saved.pop()
                if os.path.exists(path):
                    os.remove(path)
                    if label == BLANK_LABEL:
                        counts[BLANK_LABEL] = max(0, counts.get(BLANK_LABEL, 0) - 1)
                    else:
                        counts[int(label)] = max(0, counts[int(label)] - 1)
                current_slot = (current_slot - 1) % n_digits
        elif key in (ord(","), ord("<")):
            current_slot = (current_slot - 1) % n_digits
        elif key in (ord("."), ord(">")):
            current_slot = (current_slot + 1) % n_digits

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()