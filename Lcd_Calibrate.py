"""
LCD Segment Eşik Ayarlayıcı (Kalibrasyon Yardımcısı)
=====================================================
Bu araç, kameranızın aydınlatma koşullarına göre
segment algılama eşiklerini görsel olarak ayarlamanızı sağlar.

Kullanım:
  python lcd_calibrate.py

Çıktı:
  lcd_config.json dosyasını günceller
"""

import cv2
import numpy as np
import json
import os

CONFIG_FILE = "lcd_config.json"


def nothing(x):
    pass


def calibrate_thresholds():
    print("=" * 50)
    print("  LCD Eşik Kalibrasyon Aracı")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!")
        return

    # Config yükle
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    # Kontrol paneli
    cv2.namedWindow("Parametreler")
    cv2.createTrackbar("Blur Kernel", "Parametreler", 5, 21, nothing)
    cv2.createTrackbar("Thresh Min", "Parametreler",
                       config.get("thresh_min", 80), 255, nothing)
    cv2.createTrackbar("Dilate", "Parametreler",
                       config.get("dilate", 1), 10, nothing)
    cv2.createTrackbar("Segment Esik %", "Parametreler",
                       config.get("seg_threshold", 20), 60, nothing)

    print("\nAyarları yapın, sonra [S] kaydet, [Q] çık")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Trackbar değerleri
        blur_k = cv2.getTrackbarPos("Blur Kernel", "Parametreler")
        if blur_k % 2 == 0:
            blur_k += 1  # tek sayı olmalı
        thresh_min = cv2.getTrackbarPos("Thresh Min", "Parametreler")
        dilate_k = cv2.getTrackbarPos("Dilate", "Parametreler")
        seg_thr = cv2.getTrackbarPos("Segment Esik %", "Parametreler")

        # Görüntü işle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        _, thresh = cv2.threshold(blur, thresh_min, 255,
                                  cv2.THRESH_BINARY_INV)
        if dilate_k > 0:
            kernel = np.ones((dilate_k, dilate_k), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        # ROI varsa göster
        if config:
            x = config.get("x", 0)
            y = config.get("y", 0)
            w = config.get("w", frame.shape[1])
            h = config.get("h", frame.shape[0])
            roi_thresh = thresh[y:y+h, x:x+w]

            # Ekranda önizleme
            preview = cv2.cvtColor(roi_thresh, cv2.COLOR_GRAY2BGR)
            ph, pw = preview.shape[:2]
            scale = min(400/pw, 120/ph)
            preview_small = cv2.resize(preview,
                                       (int(pw*scale), int(ph*scale)))

            # Ana frame'e ROI kutusunu çiz
            display = frame.copy()
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Threshold önizlemesini üste yerleştir
            py, px = 10, 10
            ey, ex = py + preview_small.shape[0], px + preview_small.shape[1]
            display[py:ey, px:ex] = preview_small

            cv2.putText(display, "ROI Threshold Onizleme",
                        (10, py + preview_small.shape[0] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            display = frame.copy()

        cv2.imshow("Kalibrasyon", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            config.update({
                "blur_kernel": blur_k,
                "thresh_min": thresh_min,
                "dilate": dilate_k,
                "seg_threshold": seg_thr,
            })
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[✓] Kaydedildi: blur={blur_k}, thresh={thresh_min}, "
                  f"dilate={dilate_k}, seg_thr={seg_thr}%")
            print(f"    → {CONFIG_FILE}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_thresholds()