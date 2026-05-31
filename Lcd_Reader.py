"""
LCD Dijital Ekran Okuyucu v3
==============================
EasyOCR + Özel Eğitilmiş Model

- EasyOCR genel okumayı yapar
- Düşük güvenli veya yanlış sonuçlarda özel model devreye girer
- Özellikle 0 ve 1 için çok daha doğru
- 4 rakamlı sayı tespiti: ilk bölme boşsa sayı 4 rakamlıdır,
  Excel'e 100'e bölünmüş ondalıklı olarak kaydedilir
  (örn: 2687 → 26.87,  bütün 5 bölmeli gösterimde 02687 → 26.87)

Kurulum:
  pip install easyocr opencv-python openpyxl numpy torch torchvision

Tuşlar:
  [S]  -> Kaydet
  [A]  -> Otomatik kayıt aç/kapat
  [C]  -> Kalibrasyon
  [+]  -> Aralığı artır
  [-]  -> Aralığı azalt
  [Q]  -> Çıkış
"""

import cv2
import numpy as np
import openpyxl
import os
import time
import json
from datetime import datetime

# ── EasyOCR ─────────────────────────────────────
try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    print("[HATA] EasyOCR kurulu degil:  pip install easyocr")
    exit(1)

# ── PyTorch (özel model) ─────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

CONFIG_FILE    = "lcd_config.json"
EXCEL_FILE     = "lcd_kayitlar.xlsx"
CUSTOM_MODEL   = "digit_model.pth"
SKLEARN_MODEL  = "digit_model_sklearn.pkl"


# ================================================================
#  Özel CNN modeli
# ================================================================
class DigitCNN(nn.Module):
    def __init__(self, n_classes=10):
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


# ================================================================
#  Model yükleyici
# ================================================================
class CustomDigitModel:
    def __init__(self):
        self.model   = None
        self.classes = None
        self.backend = None
        self.device  = None
        self._load()

    def _load(self):
        if TORCH_OK and os.path.exists(CUSTOM_MODEL):
            try:
                ckpt = torch.load(CUSTOM_MODEL, map_location="cpu",
                                  weights_only=False)
                self.classes = ckpt["classes"]
                n_classes    = ckpt["n_classes"]
                net = DigitCNN(n_classes=n_classes)
                net.load_state_dict(ckpt["model_state"])
                net.eval()
                self.model   = net
                self.backend = "pytorch"
                print(f"[OK] Ozel model yuklendi (PyTorch)  siniflar={self.classes}")
                return
            except Exception as e:
                print(f"[!] PyTorch model yuklenemedi: {e}")

        if os.path.exists(SKLEARN_MODEL):
            try:
                import pickle
                with open(SKLEARN_MODEL, "rb") as f:
                    data = pickle.load(f)
                self.model   = data["model"]
                self.classes = data["classes"]
                self.backend = "sklearn"
                print(f"[OK] Ozel model yuklendi (sklearn)  siniflar={self.classes}")
                return
            except Exception as e:
                print(f"[!] Sklearn model yuklenemedi: {e}")

        print("[!] Ozel model bulunamadi. Sadece EasyOCR kullanilacak.")

    def is_ready(self):
        return self.model is not None

    def predict(self, roi_gray_32x32):
        """
        Döndürür: (etiket, güven)
          etiket → int (0-9) veya "bos" veya None
        """
        if not self.is_ready():
            return None, 0.0
        img = roi_gray_32x32.astype(np.float32) / 255.0
        if self.backend == "pytorch":
            tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
            best_idx   = probs.argmax().item()
            confidence = probs[best_idx].item()
            label      = self.classes[best_idx]   # int veya "bos"
            return label, confidence
        elif self.backend == "sklearn":
            flat  = img.reshape(1, -1)
            idx   = self.model.predict(flat)[0]
            prob  = self.model.predict_proba(flat)[0]
            label = self.classes[idx]             # int veya "bos"
            return label, prob.max()
        return None, 0.0


# ================================================================
#  Görüntü ön işleme
# ================================================================
def preprocess_for_ocr(roi):
    h, w = roi.shape[:2]
    scale = max(1, int(80 / h))
    if scale > 1:
        roi = cv2.resize(roi, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def preprocess_digit(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_CUBIC)


# ================================================================
#  İlk bölmenin boş (segment yok) olup olmadığını kontrol et
# ================================================================
def is_slot_empty(roi_slot):
    """
    Bölmedeki piksellerin büyük çoğunluğu açık renk (LCD arka plan) ise
    o bölmede rakam yok demektir → True döner.
    """
    gray = cv2.cvtColor(roi_slot, cv2.COLOR_BGR2GRAY) if len(roi_slot.shape) == 3 else roi_slot
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    # Segment piksel oranı: koyu piksel (segment) ne kadar?
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.count_nonzero(dark) / dark.size
    # %5'ten az koyu piksel varsa bölme boş kabul edilir
    return dark_ratio < 0.05


# ================================================================
#  Ham sayı metnini ondalıklı değere çevir
# ================================================================
def format_value(raw_text, is_4digit):
    """
    raw_text : "27658" veya "2658" gibi sadece rakam içeren string
    is_4digit: True ise son 2 rakam virgülden sonra gelir

    Döndürür: (display_str, float_value)
      display_str → ekranda gösterilecek metin  "268.75"
      float_value → Excel'e yazılacak sayı       268.75
    """
    digits_only = raw_text.replace(".", "").replace("-", "").strip()

    if not digits_only.isdigit():
        return raw_text, None

    if is_4digit:
        # 4 rakamlı: son 2 hane ondalık
        if len(digits_only) >= 2:
            int_part  = digits_only[:-2] or "0"
            dec_part  = digits_only[-2:]
            display   = f"{int_part}.{dec_part}"
            try:
                value = float(display)
            except ValueError:
                value = None
            return display, value
        else:
            return raw_text, None
    else:
        # 5 rakamlı: sayıyı olduğu gibi yaz
        try:
            value = float(digits_only)
        except ValueError:
            value = None
        return digits_only, value


# ================================================================
#  EasyOCR başlatma
# ================================================================
print("[...] EasyOCR modeli yukleniyor...")
_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("[OK] EasyOCR hazir.")

custom_model = CustomDigitModel()

CONFIDENCE_THRESHOLD = 0.75


# ================================================================
#  Okuma: EasyOCR + özel model + 4-rakam tespiti
# ================================================================
def read_display(frame, config):
    x, y, w, h = config["x"], config["y"], config["w"], config["h"]
    n_digits    = config.get("n_digits", 5)
    roi = frame[y:y+h, x:x+w]
    step = w // n_digits

    # ── İlk bölme boş mu? (4 rakamlı sayı tespiti) ────────
    first_slot = roi[:, 0:step]
    is_4digit  = is_slot_empty(first_slot)

    # ── EasyOCR okuması ───────────────────────────────────
    processed  = preprocess_for_ocr(roi)
    results    = _reader.readtext(
        processed,
        allowlist="0123456789.-",
        detail=1,
        paragraph=False,
    )
    easyocr_text = ""
    easyocr_conf = 0.0
    if results:
        best = max(results, key=lambda r: r[2] * len(r[1]))
        easyocr_text = best[1].strip().replace(" ", "")
        easyocr_conf = best[2]

    # ── Özel model okuması ────────────────────────────────
    custom_text = ""
    custom_conf = 0.0
    if custom_model.is_ready():
        digit_chars = []
        for i in range(n_digits):
            digit_roi = roi[:, i*step:(i+1)*step]
            d32       = preprocess_digit(digit_roi)
            label, conf = custom_model.predict(d32)
            digit_chars.append((label, conf))

        # "bos" etiketli bölmeleri atla, rakamları birleştir
        custom_text = "".join(
            str(lbl) for lbl, _ in digit_chars if lbl != "bos"
        )
        custom_conf = float(np.mean([c for _, c in digit_chars]))

    # ── Sonuç birleştirme ─────────────────────────────────
    if custom_model.is_ready():
        if easyocr_conf < CONFIDENCE_THRESHOLD:
            raw_text   = custom_text
            final_conf = custom_conf
            source     = "OZEL"
        elif easyocr_text != custom_text:
            raw_text   = custom_text
            final_conf = custom_conf
            source     = "OZEL(duzeltildi)"
        else:
            raw_text   = easyocr_text
            final_conf = easyocr_conf
            source     = "EASYOCR"
    else:
        raw_text   = easyocr_text
        final_conf = easyocr_conf
        source     = "EASYOCR"

    # ── Ondalık formatlama ────────────────────────────────
    display_text, float_value = format_value(raw_text, is_4digit)

    debug_info = {
        "easyocr":  (easyocr_text, easyocr_conf),
        "custom":   (custom_text,  custom_conf),
        "final":    (display_text, final_conf),
        "source":   source,
        "is_4digit": is_4digit,
        "float_value": float_value,
    }
    return display_text, final_conf, debug_info


# ================================================================
#  Excel kayıt — sadece 100'e bölünmüş sayı
# ================================================================
def save_to_excel(raw_digits, confidence=None, excel_file=EXCEL_FILE):
    if os.path.exists(excel_file):
        wb = openpyxl.load_workbook(excel_file)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "LCD Kayitlari"

    digits_only = "".join(c for c in str(raw_digits) if c.isdigit())
    try:
        value = int(digits_only) / 100.0
    except (ValueError, ZeroDivisionError):
        value = raw_digits

    ws.append([value])
    wb.save(excel_file)
    return ws.max_row - 1


# ================================================================
#  Kalibrasyon
# ================================================================
def calibrate(cap):
    print("\n[KALIBRASYON] Fare ile ekrani secin, ENTER=onayla, ESC=iptal")
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        return None

    roi_coords = cv2.selectROI(
        "Ekrani Secin: Surukle, ENTER=Onayla, ESC=Iptal",
        frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()

    x, y, w, h = [int(v) for v in roi_coords]
    if w == 0 or h == 0:
        return None

    n_str    = input("Kac rakam var? (varsayilan 5): ").strip()
    n_digits = int(n_str) if n_str.isdigit() else 5

    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    config.update({"x": x, "y": y, "w": w, "h": h, "n_digits": n_digits})
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Kaydedildi -> {CONFIG_FILE}")
    return config


# ================================================================
#  Ana program
# ================================================================
def main():
    print()
    print("=" * 60)
    print("  LCD Okuyucu v3  --  EasyOCR + Ozel Model")
    print("  [S] Kaydet  [A] Oto  [C] Kalibrasyon  [Q] Cik")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera acilamadi!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    config = None
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    if not config or not all(k in config for k in ("x", "y", "w", "h")):
        config = calibrate(cap)
        if config is None:
            cap.release()
            return

    auto_save     = False
    auto_interval = 5
    last_auto_t   = 0
    last_reading  = ""
    last_conf     = 0.0
    save_count    = 0
    frame_skip    = 0
    OCR_EVERY     = 3
    last_debug    = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        x, y, w, h = config["x"], config["y"], config["w"], config["h"]

        frame_skip += 1
        if frame_skip >= OCR_EVERY:
            frame_skip = 0
            reading, conf, debug = read_display(frame, config)
            if reading:
                last_reading = reading
                last_conf    = conf
                last_debug   = debug

        # ROI kutusu
        color = (0, 220, 80) if last_reading else (80, 80, 80)
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        if last_reading:
            conf_pct  = int(last_conf * 100)
            source    = last_debug.get("source", "")
            is_4d     = last_debug.get("is_4digit", False)
            mode_tag  = "4-RAKAM" if is_4d else "5-RAKAM"
            label     = f"{last_reading}  ({conf_pct}%  {source}  {mode_tag})"

            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(display,
                          (x, y - lh - 16), (x + lw + 12, y),
                          (0, 0, 0), -1)
            cv2.putText(display, label, (x+6, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Debug satırı
            eocr_t, eocr_c = last_debug.get("easyocr", ("", 0))
            cust_t, cust_c = last_debug.get("custom",  ("", 0))
            fval           = last_debug.get("float_value", None)
            fval_str       = f"  Excel={fval}" if fval is not None else ""
            if eocr_t or cust_t:
                debug_line = (f"EasyOCR:{eocr_t}({int(eocr_c*100)}%)  "
                              f"Ozel:{cust_t}({int(cust_c*100)}%){fval_str}")
                cv2.putText(display, debug_line, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 180, 80), 1)

        # Otomatik kayıt
        if auto_save and last_reading:
            now_t = time.time()
            if now_t - last_auto_t >= auto_interval:
                save_count  = save_to_excel(last_reading, last_conf)
                last_auto_t = now_t
                print(f"[OTO] {last_reading}  {int(last_conf*100)}%"
                      f"  satir={save_count}")

        # Alt panel
        H, W = display.shape[:2]
        cv2.rectangle(display, (0, H-90), (W, H), (20, 20, 20), -1)
        auto_str = f"ACIK({auto_interval}s)" if auto_save else "KAPALI"
        cv2.putText(display,
                    f"[S]Kaydet [A]Oto:{auto_str} [+/-]Aralik "
                    f"[C]Kalibrasyon [Q]Cik",
                    (10, H-58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180, 180, 180), 1)
        model_status = ("Ozel model: AKTIF" if custom_model.is_ready()
                        else "Ozel model: YOK (collect+train calistirin)")
        cv2.putText(display,
                    f"{last_reading or '-'}  Kayit:{save_count}  {model_status}",
                    (10, H-28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (100, 230, 130), 1)

        cv2.imshow("LCD Okuyucu v3", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            if last_reading:
                save_count = save_to_excel(last_reading, last_conf)
                print(f"[KAYIT] {last_reading}  {int(last_conf*100)}%"
                      f"  satir={save_count}")
            else:
                print("[!] Okuma yok.")
        elif key in (ord("a"), ord("A")):
            auto_save = not auto_save
            print(f"[OTO] {'ACILDI' if auto_save else 'KAPATILDI'}")
        elif key == ord("+"):
            auto_interval = min(auto_interval + 1, 60)
        elif key == ord("-"):
            auto_interval = max(auto_interval - 1, 1)
        elif key in (ord("c"), ord("C")):
            config = calibrate(cap)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] {save_count} kayit -> {EXCEL_FILE}")


if __name__ == "__main__":
    main()