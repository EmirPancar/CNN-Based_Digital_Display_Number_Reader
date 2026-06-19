# 📟 Dijital Ekran Sayı Okuyucu

> Fiziksel dijital ekranlardaki sayıları gerçek zamanlı okuyup Excel'e kaydeden; standart OCR hatalarını sıfırdan eğitilmiş özel bir PyTorch CNN modeliyle çözen uçtan uca hibrit görüntü işleme sistemi.

Bu proje, hazır OCR kütüphanelerinin dijital ekranlarda yaşadığı kronik okuma hatalarını (1'i 7 sanma, 0'ı 8 ile karıştırma) çözmek amacıyla tasarlanmış uçtan uca bir Görüntü İşleme boru hattıdır.

## ✨ Öne Çıkan Özellikler

- **Hibrit OCR Motoru:** Hızlı okumalar için EasyOCR kullanılırken, düşük güven skorlarında sıfırdan eğitilmiş Özel PyTorch CNN modeli devreye girerek sonuçları doğrular veya düzeltir.
- **Kendi Veri Setini Üretme:** Farklı ortam ışıklarına ve ekran tiplerine adapte olabilmek için özel veri toplama ve etiketleme aracı içerir (`collect_digits.py`).
- **Dinamik Ortam Kalibrasyonu:** Işık parlamaları ve ekran açıları için canlı ROI (İlgili Alan), blur ve threshold ayarlama aracı barındırır.
- **Akıllı Formatlama:** Görüntüdeki "boş (blank)" segmentleri algılayarak ekrandaki sayının basamak değerini anlar. (Örn: İlk hane boşsa sayıyı 100'e bölerek ondalıklı float formatında `26.87` olarak kaydeder).
- **Otomatik Raporlama:** Okunan doğrulanmış verileri anlık olarak bir Excel dosyasına kaydeder.
- **Yedek Sistemi:** Sistemde PyTorch kurulu değilse veya GPU yoksa, Scikit-Learn SVM modeli ile otomatik yedekleme mekanizması çalışır.

## 🛠️ Kullanılan Teknolojiler

- **Dil:** Python 3.8+
- **Derin Öğrenme / ML:** PyTorch, Scikit-Learn
- **Görüntü İşleme:** OpenCV, NumPy
- **OCR:** EasyOCR
- **Veri Kayıt:** Openpyxl, JSON

## 📂 Proje Mimarisi ve Dosya Yapısı

Sistem 4 ana aşamadan oluşur:

1. `lcd_calibrate.py` ➔ Kameranın ortam ışığına göre Threshold, Blur ve Dilate ayarlarının yapıldığı ve ROI bölgesinin seçildiği kalibrasyon aracı.
2. `collect_digits.py` ➔ Seçilen bölgeden 0-9 rakamlarını ve "Boş (Blank)" ekran görüntülerini toplayıp etiketleyen manuel veri toplama arayüzü.
3. `train_digits.py` ➔ Toplanan veri seti ile veri artırma teknikleri kullanılarak 32x32 boyutlarında 11 Sınıflı bir Evrişimli Sinir Ağı (CNN) eğiten script.
4. `lcd_reader.py` ➔ Hibrit yapıyı çalıştıran, kameradan canlı veriyi okuyan ve Excel'e kaydeden ana çıkarım programı.

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin
Sistemi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
```bash
pip install torch torchvision opencv-python numpy easyocr scikit-learn openpyxl matplotlib
```

### 2. Kullanım Adımları

Projenin kendi çalışma ortamınızda %100 doğrulukla çalışması için aşağıdaki adımları sırasıyla izleyin:

**Adım 1: Kalibrasyon**
```bash
python lcd_calibrate.py
```
*Açılan pencerede kameranızdan gelen dijital ekranın üzerindeki rakamlar netleşene kadar trackbar'ları ayarlayın. Ekranı seçin ve `S` tuşu ile `lcd_config.json` dosyasına kaydedin.*

**Adım 2: Veri Toplama**
```bash
python collect_digits.py
```
*Ekrandaki sayılar değiştikçe klavyenizdeki numaralara (0-9) basarak örnekleri toplayın. Boş bir alan geldiğinde klavyeden `Boşluk (Space)` tuşuna basarak sistemin boşlukları öğrenmesini sağlayın. (Her sınıftan en az 30 örnek toplanması önerilir).*

**Adım 3: Model Eğitimi**
```bash
python train_digits.py
```
*Topladığınız verilerle arkaplanda PyTorch CNN modeliniz eğitilecek ve `digit_model.pth` adında size özel ağırlık dosyası oluşturulacaktır.*

**Adım 4: Gerçek Zamanlı Okuma ve Kayıt**
```bash
python lcd_reader.py
```
*Sistem çalışmaya başlar. `S` tuşu ile manuel kayıt alabilir veya `A` tuşu ile otomatik kaydı başlatabilirsiniz. Veriler `lcd_kayitlar.xlsx` dosyasına yazılacaktır.*

## 🧠 Model Mimari Özeti (CNN)
Sınıflandırıcı, düşük çözünürlüklü 32x32 gri tonlamalı görüntülerde segment özelliklerini yakalamak için özel olarak tasarlanmıştır. 
- 3 Adet Evrişim (Convolution) Katmanı (16, 32, 64 Filtre)
- ReLU Aktivasyonları ve Max Pooling
- Aşırı öğrenmeyi engellemek için %40 Dropout içeren Fully Connected Layer.
- Çıktı: 11 Sınıf (0'dan 9'a Rakamlar + "bos" Sınıfı).

## 👨‍💻 Geliştirici
Bu proje, hazır modellerin yetersiz kaldığı endüstriyel ortamlarda Görüntü İşleme ve Derin Öğrenme pratiklerini entegre ederek uçtan uca bir çözüm üretmek amacıyla geliştirilmiştir.
