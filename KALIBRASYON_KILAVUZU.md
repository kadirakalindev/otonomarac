# Otonom Araç Kalibrasyon Kılavuzu

Bu kılavuz, otonom araç projesindeki şerit tespit sistemini kalibre etmek için kullanılan kalibrasyon aracını ve adımları açıklamaktadır.

## Kalibrasyon Nedir ve Neden Önemlidir?

Kalibrasyon işlemi, şerit tespit algoritmasının doğru çalışması için gerekli olan parametrelerin (perspektif dönüşümü, renk filtreleri, kenar algılama eşikleri vb.) ayarlanmasını içerir. İyi bir kalibrasyon:

1. Şeritlerin doğru algılanmasını sağlar
2. Farklı ışık koşullarında daha tutarlı sonuçlar üretir
3. İşleme hızını optimize eder
4. Aracın yönlendirme doğruluğunu artırır

## Görsel Kalibrasyon Aracı (`kalibrasyon_optimize.py`)

Projede kullanılan kalibrasyon aracı, kameradan alınan görüntü üzerinde tıklama yaparak perspektif dönüşüm noktalarını seçmenizi sağlar. Hafif ve optimize edilmiş yapısıyla, sorunsuz bir kalibrasyon deneyimi sunar.

**Kullanım:**
```
python kalibrasyon_optimize.py [--camera 0] [--resolution 640x480] [--output serit_kalibrasyon.json]
```

**Parametreler:**
- `--camera`: Kullanılacak kamera numarası (varsayılan: 0)
- `--resolution`: Kamera çözünürlüğü (varsayılan: 640x480)
- `--output`: Kalibrasyon verilerinin kaydedileceği dosya (varsayılan: serit_kalibrasyon.json)

**Adımlar:**
1. Program kameradan gelen görüntüyü gösterir
2. Şerit tespiti için 4 nokta seçmeniz gerekir:
   - Sol üst (şeridin sol üst köşesi)
   - Sağ üst (şeridin sağ üst köşesi) 
   - Sol alt (genellikle ekranın sol alt köşesi)
   - Sağ alt (genellikle ekranın sağ alt köşesi)
3. Noktaları seçtikten sonra 'S' tuşuna basarak kaydedebilir veya 'R' tuşuna basarak yeniden başlatabilirsiniz
4. ESC tuşu ile programdan çıkabilirsiniz

## Kalibrasyon İş Akışı

### Adım 1: Kalibrasyon Başlatma Aracını Çalıştırın

Kalibrasyon başlatma aracını çalıştırarak kalibrasyon işlemini başlatın:

```
python kalibrasyon_basla.py
```

### Adım 2: Şerit Kalibrasyonu Seçeneğini Seçin

Ana menüden "1. Şerit Kalibrasyonu (kalibrasyon_optimize.py)" seçeneğini seçin ve istediğiniz çözünürlüğü belirleyin.

### Adım 3: Kalibrasyon Noktalarını Belirleyin

Açılan pencerede şerit tespiti için 4 noktayı şu sırayla belirleyin:
- P0: Sol üst nokta (şeridin sol üst köşesi)
- P1: Sağ üst nokta (şeridin sağ üst köşesi)
- P2: Sol alt nokta (genellikle ekranın sol alt köşesi)
- P3: Sağ alt nokta (genellikle ekranın sağ alt köşesi)

### Adım 4: Kalibrasyon Test Edin

Kalibrasyon işlemi tamamlandıktan sonra, ana menüden "2. Kamera Testi" seçeneğini seçerek kalibrasyonu test edin.

## Kalibrasyon İpuçları

1. **İyi Işık Koşulları**: Kalibrasyon yaparken iyi aydınlatılmış bir ortam kullanın

2. **Belirgin Şeritler**: Test ortamında belirgin şeritleri olan bir test parkuru tercih edin

3. **Stabil Kamera Pozisyonu**: Kalibrasyon sırasında ve sonrasında kameranın pozisyonunu değiştirmeyin

4. **Doğru Nokta Seçimi**:
   - Sol ve sağ üst noktaları, şeritlerin orta uzaklıktaki noktalarına yerleştirin
   - Alt noktaları genellikle ekranın alt kenarlarına yerleştirin
   - Kuş bakışı görünümde paralel çizgiler elde etmeye çalışın

5. **İterasyonlu Test**: Kalibrasyon yaptıktan sonra test edin, gerekirse tekrar kalibre edin

## Kalibrasyon Parametreleri

Kalibrasyon dosyasındaki (`serit_kalibrasyon.json`) temel parametreler şunlardır:

```json
{
  "src_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // Kaynak perspektif noktaları
  "dst_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // Hedef perspektif noktaları
  "resolution": {"width": 640, "height": 480},          // Çözünürlük
  "canny_low_threshold": 50,                           // Canny kenar algılama alt eşiği
  "canny_high_threshold": 150,                         // Canny kenar algılama üst eşiği
  "blur_kernel_size": 5,                               // Bulanıklaştırma çekirdek boyutu
  "hough_threshold": 15,                               // Hough çizgi tespiti eşiği
  "min_line_length": 15,                               // Minimum çizgi uzunluğu
  "max_line_gap": 100,                                 // Maksimum çizgi aralığı
  "white_lower": [0, 0, 210],                          // Beyaz renk filtresi alt değer (HSV)
  "white_upper": [180, 30, 255],                       // Beyaz renk filtresi üst değer (HSV)
  "yellow_lower": [15, 80, 120],                       // Sarı renk filtresi alt değer (HSV)
  "yellow_upper": [35, 255, 255]                       // Sarı renk filtresi üst değer (HSV)
}
```

## Sorun Giderme

1. **Şeritler Algılanmıyor**:
   - Perspektif noktalarını yeniden ayarlayın
   - Renk filtreleri değerlerini ortam ışık koşullarına göre ayarlayın
   - Canny eşik değerlerini değiştirin

2. **Şerit Tespiti Kararsız**:
   - `max_line_gap` değerini artırın
   - `min_line_length` değerini azaltın
   - Daha düşük çözünürlük kullanın (320x240 önerilir)

3. **İşlem Hızı Yavaş**:
   - Daha düşük çözünürlük kullanın
   - `blur_kernel_size` değerini azaltın
   - Debug modunu kapatın

## İletişim ve Destek

Kalibrasyon araçları veya şerit tespit sistemi ile ilgili sorularınız için proje yöneticisiyle iletişime geçin. 