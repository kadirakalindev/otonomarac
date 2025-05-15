# Otonom Araç Kalibrasyon Kılavuzu

Bu kılavuz, otonom araç projesindeki şerit tespit sistemini kalibre etmek için oluşturulan araçları ve adımları içermektedir.

## Kalibrasyon Nedir ve Neden Önemlidir?

Kalibrasyon işlemi, şerit tespit algoritmasının doğru çalışması için gerekli olan parametrelerin (perspektif dönüşümü, renk filtreleri, kenar algılama eşikleri vb.) ayarlanmasını içerir. İyi bir kalibrasyon:

1. Şeritlerin doğru algılanmasını sağlar
2. Farklı ışık koşullarında daha tutarlı sonuçlar üretir
3. İşleme hızını optimize eder
4. Aracın yönlendirme doğruluğunu artırır

## Kalibrasyon Araçları

Projede üç farklı kalibrasyon aracı bulunmaktadır:

### 1. Görsel Kalibrasyon Aracı (`kalibrasyon_optimize.py`)

Bu araç, kameradan alınan görüntü üzerinde tıklama yaparak perspektif dönüşüm noktalarını seçmenizi sağlar. Hafif ve optimize edilmiş versiyondur, donma sorunlarını çözmek için geliştirilmiştir.

**Kullanım:**
```
python3 kalibrasyon_optimize.py [--camera 0] [--resolution 320x240] [--output calibration.json]
```

**Adımlar:**
1. Program kameradan gelen görüntüyü gösterir
2. Şerit tespiti için 4 nokta seçmeniz gerekir:
   - Sol üst (şeridin sol üst köşesi)
   - Sağ üst (şeridin sağ üst köşesi) 
   - Sol alt (genellikle ekranın sol alt köşesi)
   - Sağ alt (genellikle ekranın sağ alt köşesi)
3. Noktaları seçtikten sonra 'S' tuşuna basarak kaydedebilir veya 'R' tuşuna basarak yeniden başlatabilirsiniz
4. ESC tuşu ile programdan çıkabilirsiniz

### 2. Komut Satırı Kalibrasyon Aracı (`kalibrasyon_olustur_optimize.py`) 

Bu araç, görsel arayüz kullanmadan komut satırı üzerinden kalibrasyon dosyası oluşturmanızı sağlar. İki şekilde kullanabilirsiniz:

**İnteraktif Mod:**
```
python3 kalibrasyon_olustur_optimize.py --interactive [--resolution 320x240] [--output calibration.json]
```
Bu mod size adım adım kalibrasyon noktalarını girmeniz için rehberlik eder.

**Hızlı Mod:**
```
python3 kalibrasyon_olustur_optimize.py --quick-mode [--resolution 320x240] [--output calibration.json]
```
Bu mod varsayılan değerlerle hızlıca kalibrasyon dosyası oluşturur.

**Manuel Nokta Girişi:**
```
python3 kalibrasyon_olustur_optimize.py --src-points "112,156" "208,156" "0,240" "320,240" [--dst-points "80,0" "240,0" "80,240" "240,240"] [--output calibration.json]
```
Bu kullanımda noktaları doğrudan komut satırından belirtebilirsiniz.

### 3. Orijinal Kalibrasyon Aracı (`kalibrasyon.py`)

Bu, projenin orijinal kalibrasyon aracıdır. Bazı durumlarda donma sorunları yaşanabilir, bu nedenle optimize edilmiş versiyonları kullanmanızı öneririz.

## Kalibrasyon İş Akışı

### Adım 1: İlk Kalibrasyon Dosyasını Oluşturun

Başlangıçta, varsayılan değerlerle bir kalibrasyon dosyası oluşturun:

```
python3 kalibrasyon_olustur_optimize.py --quick-mode
```

### Adım 2: Kalibrasyon Test Edin

Oluşturulan kalibrasyon dosyasını test edin:

```
python3 kamera_test.py --debug
```

veya bir video dosyası üzerinde:

```
python3 video_test.py --video <video_dosyası> --debug
```

### Adım 3: Kalibrasyon İyileştirin

Sonuçlar tatmin edici değilse, görsel kalibrasyon aracını kullanarak daha iyi sonuçlar elde etmeyi deneyin:

```
python3 kalibrasyon_optimize.py
```

### Adım 4: İnce Ayar Yapın

Gerekirse kalibrasyon parametrelerini interaktif modda ince ayarlayın:

```
python3 kalibrasyon_olustur_optimize.py --interactive
```

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

Kalibrasyon dosyasındaki (`calibration.json`) temel parametreler şunlardır:

```json
{
  "src_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // Kaynak perspektif noktaları
  "dst_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // Hedef perspektif noktaları
  "resolution": {"width": 320, "height": 240},          // Çözünürlük
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

1. **Kalibrasyon Aracı Donuyor**: `kalibrasyon_optimize.py` veya `kalibrasyon_olustur_optimize.py` araçlarını kullanın

2. **Şeritler Algılanmıyor**:
   - Perspektif noktalarını yeniden ayarlayın
   - Renk filtreleri değerlerini ortam ışık koşullarına göre ayarlayın
   - Canny eşik değerlerini değiştirin

3. **Şerit Tespiti Kararsız**:
   - `max_line_gap` değerini artırın
   - `min_line_length` değerini azaltın
   - Daha düşük çözünürlük kullanın (320x240 önerilir)

4. **İşlem Hızı Yavaş**:
   - Daha düşük çözünürlük kullanın
   - `blur_kernel_size` değerini azaltın
   - Debug modunu kapatın

## İletişim ve Destek

Kalibrasyon araçları veya şerit tespit sistemi ile ilgili sorularınız için proje yöneticisiyle iletişime geçin. 