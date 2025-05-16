# KALIBRASYON KILAVUZU

Bu kılavuz, otonom araç projesindeki kalibrasyon ve test araçlarının kullanımını açıklamaktadır. Projedeki performans iyileştirmeleri ve yeni özellikler dahil edilmiştir.

## İçindekiler

1. [Araçlar ve Özellikleri](#araçlar-ve-özellikleri)
2. [Kalibrasyon İşlemi](#kalibrasyon-işlemi)
3. [Kamera Testi](#kamera-testi)
4. [Performans Modları](#performans-modları)
5. [Hata Giderme](#hata-giderme)
6. [İpuçları ve Öneriler](#i̇puçları-ve-öneriler)

## Araçlar ve Özellikleri

Projede yer alan temel araçlar:

### 1. `kalibrasyon_optimize.py`

Görsel kalibrasyon aracı. Kameradan gelen görüntü üzerinde interaktif olarak kalibrasyon yapmanızı sağlar.

**Özellikler:**
- İyileştirilmiş kamera kontrolü (donma sorunu çözüldü)
- Görsel kalibrasyon arayüzü
- Kalibrasyon verilerini JSON formatında kaydetme
- Farklı çözünürlük seçenekleri
- Performans modları desteği

**Kullanım:**
```bash
python3 kalibrasyon_optimize.py --camera 0 --resolution 320x240 --output calibration.json --performance balanced
```

### 2. `kalibrasyon_olustur_optimize.py`

Komut satırı tabanlı kalibrasyon aracı. İnteraktif mod ile kalibrasyon parametrelerini ayarlayabilirsiniz.

**Özellikler:**
- İnteraktif mod (trackbar ile parametreleri ayarlama)
- Hızlı mod (otomatik kalibrasyon)
- Bellek optimizasyonları
- Adaptif renk filtreleme desteği

**Kullanım:**
```bash
python3 kalibrasyon_olustur_optimize.py --interactive --performance balanced
```

### 3. `kamera_test.py`

Kamera ve şerit tespit algoritmalarını test etmenizi sağlar.

**Özellikler:**
- Gerçek zamanlı şerit tespiti ve gösterimi
- Direksiyon açısı göstergesi
- Performans metrikleri (FPS, işleme süresi)
- Farklı performans modları
- Histogram görüntüleme
- Bellek optimizasyonları ve kare atlama desteği

**Kullanım:**
```bash
python3 kamera_test.py --camera 0 --resolution 320x240 --histogram --performance balanced
```

### 4. `video_test.py`

Video dosyaları üzerinde şerit tespit algoritmalarını test etmenizi sağlar.

**Özellikler:**
- Video oynatma kontrolü (duraklat/devam et, hızlandır/yavaşlat)
- Performans modları desteği
- Bellek optimizasyonları
- Görüntü kaydetme

**Kullanım:**
```bash
python3 video_test.py --video test_video.mp4 --resolution 320x240 --performance balanced
```

### 5. `goruntu_yakala.py`

Test görüntülerini yakalamak için basit bir araç.

**Kullanım:**
```bash
python3 goruntu_yakala.py --output test_goruntu.jpg
```

## Kalibrasyon İşlemi

Kalibrasyon, şerit tespitinin doğru çalışması için son derece önemlidir. Aşağıdaki adımları izleyin:

### Hızlı Başlangıç

1. Kalibrasyon aracını çalıştırın:
   ```bash
   python3 kalibrasyon_optimize.py --camera 0 --resolution 320x240
   ```

2. Aşağıdaki tuşları kullanarak görsel kalibrasyon yapın:
   - `R`: Renk eşiklerini sıfırla
   - `S`: Mevcut kalibrasyonu kaydet
   - `ESC`: Çıkış
   - Mouse ile perspektif noktalarını ayarlayın

### Detaylı Kalibrasyon

Daha detaylı bir kalibrasyon için interaktif mod kullanın:

```bash
python3 kalibrasyon_olustur_optimize.py --interactive
```

Bu mod, trackbar'lar aracılığıyla tüm parametreleri hassas bir şekilde ayarlamanızı sağlar.

## Kamera Testi

Kalibrasyon sonrasında şerit tespitini test etmek için:

```bash
python3 kamera_test.py --camera 0 --histogram
```

**Tuş Kontrolleri:**
- `q`: Çıkış
- `space`: Duraklat/Devam Et
- `s`: Görüntüyü Kaydet
- `p`: Performans Modunu Değiştir

## Performans Modları

Tüm araçlar üç farklı performans modunu destekler:

### 1. `speed` (Hız)

Bu mod, düşük işlem gücüne sahip cihazlarda daha hızlı çalışmak için optimize edilmiştir.

**Özellikler:**
- Düşük çözünürlük işleme
- Kare atlama özelliği
- Basitleştirilmiş algoritma

**Kullanım:**
```bash
python3 kamera_test.py --performance speed
```

### 2. `balanced` (Dengeli)

Varsayılan mod. Performans ve doğruluk arasında denge sağlar.

**Özellikler:**
- Orta seviye çözünürlük
- Adaptif renk filtreleme
- Optimum bellek kullanımı

**Kullanım:**
```bash
python3 kamera_test.py --performance balanced
```

### 3. `quality` (Kalite)

En yüksek doğruluk için optimize edilmiş mod. Daha güçlü donanım gerektirir.

**Özellikler:**
- Tam çözünürlük işleme
- Gelişmiş kenar algılama
- Tam şerit izleme modeli

**Kullanım:**
```bash
python3 kamera_test.py --performance quality
```

## Hata Giderme

### Yaygın Sorunlar ve Çözümleri

1. **Motor sürücüsü aşırı ısınması:** 
   - `main.py` dosyasında `max_speed` ve `pwm_frequency` değerlerini düşürün
   - PID parametrelerini daha yumuşak kontrol için ayarlayın

2. **Şerit tespiti sorunları:**
   - Adaptif renk filtreleme için `use_adaptive_color` parametresini etkinleştirin
   - Işık koşullarına göre kalibrasyon yapın

3. **Performans sorunları:**
   - Düşük işlem gücü olan cihazlarda `--performance speed` modunu kullanın
   - İşlem çözünürlüğünü düşürün: `--resolution 160x120`

4. **Kamera bağlantı sorunları:**
   - Farklı bir kamera ID'si deneyin: `--camera 1`
   - Kamera çözünürlüğünün desteklendiğinden emin olun

## İpuçları ve Öneriler

1. **Şerit algılama performansını artırmak için:**
   - Yeterli ışık koşullarında kalibrasyon yapın
   - Şerit çizgilerini içeren test görüntüleri toplayın
   - Adaptif renk filtreleme özelliğini kullanın

2. **Motor kontrolü için:**
   - Ramping mekanizmasını kullanarak ani hız değişimlerini önleyin
   - Düşük pwm_frequency (20-30Hz) ile motor sürücüsünün ısınmasını azaltın
   - Otomatik PID ayarlama özelliğini etkinleştirin

3. **Çalışma zamanı optimizasyonları:**
   - Düşük işlem gücüne sahip cihazlarda 'speed' performans modunu kullanın
   - Periyodik bellek temizliği için gc.collect() çağrısı projeye eklenmiştir
   - Önbellek tamponları ile bellek kullanımı optimize edilmiştir

4. **Kalibrasyon işlemi için:**
   - Kalibrasyon için araç üzerindeki kameranın gerçek konumunu kullanın
   - Perspektif dönüşümü için yol üzerindeki çizgileri referans alın
   - Kalibrasyon sonrası mutlaka kamera_test.py ile test edin

## Örnek Çalıştırma Komutları

### Kalibrasyon ve Test Akışı:

```bash
# 1. Görsel kalibrasyon
python3 kalibrasyon_optimize.py --camera 0 --resolution 320x240 --output calibration.json

# 2. Kalibrasyon parametrelerini ayarlama
python3 kalibrasyon_olustur_optimize.py --interactive

# 3. Kamera testi (histogram ve dengeli performans ile)
python3 kamera_test.py --camera 0 --histogram --performance balanced

# 4. Ana programı çalıştırma (optimize edilmiş ayarlarla)
python3 main.py --calibration calibration.json --performance balanced
```

Bu kılavuzdaki komutlar ve parametreler, projenin mevcut durumuna göre güncellenmiştir. Yeni eklenen performans iyileştirmeleri ve özellikler dahil edilmiştir. 