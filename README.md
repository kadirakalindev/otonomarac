# Otonom Araç Projesi

Bu proje, şerit takibi yapabilen bir otonom araç prototipi geliştirmek için tasarlanmıştır. Raspberry Pi tabanlı bir robot aracın kamera kullanarak şeritleri tespit etmesi ve bu şeritleri takip etmesi amaçlanmaktadır.

## Proje Bileşenleri

### Donanım Gereksinimleri

- Raspberry Pi (3B+ veya 4B önerilir)
- Pi Kamera veya USB Kamera
- Motor Sürücü Kartı (L298N veya benzeri)
- 2 adet DC Motor
- Güç Kaynağı (Powerbank veya Pil Paketi)
- Şasi ve Tekerlekler
- Jumper Kablolar

### Yazılım Bileşenleri

1. **Motor Kontrolü**: DC motorların hız ve yön kontrolü
2. **Şerit Tespiti**: Kamera görüntüsünden şeritleri tespit etme
3. **PID Kontrolü**: Şerit takibi için hassas kontrol algoritması
4. **Kalibrasyon Araçları**: Kamera ve şerit tespiti için kalibrasyon
5. **Test Araçları**: Bileşenlerin test edilmesi için yardımcı programlar

## Kurulum

### Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### GPIO Pin Yapılandırması

Varsayılan GPIO pin yapılandırması:

self.in1_pin = 36  # Sağ motor ileri
        self.in2_pin = 38  # Sağ motor geri
        self.in3_pin = 16  # Sol motor ileri
        self.in4_pin = 28  # Sol motor geri
        self.en_a_pin = 32  # Sağ motor hız kontrolü
        self.en_b_pin = 13  # Sol motor hız kontrolü

Bu pinleri kendi donanım yapılandırmanıza göre değiştirebilirsiniz.

## Kullanım

### Kalibrasyon Başlatma Aracı

Kalibrasyon ve test araçlarını kullanmak için başlatma aracını çalıştırın:

```bash
python kalibrasyon_basla.py
```

Bu araç size aşağıdaki seçenekleri sunar:

1. **İnteraktif Şerit Kalibrasyonu**: Kamera görüntüsü üzerinde 5 nokta seçerek şerit takibi için kalibrasyon yapın.
2. **Kamera Testi**: Kalibrasyonun doğru çalışıp çalışmadığını kontrol edin.
3. **Motor Testi**: Motorların doğru çalışıp çalışmadığını test edin.
4. **Tam Sistem Testi**: Şerit takibi ve motor kontrolünü birlikte test edin.
5. **Yardım ve Bilgi**: Kullanım talimatları ve ipuçları.

### İnteraktif Şerit Kalibrasyonu

Şerit takibi için kamera kalibrasyonu yapmak için:

```bash
python interaktif_kalibrasyon.py --resolution 640x480 --output serit_kalibrasyon.json
```

Kalibrasyon sırasında şeritler üzerinde 5 nokta seçmeniz gerekiyor:
1. Sol şeridin alt noktası
2. Sol şeridin üst noktası
3. Orta şeridin üst noktası (takip edilecek merkez)
4. Sağ şeridin üst noktası
5. Sağ şeridin alt noktası

### Kamera Testi

Kamera ve şerit tespitini test etmek için:

```bash
python kamera_test.py --resolution 640x480 --calibration serit_kalibrasyon.json --debug
```

### Motor Testi

Motorların doğru çalışıp çalışmadığını test etmek için:

```bash
python motor_test.py --interactive
```

veya otomatik test için:

```bash
python motor_test.py --speed 50 --duration 2.0
```

### Tam Sistem Testi

Şerit takibi ve motor kontrolünü birlikte test etmek için:

```bash
python main.py --debug --calibration serit_kalibrasyon.json
```

## PID Parametreleri

Şerit takibi için PID parametreleri `motor_control.py` dosyasında ayarlanabilir:

- `kp`: Orantısal katsayı (varsayılan: 0.3)
- `ki`: İntegral katsayısı (varsayılan: 0.0005)
- `kd`: Türev katsayısı (varsayılan: 0.15)

## Sorun Giderme

### Kamera Sorunları

- Kamera başlatılamazsa, bağlantıları kontrol edin.
- Raspberry Pi'da kamera arabiriminin etkinleştirildiğinden emin olun: `sudo raspi-config`
- USB kamera kullanıyorsanız, doğru kamera ID'sini belirtin.

### Motor Sorunları

- Motorlar çalışmıyorsa, pin bağlantılarını kontrol edin.
- Motor sürücü kartının güç kaynağını kontrol edin.
- `motor_test.py` ile motorları test edin.

### Şerit Tespit Sorunları

- Işık koşullarının yeterli olduğundan emin olun.
- Kamera kalibrasyonunu yeniden yapın.
- Canny ve Hough parametrelerini ayarlayın.

## Geliştirme

Projeyi geliştirmek için aşağıdaki dosyaları inceleyebilirsiniz:

- `motor_control.py`: Motor kontrolü ve PID algoritması
- `lane_detection.py`: Şerit tespiti algoritması
- `main.py`: Ana program döngüsü

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız veya önerileriniz için lütfen iletişime geçin.

---

**Not**: Bu proje eğitim amaçlıdır ve gerçek trafik ortamında kullanım için tasarlanmamıştır. 