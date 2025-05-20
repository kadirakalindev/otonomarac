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
6. **Video Simülasyonu**: Gerçek pist olmadan test için video tabanlı simülasyon

## Çalışma Modları

Bu proje iki farklı modda çalışabilir:

1. **Gerçek Pist Modu**: Gerçek bir pistte çalışma için tasarlanmıştır
2. **Video Modu**: Test ve geliştirme için pist videosu üzerinde çalışma

### Mod Geçişi Yönetimi

Kalibrasyon profilleri, her mod için ayrı kalibrasyon ayarları kullanabilmenizi sağlar:

```bash
# Kalibrasyon profillerini listeleme
python kalibrasyon_yonetici.py --list

# Gerçek pist moduna geçiş
python kalibrasyon_yonetici.py --use real

# Video moduna geçiş
python kalibrasyon_yonetici.py --use video

# Mevcut kalibrasyonu belirli bir profile kaydetme
python kalibrasyon_yonetici.py --save video
```

## Kurulum

### Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### GPIO Pin Yapılandırması

Varsayılan GPIO pin yapılandırması:

- Sol Motor İleri: GPIO 17
- Sol Motor Geri: GPIO 27
- Sağ Motor İleri: GPIO 22
- Sağ Motor Geri: GPIO 23
- Sol Motor PWM: GPIO 13
- Sağ Motor PWM: GPIO 12

Bu pinleri kendi donanım yapılandırmanıza göre değiştirebilirsiniz.

## Kullanım

### Video Pist Simülasyonu

Gerçek piste erişim olmadan test ve geliştirme için:

```bash
python pist_simulatoru.py --video test_pist.mp4 --resolution 640x480
```

Simülatör kontrolleri:

- **ESC/Q**: Çıkış
- **SPACE**: Duraklat/Devam
- **S**: Kalibrasyon kaydet
- **C**: Kalibrasyon modu aç/kapat
- **R**: Kareleri sıfırla
- **1,2,3,4**: Farklı görüntüleri aç/kapat
- **+/-**: Video hızını değiştir
- **WASD**: Canny parametrelerini ayarla
- **IJKL**: Hough parametrelerini ayarla

### Video Modu İle Çalıştırma

```bash
python main.py --mode video --video-path test_pist.mp4 --debug
```

### Gerçek Pist Modu İle Çalıştırma

```bash
python main.py --mode real --debug
```

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
python kalibrasyon_optimize.py --resolution 640x480 --output serit_kalibrasyon.json
```

### Video ve Gerçek Pist Arasında Kalibrasyon Geçişi

```bash
# Gerçek pist kalibrasyonunu video kalibrasyonuna kopyala
python kalibrasyon_yonetici.py --copy real video

# Video kalibrasyonunu gerçek pist kalibrasyonuna kopyala
python kalibrasyon_yonetici.py --copy video real
```

## Pist Videosu Hazırlama

Eğer gerçek piste erişiminiz yoksa, aşağıdaki adımları izleyerek test için video yakalayabilirsiniz:

1. **Test pisti oluşturun**: Siyah zemin üzerine beyaz bant ile şerit oluşturun
2. **Video kaydedin**: Telefonunuz veya kameranız ile üstten bakacak şekilde video çekin
3. **Videoyu aktarın**: Bilgisayarınıza aktarıp simülasyon ile kullanın

Örnek video formatları: MP4, AVI (720p veya daha düşük çözünürlük önerilir)

## PID Parametreleri

Şerit takibi için PID parametreleri `motor_control.py` dosyasında ayarlanabilir:

- `kp`: Orantısal katsayı (varsayılan: 0.5)
- `ki`: İntegral katsayısı (varsayılan: 0.001)
- `kd`: Türev katsayısı (varsayılan: 0.25)

## Yarışa Hazırlık Stratejisi

1. **Simülasyonda Test**: Video modu ile algoritmayı test edin
2. **Kalibrasyon Profili Oluşturun**: Video için optimum ayarları kaydedin
3. **Gerçek Piste Uyum**: Pist üzerinde çalıştırmadan önce kalibrasyonu güncelleyin
4. **Profillerinizi Koruyun**: Her iki mod için de kalibrasyon profillerini saklayın
5. **Hızlı Geçiş**: Yarış sırasında `kalibrasyon_yonetici.py` aracı ile hızlıca gerçek pist profiline geçiş yapın

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
- Pist simülatörünü kullanarak kalibrasyon parametrelerini ayarlayın.
- Canny ve Hough parametrelerini interaktif olarak optimize edin.
- Gerçek pistte test etmeden önce video üzerinde senaryoları test edin.

## Video-Gerçek Pist Geçiş İpuçları

- **Hızlı Kalibrasyon**: Gerçek piste ulaşır ulaşmaz kalibrasyon_optimize.py aracını çalıştırın
- **Referans Noktaları**: Gerçek pist üzerinde belirgin referans noktaları seçin
- **Parametre Taşıma**: Video modundaki başarılı parametreleri başlangıç noktası olarak kullanın
- **Test Sürüşü**: Tam hız denemesi öncesinde düşük hızda bir test sürüşü yapın
- **Felaket Modu**: Beklenmeyen sorunlar için manuel kontrol modunu hazır tutun

## Geliştirme

Projeyi geliştirmek için aşağıdaki dosyaları inceleyebilirsiniz:

- `motor_control.py`: Motor kontrolü ve PID algoritması
- `lane_detection.py`: Şerit tespiti algoritması
- `main.py`: Ana program döngüsü
- `pist_simulatoru.py`: Video tabanlı simülasyon aracı
- `kalibrasyon_yonetici.py`: Kalibrasyon profilleri yönetimi

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız veya önerileriniz için lütfen iletişime geçin.

---

**Not**: Bu proje eğitim amaçlıdır ve gerçek trafik ortamında kullanım için tasarlanmamıştır. 