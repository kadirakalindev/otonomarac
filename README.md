# Otonom Araç Projesi

Raspberry Pi 5 ve kamera kullanarak oluşturulmuş şerit takibi yapabilen otonom araç projesi.

## Özellikler

- Şerit tespiti ve takibi
- Yaya geçidi ve hemzemin geçit tespiti
- Motorların PID kontrolü ile hassas şerit takibi
- Şerit kaybı durumunda kurtarma stratejisi
- Hata toleransı ve güvenli çalışma

## Donanım Gereksinimleri

- Raspberry Pi 5 (8GB RAM önerilen)
- Raspberry Pi Camera 3
- L298N Motor Sürücü
- 2 adet DC Motor
- Li-Po Pil ve 5V 5A Voltaj Regülatörü

### Motor Pin Bağlantıları (BOARD numaralandırması)

- **Sol Motor:**
  - IN1: BOARD 16 (BCM 23)
  - IN2: BOARD 18 (BCM 24)
  - PWM Enable: BOARD 12 (BCM 18)

- **Sağ Motor:**
  - IN1: BOARD 36 (BCM 16)
  - IN2: BOARD 38 (BCM 20)
  - PWM Enable: BOARD 32 (BCM 12)

## Yazılım Gereksinimleri

Gerekli Python paketlerinin yüklenmesi için:

```bash
pip install -r requirements.txt
```

## Kurulum

1. Python bağımlılıklarını yükleyin:

```bash
pip install -r requirements.txt
```

2. Raspberry Pi'de kamera arayüzünü etkinleştirin:

```bash
sudo raspi-config
```

3. GPIO pin bağlantılarını yukarıda belirtilen şekilde yapın.

## Kullanım

### Ana Programı Çalıştırma:

```bash
python main.py --resolution 320x240 --debug
```

Parametreler:
- `--resolution`: Kamera çözünürlüğü (WIDTHxHEIGHT formatında)
- `--debug`: Debug modunu etkinleştirir (görüntü penceresi gösterilir)

### Motor Testi:

```bash
python motor_control.py
```

### Şerit Tespiti Testi:

```bash
python lane_detection.py --image test_image.jpg
```

## Dosyalar

- `main.py`: Ana program
- `lane_detection.py`: Şerit tespiti algoritmaları
- `motor_control.py`: Motor kontrol fonksiyonları
- `requirements.txt`: Python bağımlılıkları

## Çalışma Prensibi

1. Kameradan alınan görüntü işlenerek şeritler tespit edilir.
2. Şerit konumlarına göre aracın merkez ofseti hesaplanır.
3. PID kontrol algoritması ile motor hızları ayarlanır.
4. Yaya geçidi tespit edilirse araç yavaşlar.
5. Şerit kaybı durumunda kurtarma stratejisi uygulanır.

## Sorun Giderme

- Şerit tespit edilemiyorsa kamera pozlamasını ve kontrast ayarlarını kontrol edin.
- Motorlar düzgün çalışmıyorsa pin bağlantılarını ve voltaj seviyesini kontrol edin.
- Debug modu ile görüntü işleme sonuçlarını görsel olarak inceleyin.

## Geliştirme

Projeyi daha da geliştirmek için:

- `lane_detection.py` içindeki görüntü işleme parametrelerini ayarlayın.
- `motor_control.py` içindeki PID parametrelerini optimize edin.
- Ana döngü hızını ve hata tolerans değerlerini çalışma ortamına göre ayarlayın. 