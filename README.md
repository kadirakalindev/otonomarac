# Otonom Araç Projesi

Bu proje, MEB Robot Yarışmaları için geliştirilen bir otonom araç yazılımıdır. Şerit takibi, trafik işareti tanıma, sollama ve park etme gibi görevleri gerçekleştirir.

## Özellikler

- Şerit algılama ve takibi
- PID kontrol ile hassas motor kontrolü
- Modüler ve optimize edilmiş yapı
- Raspberry Pi 5 ve Raspberry Pi Camera 3 uyumluluğu

## Donanım Gereksinimleri

- Raspberry Pi 5 (8GB RAM)
- Raspberry Pi Camera 3
- L298N Motor Sürücü
- 2 adet DC Motor
- 5V 5A Voltaj Regülatörü
- Li-Po Pil

## Kurulum

### 1. Raspberry Pi OS Kurulumu

Raspberry Pi 5 üzerine son sürüm Raspberry Pi OS'u kurun:

```bash
# Sistemi güncelleyin
sudo apt update
sudo apt full-upgrade -y

# Gerekli paketleri yükleyin
sudo apt install -y python3-pip python3-dev git cmake build-essential
```

### 2. Gerekli Kütüphanelerin Kurulumu

```bash
# OpenCV ve temel kütüphaneler
sudo apt install -y python3-opencv
pip3 install opencv-contrib-python
pip3 install numpy matplotlib

# Picamera2 (Raspberry Pi Camera 3 için)
sudo apt install -y python3-picamera2

# GPIO kütüphaneleri
sudo apt install -y python3-gpiozero
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

# Diğer gereksinimler
pip3 install simple-pid
```

### 3. Projeyi İndirme

```bash
# Projeyi klonlayın
git clone https://github.com/kullanici/otonomarac.git
cd otonomarac
```

## Kullanım

### Temel Kullanım

```bash
# Temel çalıştırma
python3 main.py

# Hata ayıklama modunda çalıştırma
python3 main.py --debug

# Farklı çözünürlükte çalıştırma
python3 main.py --resolution 320x240 --fps 60
```

### Pin Konfigürasyonu

Varsayılan olarak kullanılan GPIO pinleri:
- Sol motor: GPIO 17 (ileri), GPIO 18 (geri)
- Sağ motor: GPIO 22 (ileri), GPIO 23 (geri)

Farklı pin numaraları kullanmak için:

```bash
python3 main.py --left-motor 5 6 --right-motor 13 19
```

L298N motor sürücüsü için PWM pinleri belirtmek için:

```bash
python3 main.py --left-pwm 12 --right-pwm 16
```

## Kalibrasyon

Şerit tespit algoritması, kameranın pozisyonuna ve parkurun ışık koşullarına göre kalibre edilmelidir. Kalibrasyon için `lane_detection.py` dosyasında şu parametreleri ayarlayabilirsiniz:

- `blur_kernel_size`: Gürültü azaltma için Gaussian blur kernel boyutu
- `canny_low_threshold` ve `canny_high_threshold`: Kenar tespiti için eşik değerleri
- `src_points` ve `dst_points`: Perspektif dönüşümü için kaynak ve hedef noktalar

## Motor Kontrolü

Motor davranışını ayarlamak için `motor_control.py` dosyasında PID parametrelerini düzenleyebilirsiniz:

- `kp`: Orantısal katsayı (tepki hızı)
- `ki`: İntegral katsayı (birikmiş hata)
- `kd`: Türev katsayı (aşırı tepkiyi önler)

## Sorun Giderme

### Kamera Görüntüsü Alınamıyor

```bash
# Kameranın doğru bağlandığını kontrol edin
vcgencmd get_camera

# Picamera2'nin yüklü olduğunu doğrulayın
pip3 list | grep picamera2
```

### Motor Kontrolü Çalışmıyor

```bash
# GPIO servisinin çalıştığını kontrol edin
systemctl status pigpiod

# Pin numaralarını kontrol edin
gpio readall
```

## Katkıda Bulunma

Projede iyileştirme yapmak veya hata raporlamak için lütfen bir issue açın veya pull request gönderin.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

## Teşekkürler

Bu proje, OpenCV ve gpiozero kütüphanelerinin sunduğu imkanlar sayesinde geliştirilmiştir. Tüm bu açık kaynak projelerin katkıda bulunanlarına teşekkür ederiz. 