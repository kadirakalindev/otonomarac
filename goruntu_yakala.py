#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Görüntü Yakalama Aracı
Bu program, kalibrasyon için kullanılacak test görüntülerini yakalar.
Raspberry Pi 5 için picamera2 kütüphanesini kullanır.
"""

import time
import argparse
import os
import logging
import cv2
from picamera2 import Picamera2

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GoruntuYakala")

def capture_image(output_file="test_goruntu.jpg", resolution=(320, 240), delay=2):
    """
    Kameradan görüntü yakalar ve kaydeder
    
    Args:
        output_file (str): Kaydedilecek dosya adı
        resolution (tuple): Çözünürlük (genişlik, yükseklik)
        delay (int): Yakalamadan önce beklenecek süre (saniye)
    """
    logger.info("Kamera başlatılıyor...")
    
    try:
        # Picamera2 ile kamera başlat
        camera = Picamera2()
        
        # Kamera yapılandırması
        camera_config = camera.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        camera.configure(camera_config)
        camera.start()
        
        # Kameranın hazırlanması için bekle
        logger.info(f"{delay} saniye bekleniyor...")
        time.sleep(delay)
        
        # Görüntü yakala
        logger.info("Görüntü yakalanıyor...")
        frame = camera.capture_array()
        
        # RGB'den BGR'a dönüştür (OpenCV formatı)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Görüntüyü kaydet
        cv2.imwrite(output_file, frame_bgr)
        logger.info(f"Görüntü kaydedildi: {output_file}")
        
        # Önizleme olarak göster (eğer görüntü ekranı varsa)
        try:
            cv2.imshow("Yakalanan Görüntü", frame_bgr)
            logger.info("Kapatmak için herhangi bir tuşa basın...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Görüntü gösterme hatası (önemli değil): {e}")
        
        # Kamerayı kapat
        camera.stop()
        logger.info("Kamera kapatıldı.")
        
        return True
    
    except Exception as e:
        logger.error(f"Görüntü yakalama hatası: {e}")
        return False

def capture_multiple_images(output_dir="kalibrasyon_goruntuleri", count=5, interval=2, resolution=(320, 240)):
    """
    Belirli aralıklarla birden fazla görüntü yakalar
    
    Args:
        output_dir (str): Görüntülerin kaydedileceği klasör
        count (int): Yakalanacak görüntü sayısı
        interval (int): Görüntüler arası bekleme süresi (saniye)
        resolution (tuple): Çözünürlük (genişlik, yükseklik)
    """
    # Klasörü oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Klasör oluşturuldu: {output_dir}")
    
    logger.info(f"{count} adet görüntü yakalanacak, {interval} saniye aralıkla.")
    
    try:
        # Picamera2 ile kamera başlat
        camera = Picamera2()
        
        # Kamera yapılandırması
        camera_config = camera.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        camera.configure(camera_config)
        camera.start()
        
        # Kameranın hazırlanması için bekle
        logger.info(f"2 saniye bekleniyor...")
        time.sleep(2)
        
        # Birden fazla görüntü yakala
        for i in range(count):
            # Dosya adı
            output_file = os.path.join(output_dir, f"goruntu_{i+1}.jpg")
            
            # Görüntü yakala
            logger.info(f"Görüntü {i+1}/{count} yakalanıyor...")
            frame = camera.capture_array()
            
            # RGB'den BGR'a dönüştür (OpenCV formatı)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Görüntüyü kaydet
            cv2.imwrite(output_file, frame_bgr)
            logger.info(f"Görüntü kaydedildi: {output_file}")
            
            # Aralıklarla yakala
            if i < count - 1:
                logger.info(f"{interval} saniye bekleniyor...")
                time.sleep(interval)
        
        # Kamerayı kapat
        camera.stop()
        logger.info("Kamera kapatıldı.")
        
        return True
    
    except Exception as e:
        logger.error(f"Çoklu görüntü yakalama hatası: {e}")
        return False

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Görüntü Yakalama Aracı')
    parser.add_argument('--output', default='test_goruntu.jpg', help='Çıktı dosyası adı')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--delay', type=int, default=2, help='Yakalamadan önce beklenecek süre (saniye)')
    
    # Çoklu görüntü yakalama
    parser.add_argument('--multi', action='store_true', help='Birden fazla görüntü yakala')
    parser.add_argument('--count', type=int, default=5, help='Yakalanacak görüntü sayısı')
    parser.add_argument('--interval', type=int, default=2, help='Görüntüler arası bekleme süresi (saniye)')
    parser.add_argument('--output-dir', default='kalibrasyon_goruntuleri', help='Görüntülerin kaydedileceği klasör')
    
    return parser.parse_args()

def main():
    """
    Ana program
    """
    # Argümanları işle
    args = parse_arguments()
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        resolution = (320, 240)
    
    # Çoklu görüntü yakalama modu
    if args.multi:
        capture_multiple_images(
            output_dir=args.output_dir,
            count=args.count,
            interval=args.interval,
            resolution=resolution
        )
    else:
        # Tek görüntü yakala
        capture_image(
            output_file=args.output,
            resolution=resolution,
            delay=args.delay
        )
    
    print("\nGörüntü Yakalama Tamamlandı!")
    print("Bu görüntüleri kalibrasyon için kullanabilirsiniz.")
    print("Kalibrasyon parametrelerini oluşturmak için:")
    print(f"   python3 kalibrasyon_olustur.py --resolution {width}x{height}")

if __name__ == "__main__":
    main() 