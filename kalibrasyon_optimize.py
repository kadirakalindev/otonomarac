#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Optimize Edilmiş Kalibrasyon Aracı
Bu program, kamera kalibrasyonu ve şerit tespiti parametrelerinin ayarlanması için
daha hafif ve donmayı engelleyen bir araç sağlar.
"""

import cv2
import numpy as np
import argparse
import json
import os
import logging
import time
import signal
import sys
from picamera2 import Picamera2

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KalibrasyonOptimize")

class KameraPerspektifAyarlayici:
    """
    Kamera perspektif dönüşümü için optimize edilmiş kalibrasyon aracı
    """
    def __init__(self, camera_resolution=(320, 240), framerate=15, camera_id=0, output_file="serit_kalibrasyon.json"):
        """
        KameraPerspektifAyarlayici sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            framerate (int): Kare hızı (fps)
            camera_id (int): Kamera ID numarası
            output_file (str): Kalibrasyon dosyası çıktı yolu
        """
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        self.framerate = framerate
        self.camera_id = camera_id
        self.output_file = output_file
        self.running = True
        
        # Varsayılan perspektif noktaları
        self.src_points = np.float32([
            [self.width * 0.35, self.height * 0.6],  # Sol üst
            [self.width * 0.65, self.height * 0.6],  # Sağ üst
            [0, self.height],                        # Sol alt
            [self.width, self.height]                # Sağ alt
        ])
        
        self.dst_points = np.float32([
            [self.width * 0.25, 0],               # Sol üst
            [self.width * 0.75, 0],               # Sağ üst
            [self.width * 0.25, self.height],     # Sol alt
            [self.width * 0.75, self.height]      # Sağ alt
        ])
        
        # Kamera nesnesi
        self.camera = None
        
        # İşaretçi indeksi (fare ile taşınacak nokta)
        self.selected_point = None
        
        # Pencere adı
        self.window_name = "Perspektif Kalibrasyon"
        
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # En son çekilen kare
        self.current_frame = None
        
        # Son işlem zamanı (FPS kontrol)
        self.last_process_time = time.time()
        
        logger.info("Kamera perspektif ayarlayıcı başlatıldı.")
    
    def _signal_handler(self, sig, frame):
        """
        Sinyal yakalama işleyicisi (CTRL+C gibi)
        """
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def _initialize_camera(self):
        """
        Kamerayı başlatır ve yapılandırır
        """
        try:
            logger.info(f"Kamera başlatılıyor (ID: {self.camera_id})...")
            self.camera = Picamera2(self.camera_id)
            
            # Kamera yapılandırması - düşük çözünürlük ve kare hızı
            camera_config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.framerate}
            )
            self.camera.configure(camera_config)
            
            # Kamera özel ayarlarını düzenleme (daha iyi görüntü kalitesi için)
            controls = {
                "AwbEnable": True,          # Otomatik beyaz dengesi
                "AeEnable": True,           # Otomatik pozlama
                "Sharpness": 1.5            # Keskinlik - netliği artır
            }
            
            try:
                self.camera.set_controls(controls)
            except Exception as e:
                logger.warning(f"Kamera kontrolleri ayarlanamadı: {e}")
                logger.warning("Varsayılan kamera ayarları kullanılacak")
            
            # Kamerayı başlat
            self.camera.start()
            
            # Kameranın dengelenmesi için kısa bir süre bekle
            time.sleep(1)
            
            logger.info("Kamera başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Fare olaylarını yakalayan geri çağırma fonksiyonu
        
        Args:
            event (int): Fare olayı (tıklama, sürükleme, vb.)
            x, y (int): Fare koordinatları
            flags, param: OpenCV geri çağırma parametreleri
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # En yakın noktayı bul
            min_dist = float('inf')
            for i, point in enumerate(self.src_points):
                dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    self.selected_point = i
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            # Noktayı güncelle
            self.src_points[self.selected_point] = [x, y]
    
    def _draw_points(self, image):
        """
        Perspektif noktalarını görüntü üzerine çizer
        
        Args:
            image (numpy.ndarray): Çizim yapılacak görüntü
            
        Returns:
            numpy.ndarray: Noktalar çizilmiş görüntü
        """
        result = image.copy()
        
        # Bölgeyi çiz
        cv2.polylines(result, [np.int32(self.src_points)], True, (0, 255, 0), 2)
        
        # Noktaları çiz
        for i, point in enumerate(self.src_points):
            x, y = point
            cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(result, f"P{i}", (int(x) + 10, int(y) + 10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Kullanım talimatları
        instructions = [
            "TALIMATLAR:",
            "- Noktalari fare ile surukleyerek konumlandirin",
            "- S tusu: Kalibrasyon kaydet",
            "- R tusu: Noktalari sifirla",
            "- ESC tusu: Cikis"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(result, text, (10, 20 + i * 20), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return result
    
    def _warp_perspective(self, image):
        """
        Görüntüye perspektif dönüşümü uygular
        
        Args:
            image (numpy.ndarray): Dönüştürülecek görüntü
            
        Returns:
            numpy.ndarray: Dönüştürülmüş görüntü
        """
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        warped = cv2.warpPerspective(image, M, (self.width, self.height))
        
        # Kuş bakışı noktaları çiz
        warped_with_points = warped.copy()
        for i, point in enumerate(self.dst_points):
            x, y = point
            cv2.circle(warped_with_points, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        return warped_with_points
    
    def _process_frame(self):
        """
        Bir kare işler ve görüntüleri günceller
        """
        # Karenin alınması ve işlenmesi
        try:
            # Yeni bir kare al
            frame = self.camera.capture_array()
            
            # BGR formatına dönüştür (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Güncel kareyi sakla
            self.current_frame = frame_bgr
            
            # Noktaları çiz
            annotated = self._draw_points(frame_bgr)
            
            # Kuş bakışı dönüşümünü hesapla
            warped = self._warp_perspective(frame_bgr)
            
            # Ekranda göster
            cv2.imshow(self.window_name, annotated)
            cv2.imshow("Kus Bakisi Gorunumu", warped)
            
            # FPS sınırlama (çok yüksek CPU kullanımını önlemek için)
            elapsed = time.time() - self.last_process_time
            delay = max(1, int(1000/self.framerate) - int(elapsed*1000))
            self.last_process_time = time.time()
            
            return delay
            
        except Exception as e:
            logger.error(f"Kare işleme hatası: {e}")
            return 100  # Hata durumunda daha uzun bekle
    
    def run(self):
        """
        Kalibrasyon aracını çalıştırır
        """
        try:
            # Kamerayı başlat
            self._initialize_camera()
            
            # Pencere oluştur ve fare olaylarını bağla
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            cv2.namedWindow("Kus Bakisi Gorunumu")
            
            logger.info("Kalibrasyon arayüzü hazır. Noktaları konumlandırın.")
            logger.info("Kaydetmek için 'S', sıfırlamak için 'R', çıkmak için 'ESC' tuşuna basın.")
            
            # Ana döngü
            while self.running:
                # Kare işleme ve FPS limitleyici
                delay = self._process_frame()
                
                # Tuş kontrolü
                key = cv2.waitKey(delay) & 0xFF
                
                # ESC ile çık
                if key == 27:
                    logger.info("Kalibrasyon iptal edildi.")
                    break
                # 's' ile kaydet
                elif key == ord('s'):
                    self.save_calibration()
                    logger.info("Kalibrasyon kaydedildi!")
                    # Kullanıcıya kayıt bildirimi göster
                    if self.current_frame is not None:
                        notification_frame = self.current_frame.copy()
                        cv2.putText(notification_frame, "KALIBRASYON KAYDEDILDI!", 
                                  (int(self.width/2)-150, int(self.height/2)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, notification_frame)
                        cv2.waitKey(1000)  # 1 saniye göster
                        
                # 'r' ile sıfırla
                elif key == ord('r'):
                    self.reset_points()
                    logger.info("Noktalar sıfırlandı.")
            
            # Temizle
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Kalibrasyon çalıştırma hatası: {e}")
            self.cleanup()
    
    def save_calibration(self, filename=None):
        """
        Kalibrasyon değerlerini JSON dosyasına kaydeder
        
        Args:
            filename (str): Kalibrasyon dosyası adı
        """
        if filename is None:
            filename = self.output_file
            
        calibration_data = {
            "src_points": self.src_points.tolist(),
            "dst_points": self.dst_points.tolist(),
            "resolution": {
                "width": self.width,
                "height": self.height
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            logger.info(f"Kalibrasyon dosyası kaydedildi: {filename}")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon kaydetme hatası: {e}")
            return False
    
    def reset_points(self):
        """
        Noktaları varsayılan değerlere sıfırlar
        """
        self.src_points = np.float32([
            [self.width * 0.35, self.height * 0.6],  # Sol üst
            [self.width * 0.65, self.height * 0.6],  # Sağ üst
            [0, self.height],                        # Sol alt
            [self.width, self.height]                # Sağ alt
        ])
    
    def cleanup(self):
        """
        Kaynakları temizler
        """
        logger.info("Kaynaklar temizleniyor...")
        
        # Kamera kapatılıyor
        if self.camera is not None:
            try:
                self.camera.stop()
                logger.info("Kamera kapatıldı.")
            except:
                pass
        
        # OpenCV pencereleri kapatılıyor
        try:
            cv2.destroyAllWindows()
            logger.info("Pencereler kapatıldı.")
        except:
            pass


def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Otonom Araç Kalibrasyon Aracı (Optimize)')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--fps', type=int, default=10, help='Kare hızı (daha düşük = daha az CPU kullanımı)')
    parser.add_argument('--camera', type=int, default=0, help='Kamera ID numarası')
    parser.add_argument('--output', default='serit_kalibrasyon.json', help='Kalibrasyon dosyası çıktı yolu')
    
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
    
    print("\nOPTİMİZE EDİLMİŞ KALİBRASYON ARACI")
    print("-------------------------------")
    print("Bu araç, kamera perspektif kalibrasyonu için optimize edilmiştir.")
    print("Orijinal kalibrasyon.py'nin yerine kullanılabilir.")
    print("\nKullanım Talimatları:")
    print("1. Kırmızı noktaları şeridin dört köşesine konumlandırın")
    print("2. 'S' tuşu ile kalibrasyon dosyasını kaydedin")
    print("3. 'R' tuşu ile noktaları sıfırlayın")
    print("4. 'ESC' tuşu ile programdan çıkın")
    print("\nBaşlatılıyor...\n")
    
    # Kamera perspektif ayarlayıcıyı başlat
    calibrator = KameraPerspektifAyarlayici(
        camera_resolution=resolution, 
        framerate=args.fps, 
        camera_id=args.camera,
        output_file=args.output
    )
    
    # Aracı çalıştır
    calibrator.run()

if __name__ == "__main__":
    main() 