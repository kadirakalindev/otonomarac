#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - İnteraktif Şerit Kalibrasyonu Aracı
Bu program, kullanıcının şerit üzerinde 5 noktayı işaretleyerek kalibrasyon yapmasını sağlar.
"""

import cv2
import numpy as np
import json
import time
import argparse
import logging
import os
import signal
import sys
from picamera2 import Picamera2

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KalibrasyonInteraktif")

class InteraktifKalibrasyon:
    """
    Kullanıcının 5 noktayı şerit üzerinde işaretleyerek kalibrasyon yapmasını sağlayan sınıf
    """
    def __init__(self, camera_resolution=(320, 240), framerate=15, camera_id=0, output_file="calibration.json"):
        """
        InteraktifKalibrasyon sınıfını başlatır.
        
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
        
        # Kalibrasyon noktaları
        self.points = []
        self.max_points = 5
        self.point_names = [
            "Sol Şerit Alt",
            "Sol Şerit Üst",
            "Orta Şerit",
            "Sağ Şerit Üst",
            "Sağ Şerit Alt"
        ]
        
        # ROI ve şerit parametreleri
        self.roi_vertices = None
        self.lane_width = None
        self.lane_center = None
        
        # Kamera nesnesi
        self.camera = None
        
        # Pencere adı
        self.window_name = "Şerit Kalibrasyonu"
        
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Son işlem zamanı (FPS kontrol)
        self.last_process_time = time.time()
        
        logger.info("İnteraktif kalibrasyon aracı başlatıldı.")
    
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
            
            # Kamera yapılandırması
            camera_config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.framerate}
            )
            self.camera.configure(camera_config)
            
            # Kamera özel ayarlarını düzenleme (daha iyi görüntü kalitesi için)
            controls = {
                "AwbEnable": True,          # Otomatik beyaz dengesi
                "AeEnable": True,           # Otomatik pozlama
                "ExposureTime": 15000,      # Pozlama süresi
                "Sharpness": 2.0            # Keskinlik
            }
            
            try:
                self.camera.set_controls(controls)
            except Exception as e:
                logger.warning(f"Kamera kontrolleri ayarlanamadı: {e}")
            
            # Kamerayı başlat
            self.camera.start()
            
            # Kameranın dengelenmesi için kısa bir süre bekle
            time.sleep(1.5)
            
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
            # Maksimum nokta sayısına ulaşılmadıysa yeni nokta ekle
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                logger.info(f"Nokta {len(self.points)} eklendi: ({x}, {y}) - {self.point_names[len(self.points)-1]}")
                
                # Tüm noktalar eklendiyse ROI ve şerit parametrelerini hesapla
                if len(self.points) == self.max_points:
                    self._calculate_parameters()
    
    def _calculate_parameters(self):
        """
        Seçilen noktalardan ROI ve şerit parametrelerini hesaplar
        """
        if len(self.points) != self.max_points:
            logger.warning("Tüm noktalar eklenmeden parametreler hesaplanamaz!")
            return
        
        # Noktaları sırala
        left_bottom = self.points[0]  # Sol Şerit Alt
        left_top = self.points[1]     # Sol Şerit Üst
        center = self.points[2]       # Orta Şerit
        right_top = self.points[3]    # Sağ Şerit Üst
        right_bottom = self.points[4] # Sağ Şerit Alt
        
        # ROI köşelerini hesapla
        self.roi_vertices = np.array([
            [left_bottom[0], left_bottom[1]],
            [left_top[0], left_top[1]],
            [right_top[0], right_top[1]],
            [right_bottom[0], right_bottom[1]]
        ], dtype=np.int32)
        
        # Şerit genişliğini hesapla (alt noktalar arasındaki mesafe)
        self.lane_width = np.sqrt((right_bottom[0] - left_bottom[0])**2 + (right_bottom[1] - left_bottom[1])**2)
        
        # Şerit merkezini hesapla
        self.lane_center = center
        
        logger.info("Kalibrasyon parametreleri hesaplandı:")
        logger.info(f"ROI: {self.roi_vertices.tolist()}")
        logger.info(f"Şerit genişliği: {self.lane_width:.2f}")
        logger.info(f"Şerit merkezi: {self.lane_center}")
    
    def _draw_calibration(self, image):
        """
        Kalibrasyon noktalarını ve şerit bilgilerini görüntü üzerine çizer
        
        Args:
            image (numpy.ndarray): Çizim yapılacak görüntü
            
        Returns:
            numpy.ndarray: Çizim yapılmış görüntü
        """
        result = image.copy()
        
        # Noktaları çiz
        for i, point in enumerate(self.points):
            cv2.circle(result, point, 5, (0, 0, 255), -1)
            cv2.putText(result, f"{i+1}: {self.point_names[i]}", 
                      (point[0] + 10, point[1]), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # ROI'yi çiz
        if self.roi_vertices is not None:
            cv2.polylines(result, [self.roi_vertices], True, (0, 255, 0), 2)
            
            # Şerit merkezi ve genişliğini göster
            if self.lane_center is not None:
                cv2.circle(result, self.lane_center, 7, (255, 0, 0), -1)
                cv2.putText(result, "Şerit Merkezi", 
                          (self.lane_center[0] + 10, self.lane_center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Talimatları göster
        instructions = [
            "TALIMATLAR:",
            "1. Şerit üzerinde 5 noktayı işaretleyin:",
            "   - Sol Şerit Alt",
            "   - Sol Şerit Üst",
            "   - Orta Şerit",
            "   - Sağ Şerit Üst",
            "   - Sağ Şerit Alt",
            f"İşaretlenen: {len(self.points)}/{self.max_points}",
            "",
            "S: Kaydet, R: Sıfırla, ESC: Çıkış"
        ]
        
        # Talimatlar için yarı saydam arka plan
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (280, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        # Talimatları yaz
        for i, text in enumerate(instructions):
            cv2.putText(result, text, (15, 30 + i * 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def _process_frame(self):
        """
        Bir kare işler ve görüntüleri günceller
        """
        try:
            # Yeni bir kare al
            frame = self.camera.capture_array()
            
            # BGR formatına dönüştür (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Kalibrasyon bilgilerini çiz
            annotated = self._draw_calibration(frame_bgr)
            
            # Ekranda göster
            cv2.imshow(self.window_name, annotated)
            
            # FPS sınırlama
            elapsed = time.time() - self.last_process_time
            delay = max(1, int(1000/self.framerate) - int(elapsed*1000))
            self.last_process_time = time.time()
            
            return delay
            
        except Exception as e:
            logger.error(f"Kare işleme hatası: {e}")
            return 100  # Hata durumunda daha uzun bekle
    
    def save_calibration(self, filename=None):
        """
        Kalibrasyon değerlerini JSON dosyasına kaydeder
        
        Args:
            filename (str): Kalibrasyon dosyası adı
        """
        if filename is None:
            filename = self.output_file
            
        if self.roi_vertices is None:
            logger.error("Kalibrasyon parametreleri hesaplanmadan kayıt yapılamaz!")
            return False
            
        calibration_data = {
            "roi_vertices": self.roi_vertices.tolist(),
            "lane_width": float(self.lane_width),
            "lane_center": list(self.lane_center),
            "resolution": {
                "width": self.width,
                "height": self.height
            },
            # Şerit tespiti için ek parametreler
            "canny_low_threshold": 30,
            "canny_high_threshold": 120,
            "blur_kernel_size": 5,
            "hough_threshold": 20,
            "min_line_length": 15,
            "max_line_gap": 40,
            # Şerit takibi için parametreler
            "center_deadzone": 0.05,
            "turn_speed_factor": 0.3
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
        Kalibrasyon noktalarını sıfırlar
        """
        self.points = []
        self.roi_vertices = None
        self.lane_width = None
        self.lane_center = None
        logger.info("Kalibrasyon noktaları sıfırlandı.")
    
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
            
            logger.info("Kalibrasyon arayüzü hazır.")
            logger.info("Şerit üzerinde 5 noktayı işaretleyin.")
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
                    if self.roi_vertices is not None:
                        self.save_calibration()
                        # Kayıt bildirimi göster
                        frame = self.camera.capture_array()
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        notification = frame_bgr.copy()
                        cv2.putText(notification, "KALIBRASYON KAYDEDILDI!", 
                                  (int(self.width/2)-150, int(self.height/2)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, notification)
                        cv2.waitKey(1000)  # 1 saniye göster
                    else:
                        logger.warning("Tüm noktalar işaretlenmeden kayıt yapılamaz!")
                # 'r' ile sıfırla
                elif key == ord('r'):
                    self.reset_points()
            
            # Temizle
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Kalibrasyon çalıştırma hatası: {e}")
            self.cleanup()
    
    def cleanup(self):
        """
        Kaynakları temizler
        """
        logger.info("Kaynaklar temizleniyor...")
        
        # Kamera kapatılıyor
        if hasattr(self, 'camera') and self.camera is not None:
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
    parser = argparse.ArgumentParser(description='Otonom Araç İnteraktif Şerit Kalibrasyonu')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--fps', type=int, default=15, help='Kare hızı')
    parser.add_argument('--camera', type=int, default=0, help='Kamera ID numarası')
    parser.add_argument('--output', default='calibration.json', help='Kalibrasyon dosyası çıktı yolu')
    
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
    
    print("\nİNTERAKTİF ŞERİT KALİBRASYONU")
    print("---------------------------")
    print("Bu araç, şerit üzerinde 5 noktayı işaretleyerek kalibrasyon yapmanızı sağlar.")
    print("\nKullanım Talimatları:")
    print("1. Şerit üzerinde aşağıdaki 5 noktayı sırayla işaretleyin:")
    print("   - Sol Şerit Alt (şeridin sol alt köşesi)")
    print("   - Sol Şerit Üst (şeridin sol üst köşesi)")
    print("   - Orta Şerit (şeridin ortası)")
    print("   - Sağ Şerit Üst (şeridin sağ üst köşesi)")
    print("   - Sağ Şerit Alt (şeridin sağ alt köşesi)")
    print("2. 'S' tuşu ile kalibrasyon dosyasını kaydedin")
    print("3. 'R' tuşu ile noktaları sıfırlayın")
    print("4. 'ESC' tuşu ile programdan çıkın")
    print("\nBaşlatılıyor...\n")
    
    # İnteraktif kalibrasyon aracını başlat
    calibrator = InteraktifKalibrasyon(
        camera_resolution=resolution, 
        framerate=args.fps, 
        camera_id=args.camera,
        output_file=args.output
    )
    
    # Aracı çalıştır
    calibrator.run()

if __name__ == "__main__":
    main() 