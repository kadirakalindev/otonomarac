#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - İnteraktif Şerit Kalibrasyon Aracı
Bu program, kullanıcının kamera görüntüsü üzerinde şerit noktalarını 
interaktif olarak seçmesine ve kalibrasyon değerlerini kaydetmesine olanak tanır.
"""

import cv2
import numpy as np
import json
import time
import logging
import os
import argparse
from picamera2 import Picamera2

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InteraktifKalibrasyon")

class InteraktifKalibrasyon:
    """İnteraktif şerit kalibrasyonu için sınıf"""
    
    def __init__(self, camera_resolution=(320, 240), output_file="serit_kalibrasyon.json"):
        """
        InteraktifKalibrasyon sınıfını başlatır
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            output_file (str): Kalibrasyon dosyası çıktı yolu
        """
        self.width, self.height = camera_resolution
        self.output_file = output_file
        self.running = True
        
        # Kalibrasyon noktaları
        self.points = []
        self.max_points = 5  # Maksimum 5 nokta seçilebilir
        self.current_point = None  # Fare ile taşınan nokta indeksi
        
        # Kamera
        self.camera = None
        self.current_frame = None
        
        # Pencere adı
        self.window_name = "Şerit Kalibrasyonu"
        
        # Nokta açıklamaları
        self.point_descriptions = [
            "Sol şeridin alt noktası",
            "Sol şeridin üst noktası",
            "Orta şeridin üst noktası",
            "Sağ şeridin üst noktası",
            "Sağ şeridin alt noktası"
        ]
        
        # Varsayılan noktalar
        self.default_points = [
            [int(self.width * 0.1), self.height],             # Sol şeridin alt noktası
            [int(self.width * 0.35), int(self.height * 0.6)], # Sol şeridin üst noktası
            [int(self.width * 0.5), int(self.height * 0.5)],  # Orta şeridin üst noktası
            [int(self.width * 0.65), int(self.height * 0.6)], # Sağ şeridin üst noktası
            [int(self.width * 0.9), self.height]              # Sağ şeridin alt noktası
        ]
        
        # Noktaları varsayılan değerlerle başlat
        self.points = self.default_points.copy()
        
        logger.info("İnteraktif kalibrasyon aracı başlatıldı")
    
    def _initialize_camera(self):
        """Kamerayı başlatır ve yapılandırır"""
        try:
            logger.info("Kamera başlatılıyor...")
            self.camera = Picamera2()
            
            # Kamera yapılandırması
            preview_config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                lores={"size": (320, 240), "format": "YUV420"},
                display="lores"
            )
            self.camera.configure(preview_config)
            
            # Kamera özel ayarları
            try:
                controls = {
                    "AwbEnable": True,          # Otomatik beyaz dengesi
                    "AeEnable": True,           # Otomatik pozlama
                    "ExposureTime": 10000,      # Pozlama süresi (mikrosaniye)
                    "AnalogueGain": 1.0,        # Analog kazanç
                    "Brightness": 0.0,          # Parlaklık
                    "Contrast": 1.2,            # Kontrast - artırıldı
                    "Sharpness": 1.5,           # Keskinlik - artırıldı
                }
                self.camera.set_controls(controls)
                logger.info("Kamera kontrolleri ayarlandı")
            except Exception as e:
                logger.warning(f"Kamera kontrolleri ayarlanırken hata: {e}")
            
            # Kamerayı başlat
            self.camera.start()
            
            # Kameranın dengelenmesi için bekle
            logger.info("Kamera dengeleniyor...")
            time.sleep(2)
            
            logger.info("Kamera başarıyla başlatıldı")
            
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Fare olaylarını yakalayan geri çağırma fonksiyonu"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Nokta seçimi veya taşıma
            min_dist = float('inf')
            selected_idx = -1
            
            for i, point in enumerate(self.points):
                dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    selected_idx = i
            
            # Eğer bir noktaya yeterince yakınsa, o noktayı seç
            if min_dist < 20:  # 20 piksel içindeyse
                self.current_point = selected_idx
            else:
                self.current_point = None
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.current_point = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.current_point is not None:
            # Noktayı taşı
            self.points[self.current_point] = [x, y]
    
    def _draw_points(self, image):
        """Kalibrasyon noktalarını görüntü üzerine çizer"""
        result = image.copy()
        
        # Noktaları ve çizgileri çiz
        if len(self.points) >= 2:
            # Şerit çizgilerini çiz
            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], False, (0, 255, 0), 2)
        
        # Noktaları çiz
        for i, point in enumerate(self.points):
            x, y = point
            # Aktif nokta farklı renkte
            color = (0, 0, 255) if i == self.current_point else (255, 0, 0)
            cv2.circle(result, (int(x), int(y)), 5, color, -1)
            cv2.putText(result, f"{i+1}", (int(x) + 10, int(y) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Açıklama metinleri
        for i, desc in enumerate(self.point_descriptions[:len(self.points)]):
            y_pos = 30 + i * 20
            cv2.putText(result, f"{i+1}: {desc}", (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Kullanım talimatları
        instructions = [
            "TALIMATLAR:",
            "- Noktalari fare ile surukleyerek konumlandirin",
            "- S tusu: Kalibrasyon kaydet",
            "- R tusu: Noktalari sifirla",
            "- ESC tusu: Cikis"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(result, text, (10, self.height - 70 + i * 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def _generate_roi_from_points(self):
        """Seçilen noktalardan ROI (Region of Interest) oluşturur"""
        if len(self.points) < 4:
            logger.warning("ROI oluşturmak için en az 4 nokta gerekli")
            return None
        
        # ROI için noktaları düzenle
        roi_points = np.array([
            self.points[0],  # Sol alt
            self.points[1],  # Sol üst
            self.points[3],  # Sağ üst
            self.points[4]   # Sağ alt
        ], dtype=np.int32)
        
        return roi_points
    
    def _generate_lane_center_line(self):
        """Orta şerit çizgisini oluşturur"""
        if len(self.points) < 5:
            logger.warning("Orta şerit çizgisi oluşturmak için 5 nokta gerekli")
            return None
        
        # Orta şerit çizgisi için noktalar
        center_line = np.array([
            [(self.points[0][0] + self.points[4][0]) // 2, self.height],  # Alt orta nokta
            self.points[2]  # Orta üst nokta
        ], dtype=np.int32)
        
        return center_line
    
    def save_calibration(self):
        """Kalibrasyon değerlerini dosyaya kaydeder"""
        if len(self.points) < 5:
            logger.warning("Kalibrasyon için 5 nokta gerekli")
            return False
        
        # ROI oluştur
        roi_vertices = self._generate_roi_from_points()
        if roi_vertices is None:
            return False
        
        # Orta şerit çizgisi
        center_line = self._generate_lane_center_line()
        if center_line is None:
            return False
        
        # Kalibrasyon verilerini hazırla
        calibration_data = {
            "roi_vertices": roi_vertices.tolist(),
            "center_line": center_line.tolist(),
            "lane_points": self.points,
            "resolution": {
                "width": self.width,
                "height": self.height
            },
            # Şerit tespiti parametreleri
            "canny_low_threshold": 30,
            "canny_high_threshold": 120,
            "blur_kernel_size": 5,
            "hough_threshold": 15,
            "min_line_length": 15,
            "max_line_gap": 40
        }
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            logger.info(f"Kalibrasyon dosyası kaydedildi: {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon kaydetme hatası: {e}")
            return False
    
    def run(self):
        """Kalibrasyon aracını çalıştırır"""
        try:
            # Kamerayı başlat
            self._initialize_camera()
            
            # Pencere oluştur ve fare olaylarını bağla
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            # Mouse callback fonksiyonunu güvenli bir şekilde bağla
            try:
                cv2.setMouseCallback(self.window_name, self._mouse_callback)
            except Exception as e:
                logger.error(f"Mouse callback hatası: {e}")
                # Alternatif yöntem deneyin
                try:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(self.window_name, lambda event, x, y, flags, param: 
                                      self._mouse_callback(event, x, y, flags, param))
                except Exception as e:
                    logger.error(f"Alternatif mouse callback de başarısız: {e}")
                    print("UYARI: Fare etkileşimi çalışmıyor. Kalibrasyon için varsayılan değerler kullanılacak.")
            
            logger.info("Kalibrasyon arayüzü hazır. Noktaları konumlandırın.")
            logger.info("Kaydetmek için 'S', sıfırlamak için 'R', çıkmak için 'ESC' tuşuna basın.")
            
            while self.running:
                # Kameradan görüntü al
                frame = self.camera.capture_array()
                
                # RGB'den BGR'a dönüştür (OpenCV formatı)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.current_frame = frame_bgr
                
                # Noktaları çiz
                annotated_frame = self._draw_points(frame_bgr)
                
                # Görüntüyü göster
                cv2.imshow(self.window_name, annotated_frame)
                
                # Tuş kontrolü
                key = cv2.waitKey(1) & 0xFF
                
                # ESC ile çık
                if key == 27:
                    logger.info("Kalibrasyon iptal edildi.")
                    break
                # 's' ile kaydet
                elif key == ord('s'):
                    if self.save_calibration():
                        # Kullanıcıya kayıt bildirimi göster
                        notification_frame = self.current_frame.copy()
                        cv2.putText(notification_frame, "KALIBRASYON KAYDEDILDI!", 
                                  (int(self.width/2)-150, int(self.height/2)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, notification_frame)
                        cv2.waitKey(1000)  # 1 saniye göster
                # 'r' ile sıfırla
                elif key == ord('r'):
                    self.points = self.default_points.copy()
                    logger.info("Noktalar sıfırlandı.")
            
            # Temizle
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Kalibrasyon çalıştırma hatası: {e}")
            self.cleanup()
    
    def cleanup(self):
        """Kaynakları temizler"""
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
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description='İnteraktif Şerit Kalibrasyon Aracı')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--output', default='serit_kalibrasyon.json', help='Kalibrasyon dosyası çıktı yolu')
    
    return parser.parse_args()

def main():
    """Ana program"""
    # Argümanları işle
    args = parse_arguments()
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        resolution = (320, 240)
    
    print("\nİNTERAKTİF ŞERİT KALİBRASYON ARACI")
    print("--------------------------------")
    print("Bu araç, şerit takibi için gereken kalibrasyon noktalarını belirlemenizi sağlar.")
    print("5 adet noktayı şeritler üzerinde konumlandırın:")
    print("  1. Sol şeridin alt noktası")
    print("  2. Sol şeridin üst noktası")
    print("  3. Orta şeridin üst noktası (takip edilecek merkez)")
    print("  4. Sağ şeridin üst noktası")
    print("  5. Sağ şeridin alt noktası")
    print("\nBaşlatılıyor...\n")
    
    # Kalibrasyon aracını başlat
    calibrator = InteraktifKalibrasyon(
        camera_resolution=resolution,
        output_file=args.output
    )
    
    # Aracı çalıştır
    calibrator.run()

if __name__ == "__main__":
    main() 