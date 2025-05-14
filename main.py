#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Ana Kontrol Programı
Bu program, kamera görüntüsünü işleyerek şerit tespiti yapar ve aracı kontrol eder.
"""

import cv2
import time
import signal
import sys
import logging
import argparse
from picamera2 import Picamera2
from lane_detection import LaneDetector
from motor_control import MotorController

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainControl")

class OtonomArac:
    """
    Otonom araç ana kontrol sınıfı
    """
    def __init__(self, 
                 camera_resolution=(640, 480),
                 framerate=30,
                 debug=False,
                 debug_fps=10,
                 left_motor_pins=(17, 18),
                 right_motor_pins=(22, 23),
                 left_pwm_pin=None,
                 right_pwm_pin=None):
        """
        OtonomArac sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            framerate (int): Kare hızı (fps)
            debug (bool): Hata ayıklama modu
            debug_fps (int): Debug modunda gösterilecek maksimum fps
            left_motor_pins (tuple): Sol motor pinleri (ileri, geri)
            right_motor_pins (tuple): Sağ motor pinleri (ileri, geri)
            left_pwm_pin (int): Sol motor PWM pini
            right_pwm_pin (int): Sağ motor PWM pini
        """
        self.debug = debug
        self.debug_fps = debug_fps
        self.running = False
        
        # Kamera başlatma
        logger.info("Kamera başlatılıyor...")
        self.camera = Picamera2()
        
        # Kamera yapılandırması
        self.camera_config = self.camera.create_preview_configuration(
            main={"size": camera_resolution, "format": "RGB888"},
            controls={"FrameRate": framerate}
        )
        self.camera.configure(self.camera_config)
        
        # Modülleri başlat
        logger.info("Şerit tespit modülü başlatılıyor...")
        self.lane_detector = LaneDetector(camera_resolution=camera_resolution, debug=debug)
        
        logger.info("Motor kontrol modülü başlatılıyor...")
        self.motor_controller = MotorController(
            left_motor_pins=left_motor_pins,
            right_motor_pins=right_motor_pins,
            left_pwm_pin=left_pwm_pin,
            right_pwm_pin=right_pwm_pin,
            max_speed=0.8,
            default_speed=0.4
        )
        
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # FPS ölçümü için değişkenler
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = 0
        
        logger.info("Otonom araç başlatıldı.")
    
    def signal_handler(self, sig, frame):
        """
        Sinyal yakalama işleyicisi (CTRL+C gibi)
        """
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.cleanup()
        sys.exit(0)
    
    def start(self):
        """
        Otonom araç sürüş döngüsünü başlatır
        """
        if self.running:
            logger.warning("Araç zaten çalışıyor!")
            return
        
        logger.info("Otonom sürüş başlatılıyor...")
        self.running = True
        
        # Kamerayı başlat
        self.camera.start()
        
        try:
            # Çalışma başlangıcında biraz beklet
            time.sleep(2)
            
            # FPS hesaplama değişkenlerini başlat
            self.fps_start_time = time.time()
            self.frame_count = 0
            
            # Debug modu için pencere oluştur
            if self.debug:
                cv2.namedWindow("Otonom Arac", cv2.WINDOW_NORMAL)
                cv2.setWindowTitle("Otonom Arac", "Otonom Arac - Serit Tespiti")
            
            # Debug modunda son frame update zamanı
            last_debug_update = 0
            debug_frame_interval = 1.0 / self.debug_fps if self.debug_fps > 0 else 0
            
            # Ana döngü
            while self.running:
                # Kameradan görüntü al
                frame = self.camera.capture_array()
                
                # Görüntüyü işle ve şeritleri tespit et
                processed_frame, center_diff = self.lane_detector.process_frame(frame)
                
                # Şeritlere göre motoru kontrol et
                self.motor_controller.follow_lane(center_diff)
                
                # FPS hesapla
                self.frame_count += 1
                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.fps_start_time = time.time()
                    self.frame_count = 0
                    logger.debug(f"FPS: {self.fps:.1f}")
                
                # Debug modunda görüntüyü göster (fps sınırlandırması ile)
                if self.debug:
                    current_time = time.time()
                    if current_time - last_debug_update >= debug_frame_interval:
                        # FPS bilgisini görüntüye ekle
                        cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", 
                                  (processed_frame.shape[1] - 120, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Görüntüyü göster
                        cv2.imshow("Otonom Arac", processed_frame)
                        last_debug_update = current_time
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
        except Exception as e:
            logger.error(f"Hata oluştu: {e}")
        finally:
            self.cleanup()
    
    def stop(self):
        """
        Otonom aracı durdurur
        """
        self.running = False
        logger.info("Otonom sürüş durduruldu.")
    
    def cleanup(self):
        """
        Kaynakları temizler ve kapatır
        """
        logger.info("Temizleme yapılıyor...")
        
        # Motorları durdur
        if hasattr(self, 'motor_controller'):
            self.motor_controller.stop()
            self.motor_controller.cleanup()
        
        # Kamerayı kapat
        if hasattr(self, 'camera'):
            self.camera.stop()
        
        # OpenCV pencerelerini kapat
        if self.debug:
            cv2.destroyAllWindows()
        
        logger.info("Temizleme tamamlandı.")

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Otonom Araç Kontrol Programı')
    parser.add_argument('--debug', action='store_true', help='Hata ayıklama modunu etkinleştirir')
    parser.add_argument('--debug-fps', type=int, default=10, help='Debug modunda gösterilecek maksimum FPS')
    parser.add_argument('--resolution', default='640x480', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--fps', type=int, default=30, help='Kare hızı')
    
    # Motor pin argümanları
    parser.add_argument('--left-motor', nargs=2, type=int, default=[17, 18], help='Sol motor pinleri (ileri geri)')
    parser.add_argument('--right-motor', nargs=2, type=int, default=[22, 23], help='Sağ motor pinleri (ileri geri)')
    parser.add_argument('--left-pwm', type=int, help='Sol motor PWM pini')
    parser.add_argument('--right-pwm', type=int, help='Sağ motor PWM pini')
    
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
        resolution = (640, 480)
    
    # Otonom aracı başlat
    arac = OtonomArac(
        camera_resolution=resolution,
        framerate=args.fps,
        debug=args.debug,
        debug_fps=args.debug_fps,
        left_motor_pins=tuple(args.left_motor),
        right_motor_pins=tuple(args.right_motor),
        left_pwm_pin=args.left_pwm,
        right_pwm_pin=args.right_pwm
    )
    
    # Sürüşü başlat
    arac.start()

if __name__ == "__main__":
    main() 