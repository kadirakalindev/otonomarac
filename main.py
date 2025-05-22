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
import os
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
                 left_motor_pins=(16, 18),    # Sol motor IN1, IN2 pinleri
                 right_motor_pins=(36, 38),   # Sağ motor IN1, IN2 pinleri
                 left_pwm_pin=12,             # Sol motor Enable pini
                 right_pwm_pin=32,            # Sağ motor Enable pini
                 use_board_pins=True,         # BOARD pin numaralandırması kullan
                 calibration_file="calibration.json"):  # Kalibrasyon dosyası yolu
        """
        OtonomArac sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            framerate (int): Kare hızı (fps)
            debug (bool): Hata ayıklama modu
            debug_fps (int): Debug modunda gösterilecek maksimum fps
            left_motor_pins (tuple): Sol motor pinleri (IN1, IN2)
            right_motor_pins (tuple): Sağ motor pinleri (IN1, IN2)
            left_pwm_pin (int): Sol motor Enable pini
            right_pwm_pin (int): Sağ motor Enable pini
            use_board_pins (bool): BOARD pin numaralandırması kullan (True) veya BCM kullan (False)
            calibration_file (str): Kalibrasyon dosyası yolu
        """
        self.debug = debug
        self.debug_fps = debug_fps
        self.running = False
        self.camera_resolution = camera_resolution
        self.framerate = framerate
        self.camera = None  # Başlangıçta None olarak tanımla
        self.calibration_file = calibration_file  # Kalibrasyon dosyası yolunu sakla
        
        # Modülleri başlatma denemesi - hata durumunda güvenli kapatma
        try:
            # Kamera başlatma - güvenli başlatma için try-except kullan
            self._initialize_camera()
            
            # Şerit tespit modülünü başlat
            logger.info("Şerit tespit modülü başlatılıyor...")
            self.lane_detector = LaneDetector(camera_resolution=camera_resolution, debug=debug)
            
            # Kalibrasyon dosyasını yükle (varsa)
            if os.path.exists(self.calibration_file):
                logger.info(f"Kalibrasyon dosyası yükleniyor: {self.calibration_file}")
                self.lane_detector.load_calibration(self.calibration_file)
            else:
                logger.warning(f"Kalibrasyon dosyası bulunamadı: {self.calibration_file}")
                logger.warning("Varsayılan değerler kullanılacak.")
            
            # Motor kontrol modülünü başlat
            logger.info("Motor kontrol modülü başlatılıyor...")
            self.motor_controller = MotorController(
                left_motor_pins=left_motor_pins,
                right_motor_pins=right_motor_pins,
                left_pwm_pin=left_pwm_pin,
                right_pwm_pin=right_pwm_pin,
                max_speed=0.7,  # Daha düşük maksimum hız (aşırı ısınmayı önlemek için)
                default_speed=0.35,  # Daha düşük varsayılan hız (aşırı ısınmayı önlemek için)
                use_board_pins=use_board_pins,
                pwm_frequency=50  # PWM frekansını düşürdük - aşırı ısınmayı azaltmak için
            )
        
        except Exception as e:
            logger.error(f"Başlatma hatası: {e}")
            # Kısmi başlatılmış kaynakları temizle
            self.cleanup()
            raise
            
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # FPS ölçümü için değişkenler
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = 0
        
        logger.info("Otonom araç başlatıldı.")
    
    def _initialize_camera(self):
        """
        Kamerayı başlat ve gerekli ayarları yap
        """
        try:
            logger.info("Kamera başlatılıyor...")
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                logger.debug("Önceki kamera kaynağı kapatıldı")
                
            self.camera = cv2.VideoCapture(0)  # Varsayılan kamera
            
            # Kamera başarıyla açıldı mı kontrol et
            if not self.camera.isOpened():
                logger.error("Kamera açılamadı!")
                return False
                
            # Kamera çözünürlüğünü ayarla
            if self.camera_resolution:
                width, height = self.camera_resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.debug(f"Kamera çözünürlüğü {width}x{height} olarak ayarlandı")
                
            # Kamera ayarlarını optimize et
            self.camera.set(cv2.CAP_PROP_EXPOSURE, 15000)  # Pozlama süresini azalt (daha iyi kontrast)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Parlaklık ayarı
            self.camera.set(cv2.CAP_PROP_CONTRAST, 130)   # Kontrast ayarı
            self.camera.set(cv2.CAP_PROP_SATURATION, 130) # Doygunluk ayarı
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 2.5)  # Keskinlik ayarı
            
            # Kamera stabilizasyonu için bekle
            time.sleep(2.0)  
            
            # Test çekimi yap
            logger.debug("Kamera test görüntüsü alınıyor...")
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.error("Kameradan test görüntüsü alınamadı!")
                return False
                
            actual_height, actual_width = frame.shape[:2]
            logger.debug(f"Kamera açıldı. Gerçek çözünürlük: {actual_width}x{actual_height}")
            
            return True
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {str(e)}")
            return False
    
    def signal_handler(self, sig, frame):
        """
        Sinyal yakalama işleyicisi (CTRL+C gibi)
        """
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.cleanup()
        sys.exit(0)
    
    def start(self):
        """
        Araç kontrol döngüsünü başlat
        """
        logger.info("Araç kontrol sistemi başlatılıyor...")
        
        # Hata sayacı
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self.running:
            try:
                # Kamera kontrolü ve başlatma
                if self.camera is None or not self.camera.isOpened():
                    logger.warning("Kamera bağlantısı yok, yeniden başlatılıyor...")
                    if not self._initialize_camera():
                        logger.error("Kamera başlatılamadı, 3 saniye bekleniyor...")
                        time.sleep(3.0)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                            self.cleanup()
                        continue
                
                # Görüntü yakala
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.error("Kameradan görüntü alınamadı!")
                    self.motor_controller.stop()
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                        self.cleanup()
                    continue
                
                # Görüntü işleme
                processed_frame, lane_info = self.lane_detector.process_frame(frame)
                
                # Şerit takibi için merkez sapması hesapla
                center_diff = self._calculate_center_diff(lane_info)
                
                # Motor kontrolü
                if center_diff is not None:
                    self.motor_controller.follow_lane(center_diff, self.motor_controller.default_speed)
                    consecutive_errors = 0  # Başarılı işlem, hata sayacını sıfırla
                else:
                    # Şerit bulunamadı
                    if self.lane_detector.detection_failures > self.lane_detector.max_detection_failures:
                        logger.warning("Şerit bulunamadı, durduruluyor!")
                        self.motor_controller.stop()
                    consecutive_errors += 1
                
                # Debug modu aktifse görüntüyü göster
                if self.debug:
                    self._display_debug_info(processed_frame, lane_info, center_diff)
                
                # İşlem başına bekleme süresi
                time.sleep(0.05)  # 50ms bekleme (20 FPS hedefi)
                
            except KeyboardInterrupt:
                logger.info("Klavye kesintisi, program durduruluyor...")
                break
            except Exception as e:
                logger.error(f"İşlem hatası: {str(e)}")
                logger.debug(traceback.format_exc())
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                    self.cleanup()
        
        # Kaynakları temizle
        self.cleanup()
        logger.info("Program sonlandırıldı.")
    
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
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.release()
            except:
                pass
        
        # OpenCV pencerelerini kapat
        if self.debug:
            cv2.destroyAllWindows()
        
        logger.info("Temizleme tamamlandı.")

    def _calculate_center_diff(self, lane_info):
        """
        Şerit merkez sapmasını hesapla
        """
        left_lane, right_lane = lane_info
        
        if left_lane is not None and right_lane is not None:
            # Her iki şerit de tespit edildi
            left_bottom_x = left_lane[0]
            right_bottom_x = right_lane[0]
            
            # Şerit merkezini hesapla
            lane_center = (left_bottom_x + right_bottom_x) // 2
            
            # Görüntü merkezinden sapma
            image_center = self.camera_resolution[0] // 2
            center_diff = lane_center - image_center
            
            return center_diff
            
        elif left_lane is not None:
            # Sadece sol şerit tespit edildi
            # Tahmini şerit genişliği (40 cm - fiziksel genişlik)
            # Yaklaşık piksel genişliği hesapla
            estimated_lane_width_px = int(self.camera_resolution[0] * 0.4)  # Görüntü genişliğinin %40'ı
            
            left_bottom_x = left_lane[0]
            estimated_right_x = left_bottom_x + estimated_lane_width_px
            
            # Tahmini şerit merkezini hesapla
            lane_center = (left_bottom_x + estimated_right_x) // 2
            image_center = self.camera_resolution[0] // 2
            
            # Daha muhafazakâr bir sapma değeri döndür (sadece bir şerit tespit edildiğinde daha az güvenilir)
            center_diff = int((lane_center - image_center) * 0.8)
            
            return center_diff
            
        elif right_lane is not None:
            # Sadece sağ şerit tespit edildi
            # Tahmini şerit genişliği
            estimated_lane_width_px = int(self.camera_resolution[0] * 0.4)
            
            right_bottom_x = right_lane[0]
            estimated_left_x = right_bottom_x - estimated_lane_width_px
            
            # Tahmini şerit merkezini hesapla
            lane_center = (estimated_left_x + right_bottom_x) // 2
            image_center = self.camera_resolution[0] // 2
            
            # Daha muhafazakâr bir sapma değeri döndür
            center_diff = int((lane_center - image_center) * 0.8)
            
            return center_diff
            
        # Her iki şerit de tespit edilemedi
        return None

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
    parser.add_argument('--left-motor', nargs=2, type=int, default=[16, 18], help='Sol motor pinleri (IN1 IN2)')
    parser.add_argument('--right-motor', nargs=2, type=int, default=[36, 38], help='Sağ motor pinleri (IN1 IN2)')
    parser.add_argument('--left-pwm', type=int, default=12, help='Sol motor Enable pini')
    parser.add_argument('--right-pwm', type=int, default=32, help='Sağ motor Enable pini')
    parser.add_argument('--use-bcm', action='store_true', help='BCM pin numaralandırması kullan (varsayılan: BOARD)')
    
    # Kalibrasyon dosyası için argüman ekle
    parser.add_argument('--calibration', default='calibration.json', help='Kalibrasyon dosyası yolu')
    
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
    try:
        logger.info("Otonom araç başlatılıyor...")
        otonom_arac = OtonomArac(
            camera_resolution=resolution,
            framerate=args.fps,
            debug=args.debug,
            debug_fps=args.debug_fps,
            left_motor_pins=tuple(args.left_motor),
            right_motor_pins=tuple(args.right_motor),
            left_pwm_pin=args.left_pwm,
            right_pwm_pin=args.right_pwm,
            use_board_pins=not args.use_bcm,
            calibration_file=args.calibration
        )
        
        # Otonom sürüşü başlat
        otonom_arac.start()
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu (CTRL+C)")
    except Exception as e:
        logger.error(f"Hata: {e}")
    
    logger.info("Program sonlandırıldı.")

if __name__ == "__main__":
    main() 