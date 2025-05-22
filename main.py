#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç Ana Program
Şerit tespiti ve takibi için ana program.
"""

import cv2
import time
import argparse
import logging
import signal
import sys
from picamera2 import Picamera2
from typing import Tuple, Dict, Any, Optional

# Yerel modülleri içe aktar
from lane_detection import LaneDetector
from motor_control import MotorController

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("otonom_arac.log")
    ]
)
logger = logging.getLogger('Main')

class OtonomArac:
    """Otonom araç ana kontrolör sınıfı"""
    
    def __init__(self, args: Dict[str, Any]):
        """
        Otonom araç kontrolörü başlatıcısı
        
        Args:
            args: Komut satırı argümanları
        """
        # Argümanları kaydet
        self.args = args
        self.debug = args.get('debug', False)
        self.camera_resolution = args.get('resolution', (320, 240))
        
        # Durum değişkenleri
        self.is_running = False
        self.camera = None
        self.lane_detector = None
        self.motor_controller = None
        
        # Hata sayacı
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.last_error_time = 0
        self.error_cooldown = 3.0  # Hata sonrası bekleme süresi (saniye)
        
        logger.info("Otonom araç kontrolörü başlatıldı")
    
    def _initialize_camera(self) -> bool:
        """
        Kamerayı yapılandır ve başlat
        
        Returns:
            bool: Başarılı olup olmadığı
        """
        try:
            # Picamera2'yi başlat
            self.camera = Picamera2()
            
            # Kamera yapılandırması
            config = self.camera.create_preview_configuration(
                main={"format": "RGB888", "size": self.camera_resolution}
            )
            self.camera.configure(config)
            
            # Kamera parametrelerini ayarla
            controls = {
                "ExposureTime": 15000,  # Pozlama süresi (μs)
                "AnalogueGain": 1.0,   # Analog kazanç
                "Sharpness": 2.5,      # Keskinlik
                "Brightness": 0.0,     # Parlaklık
                "Contrast": 1.0,       # Kontrast
                "Saturation": 1.2,     # Doygunluk
                "NoiseReductionMode": 1 # Gürültü azaltma modu
            }
            self.camera.set_controls(controls)
            
            # Kamerayı başlat
            self.camera.start()
            
            # Kamera ısınma süresi
            time.sleep(0.5)
            
            logger.info(f"Kamera başlatıldı: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
            
            # Test frame al
            test_frame = self.camera.capture_array()
            if test_frame is None or len(test_frame.shape) != 3:
                logger.error("Kamera başlatma başarılı ancak geçerli frame alınamadı")
                return False
                
            actual_height, actual_width = test_frame.shape[:2]
            logger.info(f"Kamera frame boyutları: {actual_width}x{actual_height}")
            
            return True
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {str(e)}")
            return False
    
    def setup(self) -> bool:
        """
        Sistem bileşenlerini başlat
        
        Returns:
            bool: Başlatma başarılı mı
        """
        try:
            # Kamerayı başlat
            if not self._initialize_camera():
                logger.error("Kamera başlatılamadı")
                return False
            
            # Şerit dedektörünü başlat
            self.lane_detector = LaneDetector(camera_resolution=self.camera_resolution)
            logger.info("Şerit dedektörü başlatıldı")
            
            # Motor kontrolörünü başlat
            self.motor_controller = MotorController()
            logger.info("Motor kontrolörü başlatıldı")
            
            return True
        except Exception as e:
            logger.error(f"Sistem başlatma hatası: {str(e)}")
            return False
    
    def cleanup(self):
        """Bileşenleri temizle ve kapat"""
        try:
            if self.camera:
                self.camera.stop()
                logger.info("Kamera kapatıldı")
            
            if self.motor_controller:
                self.motor_controller.cleanup()
                logger.info("Motor kontrolörü temizlendi")
            
            if self.debug:
                cv2.destroyAllWindows()
                logger.info("OpenCV pencereleri kapatıldı")
                
            logger.info("Sistem temizlendi ve kapatıldı")
        except Exception as e:
            logger.error(f"Temizleme sırasında hata: {str(e)}")
    
    def signal_handler(self, sig, frame):
        """Sinyal işleyicisi (Ctrl+C için)"""
        logger.info("Programı sonlandırma sinyali alındı")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        """Çalışmayı durdur"""
        self.is_running = False
        if self.motor_controller:
            self.motor_controller.stop()
        logger.info("Sistem durduruldu")
    
    def start(self):
        """Ana döngüyü başlat"""
        if not self.setup():
            logger.error("Sistem başlatılamadı. Program sonlandırılıyor.")
            return
        
        self.is_running = True
        logger.info("Otonom sürüş başlatılıyor...")
        
        # Sinyal işleyicisini ayarla
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Ana döngü
        try:
            while self.is_running:
                loop_start_time = time.time()
                
                try:
                    # Kameradan görüntü al
                    frame = self.camera.capture_array()
                    
                    # RGB'den BGR'ye dönüşüm (OpenCV için)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Şerit tespiti yap
                    result = self.lane_detector.process_frame(frame)
                    
                    # Şerit tespiti sonuçlarını kontrol et
                    if result["lanes_detected"]:
                        # Başarılı tespit, hata sayacını sıfırla
                        self.consecutive_errors = 0
                        
                        # Yaya geçidi tespiti kontrolü
                        crossing_detected = result.get("crossing_detected", False)
                        
                        # Şerit takibi yap
                        self.motor_controller.follow_lane(result, crossing_detected)
                    else:
                        # Şerit tespit edilemedi, hata sayacını artır
                        self.consecutive_errors += 1
                        logger.warning(f"Şerit tespit edilemedi. Hata sayısı: {self.consecutive_errors}/{self.max_consecutive_errors}")
                        
                        # Hata sayısı limiti aştıysa
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            # Aracı durdur
                            logger.error("Maksimum hata sayısına ulaşıldı. Araç durduruluyor.")
                            self.motor_controller.stop()
                            
                            # Soğuma süresi
                            current_time = time.time()
                            if current_time - self.last_error_time > self.error_cooldown:
                                logger.info(f"Sistem {self.error_cooldown} saniye bekledikten sonra tekrar başlatılacak.")
                                time.sleep(self.error_cooldown)
                                self.consecutive_errors = 0
                                self.last_error_time = current_time
                    
                    # Debug modunda görüntüleri göster
                    if self.debug:
                        cv2.imshow("Lane Detection", result["visualization"])
                        
                        # ESC tuşu ile çık
                        if cv2.waitKey(1) == 27:
                            break
                    
                    # Döngü hızını kontrol et
                    loop_time = time.time() - loop_start_time
                    if loop_time < 0.03:  # Hedef: ~30 FPS
                        sleep_time = 0.03 - loop_time
                        time.sleep(sleep_time)
                        
                except Exception as e:
                    logger.error(f"İşleme hatası: {str(e)}")
                    self.consecutive_errors += 1
                    
                    # Hata sayısı limiti aştıysa
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logger.error("Maksimum hata sayısına ulaşıldı.")
                        self.motor_controller.stop()
                        time.sleep(self.error_cooldown)
                        
                        # Kamerayı yeniden başlatmayı dene
                        logger.info("Kamera yeniden başlatılıyor...")
                        self.camera.stop()
                        time.sleep(1.0)
                        if not self._initialize_camera():
                            logger.error("Kamera yeniden başlatılamadı. Program sonlandırılıyor.")
                            break
                        
                        self.consecutive_errors = 0
        finally:
            self.cleanup()
            logger.info("Program sonlandırıldı")

def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    Çözünürlük stringini ayrıştırır (örn: '320x240' -> (320, 240))
    
    Args:
        resolution_str: Çözünürlük string formatında (WIDTHxHEIGHT)
        
    Returns:
        Tuple[int, int]: (width, height) çözünürlük
    """
    try:
        width, height = resolution_str.lower().split('x')
        return (int(width), int(height))
    except:
        logger.error(f"Geçersiz çözünürlük formatı: {resolution_str}, varsayılan değer kullanılıyor.")
        return (320, 240)

def main():
    """Ana fonksiyon"""
    # Argüman ayrıştırıcı
    parser = argparse.ArgumentParser(description='Otonom Araç Kontrol Programı')
    parser.add_argument('--debug', action='store_true', help='Debug modu (görüntüleri göster)')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (WIDTHxHEIGHT formatında)')
    args = parser.parse_args()
    
    # Çözünürlüğü ayrıştır
    resolution = parse_resolution(args.resolution)
    
    # Argümanları sözlüğe dönüştür
    args_dict = {
        'debug': args.debug,
        'resolution': resolution
    }
    
    # Otonom aracı başlat
    controller = OtonomArac(args_dict)
    controller.start()

if __name__ == "__main__":
    main() 