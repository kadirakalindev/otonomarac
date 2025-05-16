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
        Kamera modülünü başlatır ve yapılandırır
        """
        try:
            logger.info("Kamera başlatılıyor...")
            self.camera = Picamera2()
            
            # Kamera yapılandırması
            self.camera_config = self.camera.create_preview_configuration(
                main={"size": self.camera_resolution, "format": "RGB888"},
                controls={"FrameRate": self.framerate}
            )
            self.camera.configure(self.camera_config)
            
            # Kamera özel ayarlarını düzenleme (daha iyi şerit tespiti için)
            controls = {
                "AwbEnable": True,          # Otomatik beyaz dengesi
                "AeEnable": True,           # Otomatik pozlama
                "AwbMode": 0,               # Auto (otomatik beyaz dengesi modu)
                "ExposureTime": 20000,      # Pozlama süresi - düşük değer = daha az bulanıklık
                "Sharpness": 2.0,           # Keskinlik - netliği artır
                "Contrast": 1.2             # Kontrast - şeritleri daha belirgin hale getir
            }
            self.camera.set_controls(controls)
            
            logger.info("Kamera yapılandırması tamamlandı.")
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
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
        
        if self.camera is None:
            try:
                self._initialize_camera()
            except Exception as e:
                logger.error(f"Kamera başlatılamadı: {e}")
                self.cleanup()
                return
        
        # Kamerayı başlat
        try:
            self.camera.start()
        except Exception as e:
            logger.error(f"Kamera akışı başlatılamadı: {e}")
            self.cleanup()
            return
        
        try:
            # Çalışma başlangıcında biraz beklet (kameranın dengelenmesi için)
            logger.info("Kamera dengeleniyor...")
            time.sleep(1.5)
            
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
            
            # Hata sayacı (sürekli hata durumunu takip etmek için)
            error_count = 0
            max_errors = 10  # Maksimum kabul edilebilir ardışık hata sayısı
            
            # Ana döngü
            while self.running:
                try:
                    # Kameradan görüntü al
                    frame = self.camera.capture_array()
                    
                    # Görüntüyü işle ve şeritleri tespit et
                    processed_frame, center_diff = self.lane_detector.process_frame(frame)
                    
                    # Şeritlere göre motoru kontrol et
                    self.motor_controller.follow_lane(center_diff)
                    
                    # Hata sayacını sıfırla (başarıyla işlendi)
                    error_count = 0
                    
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
                            
                            # Tuş basımlarını kontrol et
                            key = cv2.waitKey(1) & 0xFF
                            
                            # q tuşuna basılırsa çık
                            if key == ord('q'):
                                logger.info("Kullanıcı tarafından durduruldu")
                                break
                                
                            # s tuşuna basılırsa görüntüyü kaydet
                            elif key == ord('s'):
                                filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                cv2.imwrite(filename, processed_frame)
                                logger.info(f"Ekran görüntüsü kaydedildi: {filename}")
                                
                except KeyboardInterrupt:
                    logger.info("Kullanıcı tarafından durduruldu (CTRL+C)")
                    break
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"İşleme hatası ({error_count}/{max_errors}): {e}")
                    
                    # Çok sayıda ardışık hata olursa durumu yeniden başlat
                    if error_count >= max_errors:
                        logger.critical(f"Çok fazla ardışık hata! Program yeniden başlatılıyor.")
                        self.stop()
                        self.cleanup()
                        # Kamerayı yeniden başlat
                        try:
                            self._initialize_camera()
                            self.camera.start()
                            time.sleep(1)  # Kameranın dengelenmesi için bekle
                        except:
                            logger.critical("Kamera yeniden başlatılamadı! Çıkılıyor.")
                            break
                
        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
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
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
        
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