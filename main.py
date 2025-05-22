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
                 calibration_file="calibration.json",
                 use_picamera=True):          # Picamera kullan (False: USB kamera)
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
            use_picamera (bool): Picamera kullan (True) veya USB kamera kullan (False)
        """
        self.debug = debug
        self.debug_fps = debug_fps
        self.running = False
        self.camera_resolution = camera_resolution
        self.framerate = framerate
        self.camera = None  # Başlangıçta None olarak tanımla
        self.calibration_file = calibration_file  # Kalibrasyon dosyası yolunu sakla
        self.use_picamera = use_picamera  # Picamera mı yoksa USB kamera mı kullanılacak
        self.stop_requested = False  # durdurma isteği
        self.resolution = camera_resolution  # Çözünürlük için alternatif değişken
        
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
        Kamerayı başlatır ve gerekli ayarları yapar (Picamera2 veya USB kamera)
        """
        try:
            logger.info(f"Kamera başlatılıyor... (Picamera: {self.use_picamera})")
            
            # Eğer önceki kamera nesnesi varsa temizle
            if self.camera is not None:
                if self.use_picamera:
                    try:
                        self.camera.stop()
                        self.camera.close()
                    except:
                        pass
                else:
                    try:
                        self.camera.release()
                    except:
                        pass
                    
                self.camera = None
                logger.debug("Önceki kamera kaynağı kapatıldı")

            # Kamerayı seçilen tipine göre başlat
            if self.use_picamera:
                try:
                    # Picamera2 için
                    from picamera2 import Picamera2
                    
                    # Picamera2 nesnesini oluştur
                    self.camera = Picamera2()
                    
                    # Kamera yapılandırması
                    preview_config = self.camera.create_preview_configuration(
                        main={"size": self.camera_resolution, "format": "RGB888"},
                        lores={"size": (320, 240), "format": "YUV420"},
                        display="lores",
                        buffer_count=4  # Buffer sayısını artır
                    )
                    self.camera.configure(preview_config)
                    
                    # Kamera özel ayarlarını düzenleme
                    try:
                        controls = {
                            "AwbEnable": True,          # Otomatik beyaz dengesi
                            "AeEnable": True,           # Otomatik pozlama
                            "ExposureTime": 15000,      # Pozlama süresi (mikrosaniye)
                            "AnalogueGain": 1.0,        # Analog kazanç
                            "Brightness": 0.0,          # Parlaklık
                            "Contrast": 1.0,            # Kontrast
                            "Sharpness": 2.5,           # Keskinlik
                            "NoiseReductionMode": 1     # Gürültü azaltma
                        }
                        self.camera.set_controls(controls)
                        logger.info("Kamera kontrolleri ayarlandı")
                    except Exception as e:
                        logger.warning(f"Kamera kontrolleri ayarlanırken hata: {e}")
                    
                    # Kamerayı başlat
                    self.camera.start()
                    
                    # Kamera dengelenmesi için bekle
                    logger.info("Kamera dengeleniyor...")
                    time.sleep(2)
                    
                    # Test görüntüsü al
                    logger.debug("Kamera test görüntüsü alınıyor...")
                    test_frame = self.camera.capture_array()
                    
                    if test_frame is None or test_frame.size == 0:
                        logger.error("Kameradan geçersiz görüntü alındı!")
                        return False
                        
                    logger.info(f"Kamera başarıyla başlatıldı. Görüntü boyutu: {test_frame.shape}")
                    
                except ImportError:
                    logger.error("Picamera2 kütüphanesi bulunamadı! USB kamera moduna geçiliyor.")
                    self.use_picamera = False
                    return self._initialize_camera()  # USB kamera ile tekrar dene
                    
                except Exception as e:
                    logger.error(f"Picamera başlatma hatası: {e}")
                    self.use_picamera = False
                    logger.warning("USB kamera moduna geçiliyor...")
                    return self._initialize_camera()  # USB kamera ile tekrar dene
            else:
                # OpenCV VideoCapture için
                try:
                    # USB kamera başlat
                    self.camera = cv2.VideoCapture(0)
                    
                    # Kamera başarıyla açıldı mı kontrol et
                    if not self.camera.isOpened():
                        logger.error("USB kamera açılamadı!")
                        return False
                    
                    # Kamera çözünürlüğünü ayarla
                    width, height = self.camera_resolution
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Kamera parametrelerini ayarla (bazı kameralar bu parametreleri desteklemeyebilir)
                    try:
                        self.camera.set(cv2.CAP_PROP_EXPOSURE, 15000)  # Pozlama
                        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Parlaklık
                        self.camera.set(cv2.CAP_PROP_CONTRAST, 130)    # Kontrast
                        self.camera.set(cv2.CAP_PROP_SATURATION, 130)  # Doygunluk
                        self.camera.set(cv2.CAP_PROP_SHARPNESS, 2.5)   # Keskinlik
                        logger.debug("Kamera parametreleri ayarlandı")
                    except:
                        logger.warning("Bazı kamera parametreleri ayarlanamadı - bu normal olabilir")
                    
                    # Kamera dengelenmesi için bekle
                    time.sleep(2)
                    
                    # Test görüntüsü al
                    ret, frame = self.camera.read()
                    
                    if not ret or frame is None:
                        logger.error("USB kameradan test görüntüsü alınamadı!")
                        return False
                    
                    logger.info(f"USB kamera başarıyla başlatıldı. Görüntü boyutu: {frame.shape}")
                    
                except Exception as e:
                    logger.error(f"USB kamera başlatma hatası: {e}")
                    return False
                    
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
    
    def _display_debug_info(self, processed_frame, lane_info, center_diff):
        """
        Debug bilgilerini ekranda gösterir
        """
        if processed_frame is None:
            return
            
        # Şerit bilgilerini çiz
        left_lane, right_lane = lane_info
        if left_lane is not None:
            cv2.line(processed_frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 2)
            
        if right_lane is not None:
            cv2.line(processed_frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 2)
            
        # Merkez sapmasını göster
        if center_diff is not None:
            cv2.putText(processed_frame, f"Merkez Farkı: {center_diff:.1f}px", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                      
        # Görüntüyü göster
        cv2.imshow("Otonom Araç", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Tuş kontrolleri
        if key == 27:  # ESC tuşu
            logger.info("Kullanıcı tarafından durduruldu (ESC)")
            self.stop_requested = True
            
    def _restart_program(self):
        """
        Programı yeniden başlat
        """
        logger.warning("Program yeniden başlatılıyor...")
        self._cleanup()
        # Python programını yeniden başlat
        python = sys.executable
        os.execl(python, python, *sys.argv)
        
    def start(self):
        """
        Araç kontrol döngüsünü başlat
        """
        logger.info("Araç kontrol sistemi başlatılıyor...")
        
        # Hata sayacı
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Ana kontrol döngüsü
        while not self.stop_requested:
            try:
                # Kamera kontrolü ve başlatma
                if self.camera is None or (not self.use_picamera and not self.camera.isOpened()):
                    logger.warning("Kamera bağlantısı yok, yeniden başlatılıyor...")
                    if not self._initialize_camera():
                        logger.error("Kamera başlatılamadı, 3 saniye bekleniyor...")
                        time.sleep(3.0)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                            self._restart_program()
                        continue
                
                # Görüntü yakala
                frame = None
                if self.use_picamera:
                    try:
                        frame = self.camera.capture_array()
                        # RGB'den BGR'ye dönüştür (OpenCV için)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        logger.error(f"Picamera görüntü yakalama hatası: {e}")
                        frame = None
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        frame = None
                
                if frame is None:
                    logger.error("Kameradan görüntü alınamadı!")
                    self.motor_controller.stop()
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                        self._restart_program()
                    continue
                
                # Görüntü işleme
                processed_frame, lane_info = self.lane_detector.process_frame(frame)
                
                # Şerit takibi için merkez sapması hesapla
                center_diff = self._calculate_center_diff(lane_info)
                
                # Motor kontrolü
                if center_diff is not None:
                    self.motor_controller.follow_lane(center_diff, self.motor_controller.default_speed)
                    consecutive_errors = max(0, consecutive_errors - 1)  # Başarılı işlem, hata sayacını azalt
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
                time.sleep(0.02)  # 20ms bekleme (50 FPS hedefi)
                
            except KeyboardInterrupt:
                logger.info("Klavye kesintisi, program durduruluyor...")
                break
            except Exception as e:
                logger.error(f"İşlem hatası: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"{max_consecutive_errors} ardışık hata! Program yeniden başlatılıyor.")
                    self._restart_program()
        
        # Kaynakları temizle
        self._cleanup()
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
                if self.use_picamera:
                    self.camera.stop()
                    self.camera.close()
                else:
                    self.camera.release()
            except Exception as e:
                logger.error(f"Kamera kapatma hatası: {e}")
        
        # OpenCV pencerelerini kapat
        if self.debug:
            cv2.destroyAllWindows()
        
        logger.info("Temizleme tamamlandı.")
        
    def _cleanup(self):
        """
        Kaynakları temizle (alias)
        """
        self.cleanup()

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
    
    # Kamera tipi seçimi için argüman ekle
    parser.add_argument('--usb-camera', action='store_true', help='USB kamera kullan (varsayılan: Picamera)')
    
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
            calibration_file=args.calibration,
            use_picamera=not args.usb_camera  # USB kamera parametresi ekle
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