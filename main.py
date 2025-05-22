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
import numpy as np

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
                 calibration_file="serit_kalibrasyon.json"):  # Kalibrasyon dosyası yolu
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
                self._load_calibration()
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
        Kamera başlatma ve ayarlaması
        """
        try:
            resolution = self.camera_resolution
            camera_config = {
                'resolution': resolution,
                'framerate': 25,
            }
            
            if self.camera_type == 'picamera2':
                from picamera2 import Picamera2
                
                self.camera = Picamera2()
                
                # Kamera yapılandırması
                config = self.camera.create_still_configuration(
                    main={"size": resolution, "format": "RGB888"}, 
                    lores={"size": (640, 480), "format": "YUV420"}, 
                    display="lores",
                )
                self.camera.configure(config)
                
                # Kamera parametrelerini iyileştir
                if hasattr(self.camera, 'set_controls'):
                    try:
                        self.camera.set_controls({
                            "ExposureTime": 15000,  # Pozlama süresini artır
                            "AnalogueGain": 1.5,   # Analog kazancı artır
                            "Brightness": 0.5,     # Parlaklığı artır
                            "Contrast": 1.2,       # Kontrastı artır
                            "Sharpness": 2.5,      # Keskinliği artır
                            "Saturation": 1.3,     # Doygunluğu artır
                            "NoiseReductionMode": 2, # Gürültü azaltma
                        })
                    except Exception as e:
                        logger.warning(f"Kamera parametreleri ayarlanamadı: {e}")
                
                self.camera.start()
                
                # Kameranın dengelenmesi için bekle
                logger.info("Kamera dengeleniyor...")
                time.sleep(3)  # Dengeleme için bekleme süresini artır
                
                logger.info(f"PiCamera2 başlatıldı: {resolution}")
            
            elif self.camera_type == 'picamera':
                import picamera
                
                self.camera = picamera.PiCamera()
                self.camera.resolution = resolution
                self.camera.framerate = camera_config['framerate']
                
                # Kamera parametreleri
                self.camera.brightness = 55      # Parlaklığı artır (0-100)
                self.camera.contrast = 60        # Kontrastı artır (0-100)
                self.camera.sharpness = 70       # Keskinliği artır (0-100)
                self.camera.saturation = 30      # Doygunluğu artır (0-100)
                self.camera.exposure_mode = 'auto'
                self.camera.awb_mode = 'auto'
                
                # Kameranın dengelenmesi için bekle
                logger.info("Kamera dengeleniyor...")
                time.sleep(3)  # Dengeleme için bekleme süresini artır
                
                logger.info(f"PiCamera başlatıldı: {resolution}")
            
            elif self.camera_type == 'opencv':
                import cv2
                
                # OpenCV kamerası
                self.camera = cv2.VideoCapture(self.camera_index)
                
                # Ayarlar
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, camera_config['framerate'])
                
                # Kamera görüntü ayarlarını iyileştir
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Parlaklığı artır (0-1)
                self.camera.set(cv2.CAP_PROP_CONTRAST, 0.6)    # Kontrastı artır (0-1)
                self.camera.set(cv2.CAP_PROP_SATURATION, 0.7)  # Doygunluğu artır (0-1)
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Pozlama ayarı
                
                # Kameranın dengelenmesi için bekle
                logger.info("Kamera dengeleniyor...")
                time.sleep(3)  # Dengeleme için bekleme süresini artır
                
                if not self.camera.isOpened():
                    raise RuntimeError("Kamera açılamadı")
                
                logger.info(f"OpenCV kamerası başlatıldı: {resolution}")
            
            else:
                raise ValueError(f"Desteklenmeyen kamera tipi: {self.camera_type}")
                
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
        
        # Hata sayacı ve yeniden başlatma değişkenleri
        error_count = 0
        max_consecutive_errors = 5
        restart_delay = 3.0  # saniye
        last_error_time = 0
        
        try:
            # Kamera başlatma ve yapılandırma
            if self.camera is None:
                self._initialize_camera()
            
            # FPS hesaplama değişkenlerini başlat
            self.fps_start_time = time.time()
            self.frame_count = 0
            
            # Debug modu için pencere oluştur
            if self.debug:
                cv2.namedWindow("Otonom Arac", cv2.WINDOW_NORMAL)
                cv2.setWindowTitle("Otonom Arac", "Otonom Arac - Serit Tespiti")
                cv2.resizeWindow("Otonom Arac", self.camera_resolution[0], self.camera_resolution[1])
            
            # Debug modunda son frame update zamanı
            last_debug_update = 0
            debug_frame_interval = 1.0 / self.debug_fps if self.debug_fps > 0 else 0
            
            # Görüntü işleme döngüsü
            while self.running:
                try:
                    # Kameradan görüntü al
                    frame = self.camera.capture_array()
                    if frame is None or frame.size == 0:
                        raise Exception("Geçersiz kamera görüntüsü")
                    
                    # BGR'ye dönüştür (OpenCV için)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
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
                    
                    # Debug modunda görüntüyü göster
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
                            
                            # Tuş kontrolü
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                logger.info("Kullanıcı tarafından durduruldu")
                                self.running = False
                                break
                            elif key == ord('s'):
                                filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                cv2.imwrite(filename, processed_frame)
                                logger.info(f"Ekran görüntüsü kaydedildi: {filename}")
                    
                except KeyboardInterrupt:
                    logger.info("Kullanıcı tarafından durduruldu (CTRL+C)")
                    self.running = False
                    break
                    
                except Exception as e:
                    error_count += 1
                    current_time = time.time()
                    
                    # Son hata üzerinden yeterli süre geçtiyse sayacı sıfırla
                    if current_time - last_error_time > 10.0:
                        error_count = 0
                    
                    last_error_time = current_time
                    logger.error(f"Kare işleme hatası ({error_count}/{max_consecutive_errors}): {e}")
                    
                    if error_count >= max_consecutive_errors:
                        logger.critical("Çok fazla ardışık hata! Yeniden başlatılıyor...")
                        self.cleanup()
                        time.sleep(restart_delay)
                        error_count = 0
                        continue
        
        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
        
        finally:
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
                self.camera.stop()
            except:
                pass
        
        # OpenCV pencerelerini kapat
        if self.debug:
            cv2.destroyAllWindows()
        
        logger.info("Temizleme tamamlandı.")

    def _load_calibration(self):
        """
        Kalibrasyon dosyasını yükler ve şerit tespiti için gerekli parametreleri ayarlar.
        kalibrasyon_optimize.py tarafından oluşturulan formata uygun olarak çalışır.
        """
        try:
            import json
            with open(self.calibration_file, 'r') as f:
                calibration = json.load(f)
            
            # kalibrasyon_optimize.py formatı kontrolü
            if 'src_points' in calibration and 'dst_points' in calibration:
                logger.info("kalibrasyon_optimize.py formatında kalibrasyon dosyası tespit edildi.")
                
                # Çözünürlük kontrolü
                if 'resolution' in calibration:
                    cal_width = calibration['resolution'].get('width', self.camera_resolution[0])
                    cal_height = calibration['resolution'].get('height', self.camera_resolution[1])
                    
                    # Çözünürlük uyumsuzluğu kontrolü
                    if cal_width != self.camera_resolution[0] or cal_height != self.camera_resolution[1]:
                        logger.warning(f"Kalibrasyon dosyası çözünürlüğü ({cal_width}x{cal_height}) " +
                                      f"kamera çözünürlüğü ({self.camera_resolution[0]}x{self.camera_resolution[1]}) ile uyumsuz.")
                        logger.warning("Kalibrasyon noktaları ölçeklendirilecek.")
                        
                        # Ölçeklendirme faktörleri
                        scale_x = self.camera_resolution[0] / cal_width
                        scale_y = self.camera_resolution[1] / cal_height
                        
                        # Noktaları ölçeklendir
                        src_points = calibration['src_points']
                        for i in range(len(src_points)):
                            src_points[i][0] *= scale_x
                            src_points[i][1] *= scale_y
                        
                        dst_points = calibration['dst_points']
                        for i in range(len(dst_points)):
                            dst_points[i][0] *= scale_x
                            dst_points[i][1] *= scale_y
                    else:
                        src_points = calibration['src_points']
                        dst_points = calibration['dst_points']
                else:
                    src_points = calibration['src_points']
                    dst_points = calibration['dst_points']
                
                # Şerit tespiti için ROI oluştur
                roi_vertices = np.array([
                    src_points[2],  # Sol alt
                    src_points[0],  # Sol üst
                    src_points[1],  # Sağ üst
                    src_points[3]   # Sağ alt
                ], dtype=np.int32)
                
                # Orta şerit çizgisi oluştur
                center_line = np.array([
                    [(src_points[2][0] + src_points[3][0]) // 2, self.camera_resolution[1]],  # Alt orta nokta
                    [(src_points[0][0] + src_points[1][0]) // 2, (src_points[0][1] + src_points[1][1]) // 2]  # Üst orta nokta
                ], dtype=np.int32)
                
                # LaneDetector'a parametreleri ayarla
                self.lane_detector.roi_vertices = roi_vertices
                self.lane_detector.center_line = center_line
                
                # Diğer parametreleri ayarla (varsayılan değerlerle)
                self.lane_detector.canny_low = calibration.get('canny_low_threshold', 30)
                self.lane_detector.canny_high = calibration.get('canny_high_threshold', 120)
                self.lane_detector.blur_kernel = calibration.get('blur_kernel_size', 5)
                self.lane_detector.min_line_length = calibration.get('min_line_length', 15)
                self.lane_detector.max_line_gap = calibration.get('max_line_gap', 40)
                
                logger.info("Kalibrasyon parametreleri başarıyla yüklendi.")
                return True
            else:
                # Eski format kalibrasyon dosyası, doğrudan LaneDetector'a yükle
                logger.info("Eski format kalibrasyon dosyası tespit edildi.")
                return self.lane_detector.load_calibration(self.calibration_file)
                
        except Exception as e:
            logger.error(f"Kalibrasyon dosyası yüklenirken hata: {e}")
            return False

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
    parser.add_argument('--calibration', default='serit_kalibrasyon.json', help='Kalibrasyon dosyası yolu')
    
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