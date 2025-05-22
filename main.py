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

class AutonomousVehicle:
    def __init__(self, camera_resolution=(320, 240), camera_framerate=30, calibration_file="serit_kalibrasyon.json", debug=False):
        """
        Ana kontrol sınıfı - araç davranışını yönetir
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            camera_framerate (int): Kamera FPS değeri
            calibration_file (str): Kalibrasyon dosyası yolu
            debug (bool): Debug modu açık/kapalı
        """
        self.camera_resolution = camera_resolution
        self.camera_framerate = camera_framerate
        self.calibration_file = calibration_file
        self.debug = debug
        
        # Sistem durumu
        self.running = False
        self.mode = "lane_following"  # lane_following, manual, obstacle_avoidance
        
        # Bileşenleri başlat
        self.camera = None
        self.lane_detector = None
        self.motor_controller = None
        
        # Hata yönetimi
        self.error_counter = 0
        self.max_consecutive_errors = 5
        self.restart_delay = 3  # saniye
        self.last_error_time = 0
        self.last_frame_time = 0
        self.frame_timeout = 1.0  # saniye
        
        # Performans metrikler
        self.fps = 0
        self.process_time = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # Viraj durumu
        self.in_curve = False
        self.curve_direction = "none"
        
        # Shutting down 
        self.is_shutting_down = False
        
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _initialize_components(self):
        """Bileşenleri başlatır"""
        try:
            # Lane detector'ı başlat
            self.lane_detector = LaneDetector(camera_resolution=self.camera_resolution, debug=self.debug)
            
            # Kalibrasyon dosyasını yükle
            self._load_calibration()
            
            # Motor kontrolcüsünü başlat
            self.motor_controller = MotorController(debug=self.debug)
            
            return True
            
        except Exception as e:
            logging.error(f"Bileşenleri başlatırken hata: {e}")
            return False
            
    def _initialize_camera(self):
        """Kamera bağlantısını başlatır"""
        try:
            from picamera2 import Picamera2
            
            # Kamera nesnesini oluştur
            self.camera = Picamera2()
            
            # Kamera konfigürasyonu
            config = self.camera.create_video_configuration(
                main={"size": self.camera_resolution, "format": "RGB888"},
                controls={
                    "FrameRate": self.camera_framerate,
                    "ExposureTime": 15000,  # Daha kısa pozlama süresi (mikrosaniye)
                    "AnalogueGain": 1.5,    # Işık hassasiyeti
                    "Sharpness": 2.5,       # Keskinlik arttırıldı
                    "Brightness": 0.1,      # Parlaklık biraz arttırıldı
                    "Contrast": 1.2,        # Kontrast arttırıldı
                    "Saturation": 1.3,      # Doygunluk arttırıldı
                    "NoiseReductionMode": 2  # Gürültü azaltma - Fast
                }
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Kameranın ısınması için bekle
            time.sleep(0.5)
            
            # Test karesi al
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                raise Exception("Kamera test karesi alınamadı")
                
            logging.info(f"Kamera başlatıldı: {self.camera_resolution[0]}x{self.camera_resolution[1]}@{self.camera_framerate}fps")
            return True
            
        except ImportError:
            logging.error("PiCamera2 kütüphanesi bulunamadı - kamera devre dışı")
            return False
            
        except Exception as e:
            logging.error(f"Kamera başlatma hatası: {e}")
            return False
            
    def _load_calibration(self):
        """Kalibrasyon dosyasını yükler"""
        try:
            import json
            import numpy as np
            import os
            
            # Kalibrasyon dosyası var mı kontrol et
            if not os.path.exists(self.calibration_file):
                logging.warning(f"Kalibrasyon dosyası bulunamadı: {self.calibration_file}")
                return False
            
            # Kalibrasyon dosyasını oku
            with open(self.calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            # Kalibrasyon_optimize.py formatını kontrol et
            if 'src_points' in calib_data and 'dst_points' in calib_data:
                logging.info("kalibrasyon_optimize.py formatında kalibrasyon dosyası yüklendi")
                
                # Kalibrasyon verisini doğrudan lane_detector'a yükle
                if self.lane_detector.load_calibration(self.calibration_file):
                    logging.info(f"Kalibrasyon dosyası başarıyla yüklendi: {self.calibration_file}")
                    
                    # Kalibrasyon çözünürlüğünü kontrol et
                    if 'resolution' in calib_data:
                        calib_res = calib_data['resolution']
                        if calib_res != self.camera_resolution:
                            logging.warning(f"Kalibrasyon çözünürlüğü ({calib_res[0]}x{calib_res[1]}) ve " +
                                         f"kamera çözünürlüğü ({self.camera_resolution[0]}x{self.camera_resolution[1]}) farklı!")
                    return True
            else:
                # Eski format dosya - lane_detector.load_calibration kullan
                if self.lane_detector.load_calibration(self.calibration_file):
                    logging.info(f"Eski format kalibrasyon dosyası başarıyla yüklendi: {self.calibration_file}")
                    return True
                else:
                    logging.error(f"Eski format kalibrasyon dosyası yüklenemedi: {self.calibration_file}")
            
            return False
            
        except Exception as e:
            logging.error(f"Kalibrasyon yükleme hatası: {e}")
            return False
            
    def process_frame(self, frame):
        """Bir kareyi işler ve motor komutlarını günceller"""
        try:
            start_time = time.time()
            
            # Şerit tespiti
            processed_frame, center_diff = self.lane_detector.process_frame(frame)
            
            if self.debug:
                # FPS hesapla
                self.frame_count += 1
                if time.time() - self.last_fps_update >= 1.0:  # Her 1 saniyede bir FPS güncelle
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_update = time.time()
                
                # FPS göster
                cv2.putText(processed_frame, f"FPS: {self.fps}",
                          (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # İşleme süresi
                process_time = time.time() - start_time
                cv2.putText(processed_frame, f"Process: {process_time*1000:.1f}ms",
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Motor kontrolü - moduna göre
            if self.mode == "lane_following":
                # Lane detector'dan viraj durumunu oku
                self.in_curve = self.lane_detector.is_curve
                self.curve_direction = self.lane_detector.curve_direction
                
                # Şerit takibi yap - viraj tespiti ile birlikte
                left_speed, right_speed = self.motor_controller.follow_lane(
                    center_diff, self.camera_resolution[0], self.lane_detector)
                
                # Motorları kontrol et
                self.motor_controller.set_motors(left_speed, right_speed)
                
            elif self.mode == "manual":
                # Manuel kontrol - burada bir şey yapmıyoruz, dışarıdan kontrol ediliyor
                pass
                
            elif self.mode == "obstacle_avoidance":
                # Engel tespit ve kaçınma modu - şimdilik eklenmedi
                pass
                
            # İşlem zamanını kaydet
            self.process_time = time.time() - start_time
            
            return processed_frame
            
        except Exception as e:
            logging.error(f"Kare işleme hatası: {e}")
            self.error_counter += 1
            return None
            
    def start(self):
        """Sistemi başlatır"""
        if self.running:
            logging.warning("Sistem zaten çalışıyor")
            return False
            
        logging.info("Sistem başlatılıyor...")
        
        # Bileşenleri başlat
        while True:
            try:
                if not self._initialize_components():
                    logging.error("Bileşenler başlatılamadı, yeniden deneniyor...")
                    time.sleep(1)
                    continue
                
                if not self._initialize_camera():
                    logging.error("Kamera başlatılamadı, yeniden deneniyor...")
                    time.sleep(1)
                    continue
                
                break  # Başarıyla başlatıldıysa döngüden çık
                
            except Exception as e:
                logging.error(f"Başlatma hatası: {e}")
                time.sleep(1)  # Biraz bekle ve yeniden dene
        
        self.running = True
        self.error_counter = 0
        self.is_shutting_down = False
        
        # Ana döngü
        try:
            logging.info("Sistem başlatıldı, ana döngü başlıyor...")
            
            while self.running and not self.is_shutting_down:
                try:
                    frame_start_time = time.time()
                    
                    # Kamera karesi al
                    if self.camera is None:
                        raise Exception("Kamera başlatılmadı")
                        
                    frame = self.camera.capture_array()
                    
                    if frame is None or frame.size == 0:
                        raise Exception("Boş kare alındı")
                        
                    # Kareyi işle
                    processed_frame = self.process_frame(frame)
                    
                    # Hata sayacını sıfırla veya artır
                    if processed_frame is not None:
                        self.error_counter = max(0, self.error_counter - 1)  # Hatalar varsa kademeli olarak azalt
                    else:
                        self.error_counter += 1
                        logging.warning(f"İşleme hatası: {self.error_counter}/{self.max_consecutive_errors}")
                        
                    # Hata kontrolleri
                    if self.error_counter > self.max_consecutive_errors:
                        logging.error(f"Maksimum ardışık hata sayısına ulaşıldı ({self.max_consecutive_errors}), yeniden başlatılıyor...")
                        self._handle_critical_error()
                        break
                        
                    # Uzun süre kare alınamazsa hataya düş
                    if time.time() - frame_start_time > self.frame_timeout:
                        logging.error(f"Kare işleme zaman aşımı: {time.time() - frame_start_time:.2f}s")
                        self.error_counter += 1
                    
                    # Debug modunda görüntüyü göster
                    if self.debug and processed_frame is not None:
                        cv2.imshow("Lane Detection", processed_frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        # Klavye kontrolü (ESC=çıkış)
                        if key == 27:  # ESC
                            logging.info("Kullanıcı tarafından durduruldu (ESC)")
                            self.running = False
                            break
                            
                except KeyboardInterrupt:
                    logging.info("Klavye kesintisi - durduruldu")
                    self.running = False
                    break
                    
                except Exception as e:
                    logging.error(f"Ana döngüde beklenmeyen hata: {e}")
                    self.error_counter += 1
                    time.sleep(0.5)  # Hata sonrası kısa bir bekleme
                
        except Exception as e:
            logging.error(f"Ana döngüde kritik hata: {e}")
            
        finally:
            # Temizlik işlemleri
            self._cleanup()
            
        return True
        
    def _handle_critical_error(self):
        """Kritik bir hata durumunda sistemi güvenli şekilde yeniden başlatır"""
        logging.warning("Kritik hata işleniyor - sistemi yeniden başlatma...")
        
        try:
            # Motorları durdur
            if self.motor_controller:
                self.motor_controller.stop()
                
            # Kamerayı kapat
            if self.camera:
                try:
                    self.camera.stop()
                    self.camera.close()
                except:
                    pass
                self.camera = None
                
            # Bekleme süresi
            logging.info(f"{self.restart_delay} saniye bekleniyor...")
            time.sleep(self.restart_delay)
            
            # Sayaçları sıfırla
            self.error_counter = 0
            
            # Yeniden başlat
            logging.info("Sistem yeniden başlatılıyor...")
            self._initialize_components()
            self._initialize_camera()
            
        except Exception as e:
            logging.error(f"Yeniden başlatma hatası: {e}")
            self.running = False
            
    def stop(self):
        """Sistemi durdurur"""
        logging.info("Sistem durduruluyor...")
        self.running = False
        self.is_shutting_down = True
        
    def _cleanup(self):
        """Kaynakları temizler"""
        logging.info("Sistem kaynakları temizleniyor...")
        
        # Motorları durdur
        try:
            if self.motor_controller:
                self.motor_controller.stop()
                self.motor_controller.cleanup()
                self.motor_controller = None
        except Exception as e:
            logging.error(f"Motor temizleme hatası: {e}")
        
        # Kamerayı kapat
        try:
            if self.camera:
                self.camera.stop()
                self.camera.close()
                self.camera = None
        except Exception as e:
            logging.error(f"Kamera temizleme hatası: {e}")
        
        # OpenCV pencerelerini kapat
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        logging.info("Sistem durduruldu")

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Otonom Araç Kontrol Sistemi")
    parser.add_argument("--debug", action="store_true", help="Debug modu")
    parser.add_argument("--width", type=int, default=320, help="Kamera genişliği")
    parser.add_argument("--height", type=int, default=240, help="Kamera yüksekliği")
    parser.add_argument("--fps", type=int, default=30, help="Kamera FPS")
    parser.add_argument("--calibration", type=str, default="serit_kalibrasyon.json", help="Kalibrasyon dosya yolu")
    
    return parser.parse_args()

def main():
    """Ana fonksiyon"""
    # Argümanları işle
    args = parse_arguments()
    
    # Otonom aracı başlat
    vehicle = AutonomousVehicle(
        camera_resolution=(args.width, args.height),
        camera_framerate=args.fps,
        calibration_file=args.calibration,
        debug=args.debug
    )
    
    try:
        # CTRL+C sinyalini yakala
        import signal
        def signal_handler(sig, frame):
            print("\nCtrl+C algılandı, program durduruluyor...")
            vehicle.stop()
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Sistemi başlat
        vehicle.start()
        
    except Exception as e:
        logging.error(f"Ana programda hata: {e}")
        
    finally:
        # Program sonlandırıldığında temizlik yap
        try:
            vehicle.stop()
        except:
            pass

if __name__ == "__main__":
    main() 