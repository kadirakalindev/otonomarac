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
import gc
import numpy as np
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
                 calibration_file="calibration.json",  # Kalibrasyon dosyası yolu
                 performance_mode="balanced"):  # Performans modu: "speed", "balanced" veya "quality"
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
            performance_mode (str): Performans modu - hız/kalite dengesi
        """
        self.debug = debug
        self.debug_fps = debug_fps
        self.running = False
        self.camera_resolution = camera_resolution
        self.framerate = framerate
        self.camera = None  # Başlangıçta None olarak tanımla
        self.calibration_file = calibration_file  # Kalibrasyon dosyası yolunu sakla
        self.performance_mode = performance_mode  # Performans modunu sakla
        
        # Performans izleme
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = 0
        self.processing_times = []
        self.max_process_time = 0
        self.skipped_frames = 0
        
        # Bellek yönetimi - önceden ayrılan tamponlar
        self.frame_buffer = None  # Kamera karesi tamponu
        self.resize_buffer = None  # Yeniden boyutlandırma tamponu
        
        # Modülleri başlatma denemesi - hata durumunda güvenli kapatma
        try:
            # Kamera başlatma - güvenli başlatma için try-except kullan
            self._initialize_camera()
            
            # Şerit tespit modülünü başlat - performans moduna göre
            logger.info(f"Şerit tespit modülü başlatılıyor (mod: {performance_mode})...")
            
            # Performans moduna göre çözünürlük seçimi
            lane_detection_resolution = self._get_lane_detection_resolution()
            
            self.lane_detector = LaneDetector(camera_resolution=lane_detection_resolution, debug=debug)
            
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
                max_speed=0.7,
                default_speed=0.35,
                use_board_pins=use_board_pins,
                pwm_frequency=25  # PWM frekansını düşürdük - aşırı ısınmayı azaltmak için
            )
        
        except Exception as e:
            logger.error(f"Başlatma hatası: {e}")
            # Kısmi başlatılmış kaynakları temizle
            self.cleanup()
            raise
            
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # RAM kullanımını optimize et
        gc.collect()
        
        logger.info("Otonom araç başlatıldı.")
    
    def _get_lane_detection_resolution(self):
        """
        Performans moduna göre şerit tespiti için çözünürlük belirler
        
        Returns:
            tuple: Şerit tespiti için çözünürlük (genişlik, yükseklik)
        """
        width, height = self.camera_resolution
        
        if self.performance_mode == "speed":
            # Hız odaklı: Düşük çözünürlük
            return (width // 2, height // 2)
        
        elif self.performance_mode == "quality":
            # Kalite odaklı: Yüksek çözünürlük
            return self.camera_resolution
        
        else:  # "balanced" veya diğer değerler
            # Dengeli mod: Orta çözünürlük
            return (width * 3 // 4, height * 3 // 4)
    
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
            
            # Tamponları önceden oluştur
            self._allocate_buffers()
            
            logger.info("Kamera yapılandırması tamamlandı.")
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
    def _allocate_buffers(self):
        """
        Performansı artırmak için görüntü tamponlarını önceden oluşturur
        """
        # Kamera çözünürlüğünde boş bir görüntü tamponu
        self.frame_buffer = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), 
                                    dtype=np.uint8)
        
        # Şerit tespiti çözünürlüğünde boş bir görüntü tamponu
        lane_detection_resolution = self._get_lane_detection_resolution()
        self.resize_buffer = np.zeros((lane_detection_resolution[1], lane_detection_resolution[0], 3), 
                                     dtype=np.uint8)
        
        logger.debug(f"Görüntü tamponları ayrıldı - Kamera: {self.frame_buffer.shape}, "
                   f"Şerit Tespiti: {self.resize_buffer.shape}")
    
    def signal_handler(self, sig, frame):
        """
        Sinyal yakalama işleyicisi (CTRL+C gibi)
        """
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.cleanup()
        sys.exit(0)
    
    def _process_frame(self, frame):
        """
        Bir kareyi işler ve sonuçları döndürür.
        Optimizasyonlar içerir.
        
        Args:
            frame (numpy.ndarray): İşlenecek kare
            
        Returns:
            tuple: (İşlenmiş kare, merkez sapması, işlem süresi)
        """
        start_time = time.time()
        
        # Performans moduna göre görüntüyü yeniden boyutlandır
        lane_detection_resolution = self._get_lane_detection_resolution()
        
        # Eğer görüntü zaten doğru boyutta değilse
        if frame.shape[1] != lane_detection_resolution[0] or frame.shape[0] != lane_detection_resolution[1]:
            # Önceden ayrılmış tamponu kullan (yeni bellek ayırmayı önler)
            resized = cv2.resize(frame, lane_detection_resolution, dst=self.resize_buffer)
        else:
            resized = frame
            
        # Görüntüyü işle ve merkezi hesapla
        processed_frame, center_diff = self.lane_detector.process_frame(resized)
        
        # İşlem süresini hesapla
        process_time = (time.time() - start_time) * 1000  # milisaniye
        
        # İşlem istatistiklerini güncelle
        self.processing_times.append(process_time)
        if len(self.processing_times) > 30:  # Son 30 kareyi tut
            self.processing_times.pop(0)
        
        # Maksimum işlem süresini güncelle
        if process_time > self.max_process_time:
            self.max_process_time = process_time
        
        # Debug modunda görüntüye istatistikler ekle
        if self.debug:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            # Görüntüye metin ekle
            text_color = (0, 255, 0)  # Yeşil
            cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(processed_frame, f"İşlem: {process_time:.1f}ms (Ort: {avg_time:.1f}ms)", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            if self.skipped_frames > 0:
                cv2.putText(processed_frame, f"Atlanan: {self.skipped_frames}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if center_diff is not None:
                # Merkez farkına göre renk değiştirme
                if abs(center_diff) < 20:
                    color = (0, 255, 0)  # İyi: yeşil
                elif abs(center_diff) < 50:
                    color = (0, 255, 255)  # Orta: sarı
                else:
                    color = (0, 0, 255)  # Kötü: kırmızı
                
                cv2.putText(processed_frame, f"Merkez Farkı: {center_diff}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return processed_frame, center_diff, process_time
    
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
            self.skipped_frames = 0
            
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
            
            # Kare atlama kontrolü için değişkenler
            frame_skip_threshold = 100  # Bu milisaniyeden uzun sürerse kare atlama düşün
            skip_frame_count = 0  # Art arda kaç kare atlanacak
            
            # Ana döngü
            while self.running:
                try:
                    # Kameradan görüntü al
                    frame = self.camera.capture_array(buffer=self.frame_buffer)
                    
                    # Kare atlamayı yönet - performans optimizasyonu için
                    should_process = True
                    
                    # Önceki işlem süresi çok uzunsa kare atla
                    if len(self.processing_times) > 0:
                        last_process_time = self.processing_times[-1]
                        if last_process_time > frame_skip_threshold:
                            skip_frame_count = max(1, int(last_process_time / 33.3))  # 30 fps için
                            
                            if skip_frame_count > 0:
                                should_process = (self.frame_count % skip_frame_count == 0)
                                if not should_process:
                                    self.skipped_frames += 1
                    
                    # Kare işle ve motoru kontrol et
                    if should_process:
                        # Görüntüyü işle ve şeritleri tespit et
                        processed_frame, center_diff, process_time = self._process_frame(frame)
                        
                        # Şeritlere göre motoru kontrol et
                        self.motor_controller.follow_lane(center_diff)
                    else:
                        # İşlemeden sadece görüntü göster (debug modunda)
                        processed_frame = frame
                        center_diff = None
                        process_time = 0
                    
                    # Hata sayacını sıfırla (başarıyla işlendi)
                    error_count = 0
                    
                    # FPS hesapla
                    self.frame_count += 1
                    elapsed_time = time.time() - self.fps_start_time
                    if elapsed_time >= 1.0:
                        self.fps = self.frame_count / elapsed_time
                        self.fps_start_time = time.time()
                        self.frame_count = 0
                        logger.debug(f"FPS: {self.fps:.1f}, Ortalama İşlem Süresi: {sum(self.processing_times) / len(self.processing_times):.1f}ms")
                        
                        # Her saniyede bir bellek temizliği
                        if self.frame_count % 30 == 0:
                            gc.collect()
                    
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
        
        # Belleği temizle
        self.frame_buffer = None
        self.resize_buffer = None
        gc.collect()
        
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
    
    # Kalibrasyon dosyası için argüman
    parser.add_argument('--calibration', default='calibration.json', help='Kalibrasyon dosyası yolu')
    
    # Performans modu için argüman
    parser.add_argument('--performance', default='balanced', choices=['speed', 'balanced', 'quality'],
                       help='Performans modu: hız/kalite dengesi')
    
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
        logger.info(f"Performans modu: {args.performance}")
        
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
            performance_mode=args.performance
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