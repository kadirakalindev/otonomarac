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
                 calibration_file="serit_kalibrasyon.json",  # Kalibrasyon dosyası yolu
                 mode="real",                 # Çalışma modu: 'real' veya 'video'
                 video_path=None,             # Video dosyası yolu (video modunda)
                 skip_frames=0,               # Video modunda atlanacak kare sayısı
                 loop_video=False,            # Video bitince başa dön
                 video_speed=1.0):            # Video oynatma hızı
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
            mode (str): Çalışma modu - 'real' veya 'video'
            video_path (str): Test videosu dosya yolu (video modunda)
            skip_frames (int): Video modunda atlanacak kare sayısı
            loop_video (bool): Video bitince başa dön
            video_speed (float): Video oynatma hızı çarpanı
        """
        self.debug = debug
        self.debug_fps = debug_fps
        self.running = False
        self.camera_resolution = camera_resolution
        self.framerate = framerate
        self.camera = None  # Başlangıçta None olarak tanımla
        self.calibration_file = calibration_file  # Kalibrasyon dosyası yolunu sakla
        
        # Çalışma modu ayarları
        self.mode = mode
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.loop_video = loop_video
        self.video_speed = video_speed
        self.video_capture = None
        self.frame_count = 0
        self.total_frames = 0
        
        # Video modu kontrolü
        if self.mode == "video" and (self.video_path is None or not os.path.exists(self.video_path)):
            logger.error(f"Video modu seçildi fakat geçerli bir video dosyası belirtilmedi: {self.video_path}")
            raise ValueError("Video modunda geçerli bir video dosyası yolu belirtilmelidir")
        
        # Modülleri başlatma denemesi - hata durumunda güvenli kapatma
        try:
            # Giriş kaynağını başlat (kamera veya video)
            self._initialize_input_source()
            
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
            
            # Motor kontrol modülünü başlat (video modunda simüle et)
            logger.info("Motor kontrol modülü başlatılıyor...")
            if self.mode == "real":
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
            else:  # video modu için sanal motor kontrolörü
                self.motor_controller = self._create_virtual_motor_controller()
        
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
        
        logger.info(f"Otonom araç başlatıldı. Mod: {self.mode.upper()}")
    
    def _create_virtual_motor_controller(self):
        """Video modu için sanal motor kontrolörü oluşturur"""
        class VirtualMotorController:
            def __init__(self):
                self.left_speed = 0
                self.right_speed = 0
                self.direction = "stop"
                logger.info("Sanal motor kontrolörü başlatıldı")
                
            def forward(self, speed=None):
                self.left_speed = self.right_speed = speed if speed is not None else 0.35
                self.direction = "forward"
                logger.debug(f"[SANAL] İleri hareket: {speed}")
                
            def backward(self, speed=None):
                self.left_speed = self.right_speed = speed if speed is not None else 0.35
                self.direction = "backward"
                logger.debug(f"[SANAL] Geri hareket: {speed}")
                
            def turn_left(self, speed=None):
                self.left_speed = 0.2
                self.right_speed = speed if speed is not None else 0.35
                self.direction = "left"
                logger.debug(f"[SANAL] Sola dönüş: {speed}")
                
            def turn_right(self, speed=None):
                self.left_speed = speed if speed is not None else 0.35
                self.right_speed = 0.2
                self.direction = "right"
                logger.debug(f"[SANAL] Sağa dönüş: {speed}")
                
            def stop(self):
                self.left_speed = self.right_speed = 0
                self.direction = "stop"
                logger.debug("[SANAL] Durduruldu")
                
            def follow_lane(self, center_diff, speed=None):
                if center_diff is None:
                    logger.debug("[SANAL] Şerit kaybedildi")
                    return
                    
                base_speed = speed if speed is not None else 0.35
                
                # Merkez farkına göre dönüş hesapla
                if abs(center_diff) < 10:  # Merkezde
                    self.left_speed = self.right_speed = base_speed
                    logger.debug(f"[SANAL] Düz ileri: {base_speed}")
                elif center_diff < 0:  # Sola dönüş
                    factor = min(1.0, abs(center_diff) / 100)
                    self.left_speed = base_speed * (1 - factor * 0.8)
                    self.right_speed = base_speed
                    logger.debug(f"[SANAL] Sola dönüş: L={self.left_speed:.2f}, R={self.right_speed:.2f}")
                else:  # Sağa dönüş
                    factor = min(1.0, abs(center_diff) / 100)
                    self.left_speed = base_speed
                    self.right_speed = base_speed * (1 - factor * 0.8)
                    logger.debug(f"[SANAL] Sağa dönüş: L={self.left_speed:.2f}, R={self.right_speed:.2f}")
            
            def cleanup(self):
                logger.info("[SANAL] Motor kontrolörü temizlendi")
                
        return VirtualMotorController()
    
    def _initialize_input_source(self):
        """Giriş kaynağını başlatır (kamera veya video)"""
        if self.mode == "real":
            self._initialize_camera()
        else:  # video modu
            self._initialize_video()
    
    def _initialize_camera(self):
        """
        Kamera modülünü başlatır ve yapılandırır
        """
        try:
            logger.info("Kamera başlatılıyor...")
            self.camera = Picamera2()
            
            # Kamera yapılandırması - daha detaylı
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
                    "ExposureTime": 10000,      # Pozlama süresi (mikrosaniye)
                    "AnalogueGain": 1.0,        # Analog kazanç
                    "Brightness": 0.0,          # Parlaklık
                    "Contrast": 1.0,            # Kontrast
                    "Sharpness": 1.0,           # Keskinlik
                    "NoiseReductionMode": 1     # Gürültü azaltma
                }
                self.camera.set_controls(controls)
                logger.info("Kamera kontrolleri ayarlandı")
            except Exception as e:
                logger.warning(f"Kamera kontrolleri ayarlanırken hata: {e}")
            
            # Kamerayı başlat
            self.camera.start()
            
            # Test görüntüsü al
            logger.info("Kamera test ediliyor...")
            for _ in range(3):  # 3 kere dene
                try:
                    test_frame = self.camera.capture_array()
                    if test_frame is not None and test_frame.size > 0:
                        logger.info(f"Kamera test başarılı. Görüntü boyutu: {test_frame.shape}")
                        break
                except Exception as e:
                    logger.warning(f"Test görüntüsü alınamadı, tekrar deneniyor: {e}")
                time.sleep(0.5)
            
            # Kameranın dengelenmesi için bekle
            logger.info("Kamera dengeleniyor...")
            time.sleep(2)
            
            logger.info("Kamera başarıyla başlatıldı.")
            
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.close()
            raise
    
    def _initialize_video(self):
        """
        Video dosyasını başlatır ve yapılandırır
        """
        try:
            logger.info(f"Video dosyası açılıyor: {self.video_path}")
            self.video_capture = cv2.VideoCapture(self.video_path)
            
            if not self.video_capture.isOpened():
                raise IOError(f"Video dosyası açılamadı: {self.video_path}")
            
            # Video özelliklerini al
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video yüklendi: {self.video_path}")
            logger.info(f"Video özellikleri: {video_width}x{video_height}, {video_fps} FPS, {self.total_frames} kare")
            
            # İstenilen sayıda kareyi atla
            if self.skip_frames > 0:
                logger.info(f"{self.skip_frames} kare atlanıyor...")
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frames)
                self.frame_count = self.skip_frames
            
            # Test karesi al
            ret, test_frame = self.video_capture.read()
            if not ret:
                raise IOError("Video karesini okuma hatası")
                
            # Video karesi boyutunu kontrol et ve gerekirse boyutlandır
            if test_frame.shape[1] != self.camera_resolution[0] or test_frame.shape[0] != self.camera_resolution[1]:
                logger.info(f"Video karesi yeniden boyutlandırılıyor: {test_frame.shape[1]}x{test_frame.shape[0]} -> {self.camera_resolution[0]}x{self.camera_resolution[1]}")
            
            # Videonun başına dön
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frames)
            self.frame_count = self.skip_frames
            
            logger.info("Video başarıyla başlatıldı.")
            
        except Exception as e:
            logger.error(f"Video başlatma hatası: {e}")
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
            raise

    def _get_next_frame(self):
        """
        Bir sonraki kareyi alır (kamera veya videodan)
        
        Returns:
            tuple: (başarı durumu, kare)
        """
        if self.mode == "real":
            try:
                # Kameradan görüntü al
                frame = self.camera.capture_array()
                if frame is None or frame.size == 0:
                    return False, None
                
                # RGB'den BGR'ye dönüştür (OpenCV için)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                return True, frame
            except Exception as e:
                logger.error(f"Kamera karesi alma hatası: {e}")
                return False, None
        else:  # video modu
            try:
                # Kare oku
                ret, frame = self.video_capture.read()
                
                # Video sonuna gelindiğinde
                if not ret:
                    if self.loop_video:
                        logger.info("Video sonuna gelindi. Başa dönülüyor...")
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frames)
                        self.frame_count = self.skip_frames
                        ret, frame = self.video_capture.read()
                        if not ret:
                            return False, None
                    else:
                        return False, None
                
                self.frame_count += 1
                
                # Video karesi boyutunu kontrol et ve gerekirse boyutlandır
                if frame.shape[1] != self.camera_resolution[0] or frame.shape[0] != self.camera_resolution[1]:
                    frame = cv2.resize(frame, self.camera_resolution)
                
                # Video hız kontrolü - gerekirse bekleme ekle
                if self.video_speed < 1.0:
                    time.sleep((1.0 - self.video_speed) / 30)  # Varsayılan 30 FPS için
                
                return True, frame
            except Exception as e:
                logger.error(f"Video karesi alma hatası: {e}")
                return False, None

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
            # Giriş kaynağını kontrol et
            if (self.mode == "real" and self.camera is None) or (self.mode == "video" and self.video_capture is None):
                self._initialize_input_source()
            
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
                    # Kare al (kamera veya video)
                    ret, frame = self._get_next_frame()
                    if not ret:
                        if self.mode == "video" and not self.loop_video:
                            logger.info("Video sonuna gelindi.")
                            break
                        else:
                            raise Exception("Geçersiz kare")
                    
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
                            # Video modu bilgisi
                            if self.mode == "video":
                                progress_text = f"Video: {self.frame_count}/{self.total_frames} ({(self.frame_count/self.total_frames*100):.1f}%)"
                                cv2.putText(processed_frame, progress_text, 
                                          (10, processed_frame.shape[0] - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # FPS bilgisini görüntüye ekle
                            fps_text = f"FPS: {self.fps:.1f}"
                            cv2.putText(processed_frame, fps_text, 
                                      (processed_frame.shape[1] - 120, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Mod bilgisini ekle
                            mode_text = f"Mod: {'GERÇEK' if self.mode == 'real' else 'VİDEO'}"
                            cv2.putText(processed_frame, mode_text, 
                                      (processed_frame.shape[1] - 120, 60), 
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
                            elif self.mode == "video":
                                if key == ord(' '):  # Boşluk = Duraklat/Devam
                                    wait_key = cv2.waitKey(0)
                                    if wait_key == ord('q'):
                                        logger.info("Kullanıcı tarafından durduruldu")
                                        self.running = False
                                        break
                    
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
        if self.mode == "real" and hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.stop()
                logger.info("Kamera kapatıldı.")
            except:
                pass
        
        # Video kaynağını kapat
        if self.mode == "video" and hasattr(self, 'video_capture') and self.video_capture is not None:
            try:
                self.video_capture.release()
                logger.info("Video kaynağı kapatıldı.")
            except:
                pass
        
        # OpenCV pencerelerini kapat
        if self.debug:
            cv2.destroyAllWindows()
            logger.info("Görüntüleme pencereleri kapatıldı.")
        
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

    def signal_handler(self, sig, frame):
        """
        Sinyal yakalama işleyicisi (CTRL+C gibi)
        """
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.cleanup()
        sys.exit(0)

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
    
    # Video modu ve gerçek pist modu arasında geçiş için argümanlar
    parser.add_argument('--mode', choices=['real', 'video'], default='real', help='Çalışma modu: gerçek pist veya video testi')
    parser.add_argument('--video-path', help='Test için video dosyası yolu (video modunda gerekli)')
    parser.add_argument('--skip-frames', type=int, default=0, help='Video modunda kaç kareyi atlayacağı')
    parser.add_argument('--loop-video', action='store_true', help='Video bitince başa dön')
    parser.add_argument('--video-speed', type=float, default=1.0, help='Video oynatma hızı çarpanı')
    
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
    
    # Video modu kontrolü
    if args.mode == "video" and not args.video_path:
        logger.error("Video modu seçildi ancak video dosya yolu belirtilmedi!")
        print("\nHata: Video modu için --video-path parametresi gereklidir.")
        print("Örnek kullanım: python main.py --mode video --video-path test_pist.mp4")
        return
    
    # Otonom aracı başlat
    try:
        mode_text = "VİDEO" if args.mode == "video" else "GERÇEK PİST"
        logger.info(f"Otonom araç {mode_text} modunda başlatılıyor...")
        print(f"\nOTONOM ARAÇ KONTROL PROGRAMI - {mode_text} MODU")
        print("----------------------------------------")
        
        if args.mode == "video":
            print(f"Video dosyası: {args.video_path}")
            print(f"Video hızı: {args.video_speed}x")
            if args.loop_video:
                print("Video döngüsü: AÇIK (video bitince başa dönecek)")
            print("\nKontroller:")
            print("  q     : Çıkış")
            print("  s     : Ekran görüntüsü")
            print("  space : Duraklat/Devam et")
            print("\nProgram başlatılıyor...")
        
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
            mode=args.mode,
            video_path=args.video_path,
            skip_frames=args.skip_frames,
            loop_video=args.loop_video,
            video_speed=args.video_speed
        )
        
        # Otonom sürüşü başlat
        otonom_arac.start()
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu (CTRL+C)")
    except FileNotFoundError as e:
        logger.error(f"Dosya bulunamadı: {e}")
        print(f"\nHata: {e}")
    except Exception as e:
        logger.error(f"Hata: {e}")
        print(f"\nHata: {e}")
    
    logger.info("Program sonlandırıldı.")

if __name__ == "__main__":
    main() 