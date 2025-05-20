#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Pist Simülatörü
Bu araç, pist videosunu kullanarak şerit tespiti ve kalibrasyon için bir simülasyon ortamı sağlar.
"""

import cv2
import numpy as np
import argparse
import time
import os
import logging
import signal
import sys
from lane_detection import LaneDetector

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PistSimulatoru")

class PistSimulatoru:
    """
    Pist simülasyonu sınıfı
    """
    def __init__(self, 
                 video_path, 
                 resolution=(640, 480), 
                 calibration_file="serit_kalibrasyon_video.json",
                 skip_frames=0,
                 loop_video=True,
                 speed=1.0):
        """
        PistSimulatoru sınıfını başlatır.
        
        Args:
            video_path (str): Pist videosu dosya yolu
            resolution (tuple): Görüntü çözünürlüğü (genişlik, yükseklik)
            calibration_file (str): Kalibrasyon dosyası yolu
            skip_frames (int): Atlanacak kare sayısı
            loop_video (bool): Video sonunda başa dönsün mü
            speed (float): Video oynatma hızı çarpanı
        """
        self.video_path = video_path
        self.width, self.height = resolution
        self.resolution = resolution
        self.calibration_file = calibration_file
        self.skip_frames = skip_frames
        self.loop_video = loop_video
        self.speed = speed
        
        self.running = True
        self.video_capture = None
        self.lane_detector = None
        
        # ROI ve kalibrasyon noktaları
        self.roi_points = []
        self.src_points = []  # Perspektif dönüşümü kaynak noktaları
        self.dst_points = []  # Perspektif dönüşümü hedef noktaları
        
        # Kalibrasyon modu
        self.calibration_mode = False
        self.selected_point = None
        
        # Kareleri gösterme ayarları
        self.show_original = True
        self.show_processed = True
        self.show_bird_eye = True
        self.show_binary = False
        
        # Canny ve Hough parametreleri
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 15
        self.min_line_length = 15
        self.max_line_gap = 30
        
        # Ekstra görselleştirme ayarları
        self.show_center_line = True
        self.show_lane_markings = True
        
        # Toplam kare sayısı ve mevcut kare
        self.total_frames = 0
        self.frame_count = 0
        
        # FPS ölçümü
        self.fps = 0
        self.frame_time = 0
        self.frame_counter = 0
        self.fps_update_time = time.time()
        
        # Temiz kapatma için sinyal yakalama
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Sinyal yakalama işleyicisi (CTRL+C gibi)"""
        logger.info("Kapatma sinyali alındı, temizleniyor...")
        self.cleanup()
        sys.exit(0)
        
    def _load_calibration(self):
        """Kalibrasyon dosyasını yükler"""
        try:
            if os.path.exists(self.calibration_file):
                import json
                with open(self.calibration_file, 'r') as f:
                    calibration = json.load(f)
                
                if "src_points" in calibration and "dst_points" in calibration:
                    self.src_points = calibration["src_points"]
                    self.dst_points = calibration["dst_points"]
                    
                    # ROI noktaları oluştur
                    self.roi_points = np.array([
                        self.src_points[2],  # Sol alt
                        self.src_points[0],  # Sol üst
                        self.src_points[1],  # Sağ üst
                        self.src_points[3]   # Sağ alt
                    ], dtype=np.int32)
                    
                # Diğer parametreleri yükle
                if "canny_low_threshold" in calibration:
                    self.canny_low = calibration["canny_low_threshold"]
                if "canny_high_threshold" in calibration:
                    self.canny_high = calibration["canny_high_threshold"]
                if "hough_threshold" in calibration:
                    self.hough_threshold = calibration["hough_threshold"]
                if "min_line_length" in calibration:
                    self.min_line_length = calibration["min_line_length"]
                if "max_line_gap" in calibration:
                    self.max_line_gap = calibration["max_line_gap"]
                
                logger.info(f"Kalibrasyon yüklendi: {self.calibration_file}")
                return True
            else:
                logger.warning(f"Kalibrasyon dosyası bulunamadı: {self.calibration_file}")
                self._create_default_calibration()
                return False
        except Exception as e:
            logger.error(f"Kalibrasyon yükleme hatası: {e}")
            self._create_default_calibration()
            return False
    
    def _create_default_calibration(self):
        """Varsayılan kalibrasyon noktaları oluşturur"""
        width, height = self.resolution
        
        # Varsayılan perspektif dönüşüm noktaları
        self.src_points = [
            [width * 0.35, height * 0.65],  # Sol üst
            [width * 0.65, height * 0.65],  # Sağ üst
            [0, height],                     # Sol alt
            [width, height]                  # Sağ alt
        ]
        
        self.dst_points = [
            [width * 0.25, 0],               # Sol üst
            [width * 0.75, 0],               # Sağ üst
            [width * 0.25, height],          # Sol alt
            [width * 0.75, height]           # Sağ alt
        ]
        
        # ROI noktaları
        self.roi_points = np.array([
            self.src_points[2],  # Sol alt
            self.src_points[0],  # Sol üst
            self.src_points[1],  # Sağ üst
            self.src_points[3]   # Sağ alt
        ], dtype=np.int32)
        
        logger.info("Varsayılan kalibrasyon noktaları oluşturuldu")
    
    def _save_calibration(self):
        """Kalibrasyon verilerini dosyaya kaydeder"""
        try:
            import json
            
            calibration = {
                "src_points": self.src_points,
                "dst_points": self.dst_points,
                "resolution": {
                    "width": self.width,
                    "height": self.height
                },
                "canny_low_threshold": self.canny_low,
                "canny_high_threshold": self.canny_high,
                "hough_threshold": self.hough_threshold,
                "min_line_length": self.min_line_length,
                "max_line_gap": self.max_line_gap
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration, f, indent=2)
            
            logger.info(f"Kalibrasyon kaydedildi: {self.calibration_file}")
            
            # Ana kalibrasyon dosyasına da kaydet
            with open("serit_kalibrasyon.json", 'w') as f:
                json.dump(calibration, f, indent=2)
                
            logger.info("Ana kalibrasyon dosyası güncellendi: serit_kalibrasyon.json")
            
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon kaydetme hatası: {e}")
            return False
    
    def _initialize_video(self):
        """Video dosyasını başlatır"""
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
                
            logger.info("Video başarıyla başlatıldı.")
            return True
            
        except Exception as e:
            logger.error(f"Video başlatma hatası: {e}")
            return False
    
    def _initialize_lane_detector(self):
        """Şerit dedektörünü başlatır"""
        try:
            self.lane_detector = LaneDetector(camera_resolution=self.resolution, debug=True)
            
            # Kalibrasyon parametrelerini ayarla
            if hasattr(self, 'roi_points') and len(self.roi_points) > 0:
                self.lane_detector.roi_vertices = self.roi_points
            
            # Orta şerit çizgisini belirle
            if hasattr(self, 'src_points') and len(self.src_points) >= 4:
                center_line = np.array([
                    [(self.src_points[2][0] + self.src_points[3][0]) // 2, self.height],  # Alt orta nokta
                    [(self.src_points[0][0] + self.src_points[1][0]) // 2, (self.src_points[0][1] + self.src_points[1][1]) // 2]  # Üst orta nokta
                ], dtype=np.int32)
                self.lane_detector.center_line = center_line
            
            # Diğer parametreleri ayarla
            self.lane_detector.canny_low = self.canny_low
            self.lane_detector.canny_high = self.canny_high
            self.lane_detector.min_line_length = self.min_line_length
            self.lane_detector.max_line_gap = self.max_line_gap
            
            logger.info("Şerit dedektörü başlatıldı")
            return True
        except Exception as e:
            logger.error(f"Şerit dedektörü başlatma hatası: {e}")
            return False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Fare etkileşimi geri çağırma fonksiyonu"""
        if not self.calibration_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # En yakın kalibrasyon noktasını bul
            min_dist = float('inf')
            closest_idx = -1
            
            for i, point in enumerate(self.src_points):
                dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
                if dist < min_dist and dist < 20:  # 20 piksel içinde olmalı
                    min_dist = dist
                    closest_idx = i
            
            self.selected_point = closest_idx
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = None
            # Noktalar değiştikçe ROI'yi güncelle
            if len(self.src_points) >= 4:
                self.roi_points = np.array([
                    self.src_points[2],  # Sol alt
                    self.src_points[0],  # Sol üst
                    self.src_points[1],  # Sağ üst
                    self.src_points[3]   # Sağ alt
                ], dtype=np.int32)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            # Noktayı güncelle
            self.src_points[self.selected_point] = [x, y]
            
            # Noktalar değiştikçe ROI'yi güncelle
            if len(self.src_points) >= 4:
                self.roi_points = np.array([
                    self.src_points[2],  # Sol alt
                    self.src_points[0],  # Sol üst
                    self.src_points[1],  # Sağ üst
                    self.src_points[3]   # Sağ alt
                ], dtype=np.int32)
    
    def _get_next_frame(self):
        """Bir sonraki kareyi alır"""
        if self.video_capture is None:
            return False, None
            
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
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Video hız kontrolü
            wait_time = 0
            if self.speed < 1.0:
                time.sleep((1.0 - self.speed) / 30)  # Varsayılan 30 FPS için
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Kare alma hatası: {e}")
            return False, None
    
    def _apply_bird_eye_transform(self, frame):
        """Kuş bakışı dönüşümü uygular"""
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            return frame
            
        try:
            src_points = np.float32(self.src_points)
            dst_points = np.float32(self.dst_points)
            
            # Perspektif dönüşüm matrisi
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Dönüşümü uygula
            warped = cv2.warpPerspective(frame, M, (self.width, self.height))
            
            return warped
        except Exception as e:
            logger.error(f"Kuş bakışı dönüşüm hatası: {e}")
            return frame
    
    def _generate_binary_view(self, frame):
        """İkili (binary) görüntü oluştur"""
        try:
            # Gri tonlamaya dönüştür
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Gürültü azaltma
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny kenar tespiti
            edges = cv2.Canny(blur, self.canny_low, self.canny_high)
            
            # İkili görüntü oluştur
            ret, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
            
            # Görselleştirme için BGR'ye dönüştür
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            return binary_bgr
            
        except Exception as e:
            logger.error(f"Binary görüntü oluşturma hatası: {e}")
            return frame
    
    def _draw_calibration_points(self, frame):
        """Kalibrasyon noktalarını çizer"""
        result = frame.copy()
        
        try:
            # ROI bölgesini çiz
            if len(self.roi_points) > 0:
                cv2.polylines(result, [self.roi_points], True, (0, 255, 0), 2)
            
            # Perspektif dönüşüm noktalarını çiz
            for i, point in enumerate(self.src_points):
                x, y = int(point[0]), int(point[1])
                cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(result, f"P{i}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            logger.error(f"Kalibrasyon noktaları çizim hatası: {e}")
        
        return result
    
    def _generate_info_overlay(self, frame):
        """Bilgi katmanı oluşturur"""
        result = frame.copy()
        
        # FPS hesapla
        current_time = time.time()
        if current_time - self.fps_update_time >= 1.0:
            self.fps = self.frame_counter / (current_time - self.fps_update_time)
            self.fps_update_time = current_time
            self.frame_counter = 0
        else:
            self.frame_counter += 1
        
        # İlerleme bilgisi
        progress = self.frame_count / self.total_frames if self.total_frames > 0 else 0
        progress_text = f"Video: {self.frame_count}/{self.total_frames} ({progress*100:.1f}%)"
        cv2.putText(result, progress_text, (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # FPS bilgisi
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(result, fps_text, (self.width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mod bilgisi
        mode_text = "MOD: KALİBRASYON" if self.calibration_mode else "MOD: SİMÜLASYON"
        cv2.putText(result, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if self.calibration_mode else (0, 255, 0), 1)
        
        # Parametre bilgileri
        param_text = f"Canny: {self.canny_low}/{self.canny_high}, Hough: {self.hough_threshold}, L: {self.min_line_length}, G: {self.max_line_gap}"
        cv2.putText(result, param_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Kalibrasyon modu yardımı
        if self.calibration_mode:
            help_text = "Fare ile noktaları sürükleyin | S: Kaydet | ESC: Çıkış | C: Kalibrasyon Modu"
            cv2.putText(result, help_text, (self.width//2 - 180, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
        
        return result
    
    def run(self):
        """Ana simülasyon döngüsünü çalıştırır"""
        # Video dosyasını başlat
        if not self._initialize_video():
            logger.error("Video başlatılamadı!")
            return False
        
        # Kalibrasyon yükle
        self._load_calibration()
        
        # Şerit dedektörünü başlat
        self._initialize_lane_detector()
        
        # Pencereler oluştur
        cv2.namedWindow("Pist Simulatoru", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Pist Simulatoru", self._mouse_callback)
        
        print("\nPİST SİMÜLATÖRÜ KONTROL KOMUTLARI:")
        print("----------------------------------")
        print("ESC/Q : Çıkış")
        print("SPACE  : Duraklat/Devam")
        print("S      : Kalibrasyon kaydet")
        print("C      : Kalibrasyon modu aç/kapat")
        print("R      : Kareleri sıfırla")
        print("1      : Orijinal görüntü aç/kapat")
        print("2      : İşlenmiş görüntü aç/kapat")
        print("3      : Kuş bakışı görüntü aç/kapat")
        print("4      : İkili görüntü aç/kapat")
        print("+/-    : Video hızını değiştir")
        
        # Ana döngü
        paused = False
        while self.running:
            if not paused:
                # Kare al
                ret, frame = self._get_next_frame()
                if not ret:
                    logger.info("Video bitti.")
                    break
                
                # Şerit tespiti yap
                processed_frame, center_diff = self.lane_detector.process_frame(frame)
                
                # Kuş bakışı dönüşümünü uygula
                bird_eye_view = self._apply_bird_eye_transform(frame)
                
                # İkili görüntü oluştur
                binary_view = self._generate_binary_view(frame)
                
                # Kalibrasyon modunda noktaları çiz
                if self.calibration_mode:
                    frame = self._draw_calibration_points(frame)
            
            # Sonuç görüntüsünü oluştur
            if self.show_original and self.show_processed and self.show_bird_eye:
                # 3'lü görüntü (orijinal + işlenmiş + kuş bakışı)
                top_row = cv2.hconcat([frame, processed_frame])
                
                # İkili görüntüyü kontrol et
                if self.show_binary:
                    bottom_row = cv2.hconcat([bird_eye_view, binary_view])
                else:
                    # İkili görüntü kapalıysa boş bir görüntü ekle
                    bottom_row = cv2.hconcat([bird_eye_view, np.zeros_like(bird_eye_view)])
                
                # Görüntüleri birleştir
                result = cv2.vconcat([top_row, bottom_row])
                
                # Görüntü boyutunu kontrol et
                max_height = 800
                if result.shape[0] > max_height:
                    scale = max_height / result.shape[0]
                    result = cv2.resize(result, None, fx=scale, fy=scale)
                
            elif self.show_original and self.show_processed:
                # 2'li görüntü (orijinal + işlenmiş)
                result = cv2.hconcat([frame, processed_frame])
            elif self.show_original:
                # Sadece orijinal
                result = frame.copy()
            elif self.show_processed:
                # Sadece işlenmiş
                result = processed_frame.copy()
            else:
                # Hiçbir görüntü seçilmemişse, orijinali göster
                result = frame.copy()
                self.show_original = True
            
            # Bilgi katmanını ekle
            result = self._generate_info_overlay(result)
            
            # Görüntüyü göster
            cv2.imshow("Pist Simulatoru", result)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC/Q = Çık
                logger.info("Kullanıcı tarafından sonlandırıldı.")
                break
                
            elif key == ord(' '):  # Boşluk = Duraklat/Devam
                paused = not paused
                if paused:
                    logger.info("Video duraklatıldı.")
                else:
                    logger.info("Video devam ediyor.")
            
            elif key == ord('s'):  # S = Kaydet
                if self._save_calibration():
                    # Kalibrasyon başarıyla kaydedildi
                    save_message = "Kalibrasyon Kaydedildi!"
                    overlay = result.copy()
                    cv2.putText(overlay, save_message, 
                              (result.shape[1]//2 - 100, result.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Pist Simulatoru", overlay)
                    cv2.waitKey(1000)  # 1 saniye bekle
                
            elif key == ord('c'):  # C = Kalibrasyon modu
                self.calibration_mode = not self.calibration_mode
                if self.calibration_mode:
                    logger.info("Kalibrasyon modu açıldı.")
                else:
                    logger.info("Kalibrasyon modu kapatıldı.")
            
            elif key == ord('r'):  # R = Kareleri sıfırla
                # Video başa al
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frames)
                self.frame_count = self.skip_frames
                logger.info(f"Video {self.skip_frames}. kareye sıfırlandı.")
            
            elif key == ord('1'):  # 1 = Orijinal görüntü aç/kapat
                self.show_original = not self.show_original
                
            elif key == ord('2'):  # 2 = İşlenmiş görüntü aç/kapat
                self.show_processed = not self.show_processed
                
            elif key == ord('3'):  # 3 = Kuş bakışı görüntü aç/kapat
                self.show_bird_eye = not self.show_bird_eye
                
            elif key == ord('4'):  # 4 = İkili görüntü aç/kapat
                self.show_binary = not self.show_binary
                
            elif key == ord('+') or key == ord('='):  # + = Video hızını artır
                self.speed = min(10.0, self.speed * 1.2)
                logger.info(f"Video hızı: {self.speed:.1f}x")
                
            elif key == ord('-'):  # - = Video hızını azalt
                self.speed = max(0.1, self.speed / 1.2)
                logger.info(f"Video hızı: {self.speed:.1f}x")
                
            elif key in [ord('w'), ord('a'), ord('s'), ord('d')]:  # Canny parametrelerini ayarla
                if key == ord('w'):  # W = Canny düşük eşiği artır
                    self.canny_low = min(255, self.canny_low + 5)
                elif key == ord('s'):  # S = Canny düşük eşiği azalt
                    self.canny_low = max(0, self.canny_low - 5)
                elif key == ord('a'):  # A = Canny yüksek eşiği azalt
                    self.canny_high = max(self.canny_low + 20, self.canny_high - 5)
                elif key == ord('d'):  # D = Canny yüksek eşiği artır
                    self.canny_high = min(255, self.canny_high + 5)
                
                # Şerit dedektörünü güncelle
                if self.lane_detector:
                    self.lane_detector.canny_low = self.canny_low
                    self.lane_detector.canny_high = self.canny_high
                logger.info(f"Canny eşikleri: {self.canny_low}/{self.canny_high}")
            
            elif key in [ord('i'), ord('j'), ord('k'), ord('l')]:  # Hough parametrelerini ayarla
                if key == ord('i'):  # I = Minimum çizgi uzunluğu artır
                    self.min_line_length = min(100, self.min_line_length + 5)
                elif key == ord('k'):  # K = Minimum çizgi uzunluğu azalt
                    self.min_line_length = max(5, self.min_line_length - 5)
                elif key == ord('j'):  # J = Maximum çizgi boşluğu azalt
                    self.max_line_gap = max(5, self.max_line_gap - 5)
                elif key == ord('l'):  # L = Maximum çizgi boşluğu artır
                    self.max_line_gap = min(100, self.max_line_gap + 5)
                
                # Şerit dedektörünü güncelle
                if self.lane_detector:
                    self.lane_detector.min_line_length = self.min_line_length
                    self.lane_detector.max_line_gap = self.max_line_gap
                logger.info(f"Hough parametreleri: L={self.min_line_length}, G={self.max_line_gap}")
        
        # Temizlik
        self.cleanup()
        return True
    
    def cleanup(self):
        """Kaynakları temizler"""
        logger.info("Simülasyon temizleniyor...")
        
        # Video kaynağını kapat
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            self.video_capture.release()
        
        # Pencereleri kapat
        cv2.destroyAllWindows()
        
        logger.info("Simülasyon temizlendi.")

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description='Otonom Araç Pist Simülatörü')
    parser.add_argument('--video', required=True, help='Pist videosu dosya yolu')
    parser.add_argument('--resolution', default='640x480', help='Görüntü çözünürlüğü (GENxYÜK)')
    parser.add_argument('--calibration', default='serit_kalibrasyon_video.json', help='Kalibrasyon dosyası yolu')
    parser.add_argument('--skip-frames', type=int, default=0, help='Atlanacak kare sayısı')
    parser.add_argument('--no-loop', action='store_true', help='Video sonunda başa dönmesin')
    parser.add_argument('--speed', type=float, default=1.0, help='Video oynatma hızı çarpanı')
    
    return parser.parse_args()

def main():
    """Ana program"""
    args = parse_arguments()
    
    # Video dosyası kontrolü
    if not os.path.exists(args.video):
        logger.error(f"Video dosyası bulunamadı: {args.video}")
        print(f"\nHata: Video dosyası bulunamadı: {args.video}")
        return False
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        resolution = (640, 480)
    
    print("\nOTONOM ARAÇ PİST SİMÜLATÖRÜ")
    print("----------------------------")
    print(f"Video dosyası: {args.video}")
    print(f"Kalibrasyon dosyası: {args.calibration}")
    print(f"Çözünürlük: {resolution[0]}x{resolution[1]}")
    print(f"Video hızı: {args.speed}x")
    print(f"Atlanacak kare sayısı: {args.skip_frames}")
    print(f"Video döngüsü: {'KAPALI' if args.no_loop else 'AÇIK'}")
    print("\nSimülasyon başlatılıyor...")
    
    try:
        # Simülatörü oluştur ve çalıştır
        simulator = PistSimulatoru(
            video_path=args.video,
            resolution=resolution,
            calibration_file=args.calibration,
            skip_frames=args.skip_frames,
            loop_video=not args.no_loop,
            speed=args.speed
        )
        
        simulator.run()
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        logger.error(f"Hata: {e}")
        print(f"\nBir hata oluştu: {e}")
    
    print("\nSimülasyon sonlandırıldı.")
    return True

if __name__ == "__main__":
    main() 