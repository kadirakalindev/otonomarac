#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Basitleştirilmiş Şerit Tespiti ve Takibi Modülü
"""

import cv2
import numpy as np
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LaneDetection")

class LaneDetector:
    def __init__(self, camera_resolution=(320, 240), debug=False):
        if not isinstance(camera_resolution, tuple) or len(camera_resolution) != 2:
            raise ValueError("camera_resolution geçersiz format")
            
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        self.debug = debug
        
        # ROI parametreleri - Daha dar ve simetrik bölge
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Geçersiz çözünürlük değerleri")
            
        self.roi_vertices = np.array([
            [self.width * 0.2, self.height],             # Sol alt
            [self.width * 0.45, self.height * 0.55],     # Sol üst (daha merkezi)
            [self.width * 0.55, self.height * 0.55],     # Sağ üst (daha merkezi) 
            [self.width * 0.8, self.height]              # Sağ alt
        ], dtype=np.int32)
        
        # Temel filtre parametreleri - hassasiyet azaltıldı
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 50     # Yükseltildi
        self.canny_high = 150   # Yükseltildi
        
        # Hough parametreleri - daha seçici
        self.rho = 2  # Hassasiyeti azalt (piksel çözünürlüğü)
        self.theta = np.pi/180
        self.min_line_length = self.height * 0.15  # Daha uzun çizgiler ara
        self.max_line_gap = self.height * 0.1      # Çizgi aralığını azalt
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.max_history_frames = 10  # Daha uzun hafıza
        self.smooth_factor = 0.7  # Yumuşatma faktörü
        self.confidence_threshold = 0.6  # Güven eşiği
        
        # Şerit kaybı için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 15  # Daha toleranslı
        self.lane_recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_timeout = 3.0  # 3 saniye kurtarma modu
        
        # Debug görüntüleri için sözlük
        self.debug_images = {}
        
    def validate_frame(self, frame):
        """Gelen kareyi doğrula"""
        if frame is None:
            raise ValueError("Boş kare alındı")
            
        if len(frame.shape) != 3:
            raise ValueError("Geçersiz kare formatı")
            
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(f"Kare boyutu uyumsuz: Beklenen {self.width}x{self.height}, Alınan {frame.shape[1]}x{frame.shape[0]}")
            
        return True
        
    def load_calibration(self, calibration_file):
        """
        Kalibrasyon dosyasından parametreleri yükler
        
        Args:
            calibration_file (str): Kalibrasyon JSON dosyasının yolu
        """
        try:
            import json
            with open(calibration_file, 'r') as f:
                calibration = json.load(f)
                
            # ROI noktalarını güncelle (eğer varsa)
            if 'src_points' in calibration:
                self.roi_vertices = np.array(calibration['src_points'], dtype=np.int32)
                
            # Filtre parametrelerini güncelle
            if 'canny_low_threshold' in calibration:
                self.canny_low = calibration['canny_low_threshold']
            if 'canny_high_threshold' in calibration:
                self.canny_high = calibration['canny_high_threshold']
            if 'blur_kernel_size' in calibration:
                self.blur_kernel = calibration['blur_kernel_size']
                
            # Hough parametrelerini güncelle
            if 'hough_threshold' in calibration:
                self.rho = calibration['hough_threshold']
            if 'min_line_length' in calibration:
                self.min_line_length = calibration['min_line_length']
            if 'max_line_gap' in calibration:
                self.max_line_gap = calibration['max_line_gap']
                
            logger.info(f"Kalibrasyon dosyası yüklendi: {calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Kalibrasyon dosyası yüklenirken hata: {e}")
            return False
        
    def preprocess_image(self, image):
        """Temel görüntü ön işleme"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Kenar tespiti
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # ROI maskesi
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked = cv2.bitwise_and(edges, mask)
        
        if self.debug:
            self.debug_images["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.debug_images["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["masked"] = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        
        return masked
    
    def detect_lane_lines(self, edges):
        """Basitleştirilmiş şerit tespiti - geliştirilmiş versiyon"""
        # Hough çizgi tespit parametreleri güncellendi
        lines = cv2.HoughLinesP(
            edges, self.rho, self.theta, 30,  # Eşik değeri yükseltildi
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            # Çizgileri filtrele
            valid_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çok kısa çizgileri filtrele
                if np.sqrt((x2-x1)**2 + (y2-y1)**2) < self.min_line_length:
                    continue
                    
                # Eğim hesapla (sonsuz eğimi önle)
                if x2 - x1 == 0:  
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Yatay çizgileri daha agresif filtrele
                if abs(slope) < 0.35:
                    continue
                
                # Aşırı dik çizgileri filtrele
                if abs(slope) > 5:
                    continue
                
                # Çizginin orta noktasını hesapla
                mid_x = (x1 + x2) / 2
                
                # Görüntünün alt yarısında olmayan çizgileri filtrele
                if (y1 + y2) / 2 < self.height * 0.6:
                    continue
                
                # Geçerli çizgileri ekle
                valid_lines.append((x1, y1, x2, y2, slope, mid_x))
            
            # Eğer yeterli çizgi kaldıysa, sıralama ve gruplama yap
            if valid_lines:
                # Eğime göre çizgileri ayır
                for x1, y1, x2, y2, slope, mid_x in valid_lines:
                    # Sol şerit çizgileri (negatif eğim)
                    if slope < 0 and mid_x < self.width * 0.6:
                        left_lines.append((x1, y1, x2, y2, slope))
                    
                    # Sağ şerit çizgileri (pozitif eğim)
                    elif slope > 0 and mid_x > self.width * 0.4:
                        right_lines.append((x1, y1, x2, y2, slope))
            
        # Şerit doğrulama: Minimum sayıda çizgi kontrolü
        if len(left_lines) < 1:
            left_lines = []
            
        if len(right_lines) < 1:
            right_lines = []
        
        return left_lines, right_lines
    
    def fit_lane_lines(self, lines, is_left=True):
        """Şerit çizgilerini uydur"""
        if not lines:
            # Şerit bulunamadı, hafızadaki son şeridi kullan
            if is_left:
                if len(self.left_fit_history) > 0:
                    # Güven değerine göre azaltılmış bir şerit tahmini yap
                    confidence = max(0.1, self.confidence_threshold - 0.1 * self.consecutive_detection_failures)
                    avg_fit = np.mean(self.left_fit_history, axis=0)
                    return avg_fit * confidence
                return None
            else:
                if len(self.right_fit_history) > 0:
                    confidence = max(0.1, self.confidence_threshold - 0.1 * self.consecutive_detection_failures)
                    avg_fit = np.mean(self.right_fit_history, axis=0)
                    return avg_fit * confidence
                return None
            
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2, _ in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        fit = np.polyfit(y_coords, x_coords, deg=1)
        
        # Önceki değerlerle yumuşatma
        last_fit = self.last_left_fit if is_left else self.last_right_fit
        if last_fit is not None:
            fit = self.smooth_factor * last_fit + (1 - self.smooth_factor) * fit
        
        # Şerit hafızasını güncelle
        if is_left:
            self.last_left_fit = fit
            self.left_fit_history.append(fit)
            if len(self.left_fit_history) > self.max_history_frames:
                self.left_fit_history.pop(0)
        else:
            self.last_right_fit = fit
            self.right_fit_history.append(fit)
            if len(self.right_fit_history) > self.max_history_frames:
                self.right_fit_history.pop(0)
            
        return fit
    
    def draw_lanes(self, image, left_fit, right_fit):
        """Şeritleri çiz ve şerit bölgesini doldur"""
        overlay = np.zeros_like(image)
        
        # Şeritleri çizebilmek için her iki şeride de ihtiyaç var
        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(self.height * 0.6, self.height, 20)
            
            # Sol şerit
            left_fitx = left_fit[0] * ploty + left_fit[1]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_left]), False, (0, 255, 0), 8)
            
            # Sağ şerit
            right_fitx = right_fit[0] * ploty + right_fit[1]
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_right]), False, (0, 255, 0), 8)
            
            # Şeritler arası alanı doldur
            pts = np.hstack((pts_left, np.fliplr(pts_right)))
            cv2.fillPoly(overlay, np.int32([pts]), (0, 100, 0))
            
            # Şerit genişliğini kontrol et - çok geniş veya dar şeritler için uyarı
            bottom_width = abs(right_fitx[-1] - left_fitx[-1])
            expected_width = self.width * 0.4  # Beklenen şerit genişliği
            
            if bottom_width < expected_width * 0.5 or bottom_width > expected_width * 1.5:
                if self.debug:
                    cv2.putText(overlay, "Gecersiz Serit Genisligi", (10, 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Sadece tek şerit algılandıysa
        elif left_fit is not None:
            ploty = np.linspace(self.height * 0.6, self.height, 20)
            left_fitx = left_fit[0] * ploty + left_fit[1]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_left]), False, (0, 255, 0), 8)
            
        elif right_fit is not None:
            ploty = np.linspace(self.height * 0.6, self.height, 20)
            right_fitx = right_fit[0] * ploty + right_fit[1]
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_right]), False, (0, 255, 0), 8)
        
        return cv2.addWeighted(image, 1, overlay, 0.5, 0)
    
    def calculate_center_diff(self, left_fit, right_fit):
        """Merkez sapmasını hesapla"""
        if left_fit is None and right_fit is None:
            return None
            
        # Alt noktadaki şerit pozisyonları
        y = self.height
        
        if left_fit is not None and right_fit is not None:
            left_x = left_fit[0] * y + left_fit[1]
            right_x = right_fit[0] * y + right_fit[1]
            lane_center = (left_x + right_x) / 2
            
        elif left_fit is not None:
            left_x = left_fit[0] * y + left_fit[1]
            lane_center = left_x + self.width * 0.25  # Tahmini şerit genişliği
            
        else:  # right_fit is not None
            right_x = right_fit[0] * y + right_fit[1]
            lane_center = right_x - self.width * 0.25  # Tahmini şerit genişliği
        
        image_center = self.width / 2
        center_diff = lane_center - image_center
        
        return center_diff
    
    def process_frame(self, frame):
        """Ana işleme fonksiyonu"""
        # Görüntüyü işle
        edges = self.preprocess_image(frame)
        
        # Şeritleri tespit et
        left_lines, right_lines = self.detect_lane_lines(edges)
        
        # Şerit kurtarma modunu kontrol et
        current_time = time.time()
        if self.lane_recovery_mode:
            if current_time - self.recovery_start_time > self.recovery_timeout:
                self.lane_recovery_mode = False
                self.consecutive_detection_failures = 0
                if self.debug:
                    logger.info("Şerit kurtarma modu tamamlandı")
        
        # Şerit tespiti başarısını kontrol et
        if not left_lines and not right_lines:
            self.consecutive_detection_failures += 1
            if self.consecutive_detection_failures > self.max_detection_failures and not self.lane_recovery_mode:
                self.lane_recovery_mode = True
                self.recovery_start_time = current_time
                if self.debug:
                    logger.warning(f"Şerit kurtarma modu başlatıldı ({self.consecutive_detection_failures} başarısız tespit)")
        else:
            # En az bir şerit bulunduğunda hata sayacını azalt
            self.consecutive_detection_failures = max(0, self.consecutive_detection_failures - 1)
        
        # Şeritleri uydur
        left_fit = self.fit_lane_lines(left_lines, is_left=True)
        right_fit = self.fit_lane_lines(right_lines, is_left=False)
        
        # Merkez sapmasını hesapla
        center_diff = self.calculate_center_diff(left_fit, right_fit)
        
        # Görselleştirme
        result = self.draw_lanes(frame, left_fit, right_fit)
        
        # Debug bilgileri
        if self.debug:
            if center_diff is not None:
                cv2.putText(result, f"Merkez Farki: {center_diff:.1f}px",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Şerit kurtarma modu bilgisi
            if self.lane_recovery_mode:
                cv2.putText(result, "SERIT KURTARMA MODU",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Hata sayacı
            cv2.putText(result, f"Hata: {self.consecutive_detection_failures}/{self.max_detection_failures}",
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ROI'yi göster
            cv2.polylines(result, [self.roi_vertices], True, (0, 0, 255), 2)
        
        return result, center_diff 