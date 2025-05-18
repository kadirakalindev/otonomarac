#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Basitleştirilmiş Şerit Tespiti ve Takibi Modülü
"""

import cv2
import numpy as np
import logging

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
        
        # ROI parametreleri - genişletildi ve doğrulama eklendi
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Geçersiz çözünürlük değerleri")
            
        self.roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.3, self.height * 0.5],  # Daha yukarı ve geniş
            [self.width * 0.7, self.height * 0.5],  # Daha yukarı ve geniş
            [self.width, self.height]
        ], dtype=np.int32)
        
        # Temel filtre parametreleri - hassasiyet artırıldı
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 30    # Düşürüldü
        self.canny_high = 120  # Düşürüldü
        
        # Hough parametreleri - hassasiyet artırıldı ve sınırlar eklendi
        self.rho = max(1, min(2, self.width / 320))  # Çözünürlüğe göre ayarla
        self.theta = np.pi/180
        self.min_line_length = max(15, self.height * 0.1)  # Görüntü boyutuna göre ayarla
        self.max_line_gap = min(40, self.height * 0.2)     # Görüntü boyutuna göre ayarla
        
        # Şerit hafızası ve yumuşatma
        self.last_left_fit = None
        self.last_right_fit = None
        self.smooth_factor = 0.8  # Yumuşatma faktörü
        self.max_memory_frames = 5  # Maksimum hafıza karesi
        
        # Hata durumu için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 10
        
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
        """Basitleştirilmiş şerit tespiti"""
        lines = cv2.HoughLinesP(
            edges, self.rho, self.theta, 20,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Yatay çizgileri filtrele
                if abs(slope) < 0.3:
                    continue
                
                # Sol ve sağ şeritleri ayır
                if slope < 0 and x1 < self.width * 0.6:
                    left_lines.append((x1, y1, x2, y2, slope))
                elif slope > 0 and x1 > self.width * 0.4:
                    right_lines.append((x1, y1, x2, y2, slope))
        
        return left_lines, right_lines
    
    def fit_lane_lines(self, lines, is_left=True):
        """Şerit çizgilerini uydur"""
        if not lines:
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
        
        if is_left:
            self.last_left_fit = fit
        else:
            self.last_right_fit = fit
            
        return fit
    
    def draw_lanes(self, image, left_fit, right_fit):
        """Şeritleri çiz"""
        overlay = np.zeros_like(image)
        
        if left_fit is not None or right_fit is not None:
            ploty = np.linspace(self.height * 0.6, self.height, 20)
            
            if left_fit is not None:
                left_fitx = left_fit[0] * ploty + left_fit[1]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_left]), False, (0, 255, 0), 8)
            
            if right_fit is not None:
                right_fitx = right_fit[0] * ploty + right_fit[1]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_right]), False, (0, 255, 0), 8)
            
            # Şeritler arası alanı doldur
            if left_fit is not None and right_fit is not None:
                pts = np.hstack((pts_left, np.fliplr(pts_right)))
                cv2.fillPoly(overlay, np.int32([pts]), (0, 100, 0))
        
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
            
            # ROI'yi göster
            cv2.polylines(result, [self.roi_vertices], True, (0, 0, 255), 2)
        
        return result, center_diff 