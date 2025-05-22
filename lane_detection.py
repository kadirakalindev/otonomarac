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
        
        # ROI parametreleri - merkezi daha geniş bir alanı kapsayacak şekilde güncellendi
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Geçersiz çözünürlük değerleri")
            
        self.roi_vertices = np.array([
            [int(self.width * 0.05), self.height],                   # Sol alt - daha geniş alt kenar
            [int(self.width * 0.25), int(self.height * 0.45)],       # Sol üst - daha geniş görüş
            [int(self.width * 0.75), int(self.height * 0.45)],       # Sağ üst - daha geniş görüş
            [int(self.width * 0.95), self.height]                    # Sağ alt - daha geniş alt kenar
        ], dtype=np.int32)
        
        # Orta şerit çizgisi (kalibrasyon ile ayarlanabilir)
        self.center_line = np.array([
            [self.width // 2, self.height],
            [self.width // 2, int(self.height * 0.45)]  # Daha yukarıya uzatıldı
        ], dtype=np.int32)
        
        # Görüntü işleme parametreleri - ışık koşullarına daha duyarlı
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 25    # Daha hassas kenar tespiti için düşürüldü
        self.canny_high = 100  # Daha hassas kenar tespiti için düşürüldü
        
        # İlave görüntü işleme parametreleri
        self.adaptive_threshold_block_size = 11  # Bölgesel ışık değişimlerine adaptasyon için
        self.adaptive_threshold_constant = 2     # Duyarlılık ayarı
        
        # Renk maskeleme için HSV eşikleri (beyaz ve sarı şeritler için)
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        self.yellow_lower = np.array([15, 80, 120])
        self.yellow_upper = np.array([35, 255, 255])
        
        # Hough parametreleri - hassasiyet artırıldı, çözünürlüğe göre ayarlandı
        self.rho = max(1, min(2, self.width / 320))  # Çözünürlüğe göre ayarla
        self.theta = np.pi/180
        self.hough_threshold = int(max(15, self.height / 16))        # Çözünürlüğe göre ayarla
        self.min_line_length = max(15, int(self.height * 0.12))      # Çözünürlüğe göre ayarla
        self.max_line_gap = min(40, int(self.height * 0.18))         # Çözünürlüğe göre ayarla
        
        # Şerit tespiti parametreleri
        self.line_filter_slope_threshold = 0.25   # Yatay çizgileri filtrelemek için eğim eşiği
        self.edge_detection_kernel = np.ones((3, 3), np.uint8)  # Kenar genişletme için kernel
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.last_center_fit = None  # Orta şerit için hafıza
        self.left_fit_history = []
        self.right_fit_history = []
        self.center_fit_history = []  # Orta şerit geçmişi
        self.max_history_frames = 15  # Hafıza uzunluğu
        self.smooth_factor = 0.8  # Yumuşatma faktörü (daha fazla yumuşatma)
        self.confidence_threshold = 0.6  # Güven eşiği
        
        # Şerit kaybı için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 15
        self.lane_recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_timeout = 3.0  # 3 saniye kurtarma modu
        
        # Debug görüntüleri için sözlük
        self.debug_images = {}
        
    def validate_frame(self, frame):
        """Gelen kareyi doğrula"""
        if frame is None:
            logger.error("Boş kare alındı")
            return False
            
        if len(frame.shape) != 3:
            logger.error(f"Geçersiz kare formatı: {frame.shape}")
            return False
            
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            logger.warning(f"Kare boyutu uyumsuz: Beklenen {self.width}x{self.height}, Alınan {frame.shape[1]}x{frame.shape[0]}")
            return False
            
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
                
            # Yeni kalibrasyon format kontrolü (kalibrasyon_optimize.py)
            if 'src_points' in calibration and 'dst_points' in calibration:
                logger.info("kalibrasyon_optimize.py formatında kalibrasyon dosyası tespit edildi.")
                
                # src_points'i ROI olarak kullan
                src_points = calibration['src_points']
                
                # ROI noktalarını güncelle
                self.roi_vertices = np.array([
                    src_points[2],  # Sol alt
                    src_points[0],  # Sol üst
                    src_points[1],  # Sağ üst
                    src_points[3]   # Sağ alt
                ], dtype=np.int32)
                
                # Orta şerit çizgisini oluştur
                self.center_line = np.array([
                    [(src_points[2][0] + src_points[3][0]) // 2, self.height],  # Alt orta nokta
                    [(src_points[0][0] + src_points[1][0]) // 2, (src_points[0][1] + src_points[1][1]) // 2]  # Üst orta nokta
                ], dtype=np.int32)
                
                logger.info("ROI ve orta şerit çizgisi güncellendi.")
            
            # Şerit tespit parametrelerini ayarla
            if 'canny_low_threshold' in calibration:
                self.canny_low = calibration['canny_low_threshold']
            if 'canny_high_threshold' in calibration:
                self.canny_high = calibration['canny_high_threshold']
            if 'blur_kernel_size' in calibration:
                self.blur_kernel = calibration['blur_kernel_size']
                if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
                    self.blur_kernel += 1
            
            # Hough parametrelerini ayarla
            if 'hough_threshold' in calibration:
                self.hough_threshold = calibration['hough_threshold']
            if 'min_line_length' in calibration:
                self.min_line_length = calibration['min_line_length']
            if 'max_line_gap' in calibration:
                self.max_line_gap = calibration['max_line_gap']
            
            # Renk parametreleri (eğer varsa)
            if 'white_lower' in calibration:
                self.white_lower = np.array(calibration['white_lower'])
            if 'white_upper' in calibration:
                self.white_upper = np.array(calibration['white_upper'])
            if 'yellow_lower' in calibration:
                self.yellow_lower = np.array(calibration['yellow_lower'])
            if 'yellow_upper' in calibration:
                self.yellow_upper = np.array(calibration['yellow_upper'])
                
            logger.info(f"Kalibrasyon dosyası yüklendi: {calibration_file}")
            return True
                
        except Exception as e:
            logger.error(f"Kalibrasyon dosyası yüklenirken hata: {e}")
            return False
        
    def preprocess_image(self, image):
        """Temel görüntü ön işleme - geliştirilmiş versiyon"""
        # Görüntü doğrulama
        if image is None or image.size == 0:
            logger.error("Geçersiz görüntü")
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # BGR'den HSV'ye dönüştür
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Renk maskesi oluştur (beyaz ve sarı şeritler için)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Maskeleri birleştir
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Gürültü azaltma
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Adaptif eşikleme - bölgesel parlaklık farklılıklarına daha iyi adapte olur
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            self.adaptive_threshold_block_size, 
            self.adaptive_threshold_constant
        )
        
        # Renk maskesi ve eşiklenmiş görüntüyü birleştir
        combined_binary = cv2.bitwise_or(thresh, color_mask)
        
        # Kenar tespiti (Canny)
        edges = cv2.Canny(combined_binary, self.canny_low, self.canny_high)
        
        # Kenarları genişlet (daha belirgin hale getir)
        edges = cv2.dilate(edges, self.edge_detection_kernel, iterations=1)
        
        # ROI maskesi
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked = cv2.bitwise_and(edges, mask)
        
        if self.debug:
            self.debug_images["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.debug_images["color_mask"] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            self.debug_images["threshold"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.debug_images["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["masked"] = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        
        return masked
    
    def detect_lane_lines(self, edges):
        """Basitleştirilmiş şerit tespiti - geliştirilmiş versiyon"""
        lines = cv2.HoughLinesP(
            edges, self.rho, self.theta, self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çizgi uzunluğu çok kısaysa atla
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length < self.min_line_length:
                    continue
                
                # Düşey çizgilerde bölünmeyi önle
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Düşük eğimli (yatay) çizgileri filtrele
                if abs(slope) < self.line_filter_slope_threshold:
                    continue
                
                # Görüntünün alt kısmında olan çizgilere daha fazla ağırlık ver
                # (genellikle şeritlerin daha güvenilir kısmı)
                y_factor = 1.0 + 0.5 * (min(y1, y2) / self.height)
                weight = line_length * abs(slope) * y_factor
                
                # Sol ve sağ şeritleri ayır
                if slope < 0 and x1 < self.width * 0.6:
                    left_lines.append((x1, y1, x2, y2, slope, weight))
                elif slope > 0 and x1 > self.width * 0.4:
                    right_lines.append((x1, y1, x2, y2, slope, weight))
        
        # Çizgileri ağırlıklarına göre sırala (en önemlileri en üstte)
        left_lines.sort(key=lambda x: x[5], reverse=True)
        right_lines.sort(key=lambda x: x[5], reverse=True)
        
        # Sadece en iyi çizgileri seç (gürültüyü azaltmak için)
        max_lines = 5  # En çok bu kadar çizgi kullan
        left_lines = left_lines[:max_lines]
        right_lines = right_lines[:max_lines]
        
        return left_lines, right_lines
    
    def detect_center_lane(self, edges):
        """Orta şeridi tespit et"""
        # Orta şeridin etrafındaki bölgeyi maskele
        center_mask = np.zeros_like(edges)
        center_width = self.width * 0.2  # Orta şerit genişliği
        
        # Orta şeridin alt ve üst noktaları
        bottom_center = self.center_line[0]
        top_center = self.center_line[1]
        
        # Orta şerit için maske oluştur
        center_roi = np.array([
            [bottom_center[0] - center_width/2, bottom_center[1]],
            [top_center[0] - center_width/2, top_center[1]],
            [top_center[0] + center_width/2, top_center[1]],
            [bottom_center[0] + center_width/2, bottom_center[1]]
        ], dtype=np.int32)
        
        cv2.fillPoly(center_mask, [center_roi], 255)
        center_edges = cv2.bitwise_and(edges, center_mask)
        
        # Orta şerit çizgilerini tespit et
        lines = cv2.HoughLinesP(
            center_edges, self.rho, self.theta, 15,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        center_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Yatay çizgileri filtrele
                if abs(slope) < 0.3:
                    continue
                
                center_lines.append((x1, y1, x2, y2, slope))
        
        if self.debug:
            self.debug_images["center_edges"] = cv2.cvtColor(center_edges, cv2.COLOR_GRAY2BGR)
        
        return center_lines
        
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
    
    def fit_center_lane(self, center_lines):
        """Orta şerit çizgisini uydur"""
        if not center_lines:
            # Şerit bulunamadı, hafızadaki son şeridi kullan
            if len(self.center_fit_history) > 0:
                confidence = max(0.1, self.confidence_threshold - 0.1 * self.consecutive_detection_failures)
                avg_fit = np.mean(self.center_fit_history, axis=0)
                return avg_fit * confidence
            return None
            
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2, _ in center_lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        fit = np.polyfit(y_coords, x_coords, deg=1)
        
        # Önceki değerlerle yumuşatma
        if self.last_center_fit is not None:
            fit = self.smooth_factor * self.last_center_fit + (1 - self.smooth_factor) * fit
        
        # Şerit hafızasını güncelle
        self.last_center_fit = fit
        self.center_fit_history.append(fit)
        if len(self.center_fit_history) > self.max_history_frames:
            self.center_fit_history.pop(0)
            
        return fit
    
    def draw_lanes(self, image, left_fit, right_fit, center_fit=None):
        """Şeritleri çiz - geliştirilmiş versiyon"""
        overlay = np.zeros_like(image)
        
        # Sol ve sağ şeritleri çiz (eğer varsa)
        if left_fit is not None or right_fit is not None:
            # Alt yarıya odaklan - şeritlerin en güvenilir kısmı
            ploty = np.linspace(self.height * 0.5, self.height, 30)  # Daha fazla nokta
            
            # Şerit güven göstergesi renkleri
            high_confidence_color = (0, 255, 0)  # Yeşil - yüksek güven
            medium_confidence_color = (0, 255, 255)  # Sarı - orta güven
            low_confidence_color = (0, 128, 255)  # Turuncu - düşük güven
            
            # Şerit güven seviyesini belirle
            left_confidence = 1.0 
            right_confidence = 1.0
            
            if len(self.left_fit_history) > 0:
                left_confidence = min(1.0, 1.0 - (0.05 * self.consecutive_detection_failures))
            if len(self.right_fit_history) > 0:
                right_confidence = min(1.0, 1.0 - (0.05 * self.consecutive_detection_failures))
            
            # Renkleri güven seviyesine göre belirle
            if left_fit is not None:
                if left_confidence > 0.8:
                    left_color = high_confidence_color
                elif left_confidence > 0.5:
                    left_color = medium_confidence_color
                else:
                    left_color = low_confidence_color
                    
                left_fitx = left_fit[0] * ploty + left_fit[1]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_left]), False, left_color, 2)
            
            if right_fit is not None:
                if right_confidence > 0.8:
                    right_color = high_confidence_color
                elif right_confidence > 0.5:
                    right_color = medium_confidence_color
                else:
                    right_color = low_confidence_color
                    
                right_fitx = right_fit[0] * ploty + right_fit[1]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_right]), False, right_color, 2)
            
            # Şeritler arası alanı doldur (sadece her iki şerit de tespit edildiğinde)
            if left_fit is not None and right_fit is not None:
                left_fitx = left_fit[0] * ploty + left_fit[1]
                right_fitx = right_fit[0] * ploty + right_fit[1]
                pts = np.hstack((
                    np.array([np.transpose(np.vstack([left_fitx, ploty]))]),
                    np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                ))
                cv2.fillPoly(overlay, np.int32([pts]), (0, 80, 0))
        
        # Orta şeridi çiz (eğer varsa) - daha belirgin
        if center_fit is not None:
            ploty = np.linspace(self.height * 0.4, self.height, 30)
            center_fitx = center_fit[0] * ploty + center_fit[1]
            pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
            
            # Orta şerit çizgisini daha kalın ve belirgin yap
            cv2.polylines(overlay, np.int32([pts_center]), False, (0, 0, 255), 5)
            
            # Orta şeridin alt noktasını belirgin göster (takip referansı)
            bottom_y = self.height
            bottom_x = int(center_fit[0] * bottom_y + center_fit[1])
            cv2.circle(overlay, (bottom_x, bottom_y), 10, (255, 0, 0), -1)
        
        # Görüntüyü daha iyi görünürlük için birleştir
        result = cv2.addWeighted(image, 1, overlay, 0.6, 0)
        
        # ROI bölgesini göster (debug modunda)
        if self.debug:
            cv2.polylines(result, [self.roi_vertices], True, (0, 255, 255), 2)
            cv2.polylines(result, [self.center_line], False, (255, 255, 0), 2)
            
        return result
    
    def calculate_center_diff(self, center_fit):
        """Merkez sapmasını hesapla - orta şeride göre"""
        if center_fit is None:
            return None
            
        # Alt noktadaki şerit pozisyonu
        y = self.height
        center_x = center_fit[0] * y + center_fit[1]
        
        # Görüntü merkezi ile şerit merkezi arasındaki fark
        image_center = self.width / 2
        center_diff = center_x - image_center
        
        return center_diff
    
    def calculate_center_diff_from_sides(self, left_fit, right_fit):
        """Sol ve sağ şeritlerden merkez sapmasını hesapla"""
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
        """Ana işleme fonksiyonu - geliştirilmiş versiyon"""
        # Kare doğrulama
        if not self.validate_frame(frame):
            # Hatalı kare, boş değer ve hata göster
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(placeholder, "GECERSIZ KARE", (self.width//3, self.height//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return placeholder, None
            
        # Görüntüyü işle
        edges = self.preprocess_image(frame)
        
        # Şeritleri tespit et (sol ve sağ)
        left_lines, right_lines = self.detect_lane_lines(edges)
        
        # Orta şeridi tespit et
        center_lines = self.detect_center_lane(edges)
        
        # Şerit kurtarma modunu kontrol et
        current_time = time.time()
        if self.lane_recovery_mode:
            if current_time - self.recovery_start_time > self.recovery_timeout:
                self.lane_recovery_mode = False
                self.consecutive_detection_failures = 0
                if self.debug:
                    logger.info("Şerit kurtarma modu tamamlandı")
        
        # Şerit tespiti başarısını kontrol et
        if not center_lines and not left_lines and not right_lines:
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
        center_fit = self.fit_center_lane(center_lines)
        
        # Merkez sapmasını hesapla - öncelikle orta şeride göre
        if center_fit is not None:
            center_diff = self.calculate_center_diff(center_fit)
        else:
            # Orta şerit bulunamadıysa, sol ve sağ şeritlere göre hesapla
            center_diff = self.calculate_center_diff_from_sides(left_fit, right_fit)
        
        # Görselleştirme
        result = self.draw_lanes(frame, left_fit, right_fit, center_fit)
        
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
            
            # Orta şerit durumu
            if center_fit is not None:
                cv2.putText(result, "Orta Serit: BULUNDU",
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(result, "Orta Serit: YOK",
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result, center_diff 