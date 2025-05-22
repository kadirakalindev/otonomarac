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
        """
        LaneDetector sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            debug (bool): Hata ayıklama modunu aktifleştirir
        """
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
            [self.width * 0.20, self.height * 0.40],  # Daha yukarı ve geniş ROI için değiştirildi
            [self.width * 0.80, self.height * 0.40],  # Daha yukarı ve geniş ROI için değiştirildi
            [self.width, self.height]
        ], dtype=np.int32)
        
        # Orta şerit çizgisi (kalibrasyon ile ayarlanabilir)
        self.center_line = np.array([
            [self.width // 2, self.height],
            [self.width // 2, int(self.height * 0.40)]  # ROI ile uyumlu hale getirildi
        ], dtype=np.int32)
        
        # Temel filtre parametreleri - hassasiyet artırıldı
        self.blur_kernel = 7  # Daha büyük blur kernel
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 20     # Daha düşük eşik - daha fazla kenar yakalanır
        self.canny_high = 80    # Daha düşük üst eşik
        
        # Hough parametreleri - hassasiyet artırıldı ve sınırlar eklendi
        self.rho = max(1, min(2, self.width / 320))  # Çözünürlüğe göre ayarla
        self.theta = np.pi/180
        self.hough_threshold = 10   # Daha da düşürüldü - daha fazla çizgi tespit edilir
        self.min_line_length = max(8, self.height * 0.05)  # Minimum çizgi uzunluğu azaltıldı
        self.max_line_gap = min(60, self.height * 0.35)     # Çizgi boşluğu artırıldı
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.last_center_fit = None  # Orta şerit için hafıza
        self.left_fit_history = []
        self.right_fit_history = []
        self.center_fit_history = []  # Orta şerit geçmişi
        self.max_history_frames = 20  # Daha uzun hafıza
        self.smooth_factor = 0.5  # Yumuşatma faktörü - daha yumuşak geçişler için azaltıldı
        self.confidence_threshold = 0.4  # Güven eşiği - daha toleranslı
        
        # Şerit kaybı için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 25  # Daha toleranslı
        self.lane_recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_timeout = 4.0  # 4 saniye kurtarma modu
        
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
            if 'roi_vertices' in calibration:
                self.roi_vertices = np.array(calibration['roi_vertices'], dtype=np.int32)
                
            # Orta şerit çizgisini güncelle (eğer varsa)
            if 'center_line' in calibration:
                self.center_line = np.array(calibration['center_line'], dtype=np.int32)
                
            # Filtre parametrelerini güncelle
            if 'blur_kernel' in calibration:
                self.blur_kernel = calibration['blur_kernel']
            if 'canny_low_threshold' in calibration:
                self.canny_low = calibration['canny_low_threshold']
            if 'canny_high_threshold' in calibration:
                self.canny_high = calibration['canny_high_threshold']
                
            # Hough parametrelerini güncelle
            if 'hough_threshold' in calibration:
                self.hough_threshold = calibration['hough_threshold']
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
            edges, self.rho, self.theta, self.hough_threshold,
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
    
    def detect_center_lane(self, edges):
        """Orta şeridi tespit et"""
        # Orta şeridin etrafındaki bölgeyi maskele
        center_mask = np.zeros_like(edges)
        center_width = self.width * 0.2  # Orta şerit genişliği
        
        # Orta şeridin alt ve üst noktaları
        bottom_center = self.center_line[0][0], self.center_line[0][1]
        top_center = self.center_line[1][0], self.center_line[1][1]
        
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
            center_edges, self.rho, self.theta, self.hough_threshold,
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
                    confidence = max(0.1, 1 - 0.1 * self.consecutive_detection_failures)
                    avg_fit = np.mean(self.left_fit_history, axis=0)
                    return avg_fit * confidence
                return None
            else:
                if len(self.right_fit_history) > 0:
                    confidence = max(0.1, 1 - 0.1 * self.consecutive_detection_failures)
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
            if len(self.left_fit_history) > 0 and len(self.right_fit_history) > 0:
                confidence = max(0.1, 1 - 0.1 * self.consecutive_detection_failures)
                avg_left_fit = np.mean(self.left_fit_history, axis=0)
                avg_right_fit = np.mean(self.right_fit_history, axis=0)
                avg_fit = (avg_left_fit + avg_right_fit) / 2
                return avg_fit * confidence
            return None
            
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2, _ in center_lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        fit = np.polyfit(y_coords, x_coords, deg=1)
        
        # Önceki değerlerle yumuşatma
        if self.last_left_fit is not None and self.last_right_fit is not None:
            avg_fit = (self.last_left_fit + self.last_right_fit) / 2
            fit = self.smooth_factor * avg_fit + (1 - self.smooth_factor) * fit
        
        # Şerit hafızasını güncelle
        self.last_left_fit = fit
        self.last_right_fit = fit
        self.left_fit_history.append(fit)
        self.right_fit_history.append(fit)
        if len(self.left_fit_history) > self.max_history_frames:
            self.left_fit_history.pop(0)
            self.right_fit_history.pop(0)
            
        return fit
    
    def draw_lanes(self, image, left_fit, right_fit, center_fit=None):
        """Şeritleri çiz"""
        overlay = np.zeros_like(image)
        
        # Sol ve sağ şeritleri çiz (eğer varsa)
        if left_fit is not None or right_fit is not None:
            ploty = np.linspace(self.height * 0.6, self.height, 20)
            
            if left_fit is not None:
                left_fitx = left_fit[0] * ploty + left_fit[1]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_left]), False, (0, 255, 0), 2)
            
            if right_fit is not None:
                right_fitx = right_fit[0] * ploty + right_fit[1]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_right]), False, (0, 255, 0), 2)
            
            # Şeritler arası alanı doldur
            if left_fit is not None and right_fit is not None:
                pts = np.hstack((pts_left, np.fliplr(pts_right)))
                cv2.fillPoly(overlay, np.int32([pts]), (0, 100, 0))
        
        # Orta şeridi çiz (eğer varsa) - daha kalın ve belirgin
        if center_fit is not None:
            ploty = np.linspace(self.height * 0.5, self.height, 20)
            center_fitx = center_fit[0] * ploty + center_fit[1]
            pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_center]), False, (0, 0, 255), 4)
            
            # Orta şeridin alt noktasını belirgin göster
            bottom_y = self.height
            bottom_x = int(center_fit[0] * bottom_y + center_fit[1])
            cv2.circle(overlay, (bottom_x, bottom_y), 8, (255, 0, 0), -1)
        
        return cv2.addWeighted(image, 1, overlay, 0.5, 0)
    
    def calculate_center_diff(self, left_fit, right_fit):
        """Merkez sapmasını hesapla - sol ve sağ şeritlere göre"""
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
        
        # Şeritleri tespit et (sol ve sağ)
        left_lines, right_lines = self.detect_lane_lines(edges)
        
        # Orta şeridi tespit et
        center_lines = self.detect_center_lane(edges)
        
        # Şeritleri uydur
        left_fit = self.fit_lane_lines(left_lines, is_left=True)
        right_fit = self.fit_lane_lines(right_lines, is_left=False)
        center_fit = self.fit_center_lane(center_lines)
        
        # Merkez sapmasını hesapla - öncelikle sol ve sağ şeritlere göre
        center_diff = self.calculate_center_diff(left_fit, right_fit)
        
        # Görselleştirme
        result = self.draw_lanes(frame, left_fit, right_fit, center_fit)
        
        # Debug bilgileri
        if self.debug:
            if center_diff is not None:
                cv2.putText(result, f"Merkez Farki: {center_diff:.1f}px",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hata sayacı
            cv2.putText(result, f"Hata: {self.consecutive_detection_failures}/{self.max_detection_failures}",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Orta şerit durumu
            if center_fit is not None:
                cv2.putText(result, "Orta Serit: BULUNDU",
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(result, "Orta Serit: YOK",
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ROI'yi göster
            cv2.polylines(result, [self.roi_vertices], True, (0, 0, 255), 2)
            
            # Orta şerit referans çizgisini göster
            cv2.line(result, 
                    (self.center_line[0][0], self.center_line[0][1]), 
                    (self.center_line[1][0], self.center_line[1][1]), 
                    (255, 255, 0), 2)
        
        return result, center_diff
        
    def detect_lanes(self, frame):
        """
        Verilen karedeki şeritleri algılar
        
        Args:
            frame (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            tuple: (sol_şerit_pozisyonu, sağ_şerit_pozisyonu, merkez_sapması, işlenmiş_görüntü)
        """
        # Frame geçerliliğini kontrol et
        if not self.validate_frame(frame):
            return None, None, None, frame if self.debug else None
        
        # Görüntü kopyasını oluştur
        processed = frame.copy()
        
        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.debug:
            self.debug_images['gray'] = gray.copy()
        
        # Gürültüyü azaltmak için Gaussian blur uygula - kernel boyutu artırıldı
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if self.debug:
            self.debug_images['blurred'] = blurred.copy()
        
        # Kenarları tespit et - daha hassas thresholds
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        if self.debug:
            self.debug_images['edges'] = edges.copy()
        
        # ROI maskesini uygula
        masked_edges = self.preprocess_image(frame)
        if self.debug:
            self.debug_images['masked_edges'] = masked_edges.copy()
        
        # Çizgileri tespit et - optimize edilmiş parametrelerle
        lines = cv2.HoughLinesP(
            masked_edges,
            self.rho,
            self.theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Şeritleri hesapla
        left_lane, right_lane, center_diff = self._process_lines(lines, frame.shape)
        
        # Şerit algılaması başarısını kontrol et
        if left_lane is None and right_lane is None:
            self.consecutive_detection_failures += 1
            if self.consecutive_detection_failures <= self.max_detection_failures and (self.last_left_fit is not None or self.last_right_fit is not None):
                # Hafızadan şeritleri kullan
                left_lane = self.last_left_fit
                right_lane = self.last_right_fit
                if left_lane is not None and right_lane is not None:
                    # Hafıza kullanımında merkez farkını hesapla
                    center_diff = self.calculate_center_diff(left_lane, right_lane)
        else:
            # Başarılı algılama - sayacı sıfırla
            self.consecutive_detection_failures = 0
            
            # Şerit hafızasını güncelle
            if left_lane is not None:
                if self.last_left_fit is not None:
                    # Yumuşatma uygula
                    left_lane = {
                        'x1': int(left_lane['x1'] * (1 - self.smooth_factor) + self.last_left_fit['x1'] * self.smooth_factor),
                        'y1': int(left_lane['y1'] * (1 - self.smooth_factor) + self.last_left_fit['y1'] * self.smooth_factor),
                        'x2': int(left_lane['x2'] * (1 - self.smooth_factor) + self.last_left_fit['x2'] * self.smooth_factor),
                        'y2': int(left_lane['y2'] * (1 - self.smooth_factor) + self.last_left_fit['y2'] * self.smooth_factor)
                    }
                self.last_left_fit = left_lane
                self.left_fit_history.append(left_lane)
                if len(self.left_fit_history) > self.max_history_frames:
                    self.left_fit_history.pop(0)
                
            if right_lane is not None:
                if self.last_right_fit is not None:
                    # Yumuşatma uygula
                    right_lane = {
                        'x1': int(right_lane['x1'] * (1 - self.smooth_factor) + self.last_right_fit['x1'] * self.smooth_factor),
                        'y1': int(right_lane['y1'] * (1 - self.smooth_factor) + self.last_right_fit['y1'] * self.smooth_factor),
                        'x2': int(right_lane['x2'] * (1 - self.smooth_factor) + self.last_right_fit['x2'] * self.smooth_factor),
                        'y2': int(right_lane['y2'] * (1 - self.smooth_factor) + self.last_right_fit['y2'] * self.smooth_factor)
                    }
                self.last_right_fit = right_lane
                self.right_fit_history.append(right_lane)
                if len(self.right_fit_history) > self.max_history_frames:
                    self.right_fit_history.pop(0)
        
        # Debug modunda görselleştirme
        if self.debug:
            # İlk aşamada temiz bir görüntü al
            lane_image = frame.copy()
            
            # ROI görselleştir
            roi_image = self.draw_lanes(lane_image.copy(), self.last_left_fit, self.last_right_fit)
            self.debug_images['roi'] = roi_image
            
            # Merkez çizgisi çiz
            cv2.line(lane_image, 
                    (self.center_line[0][0], self.center_line[0][1]), 
                    (self.center_line[1][0], self.center_line[1][1]), 
                    (255, 255, 0), 2)
            
            # Bulunan şeritleri çiz
            if left_lane:
                cv2.line(lane_image, 
                        (left_lane['x1'], left_lane['y1']), 
                        (left_lane['x2'], left_lane['y2']), 
                        (0, 255, 0), 8)
            
            if right_lane:
                cv2.line(lane_image, 
                        (right_lane['x1'], right_lane['y1']), 
                        (right_lane['x2'], right_lane['y2']), 
                        (0, 0, 255), 8)
            
            # Merkez sapması bilgisini ekle
            if center_diff is not None:
                direction = "MERKEZ"
                if center_diff < -10:
                    direction = "SOLA"
                elif center_diff > 10:
                    direction = "SAĞA"
                
                cv2.putText(lane_image, 
                           f"Sapma: {center_diff:.1f}px {direction}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 255), 
                           2)
            
            processed = lane_image
        
        return left_lane, right_lane, center_diff, processed if self.debug else None
        
    def _process_lines(self, lines, frame_shape):
        """
        Verilen çizgileri işleyerek şeritleri hesaplar
        
        Args:
            lines (numpy.ndarray): Hough transform çıktısı
            frame_shape (tuple): Görüntü boyutları
            
        Returns:
            tuple: (sol_şerit_pozisyonu, sağ_şerit_pozisyonu, merkez_sapması)
        """
        left_lane = None
        right_lane = None
        center_diff = None
        
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
                if slope < 0 and x1 < frame_shape[1] * 0.6:
                    left_lane = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                elif slope > 0 and x1 > frame_shape[1] * 0.4:
                    right_lane = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        
        if left_lane is not None and right_lane is not None:
            center_diff = self.calculate_center_diff(left_lane, right_lane)
        
        return left_lane, right_lane, center_diff
        
    def calculate_center_diff(self, left_fit, right_fit):
        """Merkez sapmasını hesapla - sol ve sağ şeritlere göre"""
        if left_fit is None and right_fit is None:
            return None
            
        # Alt noktadaki şerit pozisyonları
        y = self.height
        
        if left_fit is not None and right_fit is not None:
            left_x = left_fit['x1'] * y + left_fit['y1']
            right_x = right_fit['x1'] * y + right_fit['y1']
            lane_center = (left_x + right_x) / 2
            
        elif left_fit is not None:
            left_x = left_fit['x1'] * y + left_fit['y1']
            lane_center = left_x + self.width * 0.25  # Tahmini şerit genişliği
            
        else:  # right_fit is not None
            right_x = right_fit['x1'] * y + right_fit['y1']
            lane_center = right_x - self.width * 0.25  # Tahmini şerit genişliği
        
        image_center = self.width / 2
        center_diff = lane_center - image_center
        
        return center_diff 