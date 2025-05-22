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
        
        # ROI parametreleri - genişletildi ve doğrulama eklendi
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Geçersiz çözünürlük değerleri")
            
        self.roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.25, self.height * 0.45],  # Daha yukarı ve geniş
            [self.width * 0.75, self.height * 0.45],  # Daha yukarı ve geniş
            [self.width, self.height]
        ], dtype=np.int32)
        
        # Orta şerit çizgisi (kalibrasyon ile ayarlanabilir)
        self.center_line = np.array([
            [self.width // 2, self.height],
            [self.width // 2, int(self.height * 0.45)]
        ], dtype=np.int32)
        
        # Temel filtre parametreleri - viraj tespiti için hassasiyet artırıldı
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 25    # Daha düşürüldü - virajlar için
        self.canny_high = 100  # Daha düşürüldü - virajlar için
        
        # Hough parametreleri - hassasiyet artırıldı ve sınırlar eklendi
        self.rho = max(1, min(2, self.width / 320))  # Çözünürlüğe göre ayarla
        self.theta = np.pi/180
        self.min_line_length = max(10, self.height * 0.05)  # Daha kısa çizgiler
        self.max_line_gap = min(50, self.height * 0.25)     # Daha yüksek aralık
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.last_center_fit = None  # Orta şerit için hafıza
        self.left_fit_history = []
        self.right_fit_history = []
        self.center_fit_history = []  # Orta şerit geçmişi
        self.max_history_frames = 15  # Daha uzun hafıza
        self.smooth_factor = 0.6  # Daha az yumuşatma (virajlar için daha hızlı tepki)
        self.confidence_threshold = 0.5  # Daha düşük güven eşiği
        
        # Viraj tespiti için değişkenler
        self.is_curve = False
        self.curve_direction = "none"  # "left", "right" veya "none"
        self.curve_angle = 0
        self.curve_confidence = 0
        self.curve_history = []
        
        # Şerit kaybı için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 20  # Daha toleranslı
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
            
            # Kalibrasyon optimize formatını tespit et
            if 'src_points' in calibration and 'dst_points' in calibration:
                # kalibrasyon_optimize.py çıktısı
                src_points = np.array(calibration['src_points'], dtype=np.float32)
                
                # ROI noktalarını güncelle
                self.roi_vertices = np.array([
                    src_points[2],  # Sol alt
                    src_points[0],  # Sol üst
                    src_points[1],  # Sağ üst
                    src_points[3]   # Sağ alt
                ], dtype=np.int32)
                
                # Orta şerit çizgisini güncelle
                self.center_line = np.array([
                    [(src_points[2][0] + src_points[3][0]) // 2, self.height],  # Alt orta nokta
                    [(src_points[0][0] + src_points[1][0]) // 2, (src_points[0][1] + src_points[1][1]) // 2]  # Üst orta nokta
                ], dtype=np.int32)
                
                logging.info("kalibrasyon_optimize.py formatında dosya yüklendi")
            else:
                # Eski format dosyası
                # ROI noktalarını güncelle (eğer varsa)
                if 'roi_vertices' in calibration:
                    self.roi_vertices = np.array(calibration['roi_vertices'], dtype=np.int32)
                    
                # Orta şerit çizgisini güncelle (eğer varsa)
                if 'center_line' in calibration:
                    self.center_line = np.array(calibration['center_line'], dtype=np.int32)
                
            # Filtre parametrelerini güncelle
            if 'canny_low_threshold' in calibration:
                self.canny_low = calibration['canny_low_threshold']
            if 'canny_high_threshold' in calibration:
                self.canny_high = calibration['canny_high_threshold']
            if 'blur_kernel_size' in calibration:
                self.blur_kernel = calibration['blur_kernel_size']
                
            # Hough parametrelerini güncelle
            if 'hough_threshold' in calibration:
                self.hough_threshold = calibration.get('hough_threshold', 20)
            if 'min_line_length' in calibration:
                self.min_line_length = calibration['min_line_length']
            if 'max_line_gap' in calibration:
                self.max_line_gap = calibration['max_line_gap']
                
            logging.info(f"Kalibrasyon dosyası yüklendi: {calibration_file}")
            return True
            
        except Exception as e:
            logging.error(f"Kalibrasyon dosyası yüklenirken hata: {e}")
            return False
        
    def preprocess_image(self, image):
        """Temel görüntü ön işleme"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma - derecesini değiştirebiliriz (bilateral filtre daha iyi kenar korur)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)  # Daha iyi kenar koruma
        
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
        # Viraj tespiti için parametreleri ayarla
        current_min_line_length = self.min_line_length
        current_max_line_gap = self.max_line_gap
        
        # Şerit kaybı durumunda farklı parametreler kullan
        if self.lane_recovery_mode or self.consecutive_detection_failures > 5:
            current_min_line_length = max(5, current_min_line_length * 0.7)  # Daha kısa çizgileri kabul et
            current_max_line_gap = min(100, current_max_line_gap * 1.5)  # Daha büyük boşlukları kabul et
        
        # Eğer viraj tespit edildiyse farklı parametreler kullan
        if self.is_curve:
            current_min_line_length = max(5, current_min_line_length * 0.8)  # Virajlarda daha kısa çizgileri kabul et
            current_max_line_gap = min(80, current_max_line_gap * 1.3)  # Virajlarda daha büyük boşlukları kabul et
        
        lines = cv2.HoughLinesP(
            edges, self.rho, self.theta, 20,
            minLineLength=current_min_line_length,
            maxLineGap=current_max_line_gap
        )
        
        left_lines = []
        right_lines = []
        center_lines = []  # Orta şerit için
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Yatay çizgileri filtrele - virajlarda açıyı biraz gevşet
                min_slope = 0.2 if not self.is_curve else 0.1
                if abs(slope) < min_slope:
                    continue
                
                # Görüntü ortasına göre konum
                mid_x = (x1 + x2) / 2
                position = mid_x - self.width / 2  # Negatif: sol taraf, Pozitif: sağ taraf
                
                # Sol ve sağ şeritleri ayır
                if slope < 0 and x1 < self.width * 0.7:  # Sol şerit (biraz daha geniş aralık)
                    left_lines.append((x1, y1, x2, y2, slope, length))
                elif slope > 0 and x1 > self.width * 0.3:  # Sağ şerit (biraz daha geniş aralık)
                    right_lines.append((x1, y1, x2, y2, slope, length))
                
                # Merkeze yakın çizgiler
                if abs(position) < self.width * 0.2:  # Merkez bölgesi
                    center_lines.append((x1, y1, x2, y2, slope, length))
        
        # Viraj tespiti
        self.detect_curve(left_lines, right_lines)
        
        return left_lines, right_lines, center_lines
    
    def detect_curve(self, left_lines, right_lines):
        """Viraj tespiti yapar"""
        # Önceki viraj durumunu hatırla
        prev_is_curve = self.is_curve
        prev_direction = self.curve_direction
        
        # Viraj tespiti için eğim analizi
        left_slopes = [slope for _, _, _, _, slope, _ in left_lines] if left_lines else []
        right_slopes = [slope for _, _, _, _, slope, _ in right_lines] if right_lines else []
        
        # Sol veya sağ şeritte yüksek eğimli çizgiler var mı?
        left_curve = False
        right_curve = False
        curve_angle = 0
        
        if left_slopes:
            avg_left_slope = sum(left_slopes) / len(left_slopes)
            left_curve = avg_left_slope < -0.8  # Daha keskin eğim
            curve_angle = max(curve_angle, abs(avg_left_slope))
        
        if right_slopes:
            avg_right_slope = sum(right_slopes) / len(right_slopes)
            right_curve = avg_right_slope > 0.8  # Daha keskin eğim
            curve_angle = max(curve_angle, abs(avg_right_slope))
        
        # Sol veya sağ çizgi sayısında dengesizlik olması da viraj belirtisi olabilir
        left_count = len(left_lines)
        right_count = len(right_lines)
        line_ratio = 0
        
        if left_count + right_count > 0:
            if left_count > right_count * 2:  # Sol tarafta çok daha fazla çizgi
                right_curve = True
                line_ratio = left_count / max(1, right_count)
            elif right_count > left_count * 2:  # Sağ tarafta çok daha fazla çizgi
                left_curve = True
                line_ratio = right_count / max(1, left_count)
        
        # Viraj tespit
        self.is_curve = left_curve or right_curve
        
        # Viraj yönü
        if left_curve and not right_curve:
            self.curve_direction = "right"  # Sağa viraj (sol şerit belirgin)
        elif right_curve and not left_curve:
            self.curve_direction = "left"  # Sola viraj (sağ şerit belirgin)
        elif left_curve and right_curve:
            # Her iki şerit de belirginse, çizgi sayılarına bakarak karar ver
            self.curve_direction = "left" if left_count < right_count else "right"
        else:
            self.curve_direction = "none"
        
        # Viraj açısı
        self.curve_angle = curve_angle
        
        # Viraj güven değeri (0-1 arası)
        confidence = 0
        if self.is_curve:
            confidence = min(1.0, max(line_ratio / 3.0, curve_angle / 1.2))
        self.curve_confidence = confidence
        
        # Viraj durumunu historia ekle
        self.curve_history.append((self.is_curve, self.curve_direction, self.curve_confidence))
        if len(self.curve_history) > 15:
            self.curve_history.pop(0)
        
        # Viraj durumu değiştiğinde log
        if prev_is_curve != self.is_curve or prev_direction != self.curve_direction:
            if self.is_curve:
                logging.debug(f"Viraj tespit edildi: {self.curve_direction} yönünde, güven: {self.curve_confidence:.2f}")
            else:
                logging.debug("Düz yol tespit edildi")
    
    def detect_center_lane(self, edges):
        """Orta şeridi tespit et"""
        # Orta şeridin etrafındaki bölgeyi maskele
        center_mask = np.zeros_like(edges)
        center_width = self.width * 0.25  # Orta şerit genişliği - biraz artırıldı
        
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
        
        # Viraj durumuna göre parametre ayarları
        min_line_length = self.min_line_length
        max_line_gap = self.max_line_gap
        
        if self.is_curve:
            # Viraj durumunda daha kısa çizgileri kabul et
            min_line_length = max(5, self.min_line_length * 0.7)
            max_line_gap = min(100, self.max_line_gap * 1.3)
        
        # Orta şerit çizgilerini tespit et
        lines = cv2.HoughLinesP(
            center_edges, self.rho, self.theta, 15,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        center_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Yatay çizgileri filtrele - virajlarda daha esnek ol
                min_slope = 0.1 if self.is_curve else 0.2
                if abs(slope) < min_slope:
                    continue
                
                center_lines.append((x1, y1, x2, y2, slope, length))
        
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
            
        # Virajlarda uzun çizgilere daha fazla ağırlık ver
        x_coords = []
        y_coords = []
        weights = []
        
        max_length = max([length for _, _, _, _, _, length in lines])
        
        for x1, y1, x2, y2, _, length in lines:
            # Uzun çizgilere daha fazla ağırlık ver
            weight = (length / max_length) ** 2
            
            # Viraj durumunda ekstra düzenleme
            if self.is_curve:
                # Virajda alt kısımdaki çizgilere daha fazla ağırlık ver
                bottom_y = max(y1, y2)
                bottom_weight = bottom_y / self.height
                weight *= (1 + bottom_weight)
            
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            weights.extend([weight, weight])
            
        # Ağırlıklı polinom uydurma
        fit = np.polyfit(y_coords, x_coords, deg=1, w=weights)
        
        # Önceki değerlerle yumuşatma - virajlarda daha az yumuşatma
        smooth = self.smooth_factor
        if self.is_curve:
            smooth = max(0.3, smooth - 0.2)  # Virajlarda daha az yumuşatma
        
        last_fit = self.last_left_fit if is_left else self.last_right_fit
        if last_fit is not None:
            fit = smooth * last_fit + (1 - smooth) * fit
        
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
            
        # Çizgilere ağırlık ver
        x_coords = []
        y_coords = []
        weights = []
        
        max_length = max([length for _, _, _, _, _, length in center_lines])
        
        for x1, y1, x2, y2, _, length in center_lines:
            # Uzun çizgilere daha fazla ağırlık ver
            weight = (length / max_length) ** 2
            
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            weights.extend([weight, weight])
            
        # Ağırlıklı polinom uydurma
        fit = np.polyfit(y_coords, x_coords, deg=1, w=weights)
        
        # Önceki değerlerle yumuşatma
        smooth = self.smooth_factor
        if self.is_curve:
            smooth = max(0.3, smooth - 0.2)  # Virajlarda daha az yumuşatma
        
        if self.last_center_fit is not None:
            fit = smooth * self.last_center_fit + (1 - smooth) * fit
        
        # Şerit hafızasını güncelle
        self.last_center_fit = fit
        self.center_fit_history.append(fit)
        if len(self.center_fit_history) > self.max_history_frames:
            self.center_fit_history.pop(0)
            
        return fit
    
    def draw_lanes(self, image, left_fit, right_fit, center_fit=None):
        """Şeritleri çiz"""
        overlay = np.zeros_like(image)
        
        # Sol ve sağ şeritleri çiz (eğer varsa)
        if left_fit is not None or right_fit is not None:
            # Çizim yüksekliği
            bottom_y = self.height
            top_y = int(self.height * (0.45 if self.is_curve else 0.5))  # Virajda daha yukarı çiz
            ploty = np.linspace(top_y, bottom_y, 20)
            
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
            # Çizim yüksekliği - orta şerit için daha uzun çiz
            bottom_y = self.height
            top_y = int(self.height * (0.4 if self.is_curve else 0.45))  # Virajda daha yukarı çiz
            ploty = np.linspace(top_y, bottom_y, 20)
            center_fitx = center_fit[0] * ploty + center_fit[1]
            pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
            cv2.polylines(overlay, np.int32([pts_center]), False, (0, 0, 255), 4)
            
            # Orta şeridin alt noktasını belirgin göster
            bottom_y = self.height
            bottom_x = int(center_fit[0] * bottom_y + center_fit[1])
            cv2.circle(overlay, (bottom_x, bottom_y), 8, (255, 0, 0), -1)
        
        return cv2.addWeighted(image, 1, overlay, 0.5, 0)
    
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
        
        # Viraj durumunda merkez değerini düzelt
        if self.is_curve:
            if self.curve_direction == "left":
                # Sola virajda merkezi biraz sağa kaydır
                center_diff *= 0.8  # Etkiyi azalt
            elif self.curve_direction == "right":
                # Sağa virajda merkezi biraz sola kaydır
                center_diff *= 0.8  # Etkiyi azalt
        
        return center_diff
    
    def process_frame(self, frame):
        """Ana işleme fonksiyonu"""
        try:
            # Geçerlilik kontrolü
            self.validate_frame(frame)
            
            # Görüntüyü işle
            edges = self.preprocess_image(frame)
            
            # Şeritleri tespit et (sol ve sağ)
            left_lines, right_lines, center_lines = self.detect_lane_lines(edges)
            
            # Şerit kurtarma modunu kontrol et
            current_time = time.time()
            if self.lane_recovery_mode:
                if current_time - self.recovery_start_time > self.recovery_timeout:
                    self.lane_recovery_mode = False
                    self.consecutive_detection_failures = 0
                    if self.debug:
                        logging.debug("Şerit kurtarma modu tamamlandı")
            
            # Şerit tespiti başarısını kontrol et
            if (not center_lines and not left_lines and not right_lines) or \
               (not self.is_curve and not center_lines):
                self.consecutive_detection_failures += 1
                if self.consecutive_detection_failures > self.max_detection_failures and not self.lane_recovery_mode:
                    self.lane_recovery_mode = True
                    self.recovery_start_time = current_time
                    if self.debug:
                        logging.warning(f"Şerit kurtarma modu başlatıldı ({self.consecutive_detection_failures} başarısız tespit)")
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
                
                # Viraj bilgisi
                if self.is_curve:
                    curve_text = f"Viraj: {self.curve_direction.upper()}, {self.curve_confidence:.2f}"
                    cv2.putText(result, curve_text, 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Orta şerit durumu
                if center_fit is not None:
                    cv2.putText(result, "Orta Serit: BULUNDU",
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "Orta Serit: YOK",
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # ROI'yi göster
                cv2.polylines(result, [self.roi_vertices], True, (0, 0, 255), 2)
                
                # Orta şerit referans çizgisini göster
                cv2.polylines(result, [self.center_line], False, (255, 255, 0), 2)
            
            return result, center_diff
            
        except Exception as e:
            logging.error(f"Kare işleme hatası: {e}")
            # Boş bir görüntü döndür
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(blank, "HATA: " + str(e),
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return blank, None
        
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
            # Virajda daha iyi şerit genişliği tahmini
            if self.is_curve and self.curve_direction == "left":
                lane_center = left_x + self.width * 0.3  # Sola virajda daha geniş şerit
            else:
                lane_center = left_x + self.width * 0.25  # Normal şerit genişliği
            
        else:  # right_fit is not None
            right_x = right_fit[0] * y + right_fit[1]
            # Virajda daha iyi şerit genişliği tahmini
            if self.is_curve and self.curve_direction == "right":
                lane_center = right_x - self.width * 0.3  # Sağa virajda daha geniş şerit
            else:
                lane_center = right_x - self.width * 0.25  # Normal şerit genişliği
        
        image_center = self.width / 2
        center_diff = lane_center - image_center
        
        return center_diff 