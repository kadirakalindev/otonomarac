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
        
        # Temel filtre parametreleri - hassasiyet artırıldı
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Blur kernel tek sayı olmalı
            self.blur_kernel += 1
            
        self.canny_low = 25    # Daha da düşürüldü
        self.canny_high = 110  # Daha da düşürüldü
        
        # Hough parametreleri - hassasiyet artırıldı ve sınırlar eklendi
        self.rho = max(1, min(2, self.width / 320))  # Çözünürlüğe göre ayarla
        self.theta = np.pi/180
        self.min_line_length = max(10, self.height * 0.08)  # Daha kısa çizgileri de kabul et
        self.max_line_gap = min(50, self.height * 0.25)     # Daha büyük boşlukları kabul et
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.max_history_frames = 20  # Daha uzun hafıza
        self.smooth_factor = 0.5  # Yumuşatma faktörü azaltıldı (daha hızlı adaptasyon)
        self.confidence_threshold = 0.6  # Güven eşiği
        
        # Şerit kaybı için değişkenler
        self.consecutive_detection_failures = 0
        self.max_detection_failures = 20  # Daha toleranslı
        self.lane_recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_timeout = 4.0  # 4 saniye kurtarma modu
        
        # Viraj tespiti için değişkenler
        self.is_turning = False
        self.turn_direction = None  # "left" veya "right"
        self.turn_start_time = 0
        self.turn_confidence = 0
        self.max_turn_confidence = 3  # Daha hızlı viraj tespiti
        self.turn_timeout = 3.0  # 3 saniye viraj modu
        
        # Şerit genişliği ve merkezi için değişkenler
        self.lane_width = None
        self.lane_center = None
        
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
                logger.info(f"ROI noktaları yüklendi: {self.roi_vertices.tolist()}")
            elif 'src_points' in calibration:
                self.roi_vertices = np.array(calibration['src_points'], dtype=np.int32)
                logger.info(f"Eski format ROI noktaları yüklendi: {self.roi_vertices.tolist()}")
                
            # Şerit genişliği ve merkezi güncelle (eğer varsa)
            if 'lane_width' in calibration:
                self.lane_width = calibration['lane_width']
                logger.info(f"Şerit genişliği yüklendi: {self.lane_width}")
                
            if 'lane_center' in calibration:
                self.lane_center = tuple(calibration['lane_center'])
                logger.info(f"Şerit merkezi yüklendi: {self.lane_center}")
                
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
        
        # Kontrast artırma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Ek parlaklık normalizasyonu
        enhanced = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Kenar tespiti
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)
        
        # Kenarları genişlet (daha iyi bağlantı için)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # ROI maskesi
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked = cv2.bitwise_and(edges, mask)
        
        if self.debug:
            self.debug_images["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.debug_images["enhanced"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            self.debug_images["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["masked"] = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        
        return masked
    
    def detect_lane_lines(self, edges):
        """Basitleştirilmiş şerit tespiti"""
        lines = cv2.HoughLinesP(
            edges, self.rho, self.theta, 15,  # Threshold azaltıldı
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
                if abs(slope) < 0.15:  # Daha düşük eğimli çizgileri de kabul et (virajlarda yardımcı olur)
                    continue
                
                # Çizgi uzunluğu
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Viraj tespiti için eğim değerlerini kaydet
                if self.is_turning:
                    # Viraj durumunda daha geniş bir aralık kabul et
                    if self.turn_direction == "left":
                        # Sol virajda sağ şeridi daha geniş bir aralıkta ara
                        if slope > 0 and x1 > self.width * 0.25:
                            right_lines.append((x1, y1, x2, y2, slope, line_length))
                        elif slope < 0 and x1 < self.width * 0.75:
                            left_lines.append((x1, y1, x2, y2, slope, line_length))
                    else:  # right turn
                        # Sağ virajda sol şeridi daha geniş bir aralıkta ara
                        if slope > 0 and x1 > self.width * 0.25:
                            right_lines.append((x1, y1, x2, y2, slope, line_length))
                        elif slope < 0 and x1 < self.width * 0.75:
                            left_lines.append((x1, y1, x2, y2, slope, line_length))
                else:
                    # Normal durumda standart aralıkları kullan
                    if slope < 0 and x1 < self.width * 0.65:  # Daha geniş sol bölge
                        left_lines.append((x1, y1, x2, y2, slope, line_length))
                    elif slope > 0 and x1 > self.width * 0.35:  # Daha geniş sağ bölge
                        right_lines.append((x1, y1, x2, y2, slope, line_length))
        
        # Viraj tespiti
        self._detect_turn(left_lines, right_lines)
        
        return left_lines, right_lines
    
    def _detect_turn(self, left_lines, right_lines):
        """Viraj tespiti yapar"""
        current_time = time.time()
        
        # Viraj modunda zaman aşımını kontrol et
        if self.is_turning and current_time - self.turn_start_time > self.turn_timeout:
            self.is_turning = False
            self.turn_direction = None
            self.turn_confidence = 0
            logger.debug("Viraj modu zaman aşımı")
        
        # Şerit sayılarını kontrol et
        left_count = len(left_lines)
        right_count = len(right_lines)
        
        # Çizgi uzunluklarını da hesaba kat
        left_length_sum = sum(line[5] for line in left_lines) if left_lines else 0
        right_length_sum = sum(line[5] for line in right_lines) if right_lines else 0
        
        # Viraj tespiti için geliştirilmiş mantık
        if (left_count > 2 * right_count + 3) or (left_length_sum > 2.5 * right_length_sum and left_count > right_count):
            if not self.is_turning or self.turn_direction != "right":
                self.turn_confidence += 1
                if self.turn_confidence >= self.max_turn_confidence:
                    self.is_turning = True
                    self.turn_direction = "right"
                    self.turn_start_time = current_time
                    self.turn_confidence = self.max_turn_confidence
                    logger.debug("Sağa viraj tespit edildi")
        elif (right_count > 2 * left_count + 3) or (right_length_sum > 2.5 * left_length_sum and right_count > left_count):
            if not self.is_turning or self.turn_direction != "left":
                self.turn_confidence += 1
                if self.turn_confidence >= self.max_turn_confidence:
                    self.is_turning = True
                    self.turn_direction = "left"
                    self.turn_start_time = current_time
                    self.turn_confidence = self.max_turn_confidence
                    logger.debug("Sola viraj tespit edildi")
        else:
            # Viraj yoksa güveni azalt
            self.turn_confidence = max(0, self.turn_confidence - 1)
            if self.turn_confidence == 0 and self.is_turning:
                self.is_turning = False
                self.turn_direction = None
                logger.debug("Viraj modu sonlandı")
    
    def fit_lane_lines(self, lines, is_left=True):
        """Şerit çizgilerini uydur"""
        if not lines:
            # Şerit bulunamadı, hafızadaki son şeridi kullan
            if is_left:
                if len(self.left_fit_history) > 0:
                    # Güven değerine göre azaltılmış bir şerit tahmini yap
                    confidence = max(0.2, self.confidence_threshold - 0.05 * self.consecutive_detection_failures)
                    avg_fit = np.mean(self.left_fit_history[-5:], axis=0)  # Son 5 kareyi kullan
                    return avg_fit * confidence
                return None
            else:
                if len(self.right_fit_history) > 0:
                    confidence = max(0.2, self.confidence_threshold - 0.05 * self.consecutive_detection_failures)
                    avg_fit = np.mean(self.right_fit_history[-5:], axis=0)  # Son 5 kareyi kullan
                    return avg_fit * confidence
                return None
            
        # Çizgileri uzunluklarına göre sırala
        lines.sort(key=lambda x: x[5], reverse=True)
        
        # En uzun %80 çizgileri kullan
        top_lines = lines[:max(1, int(len(lines) * 0.8))]
        
        x_coords = []
        y_coords = []
        weights = []  # Çizgi uzunluğuna göre ağırlıklar
        
        for x1, y1, x2, y2, _, length in top_lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            weights.extend([length, length])  # Her nokta için çizgi uzunluğunu ağırlık olarak kullan
            
        # Ağırlıklı polinom uydurma
        fit = np.polyfit(y_coords, x_coords, deg=1, w=weights)
        
        # Önceki değerlerle yumuşatma
        last_fit = self.last_left_fit if is_left else self.last_right_fit
        
        # Viraj durumunda yumuşatma faktörünü azalt (daha hızlı adaptasyon)
        smooth_factor = self.smooth_factor
        if self.is_turning:
            smooth_factor = max(0.2, smooth_factor - 0.3)
            
        if last_fit is not None:
            fit = smooth_factor * last_fit + (1 - smooth_factor) * fit
        
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
                
                # Şerit orta çizgisini çiz
                middle_fitx = (left_fitx + right_fitx) / 2
                pts_middle = np.array([np.transpose(np.vstack([middle_fitx, ploty]))])
                cv2.polylines(overlay, np.int32([pts_middle]), False, (0, 0, 255), 2)
        
        result = cv2.addWeighted(image, 1, overlay, 0.5, 0)
        
        # Viraj bilgisini ekle
        if self.is_turning:
            turn_text = f"{'SOL' if self.turn_direction == 'left' else 'SAĞ'} VİRAJ"
            cv2.putText(result, turn_text, 
                      (self.width - 150, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
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
            
            # Şerit genişliğini güncelle (kalibrasyon için)
            current_width = right_x - left_x
            if self.lane_width is None:
                self.lane_width = current_width
            else:
                # Şerit genişliğini yumuşat
                self.lane_width = 0.9 * self.lane_width + 0.1 * current_width
            
        elif left_fit is not None:
            left_x = left_fit[0] * y + left_fit[1]
            # Kalibrasyon dosyasından şerit genişliği varsa kullan
            if self.lane_width is not None:
                lane_center = left_x + self.lane_width / 2
            else:
                lane_center = left_x + self.width * 0.25  # Tahmini şerit genişliği
            
        else:  # right_fit is not None
            right_x = right_fit[0] * y + right_fit[1]
            # Kalibrasyon dosyasından şerit genişliği varsa kullan
            if self.lane_width is not None:
                lane_center = right_x - self.lane_width / 2
            else:
                lane_center = right_x - self.width * 0.25  # Tahmini şerit genişliği
        
        # Kalibrasyon dosyasından şerit merkezi varsa kullan
        if self.lane_center is not None:
            image_center = self.lane_center[0]
        else:
            image_center = self.width / 2
            
        center_diff = lane_center - image_center
        
        # Viraj durumunda merkez farkını düzelt
        if self.is_turning:
            if self.turn_direction == "left":
                # Sol virajda merkezi biraz sola kaydır
                center_diff -= self.width * 0.08
            else:  # right turn
                # Sağ virajda merkezi biraz sağa kaydır
                center_diff += self.width * 0.08
        
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
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Şerit kurtarma modu bilgisi
            if self.lane_recovery_mode:
                cv2.putText(result, "SERIT KURTARMA MODU",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Hata sayacı
            cv2.putText(result, f"Hata: {self.consecutive_detection_failures}/{self.max_detection_failures}",
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Sol ve sağ şerit sayısı
            cv2.putText(result, f"Sol: {len(left_lines)}, Sağ: {len(right_lines)}",
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # ROI'yi göster
            cv2.polylines(result, [self.roi_vertices], True, (0, 0, 255), 2)
        
        return result, center_diff 