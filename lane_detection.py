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
        Şerit tespit sınıfı başlatma
        """
        # Kamera çözünürlüğü
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        
        # ROI (İlgi Bölgesi) ayarları - algılama bölgesi
        self.roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.35, self.height * 0.55],  # Daha geniş ROI
            [self.width * 0.65, self.height * 0.55],  # Daha geniş ROI
            [self.width, self.height]
        ], dtype=np.int32)
        
        # Görüntü işleme parametreleri
        self.blur_kernel = 5
        self.canny_low = 35  # Canny alt eşik değeri - daha düşük
        self.canny_high = 120  # Canny üst eşik değeri - daha düşük
        
        # Hough dönüşümü parametreleri
        self.rho = 1
        self.theta = np.pi / 180
        self.hough_threshold = 20  # Daha düşük eşik değeri
        self.min_line_length = int(self.height * 0.08)  # Daha kısa çizgiler
        self.max_line_gap = int(self.height * 0.15)  # Daha büyük boşluk toleransı
        
        # Şerit genişliği ile ilgili parametreler
        self.min_lane_width = int(self.width * 0.1)   # Şeritler arası minimum genişlik
        self.max_lane_width = int(self.width * 0.85)  # Şeritler arası maksimum genişlik
        
        # Hafıza ve düzeltme faktörleri
        self.left_line_memory = None
        self.right_line_memory = None
        self.memory_factor = 0.7  # Hafıza faktörü - daha yüksek değer
        self.smoothing_factor = 0.6  # Düzeltme faktörü - daha yüksek değer
        self.max_detection_failures = 20  # Daha yüksek hata toleransı
        self.detection_failures = 0
        
        self.debug = debug
        
        # Şerit hafızası ve yumuşatma - geliştirildi
        self.last_left_fit = None
        self.last_right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.max_history_frames = 20  # Daha uzun hafıza
        self.smooth_factor = 0.6  # Yumuşatma faktörü - daha hızlı uyum
        self.confidence_threshold = 0.5  # Güven eşiği - düşürüldü
        
        # Şerit kaybı için değişkenler - daha hoşgörülü
        self.consecutive_detection_failures = 0
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
        """
        Kenar görüntüsü üzerinde Hough dönüşümü ile şeritleri tespit eder
        """
        # Hough dönüşümü ile çizgileri tespit et
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        left_lines = []
        right_lines = []
        
        # Çizgi tespit edilemediyse boş şeritler döndür
        if lines is None:
            self.detection_failures += 1
            logger.debug(f"Şerit çizgileri tespit edilemedi ({self.detection_failures}/{self.max_detection_failures})")
            return (None, None)
            
        # Tespit edilen tüm çizgilerden şeritleri belirle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Çizginin eğimini hesapla
            # Eğim = 0 yatay, eğim = ∞ dikey çizgi anlamına gelir
            if x2 - x1 == 0:  # Sıfıra bölme hatasından kaçınmak için
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Çizgi uzunluğu hesapla
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Çok kısa çizgileri görmezden gel
            if line_length < self.min_line_length * 0.8:
                continue
                
            # Çizgileri eğimlerine göre sınıflandır
            # Yatay çizgileri filtrele - daha az agresif filtreleme
            if abs(slope) < 0.25:
                continue
                
            # Çizgiler sadece alt yarıda mı kontrol et
            lower_half = (y1 > self.height * 0.5) or (y2 > self.height * 0.5)
            if not lower_half:
                continue
            
            # Eğim sol ve sağ şeritleri belirlemek için kullanılır
            if slope < 0:  # Negatif eğim = sol şerit (sol üst -> sağ alt)
                left_lines.append((x1, y1, x2, y2, slope))
            else:  # Pozitif eğim = sağ şerit (sol alt -> sağ üst)
                right_lines.append((x1, y1, x2, y2, slope))
        
        # Her iki tarafta da minimum sayıda çizgi var mı kontrol et
        min_lines_required = 1  # En az bir çizgi gerekli
        if len(left_lines) < min_lines_required and len(right_lines) < min_lines_required:
            self.detection_failures += 1
            logger.debug(f"Yeterli şerit çizgisi bulunamadı - Sol: {len(left_lines)}, Sağ: {len(right_lines)} ({self.detection_failures}/{self.max_detection_failures})")
            return (None, None)
            
        # Şerit başarıyla tespit edildiyse hata sayacını azalt
        if self.detection_failures > 0:
            self.detection_failures = max(0, self.detection_failures - 2)
        
        # Sol ve sağ şerit çizgilerini hesapla
        left_lane = self._calculate_lane_line(left_lines, True)
        right_lane = self._calculate_lane_line(right_lines, False)
        
        # Şeritler arasındaki genişliği kontrol et
        if left_lane is not None and right_lane is not None:
            # Şeritler arasındaki mesafe
            lane_width = abs(right_lane[0] - left_lane[0])  # Şerit tabanları arasındaki mesafe
            
            # Şerit genişliği uygun aralıkta mı kontrol et
            if lane_width < self.min_lane_width:
                logger.debug(f"Şerit çok dar: {lane_width}px < {self.min_lane_width}px")
                # En güvenilir şeridi seç ve diğerini yok say
                if len(left_lines) > len(right_lines):
                    right_lane = None
                else:
                    left_lane = None
            elif lane_width > self.max_lane_width:
                logger.debug(f"Şerit çok geniş: {lane_width}px > {self.max_lane_width}px")
                # En güvenilir şeridi seç ve diğerini yok say
                if len(left_lines) > len(right_lines):
                    right_lane = None
                else:
                    left_lane = None
        
        return (left_lane, right_lane)
        
    def _calculate_lane_line(self, lines, is_left):
        """
        Tespit edilen çizgilerden şerit çizgisini hesaplar
        """
        if not lines:
            return None
            
        # Çizgilerin ortalamasını al
        x_bottom_sum = 0
        x_top_sum = 0
        count = 0
        
        for x1, y1, x2, y2, slope in lines:
            # Alt noktaya göre sıralayın
            if y1 > y2:
                x_bottom, y_bottom = x1, y1
                x_top, y_top = x2, y2
            else:
                x_bottom, y_bottom = x2, y2
                x_top, y_top = x1, y1
                
            # En alt ve en üst noktaları görüntü sınırlarına ekstrapole et
            if y_bottom != self.height:
                x_bottom = x_bottom + (self.height - y_bottom) * slope
                y_bottom = self.height
                
            if y_top != 0:
                x_top = x_top - y_top / slope
                y_top = 0
                
            x_bottom_sum += x_bottom
            x_top_sum += x_top
            count += 1
        
        if count > 0:
            x_bottom_avg = int(x_bottom_sum / count)
            x_top_avg = int(x_top_sum / count)
            
            # Hafıza faktörünü kullanarak yumuşatma yap
            lane_memory = self.left_line_memory if is_left else self.right_line_memory
            
            if lane_memory is not None:
                x_bottom_smooth = int(lane_memory[0] * self.memory_factor + x_bottom_avg * (1 - self.memory_factor))
                x_top_smooth = int(lane_memory[2] * self.memory_factor + x_top_avg * (1 - self.memory_factor))
            else:
                x_bottom_smooth = x_bottom_avg
                x_top_smooth = x_top_avg
                
            # Hafızayı güncelle
            if is_left:
                self.left_line_memory = (x_bottom_smooth, self.height, x_top_smooth, 0)
            else:
                self.right_line_memory = (x_bottom_smooth, self.height, x_top_smooth, 0)
                
            return (x_bottom_smooth, self.height, x_top_smooth, 0)
            
        return None
    
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
            
            # Gerçek şerit genişliği yaklaşık 40 cm, bunu piksel genişliği ile ilişkilendir
            lane_width = right_x - left_x
            expected_width = self.width * 0.4  # Beklenen şerit genişliği
            
            # Şerit genişliği kontrolü - çok dar veya çok geniş şeritler için ayarlama
            if lane_width < expected_width * 0.4 or lane_width > expected_width * 1.8:
                if self.debug:
                    logger.warning(f"Anormal şerit genişliği: {lane_width:.1f}px")
                
                # Şerit genişliğini düzeltmek yerine hafızadaki son geçerli genişliği kullan
                if self.last_left_fit is not None and self.last_right_fit is not None:
                    last_left_x = self.last_left_fit[0] * y + self.last_left_fit[1]
                    last_right_x = self.last_right_fit[0] * y + self.last_right_fit[1]
                    last_width = last_right_x - last_left_x
                    
                    # Son şerit genişliği makul ise onu kullan
                    if expected_width * 0.4 <= last_width <= expected_width * 1.8:
                        lane_center = (left_x + right_x) / 2
                    else:
                        # Merkezi tahmin et
                        lane_center = self.width / 2
                else:
                    # Merkezi tahmin et
                    lane_center = self.width / 2
            else:
                lane_center = (left_x + right_x) / 2
            
        elif left_fit is not None:
            left_x = left_fit[0] * y + left_fit[1]
            lane_center = left_x + self.width * 0.2  # Tahmini şerit genişliği
            
        else:  # right_fit is not None
            right_x = right_fit[0] * y + right_fit[1]
            lane_center = right_x - self.width * 0.2  # Tahmini şerit genişliği
        
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