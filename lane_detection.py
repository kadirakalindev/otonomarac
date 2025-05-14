#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Şerit Tespiti ve Takibi Modülü
Bu modül, görüntü işleme teknikleri kullanarak şerit tespiti ve takibi yapar.
"""

import cv2
import numpy as np
import time
import logging
import os

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LaneDetection")

class LaneDetector:
    """
    Şerit tespit ve takip işlemlerini gerçekleştiren sınıf.
    """
    def __init__(self, camera_resolution=(640, 480), debug=False):
        """
        LaneDetector sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
            debug (bool): Hata ayıklama modu açık/kapalı
        """
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        self.debug = debug
        
        # Perspektif dönüşümü için kaynak ve hedef noktaları
        # Not: Bu değerler kamera pozisyonuna göre kalibre edilmelidir
        self.src_points = np.float32([
            [self.width * 0.25, self.height * 0.6],  # Sol üst
            [self.width * 0.75, self.height * 0.6],  # Sağ üst
            [0, self.height],                       # Sol alt
            [self.width, self.height]                # Sağ alt
        ])
        
        self.dst_points = np.float32([
            [self.width * 0.25, 0],               # Sol üst
            [self.width * 0.75, 0],               # Sağ üst
            [self.width * 0.25, self.height],     # Sol alt
            [self.width * 0.75, self.height]      # Sağ alt
        ])
        
        # Perspektif dönüşüm matrisleri
        self.perspective_M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_perspective_M = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
        # Filtre parametreleri
        self.blur_kernel_size = 5
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        
        # Hough dönüşümü parametreleri
        self.rho = 1
        self.theta = np.pi/180
        self.hough_threshold = 20
        self.min_line_length = 20
        self.max_line_gap = 300
        
        # Renk filtresi parametreleri (Beyaz ve sarı şeritler için)
        # HSV renk aralıkları
        self.yellow_lower = np.array([15, 80, 120])
        self.yellow_upper = np.array([35, 255, 255])
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        # Son şerit çizgileri (hafıza)
        self.last_left_lane = None
        self.last_right_lane = None
        self.smoothing_factor = 0.8  # Yeni ve eski şerit değerlerini birleştirme faktörü
        
        # Şerit kaybı durumunu takip etme
        self.lost_lane_counter = 0
        self.max_lost_lane_frames = 10  # Bu kadar kare boyunca şerit bulunamazsa sıfırla
        
        # Debug görüntüleri
        self.debug_images = {}
        
        # Şerit renkleri
        self.LANE_COLOR = (0, 255, 0)  # Parlak yeşil
        self.LANE_THICKNESS = 8  # Daha kalın çizgi
        
        # Kalibrasyon dosyasını yükle (varsa)
        self.load_calibration()
        
        logger.info("Şerit tespit modülü başlatıldı.")
    
    def load_calibration(self, filename="calibration.json"):
        """
        Kalibrasyon dosyasını yükler (varsa)
        """
        import json
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    calibration = json.load(f)
                
                # Perspektif dönüşüm noktalarını güncelle
                if 'src_points' in calibration and 'dst_points' in calibration:
                    self.src_points = np.float32(calibration['src_points'])
                    self.dst_points = np.float32(calibration['dst_points'])
                    self.perspective_M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                    self.inverse_perspective_M = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
                    logger.info("Kalibrasyon dosyası yüklendi.")
                
                # Kenar tespiti parametrelerini güncelle
                if 'canny_low_threshold' in calibration and 'canny_high_threshold' in calibration:
                    self.canny_low_threshold = calibration['canny_low_threshold']
                    self.canny_high_threshold = calibration['canny_high_threshold']
                
                # Hough parametrelerini güncelle
                if 'hough_threshold' in calibration:
                    self.hough_threshold = calibration['hough_threshold']
                if 'min_line_length' in calibration:
                    self.min_line_length = calibration['min_line_length']
                if 'max_line_gap' in calibration:
                    self.max_line_gap = calibration['max_line_gap']
                
                logger.info("Kalibrasyon parametreleri yüklendi.")
                
            except Exception as e:
                logger.warning(f"Kalibrasyon dosyası yüklenemedi: {e}")
    
    def apply_color_filter(self, image):
        """
        Görüntüye renk filtresi uygular, beyaz ve sarı şeritleri belirginleştirir.
        
        Args:
            image (numpy.ndarray): İşlenecek renkli görüntü
            
        Returns:
            numpy.ndarray: Renk filtrelenmiş binary görüntü
        """
        # BGR'dan HSV'ye dönüştür
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Beyaz ve sarı renk maskeleri oluştur
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Maskeleri birleştir
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Morfolojik işlemler (gürültü azaltma ve şerit kalınlaştırma)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Debug için maskeleri kaydet
        if self.debug:
            white_mask_colored = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            yellow_mask_colored = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            self.debug_images["white_mask"] = white_mask_colored
            self.debug_images["yellow_mask"] = yellow_mask_colored
            self.debug_images["color_filter"] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        return combined_mask
    
    def preprocess_image(self, image):
        """
        Görüntüyü ön işlemden geçirir.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            numpy.ndarray: İşlenmiş görüntü
        """
        # Renk filtresi uygula (beyaz ve sarı şeritleri belirginleştir)
        color_filtered = self.apply_color_filter(image)
        
        # Gri tonlamaya dönüştürme (eğer renk filtresi kullanılmazsa)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü giderme (Gaussian Blur)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Kenar tespiti (Canny)
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # Adaptif eşikleme (değişken ışık koşulları için)
        _, binary_adaptive = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Renk filtresi ve kenar tespitini birleştir
        combined = cv2.bitwise_or(color_filtered, edges)
        
        # İlgi bölgesi belirleme (ROI)
        roi_mask = np.zeros_like(combined)
        roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.4, self.height * 0.6],
            [self.width * 0.6, self.height * 0.6],
            [self.width, self.height]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        masked_edges = cv2.bitwise_and(combined, roi_mask)
        
        # Debug görüntülerini kaydet
        if self.debug:
            self.debug_images["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.debug_images["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["binary_adaptive"] = cv2.cvtColor(binary_adaptive, cv2.COLOR_GRAY2BGR)
            self.debug_images["masked_edges"] = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["combined"] = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        return masked_edges
    
    def warp_perspective(self, image):
        """
        Görüntüyü kuş bakışı görünümüne dönüştürür.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            numpy.ndarray: Kuş bakışı görünümüne dönüştürülmüş görüntü
        """
        warped = cv2.warpPerspective(image, self.perspective_M, (self.width, self.height))
        return warped
    
    def detect_lane_lines(self, image):
        """
        Görüntüde şerit çizgilerini tespit eder.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            tuple: Sol ve sağ şerit çizgisi parametreleri (eğim, kesişim noktası)
        """
        # Ön işleme
        processed = self.preprocess_image(image)
        
        # Kuş bakışı dönüşümü
        bird_eye_view = self.warp_perspective(processed)
        
        # Debug görüntülerini kaydet
        if self.debug:
            self.debug_images["bird_eye_view"] = cv2.cvtColor(bird_eye_view, cv2.COLOR_GRAY2BGR)
        
        # Hough dönüşümü ile çizgi tespiti
        lines = cv2.HoughLinesP(
            bird_eye_view, 
            self.rho, 
            self.theta, 
            self.hough_threshold, 
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        # Sol ve sağ şerit çizgileri için boş listeler
        left_lines = []
        right_lines = []
        
        # Tespit edilen çizgileri sol ve sağ olarak sınıflandırma
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çizginin eğimini hesapla
                if x2 - x1 == 0:  # Dikey çizgiden kaçın
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Çok küçük eğimli yatay çizgileri filtrele
                if abs(slope) < 0.2:
                    continue
                
                # Eğime göre sınıflandır
                if slope < 0:  # Negatif eğim = Sol şerit
                    left_lines.append(line[0])
                else:  # Pozitif eğim = Sağ şerit
                    right_lines.append(line[0])
        
        # Şeritleri ortalama hesaplayarak bul
        left_lane = self._find_average_line(left_lines)
        right_lane = self._find_average_line(right_lines)
        
        # Şerit kaybı durumunu takip et
        if left_lane is None and right_lane is None:
            self.lost_lane_counter += 1
        else:
            self.lost_lane_counter = 0
            
        # Belirli bir süre şerit bulunamazsa durumu sıfırla
        if self.lost_lane_counter > self.max_lost_lane_frames:
            self.last_left_lane = None
            self.last_right_lane = None
            logger.warning("Şerit bulunamadı, hafıza sıfırlandı!")
            self.lost_lane_counter = 0
        
        # Şerit devamsızlığına karşı düzeltme (önceki karelerden bilgi kullan)
        left_lane = self._smooth_lane(left_lane, self.last_left_lane)
        right_lane = self._smooth_lane(right_lane, self.last_right_lane)
        
        # Önceki şeritleri güncelle
        self.last_left_lane = left_lane
        self.last_right_lane = right_lane
        
        # Ham çizgileri debug için sakla
        if self.debug:
            self.raw_lines = {
                "left_lines": left_lines,
                "right_lines": right_lines
            }
        
        return left_lane, right_lane
    
    def _find_average_line(self, lines):
        """
        Verilen çizgilerin ortalama eğimini ve kesişim noktasını hesaplar.
        
        Args:
            lines (list): Çizgi noktaları listesi
            
        Returns:
            tuple: (eğim, kesişim) veya None
        """
        if not lines:
            return None
            
        x_sum = 0
        y_sum = 0
        m_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Eğim hesapla
            if x2 - x1 == 0:  # Dikey çizgiden kaçın
                continue
                
            m = (y2 - y1) / (x2 - x1)
            
            # Kesişim hesapla (y = mx + b -> b = y - mx)
            b = y1 - m * x1
            
            m_sum += m
            x_sum += (x1 + x2) / 2
            y_sum += (y1 + y2) / 2
        
        if len(lines) > 0:
            # Ortalama eğim ve nokta
            m_avg = m_sum / len(lines)
            x_avg = x_sum / len(lines)
            y_avg = y_sum / len(lines)
            
            # Ortalama kesişim
            b_avg = y_avg - m_avg * x_avg
            
            return (m_avg, b_avg)
        else:
            return None
    
    def _smooth_lane(self, current_lane, previous_lane):
        """
        Şerit devamsızlığına karşı yumuşatma uygular.
        
        Args:
            current_lane (tuple): Mevcut karedeki şerit parametreleri (eğim, kesişim)
            previous_lane (tuple): Önceki karedeki şerit parametreleri
            
        Returns:
            tuple: Yumuşatılmış şerit parametreleri
        """
        # Eğer mevcut şerit tespit edilemedi ise, öncekini döndür
        if current_lane is None:
            return previous_lane
            
        # Eğer önceki şerit yoksa, mevcut şeridi döndür
        if previous_lane is None:
            return current_lane
            
        # Yumuşatılmış şerit parametrelerini hesapla
        smooth_m = self.smoothing_factor * previous_lane[0] + (1 - self.smoothing_factor) * current_lane[0]
        smooth_b = self.smoothing_factor * previous_lane[1] + (1 - self.smoothing_factor) * current_lane[1]
        
        return (smooth_m, smooth_b)
    
    def draw_lanes(self, image, left_lane, right_lane, color=None, thickness=None):
        """
        Tespit edilen şeritleri görüntü üzerine çizer.
        
        Args:
            image (numpy.ndarray): Şeritlerin çizileceği görüntü
            left_lane (tuple): Sol şerit parametreleri (eğim, kesişim)
            right_lane (tuple): Sağ şerit parametreleri (eğim, kesişim)
            color (tuple): Çizgi rengi (B, G, R), None ise varsayılan renk kullanılır
            thickness (int): Çizgi kalınlığı, None ise varsayılan kalınlık kullanılır
            
        Returns:
            numpy.ndarray: Şeritler çizilmiş görüntü
        """
        if color is None:
            color = self.LANE_COLOR
            
        if thickness is None:
            thickness = self.LANE_THICKNESS
        
        lane_image = np.zeros_like(image)
        
        # Ham çizgileri göster (sadece debug modunda)
        if self.debug and hasattr(self, 'raw_lines'):
            # Sol şerit için ham çizgileri göster
            for line in self.raw_lines.get("left_lines", []):
                x1, y1, x2, y2 = line
                cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Mavi renk
                
            # Sağ şerit için ham çizgileri göster
            for line in self.raw_lines.get("right_lines", []):
                x1, y1, x2, y2 = line
                cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Kırmızı renk
        
        # Şerit kaybı durumunda uyarı mesajı
        if left_lane is None and right_lane is None:
            if self.debug:
                cv2.putText(image, "UYARI: Serit bulunamadi!", 
                          (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return image
        
        # Şerit çizimi (kuş bakışı görünümünde)
        if left_lane is not None:
            m_left, b_left = left_lane
            y1 = self.height
            y2 = int(self.height * 0.6)
            
            # Değerler sınırların içinde mi kontrol et
            try:
                x1_left = int((y1 - b_left) / m_left)
                x2_left = int((y2 - b_left) / m_left)
                
                # Şerit çizgisini çiz
                cv2.line(lane_image, (x1_left, y1), (x2_left, y2), color, thickness)
                
                # Pulsating effect (debug modunda)
                if self.debug:
                    pulse_color = (0, 255, 255)  # Sarı
                    pulse_width = int(thickness / 2)
                    
                    # Daha görünür bir efekt için ana şeritin üzerine ince bir çizgi
                    cv2.line(lane_image, (x1_left, y1), (x2_left, y2), pulse_color, pulse_width)
            except:
                logger.warning("Sol şerit çizimi hatası. Değerler geçersiz.")
        
        if right_lane is not None:
            m_right, b_right = right_lane
            y1 = self.height
            y2 = int(self.height * 0.6)
            
            try:
                x1_right = int((y1 - b_right) / m_right)
                x2_right = int((y2 - b_right) / m_right)
                
                # Şerit çizgisini çiz
                cv2.line(lane_image, (x1_right, y1), (x2_right, y2), color, thickness)
                
                # Pulsating effect (debug modunda)
                if self.debug:
                    pulse_color = (0, 255, 255)  # Sarı
                    pulse_width = int(thickness / 2)
                    
                    # Daha görünür bir efekt için ana şeritin üzerine ince bir çizgi
                    cv2.line(lane_image, (x1_right, y1), (x2_right, y2), pulse_color, pulse_width)
            except:
                logger.warning("Sağ şerit çizimi hatası. Değerler geçersiz.")
        
        # Şeritler arasını doldur (eğer her iki şerit de mevcutsa)
        if left_lane is not None and right_lane is not None:
            try:
                m_left, b_left = left_lane
                m_right, b_right = right_lane
                
                y1 = self.height
                y2 = int(self.height * 0.6)
                
                x1_left = int((y1 - b_left) / m_left)
                x2_left = int((y2 - b_left) / m_left)
                x1_right = int((y1 - b_right) / m_right)
                x2_right = int((y2 - b_right) / m_right)
                
                pts = np.array([
                    [x1_left, y1],
                    [x2_left, y2],
                    [x2_right, y2],
                    [x1_right, y1]
                ], dtype=np.int32)
                
                cv2.fillPoly(lane_image, [pts], (0, 100, 0))
            except:
                logger.warning("Şerit dolgu hatası. Değerler geçersiz.")
        
        # Görüntüye şeritleri ekle
        result = cv2.addWeighted(image, 1, lane_image, 0.5, 0)
        
        return result
    
    def calculate_lane_center(self, left_lane, right_lane):
        """
        Tespit edilen şeritlerin merkezini hesaplar.
        
        Args:
            left_lane (tuple): Sol şerit parametreleri
            right_lane (tuple): Sağ şerit parametreleri
            
        Returns:
            int: Merkez nokta x koordinatı
        """
        if left_lane is None or right_lane is None:
            return None
        
        # Resmin alt kısmındaki şerit pozisyonları
        y = self.height
        
        try:
            m_left, b_left = left_lane
            m_right, b_right = right_lane
            
            x_left = int((y - b_left) / m_left)
            x_right = int((y - b_right) / m_right)
            
            # Merkezler
            lane_center = (x_left + x_right) // 2
            image_center = self.width // 2
            
            # Merkez farkı (pozitif = sağa kayma, negatif = sola kayma)
            center_diff = lane_center - image_center
            
            return center_diff
        except:
            logger.warning("Şerit merkezi hesaplanamadı. Değerler geçersiz.")
            return None
    
    def create_debug_view(self, original_frame, processed_frame, center_diff):
        """
        Debug görünümü oluşturur. Tüm debug görüntülerini tek bir pencerede birleştirir.
        
        Args:
            original_frame (numpy.ndarray): Orijinal kare
            processed_frame (numpy.ndarray): İşlenmiş kare (şeritler çizilmiş)
            center_diff (int): Merkez pozisyon farkı
            
        Returns:
            numpy.ndarray: Birleştirilmiş debug görünümü
        """
        # Görüntüleri yeniden boyutlandır
        debug_width = 320
        debug_height = 240
        debug_size = (debug_width, debug_height)  # (genişlik, yükseklik)
        
        # Birleştirilmiş görüntü için zemin oluştur (2x2 grid)
        # OpenCV görüntüleri (yükseklik, genişlik, kanal) formatında
        debug_view = np.zeros((debug_height*2, debug_width*2, 3), dtype=np.uint8)
        
        # Orijinal görüntüyü sol üst köşeye yerleştir
        resized_original = cv2.resize(original_frame, debug_size)
        debug_view[0:debug_height, 0:debug_width] = resized_original
        
        # İşlenmiş görüntüyü (şeritler çizilmiş) sağ üst köşeye yerleştir
        resized_processed = cv2.resize(processed_frame, debug_size)
        debug_view[0:debug_height, debug_width:debug_width*2] = resized_processed
        
        # İlk olarak kuş bakışı görüntüsünü sol alt köşeye yerleştir
        if "bird_eye_view" in self.debug_images:
            resized_bird = cv2.resize(self.debug_images["bird_eye_view"], debug_size)
            debug_view[debug_height:debug_height*2, 0:debug_width] = resized_bird
        
        # Sonra renk filtresi görüntüsünü sağ alt köşeye yerleştir
        if "color_filter" in self.debug_images:
            resized_color = cv2.resize(self.debug_images["color_filter"], debug_size)
            debug_view[debug_height:debug_height*2, debug_width:debug_width*2] = resized_color
        # Eğer renk filtresi yoksa, kenar görüntüsünü göster
        elif "edges" in self.debug_images:
            resized_edges = cv2.resize(self.debug_images["edges"], debug_size)
            debug_view[debug_height:debug_height*2, debug_width:debug_width*2] = resized_edges
        
        # Şerit durumu bilgisini ekle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        
        # Merkez farkı bilgisini ekle
        if center_diff is not None:
            # Merkez farkının mutlak değerine göre renk değiştir (kırmızı: çok sapma, yeşil: az sapma)
            if abs(center_diff) > 100:
                center_color = (0, 0, 255)  # Kırmızı
            elif abs(center_diff) > 50:
                center_color = (0, 165, 255)  # Turuncu
            else:
                center_color = (0, 255, 0)  # Yeşil
                
            cv2.putText(debug_view, f"Merkez Farki: {center_diff}px", 
                      (10, 30), font, font_scale*1.4, center_color, 2)
        else:
            cv2.putText(debug_view, "Merkez Farki: Bilinmiyor", 
                      (10, 30), font, font_scale*1.4, (0, 0, 255), 2)
        
        # Şerit kaybı durumunda uyarı ekle
        if self.lost_lane_counter > 0:
            cv2.putText(debug_view, f"Serit Kaybi: {self.lost_lane_counter}/{self.max_lost_lane_frames}", 
                      (debug_width + 10, 30), font, font_scale*1.4, (0, 0, 255), 2)
        
        # Başlıklar ekle
        cv2.putText(debug_view, "Orijinal", (10, 15), font, font_scale, font_color, font_thickness)
        cv2.putText(debug_view, "Serit Tespiti", (debug_width+10, 15), font, font_scale, font_color, font_thickness)
        cv2.putText(debug_view, "Kus Bakisi", (10, debug_height+15), font, font_scale, font_color, font_thickness)
        cv2.putText(debug_view, "Renk Filtresi/Kenarlar", (debug_width+10, debug_height+15), font, font_scale, font_color, font_thickness)
        
        return debug_view
    
    def process_frame(self, frame):
        """
        Bir görüntü karesini işler ve sonuçları döndürür.
        
        Args:
            frame (numpy.ndarray): İşlenecek görüntü karesi
            
        Returns:
            tuple: (İşlenmiş görüntü, merkez pozisyon farkı)
        """
        start_time = time.time()
        
        # Debug görüntülerini temizle
        self.debug_images = {}
        
        # Görüntüyü yeniden boyutlandır
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Şeritleri tespit et
        left_lane, right_lane = self.detect_lane_lines(frame)
        
        # Şeritleri belirgin yeşil çizgilerle çiz
        result = self.draw_lanes(frame, left_lane, right_lane)
        
        # Merkez hesapla
        center_diff = self.calculate_lane_center(left_lane, right_lane)
        
        # İşlem süresini hesapla
        process_time = time.time() - start_time
        
        # Debug modunda merkez çiz
        if self.debug:
            # Ana görüntüye işlem süresini ekle
            cv2.putText(result, f"Islem: {process_time*1000:.1f}ms", 
                      (self.width - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Şerit kaybı durumunda uyarı ekle
            if self.lost_lane_counter > 0:
                cv2.putText(result, f"Serit Kaybi: {self.lost_lane_counter}/{self.max_lost_lane_frames}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Ana görüntüde merkezleri çiz
            if center_diff is not None:
                image_center = self.width // 2
                lane_center = image_center + center_diff
                
                # Çizgi kalınlığını sapma miktarına göre ayarla (daha fazla sapma = daha kalın çizgi)
                line_thickness = max(2, min(5, abs(center_diff) // 20))
                
                cv2.circle(result, (image_center, self.height - 30), 5, (255, 0, 0), -1)  # Görüntü merkezi (mavi)
                cv2.circle(result, (lane_center, self.height - 30), 5, (0, 0, 255), -1)   # Şerit merkezi (kırmızı)
                
                # Merkez çizgisini sapma miktarına göre renklendir
                if abs(center_diff) > 100:
                    line_color = (0, 0, 255)  # Kırmızı (çok sapma)
                elif abs(center_diff) > 50:
                    line_color = (0, 165, 255)  # Turuncu (orta sapma)
                else:
                    line_color = (0, 255, 0)  # Yeşil (az sapma)
                
                cv2.line(result, (image_center, self.height - 30), 
                       (lane_center, self.height - 30), line_color, line_thickness)
                
                # Merkez farkını yaz
                cv2.putText(result, f"Merkez Farki: {center_diff}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                
            # Birleştirilmiş debug görünümü oluştur
            debug_view = self.create_debug_view(frame, result, center_diff)
            
            # Debug görünümünü göster (ana görüntü yerine)
            result = debug_view
        
        return result, center_diff 