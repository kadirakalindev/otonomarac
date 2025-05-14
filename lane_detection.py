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
        
        # Son şerit çizgileri (hafıza)
        self.last_left_lane = None
        self.last_right_lane = None
        self.smoothing_factor = 0.8  # Yeni ve eski şerit değerlerini birleştirme faktörü
        
        # Debug görüntüleri
        self.debug_images = {}
        
        logger.info("Şerit tespit modülü başlatıldı.")
    
    def preprocess_image(self, image):
        """
        Görüntüyü ön işlemden geçirir.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            numpy.ndarray: İşlenmiş görüntü
        """
        # Gri tonlamaya dönüştürme
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü giderme (Gaussian Blur)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Kenar tespiti (Canny)
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # İlgi bölgesi belirleme (ROI)
        roi_mask = np.zeros_like(edges)
        roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.4, self.height * 0.6],
            [self.width * 0.6, self.height * 0.6],
            [self.width, self.height]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Debug görüntülerini kaydet
        if self.debug:
            self.debug_images["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.debug_images["masked_edges"] = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        
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
        
        # Şerit devamsızlığına karşı düzeltme (önceki karelerden bilgi kullan)
        left_lane = self._smooth_lane(left_lane, self.last_left_lane)
        right_lane = self._smooth_lane(right_lane, self.last_right_lane)
        
        # Önceki şeritleri güncelle
        self.last_left_lane = left_lane
        self.last_right_lane = right_lane
        
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
    
    def draw_lanes(self, image, left_lane, right_lane, color=(0, 255, 0), thickness=5):
        """
        Tespit edilen şeritleri görüntü üzerine çizer.
        
        Args:
            image (numpy.ndarray): Şeritlerin çizileceği görüntü
            left_lane (tuple): Sol şerit parametreleri (eğim, kesişim)
            right_lane (tuple): Sağ şerit parametreleri (eğim, kesişim)
            color (tuple): Çizgi rengi (B, G, R)
            thickness (int): Çizgi kalınlığı
            
        Returns:
            numpy.ndarray: Şeritler çizilmiş görüntü
        """
        lane_image = np.zeros_like(image)
        
        # Şerit çizimi (kuş bakışı görünümünde)
        if left_lane is not None:
            m_left, b_left = left_lane
            y1 = self.height
            y2 = int(self.height * 0.6)
            x1_left = int((y1 - b_left) / m_left)
            x2_left = int((y2 - b_left) / m_left)
            
            cv2.line(lane_image, (x1_left, y1), (x2_left, y2), color, thickness)
        
        if right_lane is not None:
            m_right, b_right = right_lane
            y1 = self.height
            y2 = int(self.height * 0.6)
            x1_right = int((y1 - b_right) / m_right)
            x2_right = int((y2 - b_right) / m_right)
            
            cv2.line(lane_image, (x1_right, y1), (x2_right, y2), color, thickness)
        
        # Şeritler arasını doldur
        if left_lane is not None and right_lane is not None:
            pts = np.array([
                [x1_left, y1],
                [x2_left, y2],
                [x2_right, y2],
                [x1_right, y1]
            ], dtype=np.int32)
            
            cv2.fillPoly(lane_image, [pts], (0, 100, 0))
        
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
        debug_size = (320, 240)  # Küçük görüntüler için boyut
        
        # Birleştirilmiş görüntü için zemin oluştur (2x2 grid)
        h, w = debug_size
        debug_view = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Orijinal görüntüyü sol üst köşeye yerleştir
        resized_original = cv2.resize(original_frame, debug_size)
        debug_view[0:h, 0:w] = resized_original
        
        # İşlenmiş görüntüyü (şeritler çizilmiş) sağ üst köşeye yerleştir
        resized_processed = cv2.resize(processed_frame, debug_size)
        debug_view[0:h, w:w*2] = resized_processed
        
        # Debug görüntülerini alt kısma yerleştir
        if "edges" in self.debug_images:
            resized_edges = cv2.resize(self.debug_images["edges"], debug_size)
            debug_view[h:h*2, 0:w] = resized_edges
            
        if "bird_eye_view" in self.debug_images:
            resized_bird = cv2.resize(self.debug_images["bird_eye_view"], debug_size)
            debug_view[h:h*2, w:w*2] = resized_bird
        
        # Üst kısma bilgi metni ekle
        font = cv2.FONT_HERSHEY_SIMPLEX
        if center_diff is not None:
            cv2.putText(debug_view, f"Merkez Farki: {center_diff}px", 
                      (10, 30), font, 0.7, (0, 0, 255), 2)
            
        # Başlıklar ekle
        cv2.putText(debug_view, "Orijinal", (10, 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_view, "Serit Tespiti", (w+10, 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_view, "Kenarlar", (10, h+15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_view, "Kus Bakisi", (w+10, h+15), font, 0.5, (255, 255, 255), 1)
        
        return debug_view
    
    def process_frame(self, frame):
        """
        Bir görüntü karesini işler ve sonuçları döndürür.
        
        Args:
            frame (numpy.ndarray): İşlenecek görüntü karesi
            
        Returns:
            tuple: (İşlenmiş görüntü, merkez pozisyon farkı)
        """
        # Debug görüntülerini temizle
        self.debug_images = {}
        
        # Görüntüyü yeniden boyutlandır
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Şeritleri tespit et
        left_lane, right_lane = self.detect_lane_lines(frame)
        
        # Şeritleri çiz
        result = self.draw_lanes(frame, left_lane, right_lane)
        
        # Merkez hesapla
        center_diff = self.calculate_lane_center(left_lane, right_lane)
        
        # Debug modunda merkez çiz
        if self.debug:
            # Ana görüntüde merkezleri çiz
            if center_diff is not None:
                image_center = self.width // 2
                lane_center = image_center + center_diff
                cv2.circle(result, (image_center, self.height - 30), 5, (255, 0, 0), -1)
                cv2.circle(result, (lane_center, self.height - 30), 5, (0, 0, 255), -1)
                cv2.line(result, (image_center, self.height - 30), 
                       (lane_center, self.height - 30), (0, 255, 255), 2)
                
                # Merkez farkını yaz
                cv2.putText(result, f"Merkez Farki: {center_diff}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Birleştirilmiş debug görünümü oluştur
            debug_view = self.create_debug_view(frame, result, center_diff)
            
            # Debug görünümünü göster (ana görüntü yerine)
            result = debug_view
        
        return result, center_diff 