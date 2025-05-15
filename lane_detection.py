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
    def __init__(self, camera_resolution=(320, 240), debug=False):
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
            [self.width * 0.35, self.height * 0.65],  # Sol üst - daha iyi perspektif için ayarlandı
            [self.width * 0.65, self.height * 0.65],  # Sağ üst
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
        
        # Filtre parametreleri - Dönüşler için iyileştirildi
        self.blur_kernel_size = 5
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        
        # Hough dönüşümü parametreleri - U dönüşleri ve virajlar için optimize edildi
        self.rho = 1
        self.theta = np.pi/180
        self.hough_threshold = 15       # Daha düşük eşik (dönüşlerde çizgileri daha kolay tespit etmek için)
        self.min_line_length = 15       # Daha kısa çizgileri de tespit et
        self.max_line_gap = 100         # Optimize edilmiş boşluk toleransı
        
        # Renk filtresi parametreleri - Siyah zemin üzerinde beyaz şeritler için optimize edildi
        # HSV renk aralıkları
        self.yellow_lower = np.array([15, 80, 120])
        self.yellow_upper = np.array([35, 255, 255])
        self.white_lower = np.array([0, 0, 210])      # Beyaz renk için optimize edildi
        self.white_upper = np.array([180, 30, 255])   # Tüm beyaz tonları için genişletildi
        
        # Son şerit çizgileri (hafıza)
        self.last_left_lane = None
        self.last_right_lane = None
        self.smoothing_factor = 0.85  # Arttırıldı - kesikli çizgiler için daha iyi takip
        
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
        Görüntüye renk filtresi uygular, beyaz şeritleri belirginleştirir.
        Siyah zemin için optimize edilmiştir.
        
        Args:
            image (numpy.ndarray): İşlenecek renkli görüntü
            
        Returns:
            numpy.ndarray: Renk filtrelenmiş binary görüntü
        """
        # BGR'dan HSV'ye dönüştür
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Gri tonlama için direkt BGR'dan dönüşüm
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Beyaz renk maskeleri oluştur (HSV ve gri tonlama kullanarak)
        white_mask_hsv = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Gri tonlama için otomatik eşikleme (Siyah zemin üzerinde beyaz şeritler için güçlü)
        _, white_mask_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Adaptif eşikleme ekle
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5
        )
        
        # HSV ve gri tonlama maskelerini birleştir
        white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_gray)
        white_mask = cv2.bitwise_or(white_mask, adaptive_mask)
        
        # Sarı maske oluştur (eğer sarı çizgiler varsa)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Tüm maskeleri birleştir
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Morfolojik işlemler (gürültü azaltma ve şerit kalınlaştırma)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Debug için maskeleri kaydet
        if self.debug:
            white_mask_colored = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            yellow_mask_colored = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            adaptive_mask_colored = cv2.cvtColor(adaptive_mask, cv2.COLOR_GRAY2BGR)
            self.debug_images["white_mask"] = white_mask_colored
            self.debug_images["yellow_mask"] = yellow_mask_colored
            self.debug_images["adaptive_mask"] = adaptive_mask_colored
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
        
        # Gri tonlamaya dönüştürme
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
        
        # İlgi bölgesi belirleme (ROI) - U dönüşleri için optimize edildi
        roi_mask = np.zeros_like(combined)
        roi_vertices = np.array([
            [0, self.height],
            [self.width * 0.35, self.height * 0.6],  # Daha geniş bir görüş açısı
            [self.width * 0.65, self.height * 0.6],
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
            
            # ROI'yi görselleştir
            roi_viz = image.copy()
            cv2.polylines(roi_viz, [roi_vertices], True, (0, 0, 255), 2)
            self.debug_images["roi"] = roi_viz
        
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
                if abs(slope) < 0.3:  # U dönüşleri için eğim toleransını artır (0.2'den 0.3'e)
                    continue
                
                # Çizgi uzunluğunu hesapla
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Çizgi açısını hesapla (derece cinsinden)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Pozisyona göre sınıflandırma
                midpoint_x = (x1 + x2) / 2
                
                # Eğim ve pozisyona göre sınıflandır
                if slope < 0 and midpoint_x < self.width * 0.7:  # Sol şerit
                    left_lines.append(line[0])
                elif slope > 0 and midpoint_x > self.width * 0.3:  # Sağ şerit
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
        Kesikli şeritler için iyileştirildi.
        
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
        # Kesikli çizgiler için smoothing_factor değerini artırdık
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
        
        # Ham çizgileri ve şeritleri çiz
        if self.debug and hasattr(self, 'raw_lines'):
            # Sol şerit için ham çizgileri işle
            left_points = []
            if self.raw_lines.get("left_lines", []):
                for line in self.raw_lines["left_lines"]:
                    x1, y1, x2, y2 = line
                    left_points.extend([(x1, y1), (x2, y2)])
                    cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Mavi renk
                
                # Sol şerit noktalarını y koordinatına göre sırala
                left_points.sort(key=lambda p: p[1], reverse=True)  # Y'ye göre büyükten küçüğe sırala
                
                # Şerit noktalarını filtrele ve pürüzsüzleştir
                filtered_left = self._smooth_points(left_points)
                
                # Şeridin üzerinde yeşil çizgi çiz
                if len(filtered_left) > 1:
                    for i in range(len(filtered_left)-1):
                        cv2.line(lane_image, filtered_left[i], filtered_left[i+1], 
                                (0, 255, 0), thickness)  # Yeşil renk
            
            # Sağ şerit için ham çizgileri işle
            right_points = []
            if self.raw_lines.get("right_lines", []):
                for line in self.raw_lines["right_lines"]:
                    x1, y1, x2, y2 = line
                    right_points.extend([(x1, y1), (x2, y2)])
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Kırmızı renk
                
                # Sağ şerit noktalarını y koordinatına göre sırala
                right_points.sort(key=lambda p: p[1], reverse=True)  # Y'ye göre büyükten küçüğe sırala
                
                # Şerit noktalarını filtrele ve pürüzsüzleştir
                filtered_right = self._smooth_points(right_points)
                
                # Şeridin üzerinde yeşil çizgi çiz
                if len(filtered_right) > 1:
                    for i in range(len(filtered_right)-1):
                        cv2.line(lane_image, filtered_right[i], filtered_right[i+1], 
                                (0, 255, 0), thickness)  # Yeşil renk
        
        # Eğer ham çizgiler yoksa ya da debug modunda değilsek, parametrik çizileri kullan
        else:
            # Şerit kaybı durumunda uyarı mesajı
            if left_lane is None and right_lane is None:
                if self.debug:
                    cv2.putText(image, "UYARI: Serit bulunamadi!", 
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return image
            
            # Sol şerit için parametrik çizim
            if left_lane is not None:
                m_left, b_left = left_lane
                
                # Y değerlerini yolun altından üstüne doğru belirli aralıklarla oluştur
                y_values = np.linspace(self.height, int(self.height * 0.6), 20)
                
                # Her y değeri için x hesapla
                points = []
                for y in y_values:
                    try:
                        x = int((y - b_left) / m_left)
                        if 0 <= x < self.width:  # Görüntü sınırları içinde kontrol et
                            points.append((x, int(y)))
                    except:
                        pass
                
                # Şerit çizgisini çiz
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(lane_image, points[i], points[i+1], color, thickness)
            
            # Sağ şerit için parametrik çizim
            if right_lane is not None:
                m_right, b_right = right_lane
                
                # Y değerlerini yolun altından üstüne doğru belirli aralıklarla oluştur
                y_values = np.linspace(self.height, int(self.height * 0.6), 20)
                
                # Her y değeri için x hesapla
                points = []
                for y in y_values:
                    try:
                        x = int((y - b_right) / m_right)
                        if 0 <= x < self.width:  # Görüntü sınırları içinde kontrol et
                            points.append((x, int(y)))
                    except:
                        pass
                
                # Şerit çizgisini çiz
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(lane_image, points[i], points[i+1], color, thickness)
            
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
    
    def _smooth_points(self, points, window_size=5, distance_threshold=30):
        """
        Şerit noktalarını filtreler ve pürüzsüzleştirir
        
        Args:
            points (list): Şerit noktaları listesi [(x1,y1), (x2,y2), ...]
            window_size (int): Pürüzsüzleştirme pencere boyutu
            distance_threshold (int): Noktalar arası maksimum uzaklık eşiği
            
        Returns:
            list: Filtrelenmiş ve pürüzsüzleştirilmiş noktalar
        """
        if len(points) < 2:
            return points
        
        # Noktaları benzersiz yap ve y değerine göre sırala
        unique_points = []
        y_values = set()
        
        for x, y in points:
            if y not in y_values:
                unique_points.append((x, y))
                y_values.add(y)
        
        unique_points.sort(key=lambda p: p[1], reverse=True)  # Y'ye göre büyükten küçüğe sırala
        
        # Uzak noktaları filtrele
        filtered_points = [unique_points[0]]  # İlk noktayı ekle
        
        for i in range(1, len(unique_points)):
            prev_x, prev_y = filtered_points[-1]
            curr_x, curr_y = unique_points[i]
            
            # Öklid mesafesi hesapla
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            if distance < distance_threshold:
                filtered_points.append((curr_x, curr_y))
        
        # Pürüzsüzleştirme (hareketli ortalama)
        if len(filtered_points) < window_size:
            return filtered_points
            
        smoothed_points = []
        half_window = window_size // 2
        
        for i in range(len(filtered_points)):
            # Pencere sınırlarını belirle
            start = max(0, i - half_window)
            end = min(len(filtered_points), i + half_window + 1)
            
            # Penceredeki x değerlerinin ortalamasını al
            window_points = filtered_points[start:end]
            avg_x = sum(p[0] for p in window_points) // len(window_points)
            
            smoothed_points.append((avg_x, filtered_points[i][1]))
        
        return smoothed_points
    
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