#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lane Detection module for Autonomous Vehicle
Şerit tespiti ve takibi için temel sınıf ve fonksiyonlar.
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional, Any

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LaneDetection')

class LaneDetector:
    """Şerit tespiti için ana sınıf"""
    
    def __init__(self, camera_resolution: Tuple[int, int] = (320, 240)):
        """
        Şerit dedektörü başlatıcı
        
        Args:
            camera_resolution: Kamera çözünürlüğü (genişlik, yükseklik)
        """
        if not isinstance(camera_resolution, tuple) or len(camera_resolution) != 2:
            raise ValueError("camera_resolution bir tuple (genişlik, yükseklik) olmalıdır")
            
        # Temel parametreler
        self.width, self.height = camera_resolution
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Kamera çözünürlüğü pozitif değerler olmalıdır")

        # ROI (İlgi Alanı) için parametreler
        self.roi_top_width_percent = 0.4  # Üst genişlik (görüntü genişliğinin yüzdesi)
        self.roi_bottom_width_percent = 0.95  # Alt genişlik
        self.roi_height_percent = 0.6  # ROI yüksekliği (görüntü yüksekliğinin yüzdesi)
        
        # Kenar tespiti parametreleri
        self.blur_kernel = 5
        if self.blur_kernel % 2 == 0:  # Kernel boyutu tek sayı olmalıdır
            self.blur_kernel += 1
            
        self.canny_low = 50
        self.canny_high = 150
        
        # Hough transform parametreleri
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 20
        self.min_line_length = int(self.height * 0.1)  # Yüksekliğin %10'u
        self.max_line_gap = int(self.height * 0.15)    # Yüksekliğin %15'i
        
        # Çizgi belleği için
        self.left_line_mem = []
        self.right_line_mem = []
        self.memory_size = 5
        self.smoothing_factor = 0.8
        
        # Tespit hata sayacı
        self.detection_failures = 0
        self.max_detection_failures = 10
        
        # Yaya geçidi ve hemzemin geçit algılama parametreleri
        self.crossing_horizontal_threshold = 0.15  # Yatay çizgilerin eşik değeri (radian)
        self.crossing_min_lines = 4  # Yaya geçidi için minimum çizgi sayısı
        
        # Yeşil çerçeve için ROI alanını hesapla
        self.calculate_roi_points()
        
        logger.info(f"Şerit dedektörü başlatıldı: {self.width}x{self.height} çözünürlükte")

    def calculate_roi_points(self):
        """ROI için köşe noktalarını hesaplar"""
        top_left_x = int(self.width * (1 - self.roi_top_width_percent) / 2)
        top_right_x = int(self.width * (1 + self.roi_top_width_percent) / 2)
        bottom_left_x = int(self.width * (1 - self.roi_bottom_width_percent) / 2)
        bottom_right_x = int(self.width * (1 + self.roi_bottom_width_percent) / 2)
        
        roi_top_y = int(self.height * (1 - self.roi_height_percent))
        roi_bottom_y = self.height
        
        # ROI köşe noktaları (sol üst, sağ üst, sağ alt, sol alt)
        self.roi_points = np.array([
            [top_left_x, roi_top_y],
            [top_right_x, roi_top_y],
            [bottom_right_x, roi_bottom_y],
            [bottom_left_x, roi_bottom_y]
        ], np.int32)
        
        logger.debug(f"ROI noktaları hesaplandı: {self.roi_points}")

    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Gelen frame'in geçerli olup olmadığını kontrol eder
        
        Args:
            frame: İşlenecek video karesi
            
        Returns:
            bool: Frame geçerli mi
        """
        if frame is None:
            logger.error("Boş frame. İşlem yapılamıyor.")
            return False
            
        if len(frame.shape) != 3:
            logger.error(f"Beklenmeyen frame boyutu: {frame.shape}")
            return False
            
        height, width = frame.shape[:2]
        if height != self.height or width != self.width:
            logger.warning(f"Frame boyutu tanımlanan çözünürlük ile uyuşmuyor. Beklenen: {self.width}x{self.height}, Alınan: {width}x{height}")
            return False
            
        return True

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frame'i şerit tespiti için ön işlemden geçirir
        
        Args:
            frame: İşlenecek video karesi
            
        Returns:
            edges: Kenar haritası
            roi_mask: ROI maskesi
        """
        # Gri tonlama dönüşümü
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Kenar tespiti
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # ROI maskesi oluşturma
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, [self.roi_points], 255)
        
        # Sadece ROI içindeki kenarları al
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        return masked_edges, roi_mask

    def detect_lane_lines(self, masked_edges: np.ndarray) -> Tuple[List, List, np.ndarray]:
        """
        Şerit çizgilerini tespit eder
        
        Args:
            masked_edges: Kenar tespiti yapılmış ve maskelenmiş görüntü
            
        Returns:
            left_lane: Sol şerit çizgileri
            right_lane: Sağ şerit çizgileri
            line_img: Çizgilerin çizildiği görüntü
        """
        line_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Hough transform ile çizgileri tespit et
        lines = cv2.HoughLinesP(
            masked_edges, self.hough_rho, self.hough_theta, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        
        left_lane = []
        right_lane = []
        
        # Eğer hiç çizgi bulunamazsa
        if lines is None:
            self.detection_failures += 1
            logger.warning(f"Şerit çizgileri tespit edilemedi. Hata sayısı: {self.detection_failures}/{self.max_detection_failures}")
            return left_lane, right_lane, line_img
            
        # Tespit başarılıysa sayacı sıfırla
        self.detection_failures = 0
        
        # Her bir çizgiyi işle
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Çok kısa çizgileri filtrele
            if abs(x2 - x1) < 10 and abs(y2 - y1) < 10:
                continue
                
            # Çizginin eğimini hesapla
            if x2 - x1 == 0:  # Dikey çizgi durumu
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Yatay çizgileri filtrele (yaya geçidi tespiti için ayrı işlenecek)
            if abs(slope) < 0.3:
                continue
                
            # Görüntünün alt yarısındaki çizgilere odaklan
            if y1 < self.height * 0.5 and y2 < self.height * 0.5:
                continue
                
            # Sol ve sağ şeritleri ayır (eğime göre)
            if slope < 0:  # Sol şerit
                left_lane.append(line[0])
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı
            else:  # Sağ şerit
                right_lane.append(line[0])
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil
        
        # En az bir sol ve bir sağ şerit bulunması gerekiyor
        if len(left_lane) == 0 or len(right_lane) == 0:
            logger.warning(f"{'Sol' if len(left_lane) == 0 else 'Sağ'} şerit tespit edilemedi.")
        
        return left_lane, right_lane, line_img

    def calculate_lane_lines(self, left_lane: List, right_lane: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Sol ve sağ şerit çizgileri için ortalama çizgileri hesaplar
        
        Args:
            left_lane: Sol şerit çizgileri
            right_lane: Sağ şerit çizgileri
            
        Returns:
            left_line: Sol şerit için ortalama çizgi [x1, y1, x2, y2]
            right_line: Sağ şerit için ortalama çizgi [x1, y1, x2, y2]
        """
        left_line = None
        right_line = None
        
        # Sol şerit hesaplama
        if len(left_lane) > 0:
            left_line = self._average_line(left_lane, "left")
            if left_line is not None:
                # Bellekteki sol şerit değerlerini güncelle
                if len(self.left_line_mem) >= self.memory_size:
                    self.left_line_mem.pop(0)
                self.left_line_mem.append(left_line)
                
                # Yumuşatma uygula
                left_line = self._smooth_line(self.left_line_mem)
        # Eğer sol şerit bulunamadı ve bellekte değer varsa
        elif len(self.left_line_mem) > 0:
            left_line = self._smooth_line(self.left_line_mem)
            
        # Sağ şerit hesaplama
        if len(right_lane) > 0:
            right_line = self._average_line(right_lane, "right")
            if right_line is not None:
                # Bellekteki sağ şerit değerlerini güncelle
                if len(self.right_line_mem) >= self.memory_size:
                    self.right_line_mem.pop(0)
                self.right_line_mem.append(right_line)
                
                # Yumuşatma uygula
                right_line = self._smooth_line(self.right_line_mem)
        # Eğer sağ şerit bulunamadı ve bellekte değer varsa
        elif len(self.right_line_mem) > 0:
            right_line = self._smooth_line(self.right_line_mem)
            
        return left_line, right_line

    def _average_line(self, lines: List, line_type: str) -> Optional[np.ndarray]:
        """
        Çizgilerin ortalamasını hesaplar
        
        Args:
            lines: Çizgi listesi
            line_type: Çizgi tipi ("left" veya "right")
            
        Returns:
            np.ndarray: Ortalama çizgi [x1, y1, x2, y2]
        """
        if not lines:
            return None
            
        # x ve y koordinatlarını topla
        x_sum, y_sum, m_sum, b_sum = 0, 0, 0, 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Eğim (m) ve sabit terim (b) hesapla: y = mx + b
            if x2 - x1 == 0:  # Dikey çizgi durumu
                continue
                
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            x_sum += (x1 + x2) / 2
            y_sum += (y1 + y2) / 2
            m_sum += m
            b_sum += b
            
        # Ortalama değerleri hesapla
        count = len(lines)
        if count == 0:
            return None
            
        x_avg = x_sum / count
        y_avg = y_sum / count
        m_avg = m_sum / count
        b_avg = b_sum / count
        
        # y = mx + b denklemi için y değerlerini hesapla
        y1 = self.height  # Alt kenar
        y2 = int(self.height * 0.6)  # Görüntünün %60'ı yukarıda
        
        # x = (y - b) / m
        if m_avg == 0:
            return None
            
        x1 = int((y1 - b_avg) / m_avg)
        x2 = int((y2 - b_avg) / m_avg)
        
        # Çizginin görüntü sınırları içinde kalmasını sağla
        border_margin = int(self.width * 0.1)  # %10 marj
        if (line_type == "left" and (x1 > self.width / 2 or x2 > self.width / 2)) or \
           (line_type == "right" and (x1 < self.width / 2 or x2 < self.width / 2)) or \
           abs(x1) > self.width + border_margin or abs(x2) > self.width + border_margin:
            logger.warning(f"Geçersiz {line_type} şerit çizgisi hesaplandı: ({x1}, {y1}), ({x2}, {y2})")
            return None
            
        return np.array([x1, y1, x2, y2])

    def _smooth_line(self, line_memory: List) -> np.ndarray:
        """
        Şerit çizgileri için yumuşatma uygular
        
        Args:
            line_memory: Önceki çizgilerin listesi
            
        Returns:
            np.ndarray: Yumuşatılmış çizgi [x1, y1, x2, y2]
        """
        if not line_memory:
            return None
            
        # Ağırlıklı ortalama hesapla (daha yeni değerlere daha fazla ağırlık ver)
        x1_sum, y1_sum, x2_sum, y2_sum = 0, 0, 0, 0
        weight_sum = 0
        
        for i, line in enumerate(line_memory):
            # Eskilere göre yeniler daha yüksek ağırlığa sahip
            weight = (i + 1) / sum(range(1, len(line_memory) + 1))
            x1, y1, x2, y2 = line
            
            x1_sum += x1 * weight
            y1_sum += y1 * weight
            x2_sum += x2 * weight
            y2_sum += y2 * weight
            weight_sum += weight
            
        # Ağırlıklı ortalama hesapla
        if weight_sum == 0:
            return line_memory[-1]  # Son değeri döndür
            
        x1_avg = int(x1_sum / weight_sum)
        y1_avg = int(y1_sum / weight_sum)
        x2_avg = int(x2_sum / weight_sum)
        y2_avg = int(y2_sum / weight_sum)
        
        return np.array([x1_avg, y1_avg, x2_avg, y2_avg])

    def detect_crossings(self, masked_edges: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Yaya geçidi ve hemzemin geçitleri tespit eder
        
        Args:
            masked_edges: Kenar tespiti yapılmış ve maskelenmiş görüntü
            
        Returns:
            bool: Geçit tespit edildi mi
            np.ndarray: Geçit çizgilerinin çizildiği görüntü
        """
        crossing_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Hough transform ile yatay çizgileri tespit et
        lines = cv2.HoughLinesP(
            masked_edges, self.hough_rho, self.hough_theta, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return False, crossing_img
            
        # Yatay çizgileri bul
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Çizginin eğimini hesapla
            if x2 - x1 == 0:  # Dikey çizgi durumu
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            angle = abs(np.arctan(slope))  # Radyan cinsinden açı
            
            # Yatay çizgileri tespit et (eğim açısı düşük)
            if angle < self.crossing_horizontal_threshold:
                horizontal_lines.append(line[0])
                cv2.line(crossing_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Sarı
        
        # Yatay çizgi sayısı eşik değerinden büyükse geçit olarak kabul et
        is_crossing = len(horizontal_lines) >= self.crossing_min_lines
        
        if is_crossing:
            logger.info(f"Yaya geçidi/hemzemin geçit tespit edildi: {len(horizontal_lines)} yatay çizgi")
            
        return is_crossing, crossing_img

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Video karesini işleyerek şerit tespiti yapar
        
        Args:
            frame: İşlenecek video karesi
            
        Returns:
            Dict: Tespit sonuçları ve işlenmiş görüntüler
        """
        result = {
            "frame": frame.copy(),
            "lanes_detected": False,
            "crossing_detected": False,
            "left_line": None,
            "right_line": None,
            "center_offset": 0.0,
            "visualization": None
        }
        
        # Frame'i doğrula
        if not self.validate_frame(frame):
            return result
            
        # Ön işleme
        masked_edges, roi_mask = self.preprocess_frame(frame)
        
        # Şerit çizgilerini tespit et
        left_lane, right_lane, line_img = self.detect_lane_lines(masked_edges)
        
        # Ortalama şerit çizgilerini hesapla
        left_line, right_line = self.calculate_lane_lines(left_lane, right_lane)
        
        # Yaya geçidi tespiti
        crossing_detected, crossing_img = self.detect_crossings(masked_edges)
        
        # Merkez ofsetini hesapla
        center_offset = self.calculate_center_offset(left_line, right_line)
        
        # Sonuç görüntüyü oluştur
        visualization = self.draw_visualization(frame, left_line, right_line, crossing_detected)
        
        # Sonuçları kaydet
        result["lanes_detected"] = (left_line is not None) or (right_line is not None)
        result["crossing_detected"] = crossing_detected
        result["left_line"] = left_line
        result["right_line"] = right_line
        result["center_offset"] = center_offset
        result["visualization"] = visualization
        result["line_img"] = line_img
        result["crossing_img"] = crossing_img
        
        return result

    def calculate_center_offset(self, left_line: Optional[np.ndarray], right_line: Optional[np.ndarray]) -> float:
        """
        Aracın merkez ofsetini hesaplar
        
        Args:
            left_line: Sol şerit çizgisi
            right_line: Sağ şerit çizgisi
            
        Returns:
            float: Merkez ofseti (-1 ile 1 arasında, negatif değerler sol tarafa kayma)
        """
        # Eğer hem sol hem de sağ şerit tespit edilmişse
        if left_line is not None and right_line is not None:
            # Alt kenar noktalarını al (y = height)
            left_x = left_line[0]
            right_x = right_line[0]
            
            # Şerit merkezi
            lane_center = (left_x + right_x) / 2
            
            # Görüntü merkezi
            image_center = self.width / 2
            
            # Merkez ofsetini hesapla ve normalize et (-1 ile 1 arasında)
            center_offset = (lane_center - image_center) / (image_center)
            return center_offset
            
        # Eğer sadece sol şerit tespit edilmişse
        elif left_line is not None:
            # Tahmini şerit genişliği (genellikle görüntü genişliğinin %60'ı)
            estimated_lane_width = self.width * 0.6
            
            # Sol şerit x koordinatı
            left_x = left_line[0]
            
            # Tahmini şerit merkezi
            estimated_lane_center = left_x + estimated_lane_width / 2
            
            # Görüntü merkezi
            image_center = self.width / 2
            
            # Merkez ofsetini hesapla ve normalize et
            center_offset = (estimated_lane_center - image_center) / image_center
            return center_offset
            
        # Eğer sadece sağ şerit tespit edilmişse
        elif right_line is not None:
            # Tahmini şerit genişliği
            estimated_lane_width = self.width * 0.6
            
            # Sağ şerit x koordinatı
            right_x = right_line[0]
            
            # Tahmini şerit merkezi
            estimated_lane_center = right_x - estimated_lane_width / 2
            
            # Görüntü merkezi
            image_center = self.width / 2
            
            # Merkez ofsetini hesapla ve normalize et
            center_offset = (estimated_lane_center - image_center) / image_center
            return center_offset
            
        # Eğer hiç şerit tespit edilmediyse
        return 0.0

    def draw_visualization(self, frame: np.ndarray, 
                          left_line: Optional[np.ndarray], 
                          right_line: Optional[np.ndarray],
                          crossing_detected: bool) -> np.ndarray:
        """
        Sonuç görüntüsünü çizer
        
        Args:
            frame: Orijinal video karesi
            left_line: Sol şerit çizgisi
            right_line: Sağ şerit çizgisi
            crossing_detected: Geçit tespit edildi mi
            
        Returns:
            np.ndarray: Görselleştirilmiş sonuç
        """
        visualization = frame.copy()
        
        # ROI bölgesini çiz
        cv2.polylines(visualization, [self.roi_points], True, (0, 255, 0), 2)
        
        # Sol şerit çizgisini çiz
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(visualization, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        
        # Sağ şerit çizgisini çiz
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(visualization, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        
        # Şerit merkezi ve araç merkezi
        if left_line is not None and right_line is not None:
            # Alt kenar noktalarını al
            left_x = left_line[0]
            right_x = right_line[0]
            
            # Şerit merkezi
            lane_center_x = int((left_x + right_x) / 2)
            
            # Gösterge çizgileri
            cv2.circle(visualization, (lane_center_x, self.height - 10), 10, (0, 255, 255), -1)
            cv2.circle(visualization, (int(self.width / 2), self.height - 10), 5, (255, 0, 0), -1)
        
        # Geçit tespit edildiyse bilgilendirme kutusu göster
        if crossing_detected:
            cv2.putText(visualization, "Gecit Tespit Edildi!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        return visualization

# Test fonksiyonu
def test_lane_detector(image_path: str = None):
    """
    Lane Detector sınıfını test eder
    
    Args:
        image_path: Test edilecek görüntü dosyası yolu (None ise kamera kullanılır)
    """
    import matplotlib.pyplot as plt
    from picamera2 import Picamera2
    
    # Dedektörü başlat
    detector = LaneDetector(camera_resolution=(320, 240))
    
    if image_path:
        # Görüntüyü oku
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Görüntü okunamadı: {image_path}")
            return
            
        # Boyutu ayarla
        frame = cv2.resize(frame, (detector.width, detector.height))
        
        # Şerit tespiti yap
        result = detector.process_frame(frame)
        
        # Sonuçları görselleştir
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Orijinal Görüntü")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(result["line_img"], cv2.COLOR_BGR2RGB))
        plt.title("Tespit Edilen Çizgiler")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(result["crossing_img"], cv2.COLOR_BGR2RGB))
        plt.title("Geçit Tespiti")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result["visualization"], cv2.COLOR_BGR2RGB))
        plt.title("Sonuç Görüntü")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Kamera kullan
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (detector.width, detector.height)}
        )
        picam2.configure(config)
        picam2.start()
        
        try:
            while True:
                # Kameradan görüntü al
                frame = picam2.capture_array()
                
                # BGR formatına dönüştür
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Şerit tespiti yap
                result = detector.process_frame(frame)
                
                # Görüntüle
                cv2.imshow("Lane Detection", result["visualization"])
                
                # ESC tuşu ile çık
                if cv2.waitKey(1) == 27:
                    break
        finally:
            picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Komut satırı argümanlarını işle
    import argparse
    
    parser = argparse.ArgumentParser(description="Şerit tespiti test programı")
    parser.add_argument("--image", help="Test edilecek görüntü dosyası yolu")
    args = parser.parse_args()
    
    # Test çalıştır
    test_lane_detector(image_path=args.image) 