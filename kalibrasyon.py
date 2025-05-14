#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kalibrasyon Aracı
Bu program, kamera kalibrasyonu ve şerit tespiti parametrelerinin ayarlanması için bir araç sağlar.
"""

import cv2
import numpy as np
import argparse
import json
import os
import logging
from picamera2 import Picamera2

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Kalibrasyon")

class KameraPerspektifAyarlayici:
    """
    Kamera perspektif dönüşümü için kalibrasyon aracı
    """
    def __init__(self, camera_resolution=(640, 480)):
        """
        KameraPerspektifAyarlayici sınıfını başlatır.
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
        """
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        
        # Varsayılan perspektif noktaları
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
        
        # Kamera başlat
        self.camera = Picamera2()
        self.camera_config = self.camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.camera.configure(self.camera_config)
        self.camera.start()
        
        # İşaretçi indeksi (fare ile taşınacak nokta)
        self.selected_point = None
        
        # Pencere adı
        self.window_name = "Perspektif Kalibrasyon"
        
        logger.info("Kamera perspektif ayarlayıcı başlatıldı.")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Fare olaylarını yakalayan geri çağırma fonksiyonu
        
        Args:
            event (int): Fare olayı (tıklama, sürükleme, vb.)
            x, y (int): Fare koordinatları
            flags, param: OpenCV geri çağırma parametreleri
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # En yakın noktayı bul
            min_dist = float('inf')
            for i, point in enumerate(self.src_points):
                dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    self.selected_point = i
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            # Noktayı güncelle
            self.src_points[self.selected_point] = [x, y]
    
    def _draw_points(self, image):
        """
        Perspektif noktalarını görüntü üzerine çizer
        
        Args:
            image (numpy.ndarray): Çizim yapılacak görüntü
            
        Returns:
            numpy.ndarray: Noktalar çizilmiş görüntü
        """
        result = image.copy()
        
        # Bölgeyi çiz
        cv2.polylines(result, [np.int32(self.src_points)], True, (0, 255, 0), 2)
        
        # Noktaları çiz
        for i, point in enumerate(self.src_points):
            x, y = point
            cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(result, f"P{i}", (int(x) + 10, int(y) + 10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result
    
    def _warp_perspective(self, image):
        """
        Görüntüye perspektif dönüşümü uygular
        
        Args:
            image (numpy.ndarray): Dönüştürülecek görüntü
            
        Returns:
            numpy.ndarray: Dönüştürülmüş görüntü
        """
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        warped = cv2.warpPerspective(image, M, (self.width, self.height))
        return warped
    
    def run(self):
        """
        Kalibrasyon aracını çalıştırır
        """
        # Pencere oluştur ve fare olaylarını bağla
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        while True:
            # Görüntü al
            frame = self.camera.capture_array()
            
            # BGR formatına dönüştür (OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Noktaları çiz
            annotated = self._draw_points(frame)
            
            # Kuş bakışı dönüşümünü göster
            warped = self._warp_perspective(frame)
            
            # Ekranda göster
            cv2.imshow(self.window_name, annotated)
            cv2.imshow("Kuş Bakışı Görünümü", warped)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            # ESC ile çık
            if key == 27:
                break
            # 's' ile kaydet
            elif key == ord('s'):
                self.save_calibration()
                logger.info("Kalibrasyon kaydedildi.")
                
            # 'r' ile sıfırla
            elif key == ord('r'):
                self.reset_points()
                logger.info("Noktalar sıfırlandı.")
        
        # Temizle
        self.cleanup()
    
    def save_calibration(self, filename="calibration.json"):
        """
        Kalibrasyon değerlerini JSON dosyasına kaydeder
        
        Args:
            filename (str): Kalibrasyon dosyası adı
        """
        calibration_data = {
            "src_points": self.src_points.tolist(),
            "dst_points": self.dst_points.tolist(),
            "resolution": {
                "width": self.width,
                "height": self.height
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)
    
    def reset_points(self):
        """
        Noktaları varsayılan değerlere sıfırlar
        """
        self.src_points = np.float32([
            [self.width * 0.25, self.height * 0.6],  # Sol üst
            [self.width * 0.75, self.height * 0.6],  # Sağ üst
            [0, self.height],                        # Sol alt
            [self.width, self.height]                # Sağ alt
        ])
    
    def cleanup(self):
        """
        Kaynakları temizler
        """
        self.camera.stop()
        cv2.destroyAllWindows()


class SeritTespitiAyarlayici:
    """
    Şerit tespiti parametrelerini ayarlama aracı
    """
    def __init__(self, camera_resolution=(640, 480)):
        """
        SeritTespitiAyarlayici sınıfını başlatır
        
        Args:
            camera_resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
        """
        self.width = camera_resolution[0]
        self.height = camera_resolution[1]
        
        # Varsayılan ayarlar
        self.blur_kernel_size = 5
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.hough_threshold = 20
        self.min_line_length = 20
        self.max_line_gap = 300
        
        # Kamera başlat
        self.camera = Picamera2()
        self.camera_config = self.camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.camera.configure(self.camera_config)
        self.camera.start()
        
        # Kalibrasyon yükle (varsa)
        self.src_points = None
        self.dst_points = None
        self.perspective_M = None
        self.load_calibration()
        
        # Pencere adları
        self.main_window = "Şerit Tespiti Ayarları"
        self.processed_window = "İşlenmiş Görüntü"
        
        logger.info("Şerit tespiti ayarlayıcı başlatıldı.")
    
    def load_calibration(self, filename="calibration.json"):
        """
        Kalibrasyon dosyasını yükler
        
        Args:
            filename (str): Kalibrasyon dosyası adı
        
        Returns:
            bool: Başarı durumu
        """
        if not os.path.exists(filename):
            logger.warning(f"Kalibrasyon dosyası bulunamadı: {filename}")
            return False
        
        try:
            with open(filename, 'r') as f:
                calibration = json.load(f)
            
            self.src_points = np.float32(calibration["src_points"])
            self.dst_points = np.float32(calibration["dst_points"])
            self.perspective_M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            
            logger.info("Kalibrasyon yüklendi.")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon yüklenemedi: {e}")
            return False
    
    def _create_trackbars(self):
        """
        Kontrol sürgülerini oluşturur
        """
        cv2.namedWindow(self.main_window)
        
        # Trackbar oluştur
        cv2.createTrackbar("Blur Kernel", self.main_window, self.blur_kernel_size, 15, self._on_blur_changed)
        cv2.createTrackbar("Canny Alt Eşik", self.main_window, self.canny_low_threshold, 255, self._on_canny_low_changed)
        cv2.createTrackbar("Canny Üst Eşik", self.main_window, self.canny_high_threshold, 255, self._on_canny_high_changed)
        cv2.createTrackbar("Hough Eşik", self.main_window, self.hough_threshold, 100, self._on_hough_threshold_changed)
        cv2.createTrackbar("Min Çizgi Uzunluğu", self.main_window, self.min_line_length, 100, self._on_min_line_length_changed)
        cv2.createTrackbar("Max Çizgi Boşluğu", self.main_window, self.max_line_gap, 500, self._on_max_line_gap_changed)
    
    def _on_blur_changed(self, value):
        """Blur kernel değiştiğinde"""
        # Çift sayı olmasını engelle
        self.blur_kernel_size = value if value % 2 == 1 else value + 1
        if self.blur_kernel_size < 1:
            self.blur_kernel_size = 1
    
    def _on_canny_low_changed(self, value):
        """Canny alt eşik değiştiğinde"""
        self.canny_low_threshold = value
    
    def _on_canny_high_changed(self, value):
        """Canny üst eşik değiştiğinde"""
        self.canny_high_threshold = value
    
    def _on_hough_threshold_changed(self, value):
        """Hough eşik değiştiğinde"""
        self.hough_threshold = value
    
    def _on_min_line_length_changed(self, value):
        """Min çizgi uzunluğu değiştiğinde"""
        self.min_line_length = value
    
    def _on_max_line_gap_changed(self, value):
        """Max çizgi boşluğu değiştiğinde"""
        self.max_line_gap = value
    
    def _process_image(self, image):
        """
        Görüntüyü işler ve şeritleri tespit eder
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            tuple: (işlenmiş görüntü, şerit çizilmiş görüntü)
        """
        # BGR'dan gri tonlamaya dönüştür
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü giderme
        # Not: blur_kernel_size en az 1 ve tek sayı olmalı
        kernel_size = max(1, self.blur_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Kenar tespiti
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # İlgi bölgesi (ROI) belirleme
        if self.src_points is not None:
            # Kalibrasyon varsa, kuş bakışı dönüşümü uygula
            warped = cv2.warpPerspective(edges, self.perspective_M, (self.width, self.height))
        else:
            # Kalibrasyon yoksa, varsayılan ROI maskesi kullan
            roi_mask = np.zeros_like(edges)
            roi_vertices = np.array([
                [0, self.height],
                [self.width * 0.4, self.height * 0.6],
                [self.width * 0.6, self.height * 0.6],
                [self.width, self.height]
            ], dtype=np.int32)
            cv2.fillPoly(roi_mask, [roi_vertices], 255)
            warped = cv2.bitwise_and(edges, roi_mask)
        
        # Çizgileri bul
        lines = cv2.HoughLinesP(
            warped,
            1,
            np.pi/180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Sonuçları görselleştir
        result = image.copy()
        
        # Tespit edilen çizgileri çiz
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return warped, result
    
    def run(self):
        """
        Ayarlama aracını çalıştırır
        """
        self._create_trackbars()
        
        while True:
            # Görüntü al
            frame = self.camera.capture_array()
            
            # BGR formatına dönüştür (OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Görüntüyü işle
            processed, lane_detected = self._process_image(frame)
            
            # Ekranda göster
            cv2.imshow(self.processed_window, processed)
            cv2.imshow(self.main_window, lane_detected)
            
            # Bilgileri görüntüde göster
            info_text = f"Blur: {self.blur_kernel_size}, Canny: {self.canny_low_threshold}/{self.canny_high_threshold}"
            info_text2 = f"Hough: {self.hough_threshold}, MinLen: {self.min_line_length}, MaxGap: {self.max_line_gap}"
            cv2.putText(lane_detected, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(lane_detected, info_text2, (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            # ESC ile çık
            if key == 27:
                break
            # 's' ile kaydet
            elif key == ord('s'):
                self.save_parameters()
                logger.info("Parametreler kaydedildi.")
        
        # Temizle
        self.cleanup()
    
    def save_parameters(self, filename="lane_params.json"):
        """
        Ayarlanan parametreleri JSON dosyasına kaydeder
        
        Args:
            filename (str): Parametre dosyası adı
        """
        parameters = {
            "blur_kernel_size": self.blur_kernel_size,
            "canny_low_threshold": self.canny_low_threshold,
            "canny_high_threshold": self.canny_high_threshold,
            "hough_threshold": self.hough_threshold,
            "min_line_length": self.min_line_length,
            "max_line_gap": self.max_line_gap
        }
        
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=4)
    
    def cleanup(self):
        """
        Kaynakları temizler
        """
        self.camera.stop()
        cv2.destroyAllWindows()


def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Otonom Araç Kalibrasyon Aracı')
    parser.add_argument('--mode', choices=['perspective', 'lane'], default='perspective',
                      help='Kalibrasyon modu (perspective: Kamera perspektifi, lane: Şerit tespiti)')
    parser.add_argument('--resolution', default='640x480', help='Kamera çözünürlüğü (GENxYÜK)')
    
    return parser.parse_args()

def main():
    """
    Ana program
    """
    # Argümanları işle
    args = parse_arguments()
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        resolution = (640, 480)
    
    # Çalıştırılacak modu seç
    if args.mode == 'perspective':
        calibrator = KameraPerspektifAyarlayici(camera_resolution=resolution)
    else:  # lane
        calibrator = SeritTespitiAyarlayici(camera_resolution=resolution)
    
    # Aracı çalıştır
    calibrator.run()

if __name__ == "__main__":
    main() 