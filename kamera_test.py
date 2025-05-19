#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kamera Test Programı
Bu program, kamera görüntüsünü ve şerit tespitini test etmek için kullanılır.
Kalibrasyon dosyasını kullanarak şerit tespitini görselleştirir.
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
import time
from datetime import datetime

class KameraTest:
    def __init__(self, resolution="640x480", calibration_file=None, debug=False):
        """Kamera test sınıfını başlatır"""
        self.debug = debug
        self.calibration_file = calibration_file
        
        # Çözünürlük ayarları
        try:
            width, height = resolution.split("x")
            self.width = int(width)
            self.height = int(height)
        except:
            print(f"Hata: Geçersiz çözünürlük formatı: {resolution}")
            print("Format: WIDTHxHEIGHT (örn: 640x480)")
            sys.exit(1)
        
        self.camera = None
        self.roi_vertices = None
        self.lane_points = None
        self.running = True
        
        # Kenar algılama parametreleri
        self.blur_kernel = 5
        self.canny_low = 30
        self.canny_high = 120
        
        # Hough transform parametreleri
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 20
        self.hough_min_line_length = 20
        self.hough_max_line_gap = 20
        
        # Kalibrasyon dosyasını yükle
        self._load_calibration()
        
        # Kamerayı başlat
        self._initialize_camera()
    
    def _load_calibration(self):
        """Kalibrasyon dosyasını yükler"""
        if self.calibration_file and os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as file:
                    data = json.load(file)
                    
                    if 'roi_vertices' in data:
                        self.roi_vertices = np.array(data['roi_vertices'], dtype=np.int32)
                    
                    if 'lane_points' in data:
                        self.lane_points = np.array(data['lane_points'], dtype=np.int32)
                    
                    print(f"Kalibrasyon dosyası yüklendi: {self.calibration_file}")
            except Exception as e:
                print(f"Kalibrasyon dosyası yüklenirken hata oluştu: {e}")
        else:
            print(f"Kalibrasyon dosyası bulunamadı: {self.calibration_file}")
            print("Varsayılan ROI kullanılacak.")
            
            # Varsayılan ROI tanımla
            self.roi_vertices = np.array([
                [0, self.height],
                [self.width * 0.3, self.height * 0.5],
                [self.width * 0.7, self.height * 0.5],
                [self.width, self.height]
            ], dtype=np.int32)
    
    def _initialize_camera(self):
        """Kamerayı başlatır ve ayarları yapılandırır"""
        try:
            # Kamera ID'sini belirle (Windows'ta genellikle 0, Linux'ta /dev/video0)
            camera_id = 0
            
            # Kamerayı aç
            self.camera = cv2.VideoCapture(camera_id)
            
            # Kamera çözünürlüğünü ayarla
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Kamera parametrelerini ayarla (bazı kameralarda çalışmayabilir)
            try:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Otomatik odaklamayı kapat
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manuel pozlama
                self.camera.set(cv2.CAP_PROP_EXPOSURE, 40)  # Pozlama değeri
            except:
                print("Uyarı: Kamera parametreleri ayarlanamadı. Varsayılan ayarlar kullanılacak.")
            
            # Kamera başlatma kontrolü
            if not self.camera.isOpened():
                print("Hata: Kamera başlatılamadı!")
                print("Simülasyon modunda devam edilecek...")
                self._use_simulation_mode()
                return
            
            # Test karesi al
            ret, frame = self.camera.read()
            if not ret:
                print("Hata: Kameradan kare okunamadı!")
                print("Simülasyon modunda devam edilecek...")
                self._use_simulation_mode()
                return
            
            # Kameranın dengelenmesi için bekle
            print("Kamera başlatılıyor, lütfen bekleyin...")
            time.sleep(2)
            print("Kamera hazır.")
            
        except Exception as e:
            print(f"Kamera başlatılırken hata oluştu: {e}")
            print("Simülasyon modunda devam edilecek...")
            self._use_simulation_mode()
    
    def _use_simulation_mode(self):
        """Kamera yerine simülasyon modu kullanır"""
        print("Simülasyon modu aktif.")
        
        # Boş bir görüntü oluştur
        self.simulation_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Şeritleri çiz
        if self.lane_points is not None and len(self.lane_points) >= 5:
            # Sol şerit
            cv2.line(self.simulation_frame, tuple(self.lane_points[0]), tuple(self.lane_points[1]), (255, 0, 0), 2)
            
            # Orta şerit
            cv2.line(self.simulation_frame, tuple(self.lane_points[1]), tuple(self.lane_points[2]), (0, 255, 0), 2)
            
            # Sağ şerit
            cv2.line(self.simulation_frame, tuple(self.lane_points[2]), tuple(self.lane_points[3]), (255, 0, 0), 2)
            cv2.line(self.simulation_frame, tuple(self.lane_points[3]), tuple(self.lane_points[4]), (255, 0, 0), 2)
        else:
            # Varsayılan şeritler
            h, w = self.height, self.width
            cv2.line(self.simulation_frame, (int(w*0.3), h), (int(w*0.3), int(h*0.6)), (255, 0, 0), 2)
            cv2.line(self.simulation_frame, (int(w*0.5), h), (int(w*0.5), int(h*0.6)), (0, 255, 0), 2)
            cv2.line(self.simulation_frame, (int(w*0.7), h), (int(w*0.7), int(h*0.6)), (255, 0, 0), 2)
        
        # ROI bölgesini çiz
        if self.roi_vertices is not None:
            cv2.polylines(self.simulation_frame, [self.roi_vertices], True, (0, 255, 255), 2)
        
        # Simülasyon bilgisi ekle
        cv2.putText(self.simulation_frame, "SİMÜLASYON MODU", (int(self.width/2)-80, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _region_of_interest(self, img):
        """Görüntüdeki ilgi bölgesini (ROI) maskeler"""
        mask = np.zeros_like(img)
        
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        # ROI bölgesini doldur
        cv2.fillPoly(mask, [self.roi_vertices], ignore_mask_color)
        
        # Orijinal görüntüyü ROI ile maskele
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def _detect_edges(self, img):
        """Kenarları tespit eder"""
        # Gri tonlamaya dönüştür
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bulanıklaştır
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Canny kenar algılama
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        return edges
    
    def _detect_lines(self, img):
        """Hough transform ile çizgileri tespit eder"""
        lines = cv2.HoughLinesP(
            img,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        return lines
    
    def _draw_lines(self, img, lines, color=(0, 255, 0), thickness=2):
        """Tespit edilen çizgileri görüntü üzerine çizer"""
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        
        return line_img
    
    def _draw_lane_points(self, img):
        """Kalibrasyon noktalarını görüntü üzerine çizer"""
        if self.lane_points is not None and len(self.lane_points) > 0:
            # Noktaları çiz
            for i, point in enumerate(self.lane_points):
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
                cv2.putText(img, str(i+1), (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Şerit çizgilerini çiz
            if len(self.lane_points) >= 5:
                # Sol şerit
                cv2.line(img, tuple(self.lane_points[0]), tuple(self.lane_points[1]), (255, 0, 0), 2)
                
                # Orta şerit
                cv2.line(img, tuple(self.lane_points[1]), tuple(self.lane_points[2]), (0, 255, 0), 2)
                
                # Sağ şerit
                cv2.line(img, tuple(self.lane_points[2]), tuple(self.lane_points[3]), (255, 0, 0), 2)
                cv2.line(img, tuple(self.lane_points[3]), tuple(self.lane_points[4]), (255, 0, 0), 2)
        
        return img
    
    def _draw_roi(self, img):
        """ROI bölgesini görüntü üzerine çizer"""
        roi_img = np.zeros_like(img)
        cv2.fillPoly(roi_img, [self.roi_vertices], (0, 255, 0, 64))
        return cv2.addWeighted(img, 1, roi_img, 0.3, 0)
    
    def _save_screenshot(self, img):
        """Ekran görüntüsünü kaydeder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kamera_test_{timestamp}.jpg"
        cv2.imwrite(filename, img)
        print(f"Ekran görüntüsü kaydedildi: {filename}")
    
    def process_frame(self, frame):
        """Kamera karesini işler ve şerit tespiti yapar"""
        # Orijinal görüntüyü kopyala
        result = frame.copy()
        
        # ROI uygula
        roi = self._region_of_interest(frame)
        
        # Kenarları tespit et
        edges = self._detect_edges(roi)
        
        # Çizgileri tespit et
        lines = self._detect_lines(edges)
        
        # Çizgileri çiz
        line_img = self._draw_lines(frame, lines)
        
        # Sonuç görüntüsünü oluştur
        result = cv2.addWeighted(result, 0.8, line_img, 1, 0)
        
        # ROI bölgesini çiz
        if self.debug:
            result = self._draw_roi(result)
        
        # Kalibrasyon noktalarını çiz
        result = self._draw_lane_points(result)
        
        # Debug bilgilerini ekle
        if self.debug:
            cv2.putText(result, f"Canny: {self.canny_low}/{self.canny_high}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, f"Hough: t={self.hough_threshold}, l={self.hough_min_line_length}, g={self.hough_max_line_gap}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Yardım bilgilerini ekle
            cv2.putText(result, "ESC: Çıkış, S: Ekran görüntüsü", (10, self.height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def run(self):
        """Ana döngü"""
        print("\nKontroller:")
        print("- ESC/Q: Programdan çık")
        print("- S: Ekran görüntüsü kaydet")
        print("- C: Canny parametrelerini değiştir")
        print("- H: Hough parametrelerini değiştir")
        
        while self.running:
            # Kamera karesi oku veya simülasyon karesi kullan
            if hasattr(self, 'simulation_frame'):
                # Simülasyon modu
                frame = self.simulation_frame.copy()
                ret = True
            else:
                # Gerçek kamera
                ret, frame = self.camera.read()
            
            if not ret:
                print("Kamera karesi okunamadı!")
                break
            
            # Kareyi işle
            processed = self.process_frame(frame)
            
            # Sonucu göster
            cv2.imshow("Kamera Test", processed)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC veya q tuşu
                self.running = False
            elif key == ord('s'):  # S tuşu
                self._save_screenshot(processed)
            elif key == ord('c'):  # C tuşu
                self._change_canny_params()
            elif key == ord('h'):  # H tuşu
                self._change_hough_params()
        
        # Temizlik
        if not hasattr(self, 'simulation_frame') and self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Program sonlandırıldı.")
    
    def _change_canny_params(self):
        """Canny parametrelerini değiştir"""
        print("\nCanny parametrelerini değiştir:")
        try:
            low = int(input(f"Düşük eşik değeri ({self.canny_low}): ") or self.canny_low)
            high = int(input(f"Yüksek eşik değeri ({self.canny_high}): ") or self.canny_high)
            
            self.canny_low = low
            self.canny_high = high
            print(f"Canny parametreleri güncellendi: {self.canny_low}/{self.canny_high}")
        except ValueError:
            print("Geçersiz değer! Parametreler değiştirilmedi.")
    
    def _change_hough_params(self):
        """Hough parametrelerini değiştir"""
        print("\nHough parametrelerini değiştir:")
        try:
            threshold = int(input(f"Eşik değeri ({self.hough_threshold}): ") or self.hough_threshold)
            min_line_length = int(input(f"Minimum çizgi uzunluğu ({self.hough_min_line_length}): ") or self.hough_min_line_length)
            max_line_gap = int(input(f"Maksimum çizgi boşluğu ({self.hough_max_line_gap}): ") or self.hough_max_line_gap)
            
            self.hough_threshold = threshold
            self.hough_min_line_length = min_line_length
            self.hough_max_line_gap = max_line_gap
            print(f"Hough parametreleri güncellendi: t={self.hough_threshold}, l={self.hough_min_line_length}, g={self.hough_max_line_gap}")
        except ValueError:
            print("Geçersiz değer! Parametreler değiştirilmedi.")

def main():
    """Ana program"""
    parser = argparse.ArgumentParser(description="Otonom Araç Kamera Test Programı")
    parser.add_argument("--resolution", default="640x480", help="Kamera çözünürlüğü (örn: 640x480)")
    parser.add_argument("--calibration", default="serit_kalibrasyon.json", help="Kalibrasyon dosyası")
    parser.add_argument("--debug", action="store_true", help="Debug modunu aktifleştir")
    args = parser.parse_args()
    
    try:
        tester = KameraTest(
            resolution=args.resolution,
            calibration_file=args.calibration,
            debug=args.debug
        )
        tester.run()
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    print("\nOTONOM ARAÇ KAMERA TEST PROGRAMI")
    print("------------------------------")
    main() 