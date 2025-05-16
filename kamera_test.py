#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kamera Test Programı
Bu program, canlı kamera görüntüsü üzerinde şerit algılama algoritmasını test eder.
Ayrıca, tespit edilen şeritlere göre dönüş açısını hesaplar ve gösterir.
Optimize edilmiş performans ve düşük bellek kullanımı.
"""

import cv2
import numpy as np
import argparse
import time
import os
import logging
import gc
from datetime import datetime
from lane_detection import LaneDetector

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KameraTest")

class KameraTest:
    def __init__(self, camera_id=0, resolution=(320, 240), performance_mode="balanced", 
                 debug=False, save_dir=None, show_histogram=False):
        """
        Kamera test sınıfı
        
        Args:
            camera_id (int): Kamera ID (webcam için genellikle 0)
            resolution (tuple): Görüntü çözünürlüğü (genişlik, yükseklik)
            performance_mode (str): Performans modu (speed, balanced, quality)
            debug (bool): Debug modunu etkinleştir
            save_dir (str): Görüntü kaydetme dizini (None ise kaydedilmez)
            show_histogram (bool): Histogram gösterimini etkinleştir
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.debug = debug
        self.save_dir = save_dir
        self.show_histogram = show_histogram
        self.performance_mode = performance_mode
        
        # Performans moduna göre işleme çözünürlüğünü belirle
        self.processing_resolution = self._get_processing_resolution(performance_mode)
        logger.info(f"İşleme çözünürlüğü: {self.processing_resolution[0]}x{self.processing_resolution[1]} (Mod: {performance_mode})")
        
        # Kamerayı başlat
        self.cap = cv2.VideoCapture(camera_id)
        
        # Kamera çözünürlüğünü ayarla
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Kamera düzgün başlatıldı mı kontrol et
        if not self.cap.isOpened():
            logger.error(f"Kamera {camera_id} açılamadı!")
            raise RuntimeError(f"Kamera {camera_id} açılamadı!")
        
        # Gerçek kamera çözünürlüğünü al
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Kamera {camera_id} başlatıldı")
        logger.info(f"Kamera çözünürlüğü: {self.actual_width}x{self.actual_height}")
        
        # Şerit dedektörünü başlat
        self.lane_detector = LaneDetector(camera_resolution=self.processing_resolution, debug=debug)
        
        # Bellek önbellek tamponları
        self.resize_buffer = np.zeros((self.processing_resolution[1], self.processing_resolution[0], 3), dtype=np.uint8)
        
        # Kaydetme dizinini oluştur
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Kayıt dizini oluşturuldu: {self.save_dir}")
        
        # Performans istatistikleri
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        self.fps_list = []
        self.center_diff_values = []
        
        # Ön işleme tamponları
        self.display_buffer = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self.histogram_buffer = None
        if self.show_histogram:
            self.histogram_buffer = np.zeros((100, resolution[0], 3), dtype=np.uint8)
        
        # FPS ölçümü için
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
        # Kare atlama modunu etkinleştir
        self.enable_frame_skipping = performance_mode == "speed"
        self.frame_skip_threshold = 50  # 50ms'den uzun işleme süresi için kare atla
        self.frame_skip_counter = 0
    
    def _get_processing_resolution(self, performance_mode):
        """
        Performans moduna göre işleme çözünürlüğünü belirler
        
        Args:
            performance_mode (str): Performans modu ('speed', 'balanced', 'quality')
            
        Returns:
            tuple: İşleme çözünürlüğü (genişlik, yükseklik)
        """
        width, height = self.resolution
        
        if performance_mode == "speed":
            # Hız odaklı: Düşük çözünürlük
            return (width // 2, height // 2)
        
        elif performance_mode == "quality":
            # Kalite odaklı: Yüksek çözünürlük
            return self.resolution
        
        else:  # "balanced" veya diğer değerler
            # Dengeli mod: Orta çözünürlük
            return (width * 3 // 4, height * 3 // 4)
    
    def _draw_steering_indicator(self, frame, center_diff):
        """
        Dönüş açısı göstergesini çizer
        
        Args:
            frame (numpy.ndarray): Çizim yapılacak görüntü
            center_diff (float): Merkez farkı (negatif: sola, pozitif: sağa)
        """
        if center_diff is None:
            return
        
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h - 30
        
        # Direksiyon simülasyonu
        radius = 25
        thickness = 2
        
        # Direksiyon çemberi
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), thickness)
        
        # Direksiyon orta çizgisi
        cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (255, 255, 255), 1)
        
        # Normalize edilmiş sapma açısı hesapla (-30 derece ile +30 derece arasında)
        max_diff = 100  # piksel cinsinden maksimum sapma
        normalized_diff = min(max(center_diff / max_diff, -1.0), 1.0)
        angle = normalized_diff * 30  # -30 ile +30 derece arasında
        
        # Sapma çizgisini çiz
        angle_rad = np.deg2rad(angle)
        end_x = int(center_x + radius * np.sin(angle_rad))
        end_y = int(center_y - radius * np.cos(angle_rad))
        
        # Sapma yönüne göre renk belirle (kırmızı: sağa, mavi: sola, yeşil: orta)
        if abs(angle) < 5:
            color = (0, 255, 0)  # Yeşil - düz
        elif angle < 0:
            color = (255, 0, 0)  # Mavi - sola
        else:
            color = (0, 0, 255)  # Kırmızı - sağa
            
        cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 3)
        
        # Açı değerini yaz
        cv2.putText(frame, f"{angle:.1f} derece", (center_x - 40, center_y + radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_histogram(self, binary_warped):
        """
        Şerit histogram görüntüsünü oluşturur
        
        Args:
            binary_warped (numpy.ndarray): İşlenmiş ikili görüntü
            
        Returns:
            numpy.ndarray: Histogram görüntüsü
        """
        if not self.show_histogram or binary_warped is None:
            return None
            
        # Tamamen siyah bir görüntü oluştur
        buffer = self.histogram_buffer
        buffer.fill(0)
        
        # Görüntünün alt yarısındaki piksel histogramını hesapla
        if len(binary_warped.shape) > 2:
            binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
            
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Histogramı normalize et
        if np.max(histogram) > 0:
            histogram = histogram / np.max(histogram) * 100
        
        # Histogramı çiz
        for i, hist_val in enumerate(histogram):
            if i < buffer.shape[1]:
                cv2.line(buffer, (i, buffer.shape[0]), (i, buffer.shape[0] - int(hist_val)), (0, 255, 0), 1)
        
        # Sol ve sağ şerit konumlarını belirle
        if np.max(histogram) > 0:
            midpoint = histogram.shape[0]//2
            left_base = np.argmax(histogram[:midpoint])
            right_base = np.argmax(histogram[midpoint:]) + midpoint
            
            # Şerit konumlarını çiz
            cv2.line(buffer, (left_base, 0), (left_base, buffer.shape[0]), (255, 0, 0), 2)
            cv2.line(buffer, (right_base, 0), (right_base, buffer.shape[0]), (0, 0, 255), 2)
        
        return buffer
    
    def run(self):
        """
        Kamera test döngüsü
        """
        paused = False
        logger.info("Kamera testi başladı. Çıkış için 'q', duraklatmak için 'space', görüntü kaydetmek için 's' tuşuna basın.")
        
        while True:
            if not paused:
                # Kareyi oku
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Kamera karesi okunamadı!")
                    break
                
                # Kare atlama modunu uygula
                if self.enable_frame_skipping and self.frame_skip_counter > 0:
                    self.frame_skip_counter -= 1
                    continue
                
                # İşleme çözünürlüğüne göre yeniden boyutlandır
                if self.processing_resolution != self.resolution:
                    processed_frame = cv2.resize(frame, self.processing_resolution, dst=self.resize_buffer)
                else:
                    processed_frame = frame
                
                # FPS ölçümü
                self.curr_frame_time = time.time()
                self.fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = self.curr_frame_time
                
                # Şerit işleme ve zamanlama
                process_start = time.time()
                lane_frame, center_diff = self.lane_detector.process_frame(processed_frame)
                process_time = (time.time() - process_start) * 1000  # milisaniye
                
                # Performans moduna göre kare atlama
                if self.enable_frame_skipping and process_time > self.frame_skip_threshold:
                    self.frame_skip_counter = max(1, int(process_time / 16.0))  # 60 fps için
                    logger.debug(f"İşlem süresi {process_time:.1f}ms - {self.frame_skip_counter} kare atlanacak")
                
                # İşlenen görüntüyü orijinal çözünürlüğe getir
                if lane_frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                    lane_frame = cv2.resize(lane_frame, self.resolution)
                
                # Performans metriklerini kaydet
                self.processing_times.append(process_time)
                self.fps_list.append(self.fps)
                if center_diff is not None:
                    self.center_diff_values.append(abs(center_diff))
                
                # Dönüş göstergesini çiz
                self._draw_steering_indicator(lane_frame, center_diff)
                
                # Histogram çiz (gerekirse)
                if self.show_histogram:
                    # Lane detector'dan ikili görüntüyü al
                    binary_warped = self.lane_detector.get_binary_warped() if hasattr(self.lane_detector, 'get_binary_warped') else None
                    hist_image = self._draw_histogram(binary_warped)
                    
                    if hist_image is not None:
                        # Histogramı ekranın altına yerleştir
                        display_height = self.resolution[1] + hist_image.shape[0]
                        display = np.zeros((display_height, self.resolution[0], 3), dtype=np.uint8)
                        display[:self.resolution[1], :] = lane_frame
                        display[self.resolution[1]:, :] = hist_image
                        lane_frame = display
                
                # İşlem bilgilerini ekle
                avg_process_time = sum(self.processing_times[-100:]) / min(len(self.processing_times), 100)
                avg_fps = sum(self.fps_list[-100:]) / min(len(self.fps_list), 100)
                
                cv2.putText(lane_frame, f"FPS: {avg_fps:.1f}", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(lane_frame, f"Süre: {avg_process_time:.1f}ms", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(lane_frame, f"Merkez Farkı: {center_diff if center_diff is not None else 'Yok'}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Performans modunu göster
                cv2.putText(lane_frame, f"Mod: {self.performance_mode}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Görüntüyü göster
                cv2.imshow("Şerit Tespiti", lane_frame)
                
                self.frame_count += 1
                self.display_buffer = lane_frame
                
                # Bellek yönetimi
                if self.frame_count % 50 == 0:
                    gc.collect()
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            # Çıkış kontrolü
            if key == ord('q'):
                logger.info("Kullanıcı tarafından durduruldu.")
                break
            elif key == ord('s'):
                # Görüntüyü kaydet
                if self.save_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.save_dir, f"lane_frame_{timestamp}.jpg")
                    cv2.imwrite(filename, self.display_buffer)
                    logger.info(f"Görüntü kaydedildi: {filename}")
            elif key == ord(' '):
                # Duraklatma/devam etme
                paused = not paused
                status = "duraklatıldı" if paused else "devam ediyor"
                logger.info(f"Kamera {status}")
            elif key == ord('p'):
                # Performans modu değiştir
                modes = ["speed", "balanced", "quality"]
                current_index = modes.index(self.performance_mode)
                next_index = (current_index + 1) % len(modes)
                self.performance_mode = modes[next_index]
                
                # Performans moduna göre parametreleri güncelle
                self.processing_resolution = self._get_processing_resolution(self.performance_mode)
                self.resize_buffer = np.zeros((self.processing_resolution[1], self.processing_resolution[0], 3), dtype=np.uint8)
                
                # Frame skipping modunu güncelle
                self.enable_frame_skipping = self.performance_mode == "speed"
                
                logger.info(f"Performans modu değiştirildi: {self.performance_mode}")
                logger.info(f"Yeni işleme çözünürlüğü: {self.processing_resolution[0]}x{self.processing_resolution[1]}")
        
        # Performans istatistiklerini yaz
        elapsed_time = time.time() - self.start_time
        avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        avg_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        
        print("\n===== PERFORMANS RAPORU =====")
        print(f"Toplam süre: {elapsed_time:.2f} saniye")
        print(f"İşlenen kare: {self.frame_count}")
        print(f"Ortalama FPS: {avg_fps:.2f}")
        print(f"Ortalama işleme süresi: {avg_process_time:.2f} ms")
        if self.center_diff_values:
            avg_center_diff = sum(self.center_diff_values) / len(self.center_diff_values)
            print(f"Ortalama şerit sapması: {avg_center_diff:.2f} piksel")
        print("=============================")
        
        # Kaynakları temizle
        self.cap.release()
        cv2.destroyAllWindows()
        
        self.resize_buffer = None
        self.display_buffer = None
        self.histogram_buffer = None
        gc.collect()

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Kamera Test Programı')
    parser.add_argument('--camera', type=int, default=0, help='Kamera ID (varsayılan: 0)')
    parser.add_argument('--resolution', default='320x240', help='Görüntü çözünürlüğü (GENxYÜK)')
    parser.add_argument('--debug', action='store_true', help='Debug modunu etkinleştir')
    parser.add_argument('--save-dir', help='Görüntü kaydetme dizini')
    parser.add_argument('--histogram', action='store_true', help='Histogram görüntüsünü göster')
    parser.add_argument('--performance', default='balanced', choices=['speed', 'balanced', 'quality'],
                      help='Performans modu: hız/kalite dengesi')
    
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
        resolution = (320, 240)
    
    # Kamera testini başlat
    try:
        tester = KameraTest(
            camera_id=args.camera,
            resolution=resolution,
            performance_mode=args.performance,
            debug=args.debug,
            save_dir=args.save_dir,
            show_histogram=args.histogram
        )
        tester.run()
        
    except Exception as e:
        logger.error(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nKamera Testi Tamamlandı!")
    print("Tuş Kontrolleri:")
    print("  q: Çıkış")
    print("  space: Duraklat/Devam Et")
    print("  s: Görüntüyü Kaydet")
    print("  p: Performans Modunu Değiştir")

if __name__ == "__main__":
    main() 