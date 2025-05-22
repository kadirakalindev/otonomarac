#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kamera Testi
Bu program, kameradan gelen görüntüyü alıp şerit tespiti yaparak gösterir.
Kalibrasyon dosyasını test etmek için ideal bir araçtır.
"""

import cv2
import numpy as np
import argparse
import time
import logging
import os
import signal
from datetime import datetime

# Özel modüller
try:
    from lane_detection import LaneDetector
except ImportError:
    print("HATA: lane_detection modülü bulunamadı!")
    print("Bu programı otonom araç projesinin ana dizininde çalıştırın.")
    import sys
    sys.exit(1)

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KameraTesti")

class CameraTest:
    """Kamera ile şerit tespitini test etmek için sınıf"""
    
    def __init__(self, resolution=(320, 240), calibration_file="calibration.json", 
                 camera_id=0, debug=False, display_fps=True, save_path=None):
        """
        Kamera testi için başlangıç ayarları
        
        Args:
            resolution (tuple): Görüntü çözünürlüğü (genişlik, yükseklik)
            calibration_file (str): Kalibrasyon dosyası yolu
            camera_id (int): Kamera ID'si
            debug (bool): Debug modu aktif/pasif
            display_fps (bool): FPS gösterme aktif/pasif
            save_path (str): Kaydedilen görüntülerin konumu
        """
        self.width, self.height = resolution
        self.camera_id = camera_id
        self.debug = debug
        self.display_fps = display_fps
        self.save_path = save_path
        self.calibration_file = calibration_file
        
        # FPS ölçümü için değişkenler
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        self.avg_fps = []
        self.processing_times = []
        
        # Lane Detection modülünü başlat
        try:
            self.lane_detector = LaneDetector(camera_resolution=(self.width, self.height), 
                                             debug=self.debug)
            
            # Kalibrasyon dosyasını yükle (varsa)
            if os.path.exists(self.calibration_file):
                self.lane_detector.load_calibration(self.calibration_file)
                
            logger.info("Şerit tespit modülü başlatıldı")
        except Exception as e:
            logger.error(f"Şerit tespit modülü başlatılırken hata: {e}")
            raise
        
        # Video yakalama özelliğini başlat
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Kamera çalışıyor mu kontrol et
            if not self.cap.isOpened():
                raise Exception(f"Kamera açılamadı (ID: {camera_id})")
            
            logger.info(f"Kamera başlatıldı (ID: {camera_id}, {self.width}x{self.height})")
        except Exception as e:
            logger.error(f"Kamera başlatılırken hata: {e}")
            raise
        
        # Video kaydı için dosya yolu
        self.video_writer = None
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            
        # Ctrl+C ile programı sonlandırma
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """CTRL+C ile çıkış yapıldığında temiz bir şekilde kapat"""
        logger.info("Test sonlandırılıyor...")
        self.close()
        import sys
        sys.exit(0)
    
    def setup_video_writer(self):
        """Video yazıcıyı hazırla"""
        if self.save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.save_path, f"lane_test_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, 20.0, 
                (self.width * 2, self.height) if self.debug else (self.width, self.height)
            )
            logger.info(f"Video kaydı başlatıldı: {video_path}")
    
    def update_fps(self):
        """FPS ölçümünü güncelle"""
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.curr_frame_time
        
        # FPS'i yumuşatmak için ortalama al (son 10 kare)
        self.avg_fps.append(fps)
        if len(self.avg_fps) > 10:
            self.avg_fps.pop(0)
        
        self.fps = sum(self.avg_fps) / len(self.avg_fps)
    
    def draw_stats(self, frame, deviation, processing_time):
        """
        İstatistikleri görüntü üzerine çiz
        
        Args:
            frame: Çizim yapılacak görüntü
            deviation: Şerit sapması (piksel)
            processing_time: İşleme süresi (ms)
        """
        # Arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # FPS
        if self.display_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # İşleme süresi
        proc_text = f"İşleme: {processing_time:.1f} ms"
        cv2.putText(frame, proc_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Sapma
        if deviation is not None:
            # Rengi sapma miktarına göre değiştir
            if abs(deviation) < 20:
                color = (0, 255, 0)  # İyi: yeşil
            elif abs(deviation) < 50:
                color = (0, 255, 255)  # Orta: sarı
            else:
                color = (0, 0, 255)  # Kötü: kırmızı
                
            dev_text = f"Sapma: {deviation:.1f} px"
            cv2.putText(frame, dev_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Sapma: -", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    def run(self):
        """Ana test döngüsü"""
        logger.info("Test başlatılıyor...")
        
        # Video yazıcıyı hazırla
        self.setup_video_writer()
        
        # Klavye kontrolleri için açıklama
        print("\nKONTROLLER:")
        print("ESC: Programı sonlandır")
        print("SPACE: Duraksat/Devam et")
        print("S: Mevcut kareyi kaydet")
        print("-----------------------")
        
        # Değişkenler
        paused = False
        frame_count = 0
        total_deviation = 0
        max_deviation = 0
        
        try:
            while True:
                # Duraklatılmışsa son kareyi göstermeye devam et
                if not paused:
                    # Kameradan görüntü al
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.error("Kameradan görüntü alınamadı!")
                        break
                    
                    orig_frame = frame.copy()
                    
                    # Şerit tespiti için süreyi ölç
                    start_time = time.time()
                    
                    # Şerit tespiti yap
                    processed_frame, center_diff = self.lane_detector.process_frame(frame)
                    
                    # İşleme süresini hesapla
                    processing_time = (time.time() - start_time) * 1000  # ms cinsinden
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 30:
                        self.processing_times.pop(0)
                    
                    # Sapma istatistiklerini güncelle
                    deviation = center_diff
                    if deviation is not None:
                        total_deviation += abs(deviation)
                        max_deviation = max(max_deviation, abs(deviation))
                        frame_count += 1
                    
                    # Görüntü boyutunu kontrol et ve gerekirse yeniden boyutlandır
                    if processed_frame.shape[:2] != (self.height, self.width):
                        logger.debug(f"İşlenmiş görüntü boyutu{processed_frame.shape[:2][::-1]} orijinalden farklı. Yeniden boyutlandırılıyor.")
                        processed_frame = cv2.resize(processed_frame, (self.width, self.height))
                    
                    # FPS hesapla
                    self.update_fps()
                
                # İstatistikleri çiz
                display_frame = processed_frame.copy()
                self.draw_stats(display_frame, deviation, 
                               sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0)
                
                # Debug modunda yan yana görüntüle
                if self.debug:
                    # Orijinal görüntü üzerine de istatistik çiz
                    orig_with_stats = orig_frame.copy()
                    self.draw_stats(orig_with_stats, deviation, 
                                   sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0)
                    
                    # İki görüntüyü birleştir
                    combined_frame = np.hstack((orig_with_stats, display_frame))
                    cv2.imshow("Kamera Testi (Orijinal vs İşlenmiş)", combined_frame)
                    
                    # Video kaydı
                    if self.video_writer:
                        self.video_writer.write(combined_frame)
                else:
                    cv2.imshow("Kamera Testi", display_frame)
                    
                    # Video kaydı
                    if self.video_writer:
                        self.video_writer.write(display_frame)
                
                # Klavye kontrolü
                key = cv2.waitKey(1) & 0xFF
                
                # ESC ile çıkış
                if key == 27:
                    break
                
                # Space ile duraklat/devam et
                elif key == 32:  # Space
                    paused = not paused
                    status = "duraklatıldı" if paused else "devam ediyor"
                    logger.info(f"Test {status}")
                
                # 's' ile kareyi kaydet
                elif key == ord('s'):
                    if self.save_path:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_name = os.path.join(self.save_path, f"lane_frame_{timestamp}.jpg")
                        frame_to_save = combined_frame if self.debug else display_frame
                        cv2.imwrite(save_name, frame_to_save)
                        logger.info(f"Kare kaydedildi: {save_name}")
                    else:
                        logger.warning("Kayıt dizini belirtilmediği için kare kaydedilemedi!")
            
            # Test sonuçlarını göster
            if frame_count > 0:
                avg_deviation = total_deviation / frame_count
                logger.info(f"Test sonuçları:")
                logger.info(f"Ortalama FPS: {self.fps:.1f}")
                logger.info(f"Ortalama işleme süresi: {sum(self.processing_times) / len(self.processing_times):.1f} ms")
                logger.info(f"Ortalama mutlak sapma: {avg_deviation:.1f} piksel")
                logger.info(f"Maksimum sapma: {max_deviation:.1f} piksel")
            
        except Exception as e:
            logger.error(f"Test sırasında hata: {e}")
        
        finally:
            self.close()
    
    def close(self):
        """Kaynakları temizle ve kapat"""
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
            
        cv2.destroyAllWindows()
        logger.info("Test sonlandırıldı ve kaynaklar temizlendi.")

def parse_args():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(
        description="Otonom Araç - Kamera Test Programı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--resolution", default="320x240",
                      help="Kamera çözünürlüğü (GENxYÜK)")
    
    parser.add_argument("--camera", type=int, default=0,
                      help="Kamera ID")
                      
    parser.add_argument("--calibration", default="calibration.json",
                      help="Kalibrasyon dosyası yolu")
    
    parser.add_argument("--debug", action="store_true",
                      help="Debug modu (orijinal ve işlenmiş görüntüleri yan yana gösterir)")
    
    parser.add_argument("--save", action="store_true",
                      help="Görüntüleri kaydet")
    
    parser.add_argument("--output", default="output",
                      help="Kaydedilen görüntülerin konumu")
    
    parser.add_argument("--no-fps", action="store_true",
                      help="FPS gösterme")
    
    return parser.parse_args()

def main():
    """Ana program"""
    args = parse_args()
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split("x"))
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}")
        width, height = 320, 240
        logger.info(f"Varsayılan çözünürlük kullanılacak: {width}x{height}")
    
    # Kayıt dizini
    save_path = args.output if args.save else None
    
    # Test nesnesini oluştur ve çalıştır
    try:
        test = CameraTest(
            resolution=(width, height),
            calibration_file=args.calibration,
            camera_id=args.camera,
            debug=args.debug,
            display_fps=not args.no_fps,
            save_path=save_path
        )
        test.run()
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    print("\nOTONOM ARAÇ KAMERA TEST PROGRAMI")
    print("--------------------------------")
    print("Bu program, kameradan gelen görüntüyü alıp şerit tespiti yapar ve sonucu gösterir.")
    print("Kalibrasyon dosyasını test etmek için kullanışlıdır.\n")
    
    exit_code = main()
    
    if exit_code == 0:
        print("\nProgram başarıyla tamamlandı.")
        print("\nİpuçları:")
        print("- Yol çizgilerini daha iyi tespit etmek için kalibrasyon yapın:")
        print("  python3 kalibrasyon_optimize.py")
        print("- Şerit tespit parametrelerini ayarlamak için:")
        print("  python3 kalibrasyon_olustur_optimize.py --interactive")
        print("- Video testi için:")
        print("  python3 video_test.py --video <video_dosyası>")
    else:
        print("\nProgram hata ile sonlandı.")
        print("'--debug' parametresi ile daha fazla bilgi alabilirsiniz.") 