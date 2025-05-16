#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Video Test Programı
Bu program, mevcut şerit tespit algoritmasını örnek video üzerinde test eder.
Video kareleri işlenirken sonuçları gerçek zamanlı olarak gösterir.
"""

import cv2
import numpy as np
import argparse
import time
import os
import logging
from lane_detection import LaneDetector

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoTest")

def process_video(video_path, output_path=None, resolution=(320, 240), play_speed=1.0, save_output=False):
    """
    Video dosyasını işler ve şerit tespiti uygular
    
    Args:
        video_path (str): İşlenecek video dosyasının yolu
        output_path (str): İşlenmiş videonun kaydedileceği yol (opsiyonel)
        resolution (tuple): İşleme çözünürlüğü (genişlik, yükseklik)
        play_speed (float): Oynatma hızı (1.0 = normal hız)
        save_output (bool): İşlenmiş videoyu kaydet
    """
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Video dosyası açılamadı: {video_path}")
        return False
    
    # Video özellikleri
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video yüklendi: {video_path}")
    logger.info(f"Orijinal çözünürlük: {original_width}x{original_height}, FPS: {fps}, Toplam kare: {total_frames}")
    
    # Video yazıcı (eğer kaydetme seçeneği seçildiyse)
    out = None
    if save_output and output_path:
        # İşlenmiş video ve orijinal videoyu yan yana göstermek için genişletilmiş boyut
        output_width = resolution[0] * 2  # Yan yana iki görüntü
        output_height = resolution[1]
        
        # Video yazıcı
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        logger.info(f"Video kaydı başlatıldı: {output_path}")
    
    # Şerit dedektörünü başlat
    lane_detector = LaneDetector(camera_resolution=resolution, debug=True)
    
    # İşlem süresi ölçümü
    start_time = time.time()
    frame_count = 0
    
    # Paused durumu
    paused = False
    
    # Performans metrikleri
    processing_times = []
    center_diff_values = []
    lost_lane_frames = 0
    total_frames_processed = 0
    
    # Çerçeve işleme döngüsü
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                logger.info("Video dosyasının sonuna ulaşıldı veya kare okunamadı.")
                break
            
            # Kareyi yeniden boyutlandır
            frame = cv2.resize(frame, resolution)
            
            # Şerit tespiti uygula ve zamanlama ölç
            process_start = time.time()
            processed_frame, center_diff = lane_detector.process_frame(frame)
            process_time = (time.time() - process_start) * 1000  # milisaniye
            
            # Performans metriklerini kaydet
            processing_times.append(process_time)
            if center_diff is not None:
                center_diff_values.append(abs(center_diff))
            else:
                lost_lane_frames += 1
                
            total_frames_processed += 1
            
            # Performans istatistiklerini hesapla
            avg_process_time = sum(processing_times) / len(processing_times)
            if center_diff_values:
                avg_center_diff = sum(center_diff_values) / len(center_diff_values)
                max_center_diff = max(center_diff_values)
            else:
                avg_center_diff = 0
                max_center_diff = 0
                
            lane_detection_rate = ((total_frames_processed - lost_lane_frames) / total_frames_processed) * 100 if total_frames_processed > 0 else 0
            
            # Boyut kontrolü - işlenmiş frame ile orijinal frame'in boyutları farklı olabilir
            if processed_frame.shape[0] != resolution[1] or processed_frame.shape[1] != resolution[0]:
                logger.info(f"İşlenmiş görüntü boyutu({processed_frame.shape[1]}x{processed_frame.shape[0]}) orijinalden farklı. Yeniden boyutlandırılıyor.")
                processed_frame = cv2.resize(processed_frame, resolution)
            
            # İşlem bilgilerini ekle
            frame_info = f"Kare: {frame_count}/{total_frames} | İşlem: {process_time:.1f}ms"
            center_info = f"Merkez Farkı: {center_diff if center_diff is not None else 'Yok'}"
            
            cv2.putText(frame, frame_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, center_info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Performans istatistiklerini ekle
            cv2.putText(frame, f"Ort. İşlem: {avg_process_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Şerit Algılama: %{lane_detection_rate:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Ort. Sapma: {avg_center_diff:.1f}px", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Debug görüntüsü ile birlikte orijinal görüntüyü birleştir
            combined_view = np.zeros((resolution[1], resolution[0]*2, 3), dtype=np.uint8)
            combined_view[:, :resolution[0]] = frame
            combined_view[:, resolution[0]:] = processed_frame
            
            # Görselleştirme için açıklama ekle
            cv2.putText(combined_view, "Orijinal Görüntü", (10, resolution[1]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_view, "İşlenmiş Görüntü", (resolution[0]+10, resolution[1]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # İlerleme çubuğu ekle
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_width = int(resolution[0] * 2 * progress)
            cv2.rectangle(combined_view, (0, resolution[1]-5), (progress_width, resolution[1]), (0, 255, 0), -1)
            
            # Videoyu kaydet (eğer etkinleştirildiyse)
            if out is not None:
                out.write(combined_view)
                
            frame_count += 1
        
        # Görüntüyü göster
        cv2.imshow("Şerit Tespiti Test", combined_view)
        
        # Tuş kontrolü
        wait_time = 0 if paused else int(1000/fps/play_speed)
        key = cv2.waitKey(wait_time) & 0xFF
        
        # Çıkış kontrolü
        if key == 27:  # ESC tuşu
            logger.info("Kullanıcı tarafından durduruldu.")
            break
        elif key == ord(' '):  # Space (boşluk) tuşu
            # Duraklatma/devam etme
            paused = not paused
            status = "duraklatıldı" if paused else "devam ediyor"
            logger.info(f"Video {status}")
        elif key == ord('s'):  # 's' tuşu
            # Kareyi kaydet
            snapshot_path = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_path, combined_view)
            logger.info(f"Anlık görüntü kaydedildi: {snapshot_path}")
        elif key == ord('>') or key == ord('.'):  # Hızlandırma
            play_speed = min(play_speed * 1.2, 10.0)
            logger.info(f"Oynatma hızı: {play_speed:.1f}x")
        elif key == ord('<') or key == ord(','):  # Yavaşlatma
            play_speed = max(play_speed / 1.2, 0.1)
            logger.info(f"Oynatma hızı: {play_speed:.1f}x")
        
        # İlerlemeyi logla (her 10 karede bir)
        if frame_count % 10 == 0 and frame_count > 0 and not paused:
            elapsed_time = time.time() - start_time
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            logger.info(f"İlerleme: %{progress:.1f} | {frame_count}/{total_frames} | Geçen süre: {elapsed_time:.1f}s")
    
    # Final performans raporu
    if total_frames_processed > 0:
        print("\n===== PERFORMANS RAPORU =====")
        print(f"Toplam işlenen kare sayısı: {total_frames_processed}")
        print(f"Ortalama işleme süresi: {sum(processing_times) / len(processing_times):.2f} ms")
        print(f"Minimum işleme süresi: {min(processing_times):.2f} ms")
        print(f"Maksimum işleme süresi: {max(processing_times):.2f} ms")
        print(f"Şerit algılama oranı: %{lane_detection_rate:.2f}")
        if center_diff_values:
            print(f"Ortalama merkez sapması: {sum(center_diff_values) / len(center_diff_values):.2f} piksel")
            print(f"Maksimum merkez sapması: {max(center_diff_values):.2f} piksel")
        print("============================\n")
    
    # Kaynakları temizle
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # İşlem istatistikleri
    elapsed_time = time.time() - start_time
    logger.info(f"Video işleme tamamlandı. Toplam süre: {elapsed_time:.1f}s | İşlenen kare: {frame_count}")
    
    return True

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Video Test Programı')
    parser.add_argument('--video', required=True, help='Test edilecek video dosyası yolu')
    parser.add_argument('--output', help='İşlenmiş video dosyası kaydedilecek yol')
    parser.add_argument('--resolution', default='320x240', help='İşleme çözünürlüğü (GENxYÜK)')
    parser.add_argument('--speed', type=float, default=1.0, help='Oynatma hızı (1.0 = normal hız)')
    parser.add_argument('--save', action='store_true', help='İşlenmiş videoyu kaydet')
    
    return parser.parse_args()

def main():
    """
    Ana program
    """
    # Argümanları işle
    args = parse_arguments()
    
    # Video dosyası kontrolü
    if not os.path.exists(args.video):
        logger.error(f"Video dosyası bulunamadı: {args.video}")
        return False
    
    # Çözünürlüğü ayrıştır
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        resolution = (320, 240)
    
    # Videoyu işle
    process_video(
        video_path=args.video,
        output_path=args.output,
        resolution=resolution,
        play_speed=args.speed,
        save_output=args.save
    )
    
    print("\nVideo İşleme Tamamlandı!")
    print("Tuş Kontrolleri:")
    print("  ESC: Çıkış")
    print("  SPACE: Duraklat/Devam Et")
    print("  S: Anlık Görüntü Kaydet")

if __name__ == "__main__":
    main() 