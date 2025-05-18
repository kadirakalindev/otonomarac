#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Şerit Tespiti Test Programı
"""

import cv2
import time
import logging
from picamera2 import Picamera2
from lane_detection import LaneDetector

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestLaneDetection")

def main():
    """
    Ana test programı
    """
    # Kamera çözünürlüğü
    width, height = 640, 480
    
    try:
        # Picamera2'yi başlat
        camera = Picamera2()
        
        # Kamera yapılandırması
        camera_config = camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        camera.configure(camera_config)
        camera.start()
        
        # Kameranın dengelenmesi için bekle
        time.sleep(2)
        
        # Şerit dedektörünü başlat
        lane_detector = LaneDetector(camera_resolution=(width, height), debug=True)
        
        logger.info("Test başlatılıyor...")
        print("Çıkmak için 'q' tuşuna basın")
        
        while True:
            # Kameradan görüntü al
            frame = camera.capture_array()
            
            # RGB'den BGR'a dönüştür (OpenCV formatı)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Şerit tespiti yap
            processed_frame, center_diff = lane_detector.process_frame(frame)
            
            # Merkez sapmasını göster
            if center_diff is not None:
                cv2.putText(processed_frame, f"Merkez Sapma: {center_diff:.1f}px",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow("Serit Tespiti", processed_frame)
            
            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Kaynakları temizle
        camera.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 