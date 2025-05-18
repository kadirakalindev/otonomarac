#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Şerit Tespit Sistemi Test Programı
"""

import cv2
import numpy as np
from lane_detection import LaneDetector

def main():
    # Kamera veya video kaynağını başlat
    cap = cv2.VideoCapture(0)  # Webcam için 0, video dosyası için dosya adı
    
    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Şerit dedektörünü başlat
    detector = LaneDetector(camera_resolution=(640, 480), debug=True)
    
    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı!")
            break
            
        # Görüntüyü işle
        result, center_diff = detector.process_frame(frame)
        
        # Sonucu göster
        cv2.imshow("Serit Tespiti", result)
        
        # Çıkış için 'q' tuşu
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 