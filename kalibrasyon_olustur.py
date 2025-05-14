#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Basitleştirilmiş Kalibrasyon Dosyası Oluşturucusu
Bu program, kalibrasyon.py çalıştırmadan manuel olarak kalibrasyon dosyası oluşturur.
"""

import json
import numpy as np
import argparse
import os
import logging

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KalibrasyonOlustur")

def create_calibration_file(width=320, height=240, src_points=None, dst_points=None, output_file="calibration.json"):
    """
    Manuel olarak belirlenen değerlerle kalibrasyon dosyası oluşturur
    
    Args:
        width (int): Görüntü genişliği
        height (int): Görüntü yüksekliği
        src_points (list): Kaynak perspektif noktaları [x,y] formatında
        dst_points (list): Hedef perspektif noktaları [x,y] formatında
        output_file (str): Çıktı dosyası adı
    """
    # Varsayılan perspektif noktaları
    if src_points is None:
        src_points = [
            [width * 0.35, height * 0.65],  # Sol üst
            [width * 0.65, height * 0.65],  # Sağ üst
            [0, height],                     # Sol alt
            [width, height]                  # Sağ alt
        ]
    
    if dst_points is None:
        dst_points = [
            [width * 0.25, 0],               # Sol üst
            [width * 0.75, 0],               # Sağ üst
            [width * 0.25, height],          # Sol alt
            [width * 0.75, height]           # Sağ alt
        ]
    
    # Şerit tespit parametreleri
    lane_params = {
        "canny_low_threshold": 50,
        "canny_high_threshold": 150,
        "blur_kernel_size": 5,
        "hough_threshold": 15,
        "min_line_length": 15,
        "max_line_gap": 100
    }
    
    # Renk filtresi parametreleri
    color_params = {
        "white_lower": [0, 0, 210],
        "white_upper": [180, 30, 255],
        "yellow_lower": [15, 80, 120],
        "yellow_upper": [35, 255, 255]
    }
    
    # Birleşik kalibrasyon dosyası
    calibration = {
        "src_points": src_points,
        "dst_points": dst_points,
        "resolution": {"width": width, "height": height},
        **lane_params,
        **color_params
    }
    
    # Dosyaya kaydet
    with open(output_file, 'w') as f:
        json.dump(calibration, f, indent=4)
    
    logger.info(f"Kalibrasyon dosyası oluşturuldu: {output_file}")
    
    return calibration

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(description='Basitleştirilmiş Kalibrasyon Dosyası Oluşturucusu')
    parser.add_argument('--resolution', default='320x240', help='Kamera çözünürlüğü (GENxYÜK)')
    parser.add_argument('--output', default='calibration.json', help='Çıktı dosyası adı')
    
    # Manuel koordinat girişi için
    parser.add_argument('--src-points', nargs='+', type=str, help='Kaynak perspektif noktaları (x1,y1 x2,y2 x3,y3 x4,y4)')
    parser.add_argument('--dst-points', nargs='+', type=str, help='Hedef perspektif noktaları (x1,y1 x2,y2 x3,y3 x4,y4)')
    
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
    except ValueError:
        logger.error(f"Geçersiz çözünürlük formatı: {args.resolution}, varsayılan kullanılacak.")
        width, height = 320, 240
    
    # Perspektif noktalarını işle
    src_points = None
    dst_points = None
    
    if args.src_points:
        try:
            src_points = []
            for point_str in args.src_points:
                x, y = map(float, point_str.split(','))
                src_points.append([x, y])
            
            if len(src_points) != 4:
                logger.warning("Tam olarak 4 kaynak noktası gerekli! Varsayılan değerler kullanılacak.")
                src_points = None
        except Exception as e:
            logger.error(f"Kaynak noktaları işlenirken hata: {e}")
            src_points = None
    
    if args.dst_points:
        try:
            dst_points = []
            for point_str in args.dst_points:
                x, y = map(float, point_str.split(','))
                dst_points.append([x, y])
            
            if len(dst_points) != 4:
                logger.warning("Tam olarak 4 hedef noktası gerekli! Varsayılan değerler kullanılacak.")
                dst_points = None
        except Exception as e:
            logger.error(f"Hedef noktaları işlenirken hata: {e}")
            dst_points = None
    
    # Kalibrasyon dosyası oluştur
    create_calibration_file(
        width=width,
        height=height,
        src_points=src_points,
        dst_points=dst_points,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
    
    print("\nKalibrasyon Rehberi:")
    print("---------------------")
    print("1. Aracınızı şeritli yolun üzerine yerleştirin")
    print("2. Test görüntüsü almak için:")
    print("   python3 goruntu_yakala.py --output test_goruntu.jpg")
    print("3. Test görüntüsünü bir fotoğraf düzenleyici ile açın")
    print("4. Şeridin görünür olduğu perspektif noktalarını belirleyin:")
    print("   - İki üst nokta: Şeritlerin görünür başlangıç noktaları")
    print("   - İki alt nokta: Görüntünün alt kenarındaki noktalar")
    print("5. Bu noktaların x,y koordinatlarını belirleyip kalibrasyon dosyasına ekleyin:")
    print("   python3 kalibrasyon_olustur.py --src-points 112,156 208,156 0,240 320,240")
    print("6. Oluşturulan kalibrasyon.json dosyasını kontrol edin")
    print("7. Şerit tespitini test etmek için:")
    print("   python3 main.py --debug") 