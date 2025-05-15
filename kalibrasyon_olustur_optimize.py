#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Basitleştirilmiş Kalibrasyon Dosyası Oluşturucusu
Bu program, görsel arayüz kullanmadan komut satırı üzerinden kalibrasyon dosyası oluşturur.
kalibrasyon.py donma sorunlarına alternatif olarak kullanılabilir.
"""

import json
import numpy as np
import argparse
import os
import logging
import sys

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
        
    Returns:
        bool: Başarı durumu
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
    
    # Şerit tespit parametreleri - Optimize edilmiş değerler
    lane_params = {
        "canny_low_threshold": 50,
        "canny_high_threshold": 150,
        "blur_kernel_size": 5,
        "hough_threshold": 15,
        "min_line_length": 15,
        "max_line_gap": 100
    }
    
    # Renk filtresi parametreleri - Optimize edilmiş değerler
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
    
    try:
        # Dosyaya kaydet
        with open(output_file, 'w') as f:
            json.dump(calibration, f, indent=4)
        
        logger.info(f"Kalibrasyon dosyası başarıyla oluşturuldu: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Kalibrasyon dosyası oluşturma hatası: {e}")
        return False

def parse_arguments():
    """
    Komut satırı argümanlarını işler
    """
    parser = argparse.ArgumentParser(
        description='Basitleştirilmiş Kalibrasyon Dosyası Oluşturucusu',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--resolution', default='320x240', 
                      help='Kamera çözünürlüğü (GENxYÜK) örnek: 320x240')
    
    parser.add_argument('--output', default='calibration.json', 
                      help='Çıktı dosyası adı (varsayılan: calibration.json)')
    
    # Manuel koordinat girişi için -  kullanıcı dostu açıklama
    parser.add_argument('--src-points', nargs='+', type=str,
                      help='Kaynak perspektif noktaları. Dört nokta gereklidir. Örnek:\n'
                           '--src-points "112,156" "208,156" "0,240" "320,240"\n'
                           'Sırayla: Sol Üst, Sağ Üst, Sol Alt, Sağ Alt')
                           
    parser.add_argument('--dst-points', nargs='+', type=str,
                      help='Hedef perspektif noktaları. Dört nokta gereklidir. Örnek:\n'
                           '--dst-points "80,0" "240,0" "80,240" "240,240"\n'
                           'Sırayla: Sol Üst, Sağ Üst, Sol Alt, Sağ Alt')
    
    parser.add_argument('--quick-mode', action='store_true',
                      help='Hızlı mod - varsayılan değerlerle hemen oluştur')
    
    # Interaktif kalibrasyon seçeneği
    parser.add_argument('--interactive', action='store_true',
                      help='İnteraktif mod - adım adım kalibrasyon noktaları oluştur')
    
    return parser.parse_args()

def interactive_calibration(resolution):
    """
    İnteraktif modda kalibrasyon noktalarını kullanıcıdan alır
    
    Args:
        resolution (tuple): Kamera çözünürlüğü (genişlik, yükseklik)
        
    Returns:
        tuple: (kaynak noktalar, hedef noktalar) veya (None, None)
    """
    width, height = resolution
    
    print("\nİNTERAKTİF KALİBRASYON MODU")
    print("---------------------------")
    print(f"Görüntü çözünürlüğü: {width}x{height}")
    print("Şerit kalibrasyonu için 4 nokta belirlemeniz gerekiyor.")
    print("Sırayla şu noktaları gireceksiniz:")
    print("1. Sol Üst Nokta (şeridin sol üst köşesi)")
    print("2. Sağ Üst Nokta (şeridin sağ üst köşesi)")
    print("3. Sol Alt Nokta (şeridin sol alt köşesi, genellikle [0, height])")
    print("4. Sağ Alt Nokta (şeridin sağ alt köşesi, genellikle [width, height])")
    print("\nHer nokta için x,y koordinatlarını girin (örnek: 100,150)")
    print("Varsayılan değer kullanmak için boş bırakın")
    
    try:
        # Varsayılan değerler
        default_src_points = [
            [width * 0.35, height * 0.65],  # Sol üst
            [width * 0.65, height * 0.65],  # Sağ üst
            [0, height],                     # Sol alt
            [width, height]                  # Sağ alt
        ]
        
        default_dst_points = [
            [width * 0.25, 0],               # Sol üst
            [width * 0.75, 0],               # Sağ üst
            [width * 0.25, height],          # Sol alt
            [width * 0.75, height]           # Sağ alt
        ]
        
        point_names = ["Sol Üst", "Sağ Üst", "Sol Alt", "Sağ Alt"]
        src_points = []
        
        # Kaynak noktaları kullanıcıdan al
        print("\nKAYNAK NOKTALARI (gerçek görüntüdeki konumlar)")
        print("---------------------------------------------")
        
        for i, name in enumerate(point_names):
            default_x, default_y = int(default_src_points[i][0]), int(default_src_points[i][1])
            
            while True:
                user_input = input(f"{name} Nokta [x,y] (varsayılan: {default_x},{default_y}): ").strip()
                
                # Boş input için varsayılan değer
                if not user_input:
                    src_points.append([default_x, default_y])
                    break
                
                # Kullanıcı girişini işle
                try:
                    x, y = map(int, user_input.split(','))
                    if 0 <= x <= width and 0 <= y <= height:
                        src_points.append([x, y])
                        break
                    else:
                        print(f"Hatalı koordinat! x:[0-{width}], y:[0-{height}] aralığında olmalı.")
                except:
                    print("Hata! Koordinatları 'x,y' formatında girin. Örnek: 100,150")
        
        # Hedef noktalar için de kullanıcıya seçenek sun
        print("\nHEDEF NOKTALARI (kuş bakışı görünümdeki konumlar)")
        print("-----------------------------------------------")
        print("Hedef noktalarını değiştirmek istiyor musunuz? (varsayılan önerilir)")
        change_dst = input("Değiştir? [e/H]: ").strip().lower()
        
        if change_dst == 'e':
            dst_points = []
            for i, name in enumerate(point_names):
                default_x, default_y = int(default_dst_points[i][0]), int(default_dst_points[i][1])
                
                while True:
                    user_input = input(f"{name} Nokta [x,y] (varsayılan: {default_x},{default_y}): ").strip()
                    
                    # Boş input için varsayılan değer
                    if not user_input:
                        dst_points.append([default_x, default_y])
                        break
                    
                    # Kullanıcı girişini işle
                    try:
                        x, y = map(int, user_input.split(','))
                        if 0 <= x <= width and 0 <= y <= height:
                            dst_points.append([x, y])
                            break
                        else:
                            print(f"Hatalı koordinat! x:[0-{width}], y:[0-{height}] aralığında olmalı.")
                    except:
                        print("Hata! Koordinatları 'x,y' formatında girin. Örnek: 100,150")
        else:
            dst_points = default_dst_points
        
        # Kullanıcıya özeti göster
        print("\nKALİBRASYON ÖZETİ:")
        print("------------------")
        print("Kaynak Noktaları:")
        for i, (x, y) in enumerate(src_points):
            print(f"  {point_names[i]}: ({x}, {y})")
        
        print("\nHedef Noktaları:")
        for i, (x, y) in enumerate(dst_points):
            print(f"  {point_names[i]}: ({x}, {y})")
            
        # Onay
        confirm = input("\nBu değerlerle kalibrasyon dosyası oluşturulsun mu? [E/h]: ").strip().lower()
        
        if confirm in ['', 'e', 'evet']:
            return src_points, dst_points
        else:
            print("Kalibrasyon iptal edildi.")
            return None, None
            
    except KeyboardInterrupt:
        print("\nİnteraktif kalibrasyon iptal edildi.")
        return None, None

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
        width, height = 320, 240
        resolution = (width, height)
    
    # İnteraktif mod
    if args.interactive:
        src_points, dst_points = interactive_calibration(resolution)
        if src_points is None:
            return
    # Komut satırı parametreleri ile çalıştır
    else:
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
    if create_calibration_file(
        width=width,
        height=height,
        src_points=src_points,
        dst_points=dst_points,
        output_file=args.output
    ):
        print("\nKalibrasyon dosyası başarıyla oluşturuldu.")
        print(f"Dosya: {os.path.abspath(args.output)}")
        print("\nŞerit tespitini test etmek için:")
        print(f"  python3 main.py --debug --resolution {width}x{height}")
        print("veya")
        print(f"  python3 video_test.py --video <video_dosyasi> --resolution {width}x{height}")
    else:
        print("\nKalibrasyon dosyası oluşturulamadı!")
        sys.exit(1)

if __name__ == "__main__":
    # Program başlangıç bilgisi
    print("\nOTONOM ARAÇ KALİBRASYON ARACI")
    print("----------------------------")
    print("Bu araç, kalibrasyon.py çalıştırmadan manuel olarak kalibrasyon dosyası oluşturur.\n")
    
    main()
    
    print("\nKalibrasyon Rehberi:")
    print("---------------------")
    print("1. Kalibrasyon dosyasını oluşturduktan sonra şerit tespit sistemini test edin")
    print("2. Sonuçlar tatmin edici değilse şunları deneyin:")
    print("   - Farklı perspektif noktaları belirleyin")
    print("   - İnteraktif mod ile adım adım kalibrasyon yapın:")
    print("     python3 kalibrasyon_olustur_optimize.py --interactive")
    print("   - Optimize edilmiş kalibrasyon aracı ile noktaları görsel olarak ayarlayın:")
    print("     python3 kalibrasyon_optimize.py")
    print("3. Sorun devam ederse, şu komutla varsayılan değerleri kullanın:")
    print("   python3 kalibrasyon_olustur_optimize.py --quick-mode") 