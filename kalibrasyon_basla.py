#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kalibrasyon Başlatma Aracı
Bu program, kalibrasyon araçlarını ve test programlarını başlatmak için basit bir arayüz sunar.
"""

import os
import sys
import time
import subprocess
import argparse

def clear_screen():
    """Ekranı temizler"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Program başlık bilgisini yazdırır"""
    clear_screen()
    print("\n" + "=" * 60)
    print("  OTONOM ARAÇ KALİBRASYON VE TEST ARACI")
    print("=" * 60)

def print_menu():
    """Ana menüyü yazdırır"""
    print("\nLütfen bir seçenek seçin:")
    print("1. İnteraktif Şerit Kalibrasyonu")
    print("2. Kamera Testi (Kalibrasyon Kontrolü)")
    print("3. Motor Testi")
    print("4. Tam Sistem Testi (Şerit Takibi)")
    print("5. Yardım ve Bilgi")
    print("0. Çıkış")
    print("\nSeçiminiz: ", end="")

def run_command(command):
    """Komutu çalıştırır ve çıkışı gösterir"""
    print(f"\nKomut çalıştırılıyor: {command}")
    print("-" * 60)
    
    try:
        process = subprocess.Popen(command, shell=True)
        process.wait()
        
        if process.returncode == 0:
            print("\nProgram başarıyla tamamlandı.")
        else:
            print(f"\nProgram hata kodu ile sonlandı: {process.returncode}")
    
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHata: {e}")
    
    input("\nDevam etmek için ENTER tuşuna basın...")

def interactive_calibration():
    """İnteraktif şerit kalibrasyonunu başlatır"""
    print_header()
    print("\nİNTERAKTİF ŞERİT KALİBRASYONU")
    print("-" * 60)
    print("Bu araç, şerit takibi için gereken kalibrasyon noktalarını belirlemenizi sağlar.")
    print("5 adet noktayı şeritler üzerinde konumlandırmanız gerekiyor.")
    print("\nÇözünürlük seçin:")
    print("1. Düşük (320x240) - Daha hızlı")
    print("2. Orta (640x480) - Önerilen")
    print("3. Yüksek (800x600) - Daha detaylı")
    
    while True:
        choice = input("\nSeçiminiz (1-3): ")
        if choice == "1":
            resolution = "320x240"
            break
        elif choice == "2":
            resolution = "640x480"
            break
        elif choice == "3":
            resolution = "800x600"
            break
        else:
            print("Geçersiz seçenek! Lütfen 1-3 arasında bir değer girin.")
    
    output_file = "serit_kalibrasyon.json"
    command = f"python interaktif_kalibrasyon.py --resolution {resolution} --output {output_file}"
    run_command(command)

def camera_test():
    """Kamera testi ve kalibrasyon kontrolü"""
    print_header()
    print("\nKAMERA TESTİ VE KALİBRASYON KONTROLÜ")
    print("-" * 60)
    print("Bu araç, kamera görüntüsünü ve kalibrasyon ayarlarını test etmenizi sağlar.")
    
    # Kalibrasyon dosyasını kontrol et
    calibration_file = "serit_kalibrasyon.json"
    if not os.path.exists(calibration_file):
        print(f"\nUYARI: Kalibrasyon dosyası ({calibration_file}) bulunamadı!")
        print("Önce kalibrasyon yapmanız önerilir.")
        
        if input("\nYine de devam etmek istiyor musunuz? (e/H): ").lower() != 'e':
            return
    
    print("\nÇözünürlük seçin:")
    print("1. Düşük (320x240) - Daha hızlı")
    print("2. Orta (640x480) - Önerilen")
    
    while True:
        choice = input("\nSeçiminiz (1-2): ")
        if choice == "1":
            resolution = "320x240"
            break
        elif choice == "2":
            resolution = "640x480"
            break
        else:
            print("Geçersiz seçenek! Lütfen 1-2 arasında bir değer girin.")
    
    command = f"python kamera_test.py --resolution {resolution} --calibration {calibration_file} --debug"
    run_command(command)

def motor_test():
    """Motor testi"""
    print_header()
    print("\nMOTOR TESTİ")
    print("-" * 60)
    print("Bu araç, motorların doğru çalışıp çalışmadığını test etmenizi sağlar.")
    print("Her motor sırayla ileri ve geri yönde çalıştırılacaktır.")
    
    if input("\nDevam etmek istiyor musunuz? (e/H): ").lower() == 'e':
        command = "python motor_test.py"
        run_command(command)

def full_system_test():
    """Tam sistem testi (şerit takibi)"""
    print_header()
    print("\nTAM SİSTEM TESTİ (ŞERİT TAKİBİ)")
    print("-" * 60)
    print("Bu araç, şerit takibi ve motor kontrolünü test etmenizi sağlar.")
    print("Aracı şeritli bir yüzeye yerleştirin ve çalıştırın.")
    
    # Kalibrasyon dosyasını kontrol et
    calibration_file = "serit_kalibrasyon.json"
    if not os.path.exists(calibration_file):
        print(f"\nUYARI: Kalibrasyon dosyası ({calibration_file}) bulunamadı!")
        print("Önce kalibrasyon yapmanız önerilir.")
        
        if input("\nYine de devam etmek istiyor musunuz? (e/H): ").lower() != 'e':
            return
    
    print("\nDebug modu seçin:")
    print("1. Normal mod")
    print("2. Debug modu (görsel geri bildirim)")
    
    debug_option = ""
    while True:
        choice = input("\nSeçiminiz (1-2): ")
        if choice == "1":
            debug_option = ""
            break
        elif choice == "2":
            debug_option = "--debug"
            break
        else:
            print("Geçersiz seçenek! Lütfen 1-2 arasında bir değer girin.")
    
    command = f"python main.py {debug_option} --calibration {calibration_file}"
    run_command(command)

def show_help():
    """Yardım ve bilgi gösterir"""
    print_header()
    print("\nYARDIM VE BİLGİ")
    print("-" * 60)
    print("\nOtonom Araç Kalibrasyon ve Test Aracı Kullanımı:")
    
    print("\n1. İnteraktif Şerit Kalibrasyonu:")
    print("   - Kamera görüntüsü üzerinde 5 nokta seçerek şerit takibi için kalibrasyon yapın.")
    print("   - Noktaları şu sırayla yerleştirin:")
    print("     1. Sol şeridin alt noktası")
    print("     2. Sol şeridin üst noktası")
    print("     3. Orta şeridin üst noktası (takip edilecek merkez)")
    print("     4. Sağ şeridin üst noktası")
    print("     5. Sağ şeridin alt noktası")
    
    print("\n2. Kamera Testi:")
    print("   - Kalibrasyonun doğru çalışıp çalışmadığını kontrol edin.")
    print("   - Şerit tespitinin doğru yapılıp yapılmadığını görsel olarak inceleyin.")
    
    print("\n3. Motor Testi:")
    print("   - Motorların doğru yönde çalışıp çalışmadığını test edin.")
    print("   - Her motor sırayla ileri ve geri yönde çalıştırılır.")
    
    print("\n4. Tam Sistem Testi:")
    print("   - Şerit takibi ve motor kontrolünü birlikte test edin.")
    print("   - Aracı şeritli bir yüzeye yerleştirin ve çalıştırın.")
    print("   - Debug modunda görsel geri bildirim alabilirsiniz.")
    
    print("\nÖnerilen Kalibrasyon Adımları:")
    print("1. İnteraktif Şerit Kalibrasyonu yapın.")
    print("2. Kamera Testi ile kalibrasyonu kontrol edin.")
    print("3. Motor Testi ile motorların doğru çalıştığını doğrulayın.")
    print("4. Tam Sistem Testi ile şerit takibini test edin.")
    
    input("\nAna menüye dönmek için ENTER tuşuna basın...")

def main():
    """Ana program"""
    parser = argparse.ArgumentParser(description="Otonom Araç Kalibrasyon ve Test Aracı")
    parser.add_argument("--no-menu", action="store_true", help="Menüyü gösterme, doğrudan interaktif kalibrasyonu başlat")
    args = parser.parse_args()
    
    if args.no_menu:
        interactive_calibration()
        return
    
    while True:
        print_header()
        print_menu()
        
        choice = input()
        
        if choice == "1":
            interactive_calibration()
        elif choice == "2":
            camera_test()
        elif choice == "3":
            motor_test()
        elif choice == "4":
            full_system_test()
        elif choice == "5":
            show_help()
        elif choice == "0":
            print("\nProgramdan çıkılıyor...")
            sys.exit(0)
        else:
            print("\nGeçersiz seçenek! Lütfen tekrar deneyin.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramdan çıkılıyor...")
        sys.exit(0) 