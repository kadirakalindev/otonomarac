#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Kalibrasyon Yönetici Aracı
Bu araç, video ve gerçek pist için ayrı kalibrasyon dosyalarını yönetmeyi sağlar.
"""

import os
import argparse
import json
import shutil
import logging
import time

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KalibrasyonYonetici")

class KalibrasyonYonetici:
    """
    Kalibrasyon dosyalarını yöneten sınıf
    """
    def __init__(self, config_file="kalibrasyon_profilleri.json"):
        """
        KalibrasyonYonetici sınıfını başlatır.
        
        Args:
            config_file (str): Kalibrasyon profilleri dosyasının yolu
        """
        self.config_file = config_file
        self.profiles = {}
        self.active_profile = None
        
        # Varsayılan profiller
        self.default_profiles = {
            "real": {
                "description": "Gerçek pist kalibrasyonu",
                "file": "serit_kalibrasyon_real.json",
                "created": None
            },
            "video": {
                "description": "Video simülasyonu kalibrasyonu",
                "file": "serit_kalibrasyon_video.json",
                "created": None
            }
        }
        
        # Yapılandırma dosyasını yükle veya oluştur
        self.load_or_create_config()
    
    def load_or_create_config(self):
        """
        Yapılandırma dosyasını yükler veya varsa oluşturur.
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.profiles = json.load(f)
                    logger.info(f"Kalibrasyon profilleri yüklendi: {self.config_file}")
                
                # Aktif profili kontrol et
                if "active_profile" in self.profiles:
                    self.active_profile = self.profiles["active_profile"]
                    logger.info(f"Aktif profil: {self.active_profile}")
                else:
                    # Aktif profil yoksa, "real" profili varsayılan olarak ayarla
                    self.active_profile = "real"
                    self.profiles["active_profile"] = self.active_profile
                    self.save_config()
                    logger.info(f"Aktif profil ayarlandı: {self.active_profile}")
            else:
                # Varsayılan profilleri oluştur
                self.profiles = self.default_profiles.copy()
                self.active_profile = "real"
                self.profiles["active_profile"] = self.active_profile
                self.save_config()
                logger.info(f"Yeni kalibrasyon profilleri oluşturuldu: {self.config_file}")
                
        except Exception as e:
            logger.error(f"Yapılandırma dosyası yüklenirken hata: {e}")
            # Hata durumunda varsayılan profilleri kullan
            self.profiles = self.default_profiles.copy()
            self.active_profile = "real"
            self.profiles["active_profile"] = self.active_profile
    
    def save_config(self):
        """
        Yapılandırma dosyasını kaydeder.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
            logger.info(f"Kalibrasyon profilleri kaydedildi: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Yapılandırma dosyası kaydedilirken hata: {e}")
            return False
    
    def get_active_profile(self):
        """
        Aktif profili döndürür.
        
        Returns:
            dict: Aktif profil verileri
        """
        if self.active_profile in self.profiles:
            return {
                "name": self.active_profile,
                **self.profiles[self.active_profile]
            }
        return None
    
    def get_active_calibration_file(self):
        """
        Aktif kalibrasyon dosyasının yolunu döndürür.
        
        Returns:
            str: Kalibrasyon dosyasının yolu
        """
        active_profile = self.get_active_profile()
        if active_profile:
            return active_profile["file"]
        return "serit_kalibrasyon.json"  # Varsayılan dosya
    
    def set_active_profile(self, profile_name):
        """
        Aktif profili ayarlar.
        
        Args:
            profile_name (str): Profil adı ("real" veya "video")
            
        Returns:
            bool: Başarı durumu
        """
        if profile_name in self.profiles:
            self.active_profile = profile_name
            self.profiles["active_profile"] = profile_name
            self.save_config()
            
            # Aktif kalibrasyon dosyasını serit_kalibrasyon.json'a kopyala
            target_file = "serit_kalibrasyon.json"
            source_file = self.profiles[profile_name]["file"]
            
            if os.path.exists(source_file):
                try:
                    shutil.copy2(source_file, target_file)
                    logger.info(f"Kalibrasyon dosyası kopyalandı: {source_file} -> {target_file}")
                except Exception as e:
                    logger.error(f"Kalibrasyon dosyası kopyalanırken hata: {e}")
                    return False
            else:
                logger.warning(f"Kalibrasyon dosyası bulunamadı: {source_file}")
                # Profil dosyası yoksa, varsayılan bir kalibrasyon dosyası oluştur
                self.create_default_calibration(source_file)
            
            return True
        else:
            logger.error(f"Profil bulunamadı: {profile_name}")
            return False
    
    def create_default_calibration(self, file_path, width=640, height=480):
        """
        Varsayılan bir kalibrasyon dosyası oluşturur.
        
        Args:
            file_path (str): Oluşturulacak dosyanın yolu
            width (int): Görüntü genişliği
            height (int): Görüntü yüksekliği
            
        Returns:
            bool: Başarı durumu
        """
        try:
            # Varsayılan değerler - ortalama bir yol konfigürasyonu
            src_points = [
                [width * 0.35, height * 0.65],  # Sol üst
                [width * 0.65, height * 0.65],  # Sağ üst
                [0, height],                     # Sol alt
                [width, height]                  # Sağ alt
            ]
            
            dst_points = [
                [width * 0.25, 0],               # Sol üst
                [width * 0.75, 0],               # Sağ üst
                [width * 0.25, height],          # Sol alt
                [width * 0.75, height]           # Sağ alt
            ]
            
            calibration_data = {
                "src_points": src_points,
                "dst_points": dst_points,
                "resolution": {
                    "width": width,
                    "height": height
                },
                "canny_low_threshold": 50,
                "canny_high_threshold": 150,
                "blur_kernel_size": 5,
                "hough_threshold": 15,
                "min_line_length": 15,
                "max_line_gap": 30
            }
            
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            logger.info(f"Varsayılan kalibrasyon dosyası oluşturuldu: {file_path}")
            
            # Aynı içeriği serit_kalibrasyon.json'a da yaz
            target_file = "serit_kalibrasyon.json"
            with open(target_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            logger.info(f"Ana kalibrasyon dosyası güncellendi: {target_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Varsayılan kalibrasyon dosyası oluşturulurken hata: {e}")
            return False
    
    def copy_calibration(self, source, target):
        """
        Bir kalibrasyon dosyasını diğerine kopyalar.
        
        Args:
            source (str): Kaynak profil adı ("real" veya "video")
            target (str): Hedef profil adı ("real" veya "video")
            
        Returns:
            bool: Başarı durumu
        """
        if source not in self.profiles or target not in self.profiles:
            logger.error(f"Geçersiz profil adı: {source} veya {target}")
            return False
        
        source_file = self.profiles[source]["file"]
        target_file = self.profiles[target]["file"]
        
        if not os.path.exists(source_file):
            logger.error(f"Kaynak dosya bulunamadı: {source_file}")
            return False
        
        try:
            shutil.copy2(source_file, target_file)
            self.profiles[target]["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_config()
            
            logger.info(f"Kalibrasyon kopyalandı: {source} -> {target}")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon kopyalanırken hata: {e}")
            return False
    
    def save_current_calibration(self, profile_name=None):
        """
        Mevcut kalibrasyon dosyasını belirtilen profile kaydeder.
        
        Args:
            profile_name (str): Profil adı (None ise aktif profil kullanılır)
            
        Returns:
            bool: Başarı durumu
        """
        if profile_name is None:
            profile_name = self.active_profile
        
        if profile_name not in self.profiles:
            logger.error(f"Geçersiz profil adı: {profile_name}")
            return False
        
        source_file = "serit_kalibrasyon.json"
        target_file = self.profiles[profile_name]["file"]
        
        if not os.path.exists(source_file):
            logger.error(f"Kaynak dosya bulunamadı: {source_file}")
            return False
        
        try:
            shutil.copy2(source_file, target_file)
            self.profiles[profile_name]["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_config()
            
            logger.info(f"Kalibrasyon kaydedildi: {profile_name} ({target_file})")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon kaydedilirken hata: {e}")
            return False
    
    def list_profiles(self):
        """
        Tüm profilleri listeler.
        
        Returns:
            list: Profil listesi
        """
        result = []
        for name, data in self.profiles.items():
            if name != "active_profile":
                result.append({
                    "name": name,
                    **data,
                    "is_active": name == self.active_profile
                })
        return result


def parse_arguments():
    """
    Komut satırı argümanlarını işler.
    """
    parser = argparse.ArgumentParser(description='Otonom Araç Kalibrasyon Yöneticisi')
    
    # Ana komutlar
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--list', action='store_true', help='Mevcut kalibrasyon profillerini listele')
    commands.add_argument('--use', choices=['real', 'video'], help='Belirtilen kalibrasyon profilini kullan')
    commands.add_argument('--save', choices=['real', 'video'], help='Mevcut kalibrasyonu belirtilen profile kaydet')
    commands.add_argument('--copy', nargs=2, metavar=('SOURCE', 'TARGET'), help='Bir kalibrasyon profilini diğerine kopyala')
    commands.add_argument('--show-active', action='store_true', help='Aktif kalibrasyon profilini göster')
    commands.add_argument('--reset', action='store_true', help='Tüm kalibrasyon profillerini sıfırla')
    
    return parser.parse_args()

def main():
    """
    Ana program
    """
    args = parse_arguments()
    
    try:
        manager = KalibrasyonYonetici()
        
        if args.list:
            profiles = manager.list_profiles()
            print("\nKALİBRASYON PROFİLLERİ:")
            print("------------------------")
            for profile in profiles:
                active_mark = "* " if profile["is_active"] else "  "
                created = profile.get("created", "Oluşturulmadı")
                print(f"{active_mark}{profile['name']}: {profile['description']}")
                print(f"   Dosya: {profile['file']}")
                print(f"   Oluşturulma: {created}")
                print()
        
        elif args.use:
            if manager.set_active_profile(args.use):
                print(f"\nAktif profil '{args.use}' olarak ayarlandı.")
                print(f"Ana kalibrasyon dosyası '{manager.get_active_calibration_file()}' ile güncellendi.")
            else:
                print(f"\nHata: '{args.use}' profili ayarlanamadı.")
        
        elif args.save:
            if manager.save_current_calibration(args.save):
                print(f"\nMevcut kalibrasyon '{args.save}' profiline kaydedildi.")
            else:
                print(f"\nHata: Mevcut kalibrasyon '{args.save}' profiline kaydedilemedi.")
        
        elif args.copy:
            source, target = args.copy
            if manager.copy_calibration(source, target):
                print(f"\n'{source}' profili '{target}' profiline kopyalandı.")
            else:
                print(f"\nHata: '{source}' profili '{target}' profiline kopyalanamadı.")
        
        elif args.show_active:
            active_profile = manager.get_active_profile()
            if active_profile:
                print("\nAKTİF KALİBRASYON PROFİLİ:")
                print("-------------------------")
                print(f"Ad: {active_profile['name']}")
                print(f"Açıklama: {active_profile['description']}")
                print(f"Dosya: {active_profile['file']}")
                created = active_profile.get("created", "Oluşturulmadı")
                print(f"Oluşturulma: {created}")
                print(f"\nBu profil kullanılarak ana kalibrasyon dosyası (serit_kalibrasyon.json) güncellendi.")
            else:
                print("\nHata: Aktif profil bulunamadı.")
        
        elif args.reset:
            confirm = input("\nTüm kalibrasyon profilleri sıfırlanacak. Emin misiniz? (e/H): ")
            if confirm.lower() == 'e':
                # Yapılandırma dosyasını sil
                if os.path.exists(manager.config_file):
                    os.remove(manager.config_file)
                
                # Yeniden oluştur
                manager = KalibrasyonYonetici()
                print("\nTüm kalibrasyon profilleri sıfırlandı.")
                
                # Varsayılan kalibrasyon dosyaları oluştur
                for profile_name in ["real", "video"]:
                    file_path = manager.profiles[profile_name]["file"]
                    manager.create_default_calibration(file_path)
                
                print("Varsayılan kalibrasyon dosyaları oluşturuldu.")
            else:
                print("\nSıfırlama iptal edildi.")
    
    except Exception as e:
        logger.error(f"Hata: {e}")
        print(f"\nBir hata oluştu: {e}")

if __name__ == "__main__":
    print("\nOTONOM ARAÇ KALİBRASYON YÖNETİCİSİ")
    print("----------------------------------")
    
    main() 