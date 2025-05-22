#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Yön Testi
Bu program motorların yönlerini test etmek için kullanılır.
Her motor sırayla 5 saniye ileri, 5 saniye geri çalıştırılır.
"""

import time
import logging
from motor_control import MotorController

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MotorTest")

def test_motor(motor_controller, motor_name, test_duration=5.0, speed=0.4):
    """
    Belirtilen motoru test eder
    
    Args:
        motor_controller: Motor kontrol nesnesi
        motor_name (str): Motor adı ('left' veya 'right')
        test_duration (float): Test süresi (saniye)
        speed (float): Test hızı (0-1 arası)
    """
    # Motoru ileri yönde çalıştır
    logger.info(f"{motor_name.upper()} motor ileri yönde test ediliyor...")
    print(f"\n{motor_name.upper()} MOTOR İLERİ >>>>>>")
    
    if motor_name == 'left':
        motor_controller._set_motor_speed('left', 'forward', speed)
        motor_controller._set_motor_speed('right', 'stop', 0)
    else:
        motor_controller._set_motor_speed('right', 'forward', speed)
        motor_controller._set_motor_speed('left', 'stop', 0)
    
    time.sleep(test_duration)
    
    # Motoru durdur
    motor_controller.stop()
    time.sleep(1)  # Kısa bir duraklama
    
    # Motoru geri yönde çalıştır
    logger.info(f"{motor_name.upper()} motor geri yönde test ediliyor...")
    print(f"{motor_name.upper()} MOTOR GERİ <<<<<<")
    
    if motor_name == 'left':
        motor_controller._set_motor_speed('left', 'backward', speed)
        motor_controller._set_motor_speed('right', 'stop', 0)
    else:
        motor_controller._set_motor_speed('right', 'backward', speed)
        motor_controller._set_motor_speed('left', 'stop', 0)
    
    time.sleep(test_duration)
    
    # Motoru durdur
    motor_controller.stop()
    time.sleep(1)  # Kısa bir duraklama

def main():
    """
    Ana test programı
    """
    try:
        # Motor kontrolcüsünü başlat
        # NOT: Pin numaralarını kendi bağlantılarınıza göre değiştirin
        motor_controller = MotorController(
            left_motor_pins=(16, 18),    # Sol motor için (IN1, IN2)
            right_motor_pins=(36, 38),   # Sağ motor için (IN1, IN2)
            left_pwm_pin=12,             # Sol motor PWM
            right_pwm_pin=32,            # Sağ motor PWM
            max_speed=0.6,               # Maksimum hızı sınırla
            default_speed=0.4            # Test için varsayılan hız
        )
        
        print("\nMOTOR YÖN TESTİ BAŞLIYOR")
        print("------------------------")
        print("Her motor 5 saniye ileri, 5 saniye geri çalışacak.")
        print("Motorların dönüş yönlerini kontrol edin.")
        print("Test sırasında herhangi bir tuşa basarak testi iptal edebilirsiniz.")
        input("\nBaşlamak için ENTER tuşuna basın...")
        
        # Sol motoru test et
        test_motor(motor_controller, 'left')
        
        print("\nSol motor testi tamamlandı.")
        input("Sağ motor testine başlamak için ENTER tuşuna basın...")
        
        # Sağ motoru test et
        test_motor(motor_controller, 'right')
        
        print("\nTest tamamlandı!")
        print("\nSONUÇLARI KONTROL EDİN:")
        print("1. Her iki motor da ileri komutunda ileri gitti mi?")
        print("2. Her iki motor da geri komutunda geri gitti mi?")
        print("\nEğer motorlardan biri ters çalışıyorsa:")
        print("- O motorun IN1 ve IN2 kablolarının yerini değiştirin")
        print("- Veya motor_control.py dosyasında ilgili motorun yön pinlerini yazılımsal olarak değiştirin")
        
    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"Test hatası: {e}")
    finally:
        if 'motor_controller' in locals():
            motor_controller.cleanup()
            logger.info("Motor kontrol kaynakları temizlendi.")

if __name__ == "__main__":
    main() 