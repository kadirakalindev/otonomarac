#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Kontrol Modülü
Bu modül, araç motorlarının kontrolünü gpiozero kütüphanesini kullanarak sağlar.
PID kontrol ile şerit takibi için gerekli motor hız ayarlarını yapar.
"""

import time
import logging
from gpiozero import OutputDevice, PWMOutputDevice
import numpy as np

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MotorControl")

# BOARD ve BCM pin numaralandırma dönüşüm tablosu
# Bu tabloda BOARD pin numaralarının BCM karşılıkları bulunuyor
BOARD_TO_BCM = {
    3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
    15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11,
    24: 8, 26: 7, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19, 36: 16,
    37: 26, 38: 20, 40: 21
}

class MotorController:
    """
    Motor kontrolünden sorumlu sınıf
    """
    def __init__(self, 
                 left_motor_pins=(16, 18),    # Sol motor için (IN1, IN2) pin numaraları
                 right_motor_pins=(36, 38),   # Sağ motor için (IN1, IN2) pin numaraları
                 left_pwm_pin=12,             # Sol motor için PWM enable pini
                 right_pwm_pin=32,            # Sağ motor için PWM enable pini
                 max_speed=1.0,               # Maksimum hız değeri (0-1 arası)
                 default_speed=0.5,           # Varsayılan hız
                 use_board_pins=True,         # BOARD pin numaralandırmasını kullan
                 pwm_frequency=100):          # PWM frekansı (Hz)
        """
        MotorController sınıfını başlatır.
        
        Args:
            left_motor_pins (tuple): Sol motor kontrol pinleri (IN1, IN2)
            right_motor_pins (tuple): Sağ motor kontrol pinleri (IN1, IN2)
            left_pwm_pin (int): Sol motor için PWM pini (Enable)
            right_pwm_pin (int): Sağ motor için PWM pini (Enable)
            max_speed (float): Maksimum hız seviyesi (0-1 arası)
            default_speed (float): Varsayılan hız seviyesi (0-1 arası)
            use_board_pins (bool): BOARD pin numaralandırmasını kullan (True) veya BCM kullan (False)
            pwm_frequency (int): PWM frekansı (Hz)
        """
        self.use_board_pins = use_board_pins
        self.max_speed = max_speed
        self.default_speed = default_speed
        
        # Pin numaralarını ayarla (BOARD -> BCM dönüşümü gerekirse)
        if use_board_pins:
            logger.info("BOARD pin numaralandırması kullanılıyor.")
            
            # BOARD pinlerini BCM pinlerine dönüştür
            try:
                left_in1_bcm = BOARD_TO_BCM.get(left_motor_pins[0])
                left_in2_bcm = BOARD_TO_BCM.get(left_motor_pins[1])
                right_in1_bcm = BOARD_TO_BCM.get(right_motor_pins[0])
                right_in2_bcm = BOARD_TO_BCM.get(right_motor_pins[1])
                left_en_bcm = BOARD_TO_BCM.get(left_pwm_pin)
                right_en_bcm = BOARD_TO_BCM.get(right_pwm_pin)
                
                # Eğer dönüşüm tablosunda bulunmayan bir pin varsa uyarı ver
                if None in [left_in1_bcm, left_in2_bcm, right_in1_bcm, right_in2_bcm, left_en_bcm, right_en_bcm]:
                    logger.warning("Bazı BOARD pinleri BCM'e dönüştürülemedi!")
                    logger.warning(f"Dönüştürülemeyen pinler: LEFT_IN1={left_motor_pins[0] if left_in1_bcm is None else ''}, "
                                 f"LEFT_IN2={left_motor_pins[1] if left_in2_bcm is None else ''}, "
                                 f"RIGHT_IN1={right_motor_pins[0] if right_in1_bcm is None else ''}, "
                                 f"RIGHT_IN2={right_motor_pins[1] if right_in2_bcm is None else ''}, "
                                 f"LEFT_EN={left_pwm_pin if left_en_bcm is None else ''}, "
                                 f"RIGHT_EN={right_pwm_pin if right_en_bcm is None else ''}")
            except Exception as e:
                logger.error(f"Pin dönüşüm hatası: {e}")
                raise
        else:
            logger.info("BCM pin numaralandırması kullanılıyor.")
            left_in1_bcm = left_motor_pins[0]
            left_in2_bcm = left_motor_pins[1]
            right_in1_bcm = right_motor_pins[0]
            right_in2_bcm = right_motor_pins[1]
            left_en_bcm = left_pwm_pin
            right_en_bcm = right_pwm_pin
        
        # Kullanılan pin numaralarını logla
        logger.info(f"Sol motor BCM pinleri: IN1={left_in1_bcm}, IN2={left_in2_bcm}, EN={left_en_bcm}")
        logger.info(f"Sağ motor BCM pinleri: IN1={right_in1_bcm}, IN2={right_in2_bcm}, EN={right_en_bcm}")
        
        # Motor kontrol ve PWM pinlerini tanımla
        try:
            # Motor kontrol pinleri
            self.left_motor_in1 = OutputDevice(left_in1_bcm, active_high=True, initial_value=False)
            self.left_motor_in2 = OutputDevice(left_in2_bcm, active_high=True, initial_value=False)
            self.right_motor_in1 = OutputDevice(right_in1_bcm, active_high=True, initial_value=False)
            self.right_motor_in2 = OutputDevice(right_in2_bcm, active_high=True, initial_value=False)
            
            # PWM pinleri
            self.left_pwm = PWMOutputDevice(left_en_bcm, frequency=pwm_frequency, initial_value=0)
            self.right_pwm = PWMOutputDevice(right_en_bcm, frequency=pwm_frequency, initial_value=0)
            
            logger.info("Motor pinleri başarıyla başlatıldı.")
            
        except Exception as e:
            logger.error(f"Motor kontrol başlatma hatası: {e}")
            self.cleanup()
            raise
        
        # Optimize edilmiş PID kontrol parametreleri
        self.kp = 0.7   # Orantısal katsayı - arttırıldı, daha hızlı tepki için
        self.ki = 0.05  # Integral katsayı - düşük değerde başlatıldı
        self.kd = 0.15  # Türev katsayı - arttırıldı, daha az salınım için
        
        # PID sınır değerleri
        self.max_integral = 100.0  # İntegral terimini sınırla (windup'u önlemek için)
        
        # Hız limitleri
        self.min_motor_speed = 0.2  # Minimum motor hızı (durma noktasını önler)
        
        # PID hesaplaması için gerekli değişkenler
        self.previous_error = 0
        self.integral = 0
        
        # Son zamanlama (dt için)
        self.last_time = time.time()
        
        # Şerit kaybı için sayaç
        self.lost_lane_counter = 0
        
        logger.info("Motor kontrol modülü başlatıldı.")
    
    def _set_motor_speed(self, motor_side, direction, speed):
        """
        Belirtilen motoru belirtilen yön ve hızda çalıştırır.
        
        Args:
            motor_side (str): 'left' veya 'right'
            direction (str): 'forward', 'backward' veya 'stop'
            speed (float): Hız değeri (0-1 arası)
        """
        # Hız değerini sınırlandır
        speed = min(max(0, speed), self.max_speed)
        
        if motor_side == 'left':
            motor_in1 = self.left_motor_in1
            motor_in2 = self.left_motor_in2
            motor_pwm = self.left_pwm
        else:  # 'right'
            motor_in1 = self.right_motor_in1
            motor_in2 = self.right_motor_in2
            motor_pwm = self.right_pwm
        
        # Yön kontrolü
        if direction == 'forward':
            motor_in1.on()
            motor_in2.off()
            motor_pwm.value = speed
        elif direction == 'backward':
            motor_in1.off()
            motor_in2.on()
            motor_pwm.value = speed
        else:  # 'stop'
            motor_in1.off()
            motor_in2.off()
            motor_pwm.value = 0
    
    def forward(self, speed=None):
        """
        Aracı ileri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed('left', 'forward', speed)
        self._set_motor_speed('right', 'forward', speed)
        
        logger.debug(f"İleri hareket: Hız={speed}")
    
    def backward(self, speed=None):
        """
        Aracı geri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed('left', 'backward', speed)
        self._set_motor_speed('right', 'backward', speed)
        
        logger.debug(f"Geri hareket: Hız={speed}")
    
    def turn_left(self, speed=None):
        """
        Aracı sola döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed('left', 'forward', speed * 0.2)  # Sol motor yavaş
        self._set_motor_speed('right', 'forward', speed)       # Sağ motor hızlı
        
        logger.debug(f"Sola dönüş: Hız={speed}")
    
    def turn_right(self, speed=None):
        """
        Aracı sağa döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed('left', 'forward', speed)        # Sol motor hızlı
        self._set_motor_speed('right', 'forward', speed * 0.2) # Sağ motor yavaş
        
        logger.debug(f"Sağa dönüş: Hız={speed}")
    
    def stop(self):
        """
        Aracı durdurur.
        """
        self._set_motor_speed('left', 'stop', 0)
        self._set_motor_speed('right', 'stop', 0)
        
        logger.debug("Araç durduruldu.")
    
    def pid_control(self, error):
        """
        PID kontrol algoritması ile motor hızlarını ayarlar.
        
        Args:
            error (int): Şeritten sapma miktarı (piksel cinsinden)
                         Pozitif değer sağa kayma, negatif değer sola kayma anlamına gelir
        
        Returns:
            tuple: Sol ve sağ motor hızları
        """
        # Zamanı hesapla
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Zaman farkı çok küçükse minimum değer kullan
        if dt < 0.01:
            dt = 0.01
            
        self.last_time = current_time
        
        # PID hesaplaması
        # İntegral biriktirme (anti-windup ile)
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        
        # Türev hesaplama (keskin değişimleri önlemek için filtrelenmiş)
        if dt > 0:
            derivative = (error - self.previous_error) / dt
            # Aşırı türev değerlerini filtrele
            derivative = np.clip(derivative, -500, 500)
        else:
            derivative = 0
        
        # PID çıkışını hesapla
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Önceki hatayı güncelle
        self.previous_error = error
        
        # Motor hızlarını ayarla
        base_speed = self.default_speed
        
        # Çıkışı sınırlandır
        output = max(min(output, base_speed), -base_speed)
        
        # Sol ve sağ motor hızlarını hesapla - geliştirilmiş algoritma
        if error > 0:  # Sağa kaymış, sola dönmeli
            # Hata büyükse, sol motoru daha fazla azalt, sağ motoru biraz artır
            left_speed = base_speed - (output * 1.2)  # Dönüş için daha fazla azalt
            right_speed = min(base_speed + (output * 0.3), self.max_speed)  # Az miktarda artır
        else:  # Sola kaymış, sağa dönmeli
            # Hata büyükse, sağ motoru daha fazla azalt, sol motoru biraz artır
            left_speed = min(base_speed - (output * 0.3), self.max_speed)  # Az miktarda artır
            right_speed = base_speed + (output * 1.2)  # Dönüş için daha fazla azalt
        
        # Hız değerlerini sınırla
        left_speed = min(max(self.min_motor_speed, left_speed), self.max_speed)
        right_speed = min(max(self.min_motor_speed, right_speed), self.max_speed)
        
        return left_speed, right_speed
    
    def follow_lane(self, center_diff):
        """
        Şerit takibi için gerekli motor komutlarını verir.
        
        Args:
            center_diff (int): Şeritten sapma miktarı (piksel cinsinden)
                             Pozitif değer sağa kayma, negatif değer sola kayma
        """
        if center_diff is None:
            # Şerit tespit edilemedi - kayıp şerit sayacını artır
            self.lost_lane_counter += 1
            
            # Kısa süreli kayıplarda son komutları koru
            if self.lost_lane_counter < 5:
                # Önceki hızlarla devam et
                return
            # Uzun süreli kayıplarda güvenli moda geç
            elif self.lost_lane_counter < 20:
                # Yavaşla ama durmadan devam et
                self.forward(self.default_speed * 0.7)
                logger.warning(f"Şerit kaybı: {self.lost_lane_counter} kare - yavaşlıyor")
                return
            else:
                # Çok uzun süre şerit yoksa dur
                logger.error("Şerit uzun süre bulunamadı - duruyor")
                self.stop()
                return
        else:
            # Şerit bulundu, sayacı sıfırla
            self.lost_lane_counter = 0
        
        # Şerit merkezden kayma miktarının mutlak değeri
        abs_diff = abs(center_diff)
        
        # Eşik değeri (bu değerden az sapmalar için düzeltme yapma)
        threshold = 5  # Daha hassas düzeltmeler için düşürüldü
        
        if abs_diff < threshold:
            # Sapma çok az ise düz git - tam hız
            self.forward(self.default_speed)
        else:
            try:
                # PID kontrolü ile motor hızlarını hesapla
                left_speed, right_speed = self.pid_control(center_diff)
                
                # Şerit sapması çok büyükse keskin dönüş yap
                if abs_diff > 150:  # Aşırı sapma durumu
                    if center_diff > 0:
                        # Çok sağa sapmış, keskin sola dön
                        self.turn_left(self.default_speed * 1.1)
                        logger.info(f"Keskin sol dönüş (sapma: {center_diff})")
                    else:
                        # Çok sola sapmış, keskin sağa dön
                        self.turn_right(self.default_speed * 1.1)
                        logger.info(f"Keskin sağ dönüş (sapma: {center_diff})")
                else:
                    # Normal sapma - PID ile hesaplanan hızları kullan
                    self._set_motor_speed('left', 'forward', left_speed)
                    self._set_motor_speed('right', 'forward', right_speed)
                    
                    if abs_diff > 50:
                        logger.debug(f"Şerit takibi (büyük sapma): Sapma={center_diff}, Sol={left_speed:.2f}, Sağ={right_speed:.2f}")
                    else:
                        logger.debug(f"Şerit takibi (küçük sapma): Sapma={center_diff}, Sol={left_speed:.2f}, Sağ={right_speed:.2f}")
                
            except Exception as e:
                logger.error(f"Şerit takibi hatası: {e}")
                self.stop()
    
    def cleanup(self):
        """
        Motorları durdurur ve kaynakları temizler.
        """
        logger.info("Motor kontrol modülü kapatılıyor...")
        try:
            # Motorları durdur
            self.stop()
            
            # Tüm pin nesnelerini kapat
            for device in [
                self.left_motor_in1, self.left_motor_in2, self.left_pwm,
                self.right_motor_in1, self.right_motor_in2, self.right_pwm
            ]:
                if hasattr(self, device.__str__()) and device is not None:
                    try:
                        device.close()
                    except:
                        pass
            
            logger.info("Motor kontrol modülü kapatıldı.")
        except Exception as e:
            logger.error(f"Motor temizleme hatası: {e}") 