#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Kontrol Modülü
Bu modül, araç motorlarının kontrolünü gpiozero kütüphanesini kullanarak sağlar.
PID kontrol ile şerit takibi için gerekli motor hız ayarlarını yapar.
"""

import time
import logging
from gpiozero import Motor, PWMOutputDevice
import numpy as np

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MotorControl")

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
                 use_board_pins=True):        # BOARD pin numaralandırmasını kullan
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
        """
        # BOARD kullanıldığında pin numaralarını BCM'ye dönüştür
        self.use_board_pins = use_board_pins
        
        # BOARD pin numaralarını gpiozero için BCM'ye dönüştür
        if use_board_pins:
            # BOARD to BCM numaralandırma dönüşümü için yaygın pinler
            board_to_bcm = {
                3: 2, 5: 3, 7: 4, 8: 14, 10: 15, 11: 17, 12: 18, 13: 27,
                15: 22, 16: 23, 18: 24, 19: 10, 21: 9, 22: 25, 23: 11,
                24: 8, 26: 7, 29: 5, 31: 6, 32: 12, 33: 13, 35: 19, 36: 16,
                37: 26, 38: 20, 40: 21
            }
            
            # Pin numaralarını BCM'ye dönüştür
            left_in1_bcm = board_to_bcm.get(left_motor_pins[0])
            left_in2_bcm = board_to_bcm.get(left_motor_pins[1])
            right_in1_bcm = board_to_bcm.get(right_motor_pins[0])
            right_in2_bcm = board_to_bcm.get(right_motor_pins[1])
            left_en_bcm = board_to_bcm.get(left_pwm_pin)
            right_en_bcm = board_to_bcm.get(right_pwm_pin)
            
            # Dönüştürme başarısız olursa uyarı ver
            missing_pins = []
            if left_in1_bcm is None: missing_pins.append(f"Sol IN1 ({left_motor_pins[0]})")
            if left_in2_bcm is None: missing_pins.append(f"Sol IN2 ({left_motor_pins[1]})")
            if right_in1_bcm is None: missing_pins.append(f"Sağ IN1 ({right_motor_pins[0]})")
            if right_in2_bcm is None: missing_pins.append(f"Sağ IN2 ({right_motor_pins[1]})")
            if left_en_bcm is None: missing_pins.append(f"Sol Enable ({left_pwm_pin})")
            if right_en_bcm is None: missing_pins.append(f"Sağ Enable ({right_pwm_pin})")
            
            if missing_pins:
                logger.warning(f"Şu pinler BOARD-BCM dönüşümünde bulunamadı: {', '.join(missing_pins)}")
                logger.warning("Doğru pin numaralarını kullandığınızdan emin olun veya BCM numaralandırması kullanın.")
            
            # BCM numaralarını kullan
            self.left_in1_pin = left_in1_bcm if left_in1_bcm is not None else left_motor_pins[0]
            self.left_in2_pin = left_in2_bcm if left_in2_bcm is not None else left_motor_pins[1]
            self.right_in1_pin = right_in1_bcm if right_in1_bcm is not None else right_motor_pins[0]
            self.right_in2_pin = right_in2_bcm if right_in2_bcm is not None else right_motor_pins[1]
            self.left_en_pin = left_en_bcm if left_en_bcm is not None else left_pwm_pin
            self.right_en_pin = right_en_bcm if right_en_bcm is not None else right_pwm_pin
        else:
            # BCM numaralarını doğrudan kullan
            self.left_in1_pin = left_motor_pins[0]
            self.left_in2_pin = left_motor_pins[1]
            self.right_in1_pin = right_motor_pins[0]
            self.right_in2_pin = right_motor_pins[1]
            self.left_en_pin = left_pwm_pin
            self.right_en_pin = right_pwm_pin
        
        # Kullanılan pin numaralarını logla
        logger.info(f"Sol motor pinleri (BCM): IN1={self.left_in1_pin}, IN2={self.left_in2_pin}, EN={self.left_en_pin}")
        logger.info(f"Sağ motor pinleri (BCM): IN1={self.right_in1_pin}, IN2={self.right_in2_pin}, EN={self.right_en_pin}")
            
        self.max_speed = max_speed
        self.default_speed = default_speed
        
        try:
            # PWM çıkışları oluştur (L298N Enable pinleri)
            logger.info("PWM çıkışları başlatılıyor...")
            self.left_pwm = PWMOutputDevice(self.left_en_pin)
            self.right_pwm = PWMOutputDevice(self.right_en_pin)
            
            # Motor nesnelerini oluştur
            logger.info("Motor nesneleri başlatılıyor...")
            self.left_motor = Motor(forward=self.left_in1_pin, backward=self.left_in2_pin, pwm=False)
            self.right_motor = Motor(forward=self.right_in1_pin, backward=self.right_in2_pin, pwm=False)
            
            # İlk başlangıçta motorları durdur
            self.stop()
            
        except Exception as e:
            logger.error(f"Motor kontrol başlatma hatası: {e}")
            # Hata durumunda temizleme yap
            if hasattr(self, 'left_pwm'):
                self.left_pwm.close()
            if hasattr(self, 'right_pwm'):
                self.right_pwm.close()
            if hasattr(self, 'left_motor'):
                self.left_motor.close()
            if hasattr(self, 'right_motor'):
                self.right_motor.close()
            raise
        
        # PID kontrol için parametreler
        # Not: Bu parametreler test edilerek ince ayar yapılmalıdır
        self.kp = 0.5  # Orantısal katsayı
        self.ki = 0.0  # Integral katsayı
        self.kd = 0.1  # Türev katsayı
        
        # PID hesaplaması için gerekli değişkenler
        self.previous_error = 0
        self.integral = 0
        
        # Son zamanlama (dt için)
        self.last_time = time.time()
        
        logger.info("Motor kontrol modülü başlatıldı.")
    
    def _set_motor_speed(self, motor, pwm_device, speed, direction):
        """
        Motor hızını ve yönünü ayarlar.
        
        Args:
            motor: Motor nesnesi (left_motor veya right_motor)
            pwm_device: PWM çıkış cihazı (left_pwm veya right_pwm)
            speed: Hız değeri (0-1 arası)
            direction: Yön ('forward' veya 'backward')
        """
        # Hız değerini sınırla
        speed = min(max(0, speed), self.max_speed)
        
        # Yöne göre motoru ayarla
        if direction == 'forward':
            motor.forward()
        else:  # backward
            motor.backward()
            
        # PWM ile hızı ayarla
        pwm_device.value = speed
    
    def forward(self, speed=None):
        """
        Aracı ileri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed(self.left_motor, self.left_pwm, speed, 'forward')
        self._set_motor_speed(self.right_motor, self.right_pwm, speed, 'forward')
        
        logger.debug(f"İleri hareket: Hız={speed}")
    
    def backward(self, speed=None):
        """
        Aracı geri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed(self.left_motor, self.left_pwm, speed, 'backward')
        self._set_motor_speed(self.right_motor, self.right_pwm, speed, 'backward')
        
        logger.debug(f"Geri hareket: Hız={speed}")
    
    def turn_left(self, speed=None):
        """
        Aracı sola döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed(self.left_motor, self.left_pwm, speed * 0.2, 'forward')  # Sol motor yavaş
        self._set_motor_speed(self.right_motor, self.right_pwm, speed, 'forward')      # Sağ motor hızlı
        
        logger.debug(f"Sola dönüş: Hız={speed}")
    
    def turn_right(self, speed=None):
        """
        Aracı sağa döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_motor_speed(self.left_motor, self.left_pwm, speed, 'forward')      # Sol motor hızlı
        self._set_motor_speed(self.right_motor, self.right_pwm, speed * 0.2, 'forward') # Sağ motor yavaş
        
        logger.debug(f"Sağa dönüş: Hız={speed}")
    
    def stop(self):
        """
        Aracı durdurur.
        """
        self.left_motor.stop()
        self.right_motor.stop()
        self.left_pwm.value = 0
        self.right_pwm.value = 0
        
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
        self.last_time = current_time
        
        # PID hesaplaması
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        # PID çıkışını hesapla
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Önceki hatayı güncelle
        self.previous_error = error
        
        # Motor hızlarını ayarla
        base_speed = self.default_speed
        
        # Çıkışı sınırlandır
        output = max(min(output, base_speed), -base_speed)
        
        # Sol ve sağ motor hızlarını hesapla
        if error > 0:  # Sağa kaymış, sola dönmeli
            left_speed = base_speed - output
            right_speed = base_speed
        else:  # Sola kaymış, sağa dönmeli
            left_speed = base_speed
            right_speed = base_speed + output
        
        # Hız değerlerini sınırla
        left_speed = min(max(0, left_speed), self.max_speed)
        right_speed = min(max(0, right_speed), self.max_speed)
        
        return left_speed, right_speed
    
    def follow_lane(self, center_diff):
        """
        Şerit takibi için gerekli motor komutlarını verir.
        
        Args:
            center_diff (int): Şeritten sapma miktarı (piksel cinsinden)
                             Pozitif değer sağa kayma, negatif değer sola kayma
        """
        if center_diff is None:
            # Eğer şerit tespit edilemezse, ileri git
            self.forward()
            return
        
        # Şerit merkezden kayma miktarının mutlak değeri
        abs_diff = abs(center_diff)
        
        # Eşik değeri (bu değerden az sapmalar için düzeltme yapma)
        threshold = 10
        
        if abs_diff < threshold:
            # Sapma az ise düz git
            self.forward()
        else:
            try:
                # PID kontrolü ile motor hızlarını hesapla
                left_speed, right_speed = self.pid_control(center_diff)
                
                # Motor hızlarını ayarla
                self._set_motor_speed(self.left_motor, self.left_pwm, left_speed, 'forward')
                self._set_motor_speed(self.right_motor, self.right_pwm, right_speed, 'forward')
                
                logger.debug(f"Şerit takibi: Sapma={center_diff}, Sol={left_speed:.2f}, Sağ={right_speed:.2f}")
            except Exception as e:
                logger.error(f"Şerit takibi hatası: {e}")
                self.stop()
    
    def cleanup(self):
        """
        Motorları durdurur ve kapatır.
        """
        logger.info("Motor kontrol modülü kapatılıyor...")
        try:
            self.stop()
            
            # PWM ve motor pinlerini kapat
            if hasattr(self, 'left_pwm'):
                self.left_pwm.close()
            if hasattr(self, 'right_pwm'):
                self.right_pwm.close()
                
            if hasattr(self, 'left_motor'):
                self.left_motor.close()
            if hasattr(self, 'right_motor'):
                self.right_motor.close()
                
            logger.info("Motor kontrol modülü kapatıldı.")
        except Exception as e:
            logger.error(f"Motor temizleme hatası: {e}") 