#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Kontrol Modülü
Bu modül, araç motorlarının kontrolünü gpiozero kütüphanesini kullanarak sağlar.
PID kontrol ile şerit takibi için gerekli motor hız ayarlarını yapar.
"""

import time
import logging
from gpiozero import Motor, OutputDevice
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
                 left_motor_pins=(17, 18),    # Sol motor için (ileri, geri) pin numaraları
                 right_motor_pins=(22, 23),   # Sağ motor için (ileri, geri) pin numaraları
                 left_pwm_pin=None,           # Sol motor için isteğe bağlı PWM enable pini
                 right_pwm_pin=None,          # Sağ motor için isteğe bağlı PWM enable pini
                 max_speed=1.0,               # Maksimum hız değeri (0-1 arası)
                 default_speed=0.5):          # Varsayılan hız
        """
        MotorController sınıfını başlatır.
        
        Args:
            left_motor_pins (tuple): Sol motor kontrol pinleri (ileri, geri)
            right_motor_pins (tuple): Sağ motor kontrol pinleri (ileri, geri)
            left_pwm_pin (int): Sol motor için isteğe bağlı PWM pini (L298N için)
            right_pwm_pin (int): Sağ motor için isteğe bağlı PWM pini (L298N için)
            max_speed (float): Maksimum hız seviyesi (0-1 arası)
            default_speed (float): Varsayılan hız seviyesi (0-1 arası)
        """
        self.max_speed = max_speed
        self.default_speed = default_speed
        
        # PWM pinleri belirtilmişse, motor sürücü için gereken enable pinlerini oluştur
        self.left_pwm = None
        self.right_pwm = None
        
        if left_pwm_pin is not None:
            self.left_pwm = OutputDevice(left_pwm_pin)
            self.left_pwm.value = 1  # PWM etkinleştir
            
        if right_pwm_pin is not None:
            self.right_pwm = OutputDevice(right_pwm_pin)
            self.right_pwm.value = 1  # PWM etkinleştir
        
        # Motor nesnelerini oluştur
        self.left_motor = Motor(forward=left_motor_pins[0], backward=left_motor_pins[1])
        self.right_motor = Motor(forward=right_motor_pins[0], backward=right_motor_pins[1])
        
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
    
    def forward(self, speed=None):
        """
        Aracı ileri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        # Hız değerini sınırla
        speed = min(max(0, speed), self.max_speed)
        
        self.left_motor.forward(speed)
        self.right_motor.forward(speed)
        
        logger.debug(f"İleri hareket: Hız={speed}")
    
    def backward(self, speed=None):
        """
        Aracı geri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        # Hız değerini sınırla
        speed = min(max(0, speed), self.max_speed)
        
        self.left_motor.backward(speed)
        self.right_motor.backward(speed)
        
        logger.debug(f"Geri hareket: Hız={speed}")
    
    def turn_left(self, speed=None):
        """
        Aracı sola döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        # Hız değerini sınırla
        speed = min(max(0, speed), self.max_speed)
        
        self.left_motor.forward(speed * 0.2)  # Sol motor yavaş
        self.right_motor.forward(speed)       # Sağ motor hızlı
        
        logger.debug(f"Sola dönüş: Hız={speed}")
    
    def turn_right(self, speed=None):
        """
        Aracı sağa döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        # Hız değerini sınırla
        speed = min(max(0, speed), self.max_speed)
        
        self.left_motor.forward(speed)       # Sol motor hızlı
        self.right_motor.forward(speed * 0.2)  # Sağ motor yavaş
        
        logger.debug(f"Sağa dönüş: Hız={speed}")
    
    def stop(self):
        """
        Aracı durdurur.
        """
        self.left_motor.stop()
        self.right_motor.stop()
        
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
            # PID kontrolü ile motor hızlarını hesapla
            left_speed, right_speed = self.pid_control(center_diff)
            
            # Motor hızlarını ayarla
            self.left_motor.forward(left_speed)
            self.right_motor.forward(right_speed)
            
            logger.debug(f"Şerit takibi: Sapma={center_diff}, Sol={left_speed:.2f}, Sağ={right_speed:.2f}")
    
    def cleanup(self):
        """
        Motorları durdurur ve kapatır.
        """
        self.stop()
        
        # PWM pinlerini kapat
        if self.left_pwm:
            self.left_pwm.close()
        if self.right_pwm:
            self.right_pwm.close()
            
        # Motor pinlerini kapat
        self.left_motor.close()
        self.right_motor.close()
        
        logger.info("Motor kontrol modülü kapatıldı.") 