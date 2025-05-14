#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Kontrol Modülü
Bu modül, araç motorlarının kontrolünü RPi.GPIO kütüphanesini kullanarak sağlar.
PID kontrol ile şerit takibi için gerekli motor hız ayarlarını yapar.
"""

import time
import logging
import RPi.GPIO as GPIO
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
            use_board_pins (bool): BOARD pin numaralandırması kullan (True) veya BCM kullan (False)
            pwm_frequency (int): PWM frekansı (Hz)
        """
        self.use_board_pins = use_board_pins
        self.max_speed = max_speed
        self.default_speed = default_speed
        self.pwm_frequency = pwm_frequency
        
        # Pin numaralandırma modunu ayarla
        if use_board_pins:
            GPIO.setmode(GPIO.BOARD)
            logger.info("BOARD pin numaralandırması kullanılıyor.")
            
            # Pin numaralarını doğrudan kullan
            self.left_in1_pin = left_motor_pins[0]
            self.left_in2_pin = left_motor_pins[1]
            self.right_in1_pin = right_motor_pins[0]
            self.right_in2_pin = right_motor_pins[1]
            self.left_en_pin = left_pwm_pin
            self.right_en_pin = right_pwm_pin
        else:
            GPIO.setmode(GPIO.BCM)
            logger.info("BCM pin numaralandırması kullanılıyor.")
            
            # Pin numaralarını doğrudan kullan
            self.left_in1_pin = left_motor_pins[0]
            self.left_in2_pin = left_motor_pins[1]
            self.right_in1_pin = right_motor_pins[0]
            self.right_in2_pin = right_motor_pins[1]
            self.left_en_pin = left_pwm_pin
            self.right_en_pin = right_pwm_pin
        
        # Kullanılan pin numaralarını logla
        logger.info(f"Sol motor pinleri: IN1={self.left_in1_pin}, IN2={self.left_in2_pin}, EN={self.left_en_pin}")
        logger.info(f"Sağ motor pinleri: IN1={self.right_in1_pin}, IN2={self.right_in2_pin}, EN={self.right_en_pin}")
        
        try:
            # Daha önce kullanılmış olabilecek pin yapılandırmalarını temizle
            GPIO.cleanup()
            
            # Pin'leri başlat
            logger.info("GPIO pinleri başlatılıyor...")
            
            # Motor kontrol pinleri
            GPIO.setup(self.left_in1_pin, GPIO.OUT)
            GPIO.setup(self.left_in2_pin, GPIO.OUT)
            GPIO.setup(self.right_in1_pin, GPIO.OUT)
            GPIO.setup(self.right_in2_pin, GPIO.OUT)
            GPIO.setup(self.left_en_pin, GPIO.OUT)
            GPIO.setup(self.right_en_pin, GPIO.OUT)
            
            # PWM nesnelerini oluştur
            self.left_pwm = GPIO.PWM(self.left_en_pin, self.pwm_frequency)
            self.right_pwm = GPIO.PWM(self.right_en_pin, self.pwm_frequency)
            
            # PWM'yi başlat (durgun durumda)
            self.left_pwm.start(0)
            self.right_pwm.start(0)
            
            # Başlangıçta motorları durdur
            self._set_left_motor('stop', 0)
            self._set_right_motor('stop', 0)
            
            logger.info("Motor pinleri başarıyla başlatıldı.")
            
        except Exception as e:
            logger.error(f"Motor kontrol başlatma hatası: {e}")
            self.cleanup()
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
    
    def _set_left_motor(self, direction, speed):
        """
        Sol motoru ayarlar.
        
        Args:
            direction (str): 'forward', 'backward' veya 'stop'
            speed (float): Hız değeri (0-1 arası)
        """
        # Hız değerini sınırla ve PWM değerine (0-100) dönüştür
        duty_cycle = min(max(0, speed), self.max_speed) * 100
        
        # Yöne göre pinleri ayarla
        if direction == 'forward':
            GPIO.output(self.left_in1_pin, GPIO.HIGH)
            GPIO.output(self.left_in2_pin, GPIO.LOW)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
        elif direction == 'backward':
            GPIO.output(self.left_in1_pin, GPIO.LOW)
            GPIO.output(self.left_in2_pin, GPIO.HIGH)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
        else:  # stop
            GPIO.output(self.left_in1_pin, GPIO.LOW)
            GPIO.output(self.left_in2_pin, GPIO.LOW)
            self.left_pwm.ChangeDutyCycle(0)
    
    def _set_right_motor(self, direction, speed):
        """
        Sağ motoru ayarlar.
        
        Args:
            direction (str): 'forward', 'backward' veya 'stop'
            speed (float): Hız değeri (0-1 arası)
        """
        # Hız değerini sınırla ve PWM değerine (0-100) dönüştür
        duty_cycle = min(max(0, speed), self.max_speed) * 100
        
        # Yöne göre pinleri ayarla
        if direction == 'forward':
            GPIO.output(self.right_in1_pin, GPIO.HIGH)
            GPIO.output(self.right_in2_pin, GPIO.LOW)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        elif direction == 'backward':
            GPIO.output(self.right_in1_pin, GPIO.LOW)
            GPIO.output(self.right_in2_pin, GPIO.HIGH)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        else:  # stop
            GPIO.output(self.right_in1_pin, GPIO.LOW)
            GPIO.output(self.right_in2_pin, GPIO.LOW)
            self.right_pwm.ChangeDutyCycle(0)
    
    def forward(self, speed=None):
        """
        Aracı ileri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_left_motor('forward', speed)
        self._set_right_motor('forward', speed)
        
        logger.debug(f"İleri hareket: Hız={speed}")
    
    def backward(self, speed=None):
        """
        Aracı geri yönde hareket ettirir.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_left_motor('backward', speed)
        self._set_right_motor('backward', speed)
        
        logger.debug(f"Geri hareket: Hız={speed}")
    
    def turn_left(self, speed=None):
        """
        Aracı sola döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_left_motor('forward', speed * 0.2)  # Sol motor yavaş
        self._set_right_motor('forward', speed)      # Sağ motor hızlı
        
        logger.debug(f"Sola dönüş: Hız={speed}")
    
    def turn_right(self, speed=None):
        """
        Aracı sağa döndürür.
        
        Args:
            speed (float): Hız seviyesi (0-1 arası)
        """
        if speed is None:
            speed = self.default_speed
            
        self._set_left_motor('forward', speed)      # Sol motor hızlı
        self._set_right_motor('forward', speed * 0.2) # Sağ motor yavaş
        
        logger.debug(f"Sağa dönüş: Hız={speed}")
    
    def stop(self):
        """
        Aracı durdurur.
        """
        self._set_left_motor('stop', 0)
        self._set_right_motor('stop', 0)
        
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
                self._set_left_motor('forward', left_speed)
                self._set_right_motor('forward', right_speed)
                
                logger.debug(f"Şerit takibi: Sapma={center_diff}, Sol={left_speed:.2f}, Sağ={right_speed:.2f}")
            except Exception as e:
                logger.error(f"Şerit takibi hatası: {e}")
                self.stop()
    
    def cleanup(self):
        """
        Motorları durdurur ve GPIO pinlerini temizler.
        """
        logger.info("Motor kontrol modülü kapatılıyor...")
        try:
            # Motorları durdur
            self.stop()
            
            # PWM'yi durdur
            if hasattr(self, 'left_pwm'):
                self.left_pwm.stop()
            if hasattr(self, 'right_pwm'):
                self.right_pwm.stop()
            
            # GPIO pinlerini temizle
            GPIO.cleanup()
                        
            logger.info("Motor kontrol modülü kapatıldı.")
        except Exception as e:
            logger.error(f"Motor temizleme hatası: {e}") 