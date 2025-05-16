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
                 max_speed=0.7,               # Maksimum hız değeri (0-1 arası) - azaltıldı
                 default_speed=0.35,          # Varsayılan hız - azaltıldı
                 use_board_pins=True,         # BOARD pin numaralandırmasını kullan
                 pwm_frequency=50):          # PWM frekansı (Hz) - azaltıldı
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
        
        # Optimize edilmiş PID kontrol parametreleri - daha yumuşak kontrol için değiştirildi
        self.kp = 0.5   # Orantısal katsayı - azaltıldı, daha yumuşak tepki için
        self.ki = 0.02  # Integral katsayı - azaltıldı
        self.kd = 0.1   # Türev katsayı - azaltıldı, daha az salınım için
        
        # PID sınır değerleri
        self.max_integral = 50.0  # İntegral terimini daha fazla sınırla
        
        # Hız limitleri
        self.min_motor_speed = 0.2  # Minimum motor hızı (durma noktasını önler)
        
        # Hız değişim limiti - ani değişimleri önlemek için
        self.max_speed_change = 0.1  # Bir adımda maksimum hız değişimi
        self.prev_left_speed = self.default_speed
        self.prev_right_speed = self.default_speed
        
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
        
        # Ani hız değişimlerini sınırla
        if motor_side == 'left':
            # Önceki hızla karşılaştır ve değişimi sınırla
            max_new_speed = self.prev_left_speed + self.max_speed_change
            min_new_speed = self.prev_left_speed - self.max_speed_change
            speed = min(max(speed, min_new_speed), max_new_speed)
            self.prev_left_speed = speed
            
            motor_in1 = self.left_motor_in1
            motor_in2 = self.left_motor_in2
            motor_pwm = self.left_pwm
        else:  # 'right'
            # Önceki hızla karşılaştır ve değişimi sınırla
            max_new_speed = self.prev_right_speed + self.max_speed_change
            min_new_speed = self.prev_right_speed - self.max_speed_change
            speed = min(max(speed, min_new_speed), max_new_speed)
            self.prev_right_speed = speed
            
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
    
    def pid_control(self, error, dt=None):
        """
        PID kontrol algoritması ile hata değerine göre düzeltme hesaplar.
        
        Args:
            error (float): Şerit merkezinden sapma hatası (-1 ile 1 arasında)
            dt (float, optional): Zaman farkı. None ise otomatik hesaplanır.
            
        Returns:
            float: Düzeltme değeri (PID kontrolü çıktısı)
        """
        # Zaman farkını hesapla
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
            dt = max(0.001, min(dt, 0.1))  # dt değerini sınırla (çok büyük veya çok küçük olmasın)
        self.last_time = current_time
        
        # Deadband (ölü bölge) uygulaması - çok küçük hatalar için tepki verme
        deadband = 0.05  # %5'lik bir ölü bölge
        if abs(error) < deadband:
            error = 0
            self.integral = 0  # Ölü bölgedeyken integrali sıfırla
        
        # PID bileşenlerini hesapla
        proportional = error
        
        # İntegral hesabı - dt ile doğru entegrasyon
        self.integral += error * dt
        
        # İntegral sınırlaması (anti-windup) - değiştirildi (daha küçük sınır)
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        
        # Türev hesabı - daha doğru (zamana göre değişim)
        if dt > 0:
            derivative = (error - self.previous_error) / dt
            # Türev filtreleme (ani değişimleri yumuşat)
            derivative = max(-1.0, min(derivative, 1.0))
        else:
            derivative = 0
            
        # PID çıktısını hesapla
        output = (self.kp * proportional) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Çıktıyı sınırlandır
        output = max(-1.0, min(output, 1.0))
        
        # Önceki hatayı güncelle
        self.previous_error = error
        
        # Debugging
        logger.debug(f"PID - Hata: {error:.4f}, P: {proportional:.4f}, I: {self.integral:.4f}, D: {derivative:.4f}, Çıktı: {output:.4f}")
        
        return output

    def follow_lane(self, center_diff, speed=None):
        """
        Şerit takibi için araç kontrolü.
        
        Args:
            center_diff (float): Şerit merkezinden sapma değeri (-1 ile 1 arasında)
                                -1: tamamen solda, 0: merkezde, 1: tamamen sağda
            speed (float, optional): Aracın temel hızı. None ise default_speed kullanılır.
        """
        # Şerit takibi için temel hızı belirle
        if speed is None:
            speed = self.default_speed
            
        # Hızı sınırla (aşırı ısınmayı önlemek için)
        speed = min(speed, self.max_speed)
        
        # Eğer şerit tespit edilemezse (center_diff None ise)
        if center_diff is None:
            # Şerit kaybı sayacını artır
            self.lost_lane_counter += 1
            
            if self.lost_lane_counter > 10:  # Şerit belirli süre boyunca bulunamadıysa
                logger.warning(f"Şerit {self.lost_lane_counter} kare boyunca bulunamadı. Durduruluyor...")
                # Şerit bulunamadığında aracı durdur veya önceki harekete devam et
                self.stop()
                return
                
            logger.warning(f"Şerit bulunamadı ({self.lost_lane_counter}). Son harekete devam ediliyor.")
            return
        else:
            # Şerit bulundu, sayacı sıfırla
            self.lost_lane_counter = 0
            
        # PID kontrol çıktısını hesapla
        correction = self.pid_control(center_diff)
        
        # Daha yumuşak dönüşler için düzeltme faktörünü azalt (aşırı ısınmayı azaltır)
        steering_factor = 0.3  # Daha önceki 0.5 değerinden daha düşük
        
        # Motor hızlarını hesapla
        left_speed = speed - (correction * steering_factor * speed)
        right_speed = speed + (correction * steering_factor * speed)
        
        # Minimum motor hızını sağla (motorların durmasını önle)
        if left_speed > 0:
            left_speed = max(left_speed, self.min_motor_speed)
        elif left_speed < 0:
            left_speed = min(left_speed, -self.min_motor_speed)
            
        if right_speed > 0:
            right_speed = max(right_speed, self.min_motor_speed)
        elif right_speed < 0:
            right_speed = min(right_speed, -self.min_motor_speed)
        
        # İleri/geri yönleri belirle
        if left_speed >= 0:
            left_direction = 'forward'
        else:
            left_direction = 'backward'
            left_speed = abs(left_speed)
            
        if right_speed >= 0:
            right_direction = 'forward'
        else:
            right_direction = 'backward'
            right_speed = abs(right_speed)
        
        # Motorları ayarlanan hızlarda çalıştır
        self._set_motor_speed('left', left_direction, left_speed)
        self._set_motor_speed('right', right_direction, right_speed)
        
        # Yapılan ayarlamaları logla
        logger.debug(f"Şerit takibi - Sapma: {center_diff:.4f}, Düzeltme: {correction:.4f}, Sol: {left_speed:.2f}, Sağ: {right_speed:.2f}")
    
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