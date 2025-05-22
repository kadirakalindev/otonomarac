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
    def __init__(self, debug=False):
        # GPIO pinlerinin konfigurasyonu
        self.in1_pin = 22  # Sağ motor ileri
        self.in2_pin = 23  # Sağ motor geri
        self.in3_pin = 17  # Sol motor ileri
        self.in4_pin = 27  # Sol motor geri
        self.en_a_pin = 12  # Sağ motor hız kontrolü
        self.en_b_pin = 13  # Sol motor hız kontrolü
        
        # Hız ve dönüş parametreleri
        self.base_speed = 50  # Temel ileri hız (0-100)
        self.turn_speed = 35   # Dönüş için kullanılacak fark (0-100)
        self.max_speed = 70    # Maksimum hız (0-100)
        self.min_speed = 20    # Minimum hız (0-100)
        self.center_deadzone = 0.08  # Merkez ölü bölge oranı (resim genişliğinin % olarak)
        
        # Dönüş limitleri
        self.max_turn_factor = 0.9  # Maksimum dönüş faktörü
        self.curve_speed_factor = 0.8  # Viraj hız azaltma faktörü
        self.motor_difference_limit = 50  # Motorlar arası maksimum fark limiti
        
        # PID parametreleri
        self.kp = 0.8  # Orantısal katsayı
        self.ki = 0.05  # İntegral katsayısı
        self.kd = 0.2   # Türev katsayısı
        
        self.error_history = []  # Son hataların hafızası
        self.max_history = 20    # Maksimum hata hafıza sayısı
        self.derivative_sample = 3  # Türev için örnek sayısı
        self.integral_sample = 10   # İntegral için örnek sayısı
        
        # Adaptif PID için değişkenler
        self.last_pid_update = 0
        self.pid_update_interval = 1.5  # 1.5 saniye
        self.adaptive_pid_enabled = True  # Adaptif PID'yi etkinleştir/devre dışı bırak
        
        # Çalışma modu
        self.debug = debug
        self.running = False
        self.mode = "follow_lane"  # follow_lane, manual, obstacle
        
        # Viraj izleme değişkenleri
        self.in_curve = False
        self.curve_direction = "none"  # left, right, none
        self.curve_start_time = 0
        self.curve_intensity = 0  # 0-1 arası değer
        
        # GPIO ve PWM ayarları
        self._setup_gpio()
        
    def _setup_gpio(self):
        """GPIO pinleri ve PWM ayarları"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            
            # Pin modlarını ayarla
            pins = [self.in1_pin, self.in2_pin, self.in3_pin, self.in4_pin, self.en_a_pin, self.en_b_pin]
            for pin in pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            # PWM pin ayarları
            self.right_pwm = GPIO.PWM(self.en_a_pin, 100)
            self.left_pwm = GPIO.PWM(self.en_b_pin, 100)
            self.right_pwm.start(0)
            self.left_pwm.start(0)
            
            if self.debug:
                print("GPIO kurulumu tamamlandı")
                
        except ImportError:
            # Raspberry Pi dışında çalıştırıldığında
            if self.debug:
                print("GPIO kütüphanesi bulunamadı - Simülasyon modunda çalışılıyor")
            self.right_pwm = None
            self.left_pwm = None
        
        except Exception as e:
            if self.debug:
                print(f"GPIO kurulumunda hata: {e}")
            self.right_pwm = None
            self.left_pwm = None
        
    def calculate_pid(self, center_diff, frame_width):
        """
        PID hesaplamasını yapar ve dönüş düzeltmesi değeri döndürür
        
        Args:
            center_diff (float): Merkez sapması (pixel)
            frame_width (int): Kare genişliği
            
        Returns:
            float: Dönüş düzeltme değeri (-1 ile 1 arasında, negatif sol, pozitif sağ)
        """
        # Hata değerini normalizasyon (-1 ile 1 arasında)
        error = center_diff / (frame_width / 2)
        
        # Merkezde küçük sapmaları ihmal et (kararlılık için)
        if abs(error) < self.center_deadzone:
            return 0
            
        # Hata geçmişini güncelle
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        # Orantısal terim
        p_term = error * self.kp
        
        # İntegral terimi
        i_samples = self.error_history[-min(len(self.error_history), self.integral_sample):]
        i_term = sum(i_samples) * self.ki
        
        # Türev terimi
        if len(self.error_history) > self.derivative_sample:
            d_samples = self.error_history[-self.derivative_sample:]
            d_term = (d_samples[-1] - d_samples[0]) * self.kd
        else:
            d_term = 0
            
        # PID çıktısı
        output = p_term + i_term + d_term
        
        # Adaptif PID katsayılarını güncelle
        self._update_adaptive_pid()
        
        # Çıktıyı -1 ile 1 arasında sınırla
        return max(-1.0, min(1.0, output))
    
    def _update_adaptive_pid(self):
        """Adaptif PID katsayılarını hata geçmişine göre günceller"""
        if not self.adaptive_pid_enabled or len(self.error_history) < 10:
            return
            
        current_time = time.time()
        if current_time - self.last_pid_update < self.pid_update_interval:
            return
            
        self.last_pid_update = current_time
        
        # Son hataların ortalaması
        recent_errors = self.error_history[-10:]
        avg_error = sum(abs(e) for e in recent_errors) / len(recent_errors)
        
        # Hataların standart sapması (kararlılık göstergesi)
        std_error = (sum((e - avg_error) ** 2 for e in recent_errors) / len(recent_errors)) ** 0.5
        
        # Katsayıları ayarla
        if avg_error > 0.5:  # Büyük hatalar - daha agresif ol
            self.kp = min(1.2, self.kp * 1.1)
            self.kd = max(0.15, self.kd * 0.9)  # Türevi azalt
        elif avg_error < 0.2:  # Küçük hatalar - daha konservatif ol
            self.kp = max(0.5, self.kp * 0.95)
            self.kd = min(0.3, self.kd * 1.05)  # Türevi artır
            
        # Kararsızlık durumunda integral etkisini azalt
        if std_error > 0.2:
            self.ki = max(0.01, self.ki * 0.9)
        else:
            self.ki = min(0.1, self.ki * 1.05)
            
        if self.debug:
            print(f"Adaptif PID: kp={self.kp:.2f}, ki={self.ki:.2f}, kd={self.kd:.2f}")
    
    def follow_lane(self, center_diff, frame_width, lane_detector=None):
        """
        Şerit takibi için motor kontrolü
        
        Args:
            center_diff (float): Merkez sapması (pixel)
            frame_width (int): Kare genişliği
            lane_detector (LaneDetector, optional): Şerit tespiti nesnesi (viraj bilgisi için)
            
        Returns:
            tuple: Sol ve sağ motor hızları
        """
        if center_diff is None:
            # Şerit tespit edilemedi, düz git
            if self.debug:
                print("Şerit tespit edilemedi, düz devam ediliyor")
            return self.base_speed, self.base_speed
        
        # Viraj durumunu kontrol et
        self._check_curve_state(lane_detector)
        
        # PID çıktısını hesapla
        correction = self.calculate_pid(center_diff, frame_width)
        
        # Temel hızı belirle
        base_speed = self.base_speed
        
        # Viraj durumunda hızı düşür
        if self.in_curve:
            curve_factor = max(0.6, 1.0 - (self.curve_intensity * self.curve_speed_factor))
            base_speed = max(self.min_speed, int(base_speed * curve_factor))
            
            if self.debug:
                print(f"Viraj hız düzenlemesi: {base_speed} (faktör: {curve_factor:.2f})")
        
        # Dönüş faktörünü hesapla
        turn_factor = abs(correction) * self.max_turn_factor
        
        # Hızları hesapla
        left_speed = right_speed = base_speed
        
        # Dönüş yönünü belirle
        if correction > 0:  # Sağa dön
            # Sağa virajda, özel dönüş davranışı
            if self.in_curve and self.curve_direction == "right":
                # Sağ motoru daha fazla yavaşlat, sol motoru daha az yavaşlat
                right_speed = max(self.min_speed, int(base_speed * (1 - turn_factor * 1.2)))
                left_speed = max(self.min_speed, int(base_speed * (1 - turn_factor * 0.5)))
            else:
                # Normal sağa dönüş
                right_speed = max(self.min_speed, int(base_speed * (1 - turn_factor)))
        else:  # Sola dön
            # Sola virajda, özel dönüş davranışı
            if self.in_curve and self.curve_direction == "left":
                # Sol motoru daha fazla yavaşlat, sağ motoru daha az yavaşlat
                left_speed = max(self.min_speed, int(base_speed * (1 - turn_factor * 1.2)))
                right_speed = max(self.min_speed, int(base_speed * (1 - turn_factor * 0.5)))
            else:
                # Normal sola dönüş
                left_speed = max(self.min_speed, int(base_speed * (1 - turn_factor)))
        
        # Motor hızları arasındaki farkı sınırla
        if abs(left_speed - right_speed) > self.motor_difference_limit:
            # Farkı koruyarak her iki motoru da yavaşlat
            avg_speed = (left_speed + right_speed) / 2
            speed_diff = min(self.motor_difference_limit, abs(left_speed - right_speed))
            
            if left_speed > right_speed:
                left_speed = int(avg_speed + speed_diff / 2)
                right_speed = int(avg_speed - speed_diff / 2)
            else:
                left_speed = int(avg_speed - speed_diff / 2)
                right_speed = int(avg_speed + speed_diff / 2)
            
            # Minimum hız kontrolü
            left_speed = max(self.min_speed, left_speed)
            right_speed = max(self.min_speed, right_speed)
            
        if self.debug:
            print(f"Şerit takibi: merkez_fark={center_diff:.1f}, düzeltme={correction:.2f}, " +
                  f"sol_hız={left_speed}, sağ_hız={right_speed}")
                  
        return left_speed, right_speed
    
    def _check_curve_state(self, lane_detector):
        """Viraj durumunu kontrol eder ve ilgili değişkenleri günceller"""
        if lane_detector is None:
            self.in_curve = False
            self.curve_direction = "none"
            self.curve_intensity = 0
            return
        
        # LaneDetector'dan viraj verilerini al
        prev_in_curve = self.in_curve
        self.in_curve = lane_detector.is_curve
        self.curve_direction = lane_detector.curve_direction
        self.curve_intensity = lane_detector.curve_confidence
        
        # Viraj başlangıç zamanını kaydet
        if not prev_in_curve and self.in_curve:
            self.curve_start_time = time.time()
            if self.debug:
                print(f"Viraj başladı: {self.curve_direction} yönünde")
        
        # Viraj bitişinde log
        elif prev_in_curve and not self.in_curve:
            curve_duration = time.time() - self.curve_start_time
            if self.debug:
                print(f"Viraj bitti: {self.curve_direction} yönünde, süre: {curve_duration:.1f}s")
    
    def set_motors(self, left_speed, right_speed):
        """
        Motor hızlarını ve yönlerini ayarlar
        
        Args:
            left_speed (int): Sol motor hızı (-100 ile 100 arasında, negatif geri)
            right_speed (int): Sağ motor hızı (-100 ile 100 arasında, negatif geri)
        """
        # Hızları 0-100 arasına sınırla
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))
        
        try:
            import RPi.GPIO as GPIO
            
            # Sol motor yönü
            if left_speed >= 0:  # İleri
                GPIO.output(self.in3_pin, GPIO.HIGH)
                GPIO.output(self.in4_pin, GPIO.LOW)
                left_pwm_value = abs(left_speed)
            else:  # Geri
                GPIO.output(self.in3_pin, GPIO.LOW)
                GPIO.output(self.in4_pin, GPIO.HIGH)
                left_pwm_value = abs(left_speed)
            
            # Sağ motor yönü
            if right_speed >= 0:  # İleri
                GPIO.output(self.in1_pin, GPIO.HIGH)
                GPIO.output(self.in2_pin, GPIO.LOW)
                right_pwm_value = abs(right_speed)
            else:  # Geri
                GPIO.output(self.in1_pin, GPIO.LOW)
                GPIO.output(self.in2_pin, GPIO.HIGH)
                right_pwm_value = abs(right_speed)
            
            # PWM ile hız kontrolü
            if self.left_pwm is not None:
                self.left_pwm.ChangeDutyCycle(left_pwm_value)
            if self.right_pwm is not None:
                self.right_pwm.ChangeDutyCycle(right_pwm_value)
            
            if self.debug:
                print(f"Motor hızları ayarlandı: sol={left_speed}, sağ={right_speed}")
            
        except Exception as e:
            if self.debug:
                print(f"Motor kontrolünde hata: {e}")
    
    def stop(self):
        """Motorları durdurur"""
        try:
            if self.left_pwm is not None and self.right_pwm is not None:
                self.left_pwm.ChangeDutyCycle(0)
                self.right_pwm.ChangeDutyCycle(0)
                
            import RPi.GPIO as GPIO
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.LOW)
            GPIO.output(self.in3_pin, GPIO.LOW)
            GPIO.output(self.in4_pin, GPIO.LOW)
            
            self.running = False
            
        except Exception as e:
            if self.debug:
                print(f"Motorları durdururken hata: {e}")
    
    def cleanup(self):
        """GPIO pinlerini temizler"""
        try:
            self.stop()
            
            if self.left_pwm is not None:
                self.left_pwm.stop()
            if self.right_pwm is not None:
                self.right_pwm.stop()
                
            import RPi.GPIO as GPIO
            GPIO.cleanup()
            
            if self.debug:
                print("GPIO pinleri temizlendi")
                
        except Exception as e:
            if self.debug:
                print(f"GPIO temizlenirken hata: {e}") 