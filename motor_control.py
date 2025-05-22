#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Motor Control module for Autonomous Vehicle
Motorların kontrolü için temel sınıf ve fonksiyonlar.
"""

import time
import logging
import threading
from typing import Tuple, Dict, Optional, Any
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MotorControl')

class MotorController:
    """DC Motorların kontrolü için sınıf"""
    
    def __init__(self, pin_config: Dict[str, int] = None):
        """
        Motor kontrol sınıfı başlatıcı
        
        Args:
            pin_config: Motor pin konfigürasyonu (isteğe bağlı, varsayılan değerler kullanılabilir)
        """
        # Varsayılan pin konfigürasyonu (BCM numaralandırması)
        default_pin_config = {
            "left_motor_in1": 23,  # BOARD 16
            "left_motor_in2": 24,  # BOARD 18
            "left_motor_pwm": 18,  # BOARD 12
            "right_motor_in1": 16, # BOARD 36
            "right_motor_in2": 20, # BOARD 38
            "right_motor_pwm": 12  # BOARD 32
        }
        
        # Pin konfigürasyonu
        self.pin_config = default_pin_config
        if pin_config:
            self.pin_config.update(pin_config)
        
        # Hız kontrolü parametreleri
        self.base_speed = 0.5  # Temel hız (%50)
        self.max_speed = 0.8   # Maksimum hız (%80)
        self.min_speed = 0.3   # Minimum hız (%30)
        self.turning_factor = 0.2  # Dönüş faktörü
        self.speed_smooth_factor = 0.7  # Hız geçiş yumuşatma faktörü
        
        # PID kontrol parametreleri
        self.kp = 0.35  # Oransal kazanç
        self.ki = 0.0005  # İntegral kazanç
        self.kd = 0.15  # Türev kazanç
        
        # PID limitleri
        self.pid_limits = {
            "kp_max": 0.5,
            "ki_max": 0.005,
            "kd_max": 0.2
        }
        
        # PID durum değişkenleri
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.last_pid_update = time.time()
        self.pid_update_interval = 1.5  # PID güncelleme aralığı (saniye)
        
        # Merkez ölü bölge (center_offset bu aralıkta ise düz git)
        self.center_deadzone = 0.08
        
        # Son hız değerleri
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        # Hız için low-pass filtre
        self.filtered_left_speed = 0.0
        self.filtered_right_speed = 0.0
        
        # Hareket durumu
        self.is_moving = False
        self.auto_mode = False  # Otonom mod
        
        # Şerit kaybı durumu
        self.lane_lost = False
        self.lane_lost_time = 0.0
        self.recovery_mode = False
        self.recovery_start_time = 0.0
        
        # Motorları başlat
        try:
            self._setup_motors()
            logger.info("Motor kontrol başlatıldı")
        except Exception as e:
            logger.error(f"Motor başlatma hatası: {str(e)}")
            raise
    
    def _setup_motors(self):
        """Motorları yapılandır"""
        # Sol motor
        self.left_motor_in1 = DigitalOutputDevice(self.pin_config["left_motor_in1"])
        self.left_motor_in2 = DigitalOutputDevice(self.pin_config["left_motor_in2"])
        self.left_motor_pwm = PWMOutputDevice(self.pin_config["left_motor_pwm"], frequency=1000)
        
        # Sağ motor
        self.right_motor_in1 = DigitalOutputDevice(self.pin_config["right_motor_in1"])
        self.right_motor_in2 = DigitalOutputDevice(self.pin_config["right_motor_in2"])
        self.right_motor_pwm = PWMOutputDevice(self.pin_config["right_motor_pwm"], frequency=1000)
        
        # Başlangıçta motorları durdur
        self.stop()
    
    def set_motor_speeds(self, left_speed: float, right_speed: float):
        """
        Sol ve sağ motor hızlarını ayarlar (-1.0 ile 1.0 arası)
        Negatif değerler geriye doğru hareketi ifade eder
        
        Args:
            left_speed: Sol motor hızı (-1.0 ile 1.0 arası)
            right_speed: Sağ motor hızı (-1.0 ile 1.0 arası)
        """
        # Hız değerlerini sınırlandır
        left_speed = max(min(left_speed, 1.0), -1.0)
        right_speed = max(min(right_speed, 1.0), -1.0)
        
        # Yumuşatma uygula
        self.filtered_left_speed = self.filtered_left_speed * self.speed_smooth_factor + left_speed * (1 - self.speed_smooth_factor)
        self.filtered_right_speed = self.filtered_right_speed * self.speed_smooth_factor + right_speed * (1 - self.speed_smooth_factor)
        
        # Sol motor kontrolü
        if self.filtered_left_speed >= 0:
            self.left_motor_in1.on()
            self.left_motor_in2.off()
            self.left_motor_pwm.value = self.filtered_left_speed
        else:
            self.left_motor_in1.off()
            self.left_motor_in2.on()
            self.left_motor_pwm.value = -self.filtered_left_speed
        
        # Sağ motor kontrolü
        if self.filtered_right_speed >= 0:
            self.right_motor_in1.on()
            self.right_motor_in2.off()
            self.right_motor_pwm.value = self.filtered_right_speed
        else:
            self.right_motor_in1.off()
            self.right_motor_in2.on()
            self.right_motor_pwm.value = -self.filtered_right_speed
        
        # Mevcut hız değerlerini güncelle
        self.current_left_speed = self.filtered_left_speed
        self.current_right_speed = self.filtered_right_speed
        
        # Hareket durumunu güncelle
        self.is_moving = (abs(self.filtered_left_speed) > 0.05 or abs(self.filtered_right_speed) > 0.05)
        
        # Log
        logger.debug(f"Motor hızları: Sol={self.filtered_left_speed:.2f}, Sağ={self.filtered_right_speed:.2f}")
    
    def forward(self, speed: float = None):
        """
        İleri hareket
        
        Args:
            speed: Hız (0.0 ile 1.0 arası, None ise base_speed kullanılır)
        """
        if speed is None:
            speed = self.base_speed
        self.set_motor_speeds(speed, speed)
        logger.debug(f"İleri: {speed}")
    
    def backward(self, speed: float = None):
        """
        Geri hareket
        
        Args:
            speed: Hız (0.0 ile 1.0 arası, None ise base_speed kullanılır)
        """
        if speed is None:
            speed = self.base_speed
        self.set_motor_speeds(-speed, -speed)
        logger.debug(f"Geri: {speed}")
    
    def left(self, speed: float = None, factor: float = None):
        """
        Sola dönüş
        
        Args:
            speed: Hız (0.0 ile 1.0 arası, None ise base_speed kullanılır)
            factor: Dönüş faktörü (None ise turning_factor kullanılır)
        """
        if speed is None:
            speed = self.base_speed
        if factor is None:
            factor = self.turning_factor
        
        left_speed = speed * (1.0 - factor * 2)
        right_speed = speed
        
        self.set_motor_speeds(left_speed, right_speed)
        logger.debug(f"Sol dönüş: {left_speed}/{right_speed}")
    
    def right(self, speed: float = None, factor: float = None):
        """
        Sağa dönüş
        
        Args:
            speed: Hız (0.0 ile 1.0 arası, None ise base_speed kullanılır)
            factor: Dönüş faktörü (None ise turning_factor kullanılır)
        """
        if speed is None:
            speed = self.base_speed
        if factor is None:
            factor = self.turning_factor
        
        left_speed = speed
        right_speed = speed * (1.0 - factor * 2)
        
        self.set_motor_speeds(left_speed, right_speed)
        logger.debug(f"Sağ dönüş: {left_speed}/{right_speed}")
    
    def stop(self):
        """Motorları durdur"""
        self.left_motor_in1.off()
        self.left_motor_in2.off()
        self.left_motor_pwm.value = 0
        
        self.right_motor_in1.off()
        self.right_motor_in2.off()
        self.right_motor_pwm.value = 0
        
        # Hız değerlerini sıfırla
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        self.filtered_left_speed = 0.0
        self.filtered_right_speed = 0.0
        self.is_moving = False
        
        logger.debug("Durduruldu")
    
    def pid_control(self, center_offset: float) -> Tuple[float, float]:
        """
        PID kontrol algoritması ile motor hızlarını hesaplar
        
        Args:
            center_offset: Merkez ofseti (-1.0 ile 1.0 arası)
            
        Returns:
            Tuple[float, float]: Sol ve sağ motor hızları
        """
        # PID katsayılarını güncelle
        now = time.time()
        if now - self.last_pid_update > self.pid_update_interval:
            # PID ince ayarı (gerekirse)
            logger.debug("PID parametreleri güncellendi")
            self.last_pid_update = now
        
        # Ölü bölgede ise düzgün git
        if abs(center_offset) < self.center_deadzone:
            center_offset = 0.0
        
        # Zaman farkını hesapla
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01
        self.last_time = now
        
        # PID bileşenlerini hesapla
        error = center_offset
        self.integral += error * dt
        
        # İntegral sınırlama
        self.integral = max(min(self.integral, self.pid_limits["ki_max"]), -self.pid_limits["ki_max"])
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # PID çıkışını hesapla
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        # P terimi sınırlama
        p_term = max(min(p_term, self.pid_limits["kp_max"]), -self.pid_limits["kp_max"])
        
        # D terimi sınırlama
        d_term = max(min(d_term, self.pid_limits["kd_max"]), -self.pid_limits["kd_max"])
        
        # PID çıkış değeri
        output = p_term + i_term + d_term
        
        # Temel hız
        base_speed = self.base_speed
        
        # Motor hızlarını hesapla
        left_speed = base_speed - output
        right_speed = base_speed + output
        
        # Hızları sınırla
        left_speed = max(min(left_speed, self.max_speed), self.min_speed)
        right_speed = max(min(right_speed, self.max_speed), self.min_speed)
        
        logger.debug(f"PID: error={error:.3f}, P={p_term:.3f}, I={i_term:.5f}, D={d_term:.3f}, output={output:.3f}")
        
        return left_speed, right_speed
    
    def calculate_center_diff(self, left_line, right_line, frame_width: int) -> float:
        """
        Şerit merkezinden araba merkezine olan farkı hesaplar
        
        Args:
            left_line: Sol şerit çizgisi
            right_line: Sağ şerit çizgisi
            frame_width: Görüntü genişliği
            
        Returns:
            float: Merkez farkı (-1.0 ile 1.0 arası)
        """
        image_center = frame_width / 2
        
        # Her iki şerit çizgisi de tespit edilmiş
        if left_line is not None and right_line is not None:
            # Alt noktaları al
            left_x = left_line[0]
            right_x = right_line[0]
            
            # Şerit genişliği kontrol
            lane_width = right_x - left_x
            
            # Şerit genişliği anormal ise (çok dar veya çok geniş)
            if lane_width < frame_width * 0.2 or lane_width > frame_width * 0.9:
                logger.warning(f"Şüpheli şerit genişliği: {lane_width} piksel")
                
                # Eğer tek şerit tespiti varmış gibi davran
                if lane_width < frame_width * 0.2:  # Çok dar
                    # Hangi şerit daha güvenilir?
                    if left_x < image_center and right_x < image_center:
                        # Her iki çizgi de solda, sağ çizgi muhtemelen yanlış
                        return self._calculate_from_single_line(left_line, frame_width, "left")
                    elif left_x > image_center and right_x > image_center:
                        # Her iki çizgi de sağda, sol çizgi muhtemelen yanlış
                        return self._calculate_from_single_line(right_line, frame_width, "right")
            
            # Şerit merkezi
            lane_center = (left_x + right_x) / 2
            
            # Merkez farkı hesapla ve normalize et
            center_diff = (lane_center - image_center) / image_center
            return center_diff
            
        # Sadece sol şerit çizgisi tespit edilmiş
        elif left_line is not None:
            return self._calculate_from_single_line(left_line, frame_width, "left")
            
        # Sadece sağ şerit çizgisi tespit edilmiş
        elif right_line is not None:
            return self._calculate_from_single_line(right_line, frame_width, "right")
            
        # Hiç şerit tespit edilmemiş
        else:
            return 0.0
    
    def _calculate_from_single_line(self, line, frame_width: int, line_type: str) -> float:
        """
        Tek bir şerit çizgisinden merkez farkını hesaplar
        
        Args:
            line: Şerit çizgisi
            frame_width: Görüntü genişliği
            line_type: Çizgi tipi ("left" veya "right")
            
        Returns:
            float: Merkez farkı (-1.0 ile 1.0 arası)
        """
        image_center = frame_width / 2
        x = line[0]  # Alt nokta
        
        # Tahmini şerit genişliği (görüntü genişliğinin %40'ı)
        estimated_lane_width = frame_width * 0.4
        
        if line_type == "left":
            # Sol şeritten tahmini merkezi hesapla
            lane_center = x + estimated_lane_width / 2
        else:  # "right"
            # Sağ şeritten tahmini merkezi hesapla
            lane_center = x - estimated_lane_width / 2
        
        # Merkez farkını hesapla ve normalize et
        center_diff = (lane_center - image_center) / image_center
        
        # Tek şerit tespitinde daha muhafazakar ol
        center_diff *= 0.8
        
        return center_diff
    
    def follow_lane(self, detection_result: Dict, crossing_detected: bool = False):
        """
        Şerit takibi yapar
        
        Args:
            detection_result: Şerit tespiti sonucu
            crossing_detected: Yaya geçidi tespit edildi mi
        """
        # Şerit takibini etkinleştir
        self.auto_mode = True
        
        # Geçit tespiti durumunda
        if crossing_detected:
            logger.info("Geçit tespit edildi, yavaşlıyor")
            self.forward(speed=self.base_speed * 0.5)
            time.sleep(0.5)  # Yavaşlama için kısa bir süre bekle
            return
        
        # Şerit tespiti durumunu kontrol et
        left_line = detection_result.get("left_line")
        right_line = detection_result.get("right_line")
        frame_width = detection_result.get("frame").shape[1]
        
        if left_line is not None or right_line is not None:
            # Şerit tespit edildi, normal sürüşe devam et
            
            # Şerit kaybı durumunu sıfırla
            if self.lane_lost:
                logger.info("Şerit tekrar tespit edildi")
                self.lane_lost = False
                self.recovery_mode = False
            
            # Merkez farkını hesapla
            center_diff = self.calculate_center_diff(left_line, right_line, frame_width)
            
            # PID kontrol ile motor hızlarını hesapla
            left_speed, right_speed = self.pid_control(center_diff)
            
            # Motorları ayarla
            self.set_motor_speeds(left_speed, right_speed)
            logger.debug(f"Şerit takibi - center_diff: {center_diff:.3f}, L: {left_speed:.2f}, R: {right_speed:.2f}")
        else:
            # Şerit tespit edilemedi
            if not self.lane_lost:
                self.lane_lost = True
                self.lane_lost_time = time.time()
                logger.warning("Şerit kaybedildi")
            
            # Ne kadar süredir şerit kaybı yaşanıyor
            lane_lost_duration = time.time() - self.lane_lost_time
            
            # Şerit kaybından sonra kurtarma stratejisi
            if lane_lost_duration < 0.5:
                # İlk 0.5 saniye boyunca aynı hızda devam et
                logger.debug("Şerit kaybı: Devam ediliyor")
            elif not self.recovery_mode:
                # Kurtarma modunu başlat
                self.recovery_mode = True
                self.recovery_start_time = time.time()
                logger.info("Kurtarma modu başlatıldı")
            
            if self.recovery_mode:
                # Son hatanın işaretine göre dönüş yönünü belirle
                recovery_time = time.time() - self.recovery_start_time
                recovery_turn_direction = 1 if self.prev_error > 0 else -1
                
                if recovery_time < 1.0:
                    # İlk 1 saniye için: hafif dönüş yap
                    turn_factor = 0.15 * recovery_turn_direction
                    left_speed = self.base_speed * (1 - turn_factor)
                    right_speed = self.base_speed * (1 + turn_factor)
                    self.set_motor_speeds(left_speed, right_speed)
                    logger.debug(f"Kurtarma modu: Hafif dönüş - L: {left_speed:.2f}, R: {right_speed:.2f}")
                elif recovery_time < 2.0:
                    # 1-2 saniye arası: biraz daha güçlü dönüş
                    turn_factor = 0.25 * recovery_turn_direction
                    left_speed = self.base_speed * (1 - turn_factor)
                    right_speed = self.base_speed * (1 + turn_factor)
                    self.set_motor_speeds(left_speed, right_speed)
                    logger.debug(f"Kurtarma modu: Güçlü dönüş - L: {left_speed:.2f}, R: {right_speed:.2f}")
                else:
                    # 2 saniyeden sonra: yavaşla ve dur
                    self.stop()
                    logger.warning("Şerit bulunamadı, durduruluyor")
    
    def cleanup(self):
        """Motorları temizle ve kapat"""
        try:
            self.stop()
            self.left_motor_in1.close()
            self.left_motor_in2.close()
            self.left_motor_pwm.close()
            
            self.right_motor_in1.close()
            self.right_motor_in2.close()
            self.right_motor_pwm.close()
            
            logger.info("Motor kontrolü kapatıldı ve temizlendi")
        except Exception as e:
            logger.error(f"Temizleme sırasında hata: {str(e)}")

# Test fonksiyonu
def test_motor_controller():
    """Motor kontrolünü test et"""
    try:
        motors = MotorController()
        
        print("İleri gidiyor...")
        motors.forward(speed=0.5)
        time.sleep(2)
        
        print("Sola dönüyor...")
        motors.left(speed=0.5)
        time.sleep(1)
        
        print("İleri gidiyor...")
        motors.forward(speed=0.5)
        time.sleep(1)
        
        print("Sağa dönüyor...")
        motors.right(speed=0.5)
        time.sleep(1)
        
        print("İleri gidiyor...")
        motors.forward(speed=0.5)
        time.sleep(1)
        
        print("Geri gidiyor...")
        motors.backward(speed=0.4)
        time.sleep(2)
        
        print("Duruyor...")
        motors.stop()
        
        # PID test
        print("\nPID test...")
        for offset in [-0.5, -0.25, 0, 0.25, 0.5]:
            left, right = motors.pid_control(offset)
            print(f"Offset: {offset}, Sol: {left:.2f}, Sağ: {right:.2f}")
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'motors' in locals():
            motors.cleanup()
            print("Motorlar temizlendi")

if __name__ == "__main__":
    print("Motor testi başlıyor...")
    test_motor_controller() 