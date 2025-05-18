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
                 pwm_frequency=25):          # PWM frekansı (Hz) - daha da azaltıldı (50Hz->25Hz)
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
            pwm_frequency (int): PWM frekansı (Hz) - aşırı ısınmayı önlemek için düşürüldü
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
        
        # Başlangıç PID kontrol parametreleri
        self.kp = 0.5   # Orantısal katsayı - azaltıldı, daha yumuşak tepki için
        self.ki = 0.02  # Integral katsayı - azaltıldı
        self.kd = 0.1   # Türev katsayı - azaltıldı, daha az salınım için
        
        # Adaptif PID sistemi için parametreler
        self.enable_adaptive_pid = True  # Adaptif PID sistemini etkinleştir
        self.performance_history = []    # Performans geçmişini tutacak liste
        self.history_max_length = 50     # Maksimum geçmiş uzunluğu
        self.error_history = []          # Hata geçmişi
        self.pid_update_interval = 2.0   # PID güncelleme aralığı (saniye)
        self.last_pid_update = time.time()
        
        # PID ayarlama limitleri
        self.kp_min, self.kp_max = 0.3, 0.8
        self.ki_min, self.ki_max = 0.01, 0.1
        self.kd_min, self.kd_max = 0.05, 0.3
        
        # PID sınır değerleri
        self.max_integral = 50.0  # İntegral terimini daha fazla sınırla
        
        # Hız limitleri
        self.min_motor_speed = 0.2  # Minimum motor hızı (durma noktasını önler)
        
        # Geliştirilmiş ramping (hız yumuşatma) mekanizması
        # Daha yavaş ve yumuşak hız değişimi
        self.max_accel = 0.05     # Hızlanma sınırı (bir adımda maksimum artış) - azaltıldı
        self.max_decel = 0.08     # Yavaşlama sınırı (bir adımda maksimum azalış) - nispeten daha hızlı
        self.prev_left_speed = 0.0  # Başlangıçta durgun
        self.prev_right_speed = 0.0 # Başlangıçta durgun
        self.prev_left_direction = 'stop'  # Başlangıçta durgun
        self.prev_right_direction = 'stop' # Başlangıçta durgun
        
        # Yön değişimi zamanlaması
        self.direction_change_delay = 0.1  # Yön değiştirme sırasında kısa bir bekleme (sn)
        self.last_direction_change = 0     # Son yön değişikliği zamanı
        
        # PID hesaplaması için gerekli değişkenler
        self.previous_error = 0
        self.integral = 0
        
        # Son zamanlama (dt için)
        self.last_time = time.time()
        
        # Şerit kaybı için sayaç
        self.lost_lane_counter = 0
        
        # Motor dengeleme faktörleri
        self.left_motor_factor = 1.0  # Sol motor hız düzeltme faktörü
        self.right_motor_factor = 1.0 # Sağ motor hız düzeltme faktörü
        self.motor_calibration_count = 0
        self.motor_speed_history = {'left': [], 'right': []}
        self.max_history_length = 50
        self.calibration_threshold = 20  # Kaç örnek toplandıktan sonra kalibrasyon yapılacak
        
        # Loglama kontrolü için değişkenler
        self.last_log_time = time.time()
        self.log_interval = 1.0  # Saniyede bir log
        self.debug_log_counter = 0
        self.log_threshold = 30  # Her 30 işlemde bir debug log
        
        logger.info("Motor kontrol modülü başlatıldı.")
        logger.info(f"Adaptif PID: {self.enable_adaptive_pid}, Başlangıç PID: kp={self.kp}, ki={self.ki}, kd={self.kd}")
    
    def _should_log(self):
        """
        Loglama yapılıp yapılmayacağını kontrol eder
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False

    def _update_pid_parameters(self):
        """
        PID parametrelerini günceller
        """
        if not self.enable_adaptive_pid or len(self.error_history) < 10:
            return
        
        current_time = time.time()
        if current_time - self.last_pid_update < self.pid_update_interval:
            return
        
        self.last_pid_update = current_time
        
        # Son hataların analizi
        recent_errors = np.array(self.error_history[-20:])
        error_mean = np.mean(np.abs(recent_errors))
        error_var = np.var(recent_errors)
        error_trend = np.mean(np.diff(recent_errors[-10:]))
        
        old_kp, old_ki, old_kd = self.kp, self.ki, self.kd
        
        # Hata ortalaması yüksekse, kontrol yanıtını güçlendir
        if error_mean > 0.5:
            self.kp = min(self.kp * 1.05, self.kp_max)  # P terimini artır (daha agresif yanıt)
            self.kd = min(self.kd * 1.05, self.kd_max)  # D terimini artır (salınımı kontrol etmek için)
        else:
            # Hata ortalaması düşükse, yanıtı yumuşat
            self.kp = max(self.kp * 0.98, self.kp_min)  # P terimini azalt (daha yumuşak yanıt)
        
        # Varyans yüksekse (salınımlar), D terimini artır, P terimini azalt
        if error_var > 0.3:
            self.kp = max(self.kp * 0.95, self.kp_min)  # P terimini azalt
            self.kd = min(self.kd * 1.1, self.kd_max)   # D terimini artır (daha iyi salınım kontrolü)
        
        # Trend (yön) negatifse, I terimini biraz artır
        if error_trend < -0.05:
            self.ki = min(self.ki * 1.05, self.ki_max)  # I terimini artır
        elif error_trend > 0.05:
            self.ki = max(self.ki * 0.95, self.ki_min)  # I terimini azalt
        
        # Sadece parametreler değiştiğinde ve log zamanı geldiyse logla
        if (old_kp != self.kp or old_ki != self.ki or old_kd != self.kd) and self._should_log():
            logger.info(f"PID güncellendi: kp={self.kp:.3f}, ki={self.ki:.3f}, kd={self.kd:.3f}")
            
    def calculate_pid(self, error):
        """
        PID kontrolü hesaplar. Hata değerine göre motorlara uygulanacak hız düzeltmesini döndürür.
        
        Args:
            error (float): Orta çizgiden sapma hatası (negatif: sola sapma, pozitif: sağa sapma)
        
        Returns:
            float: PID çıkışı (motor hız düzeltme değeri)
        """
        # Hata geçmişini güncelle
        self.error_history.append(error)
        if len(self.error_history) > self.history_max_length:
            self.error_history.pop(0)
            
        # PID parametrelerini gerekirse güncelle
        self._update_pid_parameters()
        
        # Zaman farkını hesapla
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # dt sıfırsa veya çok küçükse
        if dt < 0.001:
            dt = 0.001
            
        # Deadband - çok küçük hatalar için düzeltme yapma
        if abs(error) < 0.05:
            return 0
        
        # PID bileşenlerini hesapla
        # Orantısal terim
        p_term = self.kp * error
        
        # İntegral terimi
        self.integral += error * dt
        # İntegral terimini sınırla (anti-windup)
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        i_term = self.ki * self.integral
        
        # Türev terimi
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Toplam PID çıkışı
        output = p_term + i_term + d_term
        
        # Bir sonraki hesaplama için hatayı kaydet
        self.previous_error = error
        
        return output
    
    def _update_motor_factors(self):
        """
        Motor hız faktörlerini günceller
        """
        if len(self.motor_speed_history['left']) < self.calibration_threshold:
            return

        # Son hızların ortalamasını al
        left_avg = sum(self.motor_speed_history['left']) / len(self.motor_speed_history['left'])
        right_avg = sum(self.motor_speed_history['right']) / len(self.motor_speed_history['right'])
        
        # Hız farkı varsa faktörleri güncelle
        if abs(left_avg - right_avg) > 0.05:  # %5'ten fazla fark varsa
            if left_avg < right_avg:
                self.left_motor_factor = min(1.5, self.left_motor_factor * 1.05)
                if self._should_log():
                    logger.info(f"Sol motor faktörü: {self.left_motor_factor:.2f}")
            else:
                self.right_motor_factor = min(1.5, self.right_motor_factor * 1.05)
                if self._should_log():
                    logger.info(f"Sağ motor faktörü: {self.right_motor_factor:.2f}")
            
            self.motor_speed_history['left'].clear()
            self.motor_speed_history['right'].clear()

    def _set_motor_speed(self, motor_side, direction, speed):
        """
        Belirtilen motoru belirtilen yön ve hızda çalıştırır.
        Motor dengeleme faktörlerini uygular.
        
        Args:
            motor_side (str): 'left' veya 'right'
            direction (str): 'forward', 'backward' veya 'stop'
            speed (float): Hız değeri (0-1 arası)
        """
        # Hız değerini sınırlandır
        speed = min(max(0, speed), self.max_speed)
        
        # Motor faktörlerini uygula
        if motor_side == 'left':
            speed = speed * self.left_motor_factor
            prev_speed = self.prev_left_speed
            prev_direction = self.prev_left_direction
            motor_in1 = self.left_motor_in1
            motor_in2 = self.left_motor_in2
            motor_pwm = self.left_pwm
        else:  # 'right'
            speed = speed * self.right_motor_factor
            prev_speed = self.prev_right_speed
            prev_direction = self.prev_right_direction
            motor_in1 = self.right_motor_in1
            motor_in2 = self.right_motor_in2
            motor_pwm = self.right_pwm
        
        # Hız geçmişini güncelle
        if direction == 'forward':
            self.motor_speed_history[motor_side].append(speed)
            if len(self.motor_speed_history[motor_side]) > self.max_history_length:
                self.motor_speed_history[motor_side].pop(0)
        
        # Yön değişimi tespiti
        direction_changed = prev_direction != direction and prev_direction != 'stop' and direction != 'stop'
        
        # Yön değişimlerinde ani geçişleri önle
        if direction_changed:
            current_time = time.time()
            # İlk önce motoru durdur
            motor_in1.off()
            motor_in2.off()
            motor_pwm.value = 0
            
            # Motorun elektromekanik olarak durması için kısa bir bekleme
            elapsed = current_time - self.last_direction_change
            if elapsed < self.direction_change_delay:
                time.sleep(self.direction_change_delay - elapsed)
            
            self.last_direction_change = time.time()
        
        # Ramping mekanizması - hedef hıza kademeli olarak geçiş
        target_speed = speed
        
        # Hız artıyor mu, azalıyor mu?
        if target_speed > prev_speed:
            # Hızlanma durumu - daha yavaş geçiş
            ramped_speed = min(target_speed, prev_speed + self.max_accel)
        else:
            # Yavaşlama durumu - daha hızlı geçiş
            ramped_speed = max(target_speed, prev_speed - self.max_decel)
        
        # Yönü ve ramping uygulanmış hızı ayarla
        if direction == 'forward':
            motor_in1.on()
            motor_in2.off()
            motor_pwm.value = ramped_speed
        elif direction == 'backward':
            motor_in1.off()
            motor_in2.on()
            motor_pwm.value = ramped_speed
        else:  # 'stop'
            motor_in1.off()
            motor_in2.off()
            motor_pwm.value = 0
            ramped_speed = 0  # Duruş durumunda hız sıfır
        
        # Önceki değerleri güncelle
        if motor_side == 'left':
            self.prev_left_speed = ramped_speed
            self.prev_left_direction = direction
        else:  # 'right'
            self.prev_right_speed = ramped_speed
            self.prev_right_direction = direction
        
        # Debug log
        if ramped_speed != speed:
            logger.debug(f"{motor_side.capitalize()} motor ramping: {prev_speed:.2f} -> {ramped_speed:.2f} (hedef: {speed:.2f})")
    
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
    
    def follow_lane(self, center_diff, speed=None):
        """
        Şerit takibi için araç kontrolü
        """
        if speed is None:
            speed = self.default_speed
            
        speed = min(speed, self.max_speed)
        
        if center_diff is None:
            self.lost_lane_counter += 1
            
            if self.lost_lane_counter > 10:
                if self._should_log():
                    logger.warning("Şerit kaybedildi, durduruluyor...")
                self.stop()
                return
            return
        else:
            self.lost_lane_counter = 0
            
        # PID kontrol çıktısını hesapla
        correction = self.calculate_pid(center_diff)
        
        # Daha yumuşak dönüşler için düzeltme faktörünü ayarla
        steering_factor = 0.4
        
        # Motor hızlarını hesapla - artık eşit temel hızlar kullanıyoruz
        base_speed = speed * 0.9  # Temel hız
        turn_speed = speed * 0.25  # Dönüş hızı
        
        # Sapma değerine göre motor hızlarını ayarla
        if abs(center_diff) < 0.15:
            left_speed = base_speed
            right_speed = base_speed
        else:
            if center_diff < 0:  # Sola dön
                left_speed = base_speed - (abs(center_diff) * turn_speed)
                right_speed = base_speed + (abs(center_diff) * turn_speed)
            else:  # Sağa dön
                left_speed = base_speed + (abs(center_diff) * turn_speed)
                right_speed = base_speed - (abs(center_diff) * turn_speed)
        
        # Hızları sınırla
        left_speed = max(min(left_speed, self.max_speed), self.min_motor_speed)
        right_speed = max(min(right_speed, self.max_speed), self.min_motor_speed)
        
        # Motor faktörlerini güncelle
        self._update_motor_factors()
        
        # Motorları çalıştır
        self._set_motor_speed('left', 'forward', left_speed)
        self._set_motor_speed('right', 'forward', right_speed)
        
        # Debug loglarını sınırla
        self.debug_log_counter += 1
        if self.debug_log_counter >= self.log_threshold:
            self.debug_log_counter = 0
            logger.debug(f"Şerit takibi - Sapma: {center_diff:.2f}, Sol: {left_speed:.2f}, Sağ: {right_speed:.2f}")
    
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