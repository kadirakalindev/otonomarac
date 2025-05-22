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
                 max_speed=0.50,              # Maksimum hızı biraz daha düşürdük
                 default_speed=0.25,          # Varsayılan hızı düşürdük
                 use_board_pins=True,         # BOARD pin numaralandırmasını kullan
                 pwm_frequency=25):          # PWM frekansı
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
        # Pin numaralarını doğrula
        if not all(isinstance(pin, int) for pin in [*left_motor_pins, *right_motor_pins, left_pwm_pin, right_pwm_pin]):
            raise ValueError("Tüm pin değerleri tam sayı olmalıdır")
            
        # Hız değerlerini doğrula
        if not 0 < max_speed <= 1 or not 0 < default_speed <= max_speed:
            raise ValueError("Geçersiz hız değerleri")
            
        # PWM frekansını doğrula
        if not 20 <= pwm_frequency <= 100:
            raise ValueError("PWM frekansı 20-100 Hz aralığında olmalıdır")
            
        self.use_board_pins = use_board_pins
        self.max_speed = max_speed
        self.default_speed = default_speed
        self.min_motor_speed = 0.15  # Minimum çalışma hızı
        
        # Motor durumu izleme
        self.motor_states = {
            'left': {'running': False, 'speed': 0, 'direction': 'stop'},
            'right': {'running': False, 'speed': 0, 'direction': 'stop'}
        }
        
        # Güvenlik limitleri
        self.max_continuous_run_time = 300  # 5 dakika
        self.overheat_cooldown_time = 60    # 1 dakika
        self.last_cooldown_time = 0
        self.run_start_time = 0
        
        # Motor sağlığı izleme
        self.motor_health = {
            'left': {'errors': 0, 'last_error_time': 0},
            'right': {'errors': 0, 'last_error_time': 0}
        }
        self.max_motor_errors = 5
        self.error_reset_time = 60  # 1 dakika
        
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
        
        # Başlangıç motor faktörleri - Sol motor faktörü azaltıldı, sağ motor faktörü artırıldı
        self.left_motor_factor = 0.85  # Sol motor faktörü azaltıldı
        self.right_motor_factor = 1.0  # Sağ motor faktörü normal seviyede
        
        # PID kontrol parametreleri - viraj dönüşleri için optimize edildi
        self.kp = 0.5    # Orantısal katsayı - daha agresif tepki
        self.ki = 0.001  # İntegral katsayı
        self.kd = 0.25   # Türev katsayı - salınımları azaltmak için artırıldı
        
        # PID sınırları
        self.kp_min, self.kp_max = 0.4, 0.8
        self.ki_min, self.ki_max = 0.0005, 0.005
        self.kd_min, self.kd_max = 0.2, 0.4
        
        # Şerit takibi için parametreler
        self.center_deadzone = 0.03  # Merkez toleransı - daha hassas
        self.turn_speed_factor = 0.4  # Dönüş hızı faktörü - daha agresif dönüşler
        
        # Adaptif PID sistemi için parametreler
        self.enable_adaptive_pid = True
        self.performance_history = []
        self.history_max_length = 50
        self.error_history = []
        self.pid_update_interval = 1.5  # Güncelleme aralığı azaltıldı
        self.last_pid_update = time.time()  # PID güncelleme zamanı başlangıcı
        
        # PID ayarlama limitleri
        self.kp_min, self.kp_max = 0.3, 0.8
        self.ki_min, self.ki_max = 0.01, 0.1
        self.kd_min, self.kd_max = 0.05, 0.3
        
        # PID sınır değerleri
        self.max_integral = 50.0  # İntegral terimini daha fazla sınırla
        
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
        PID kontrolü hesaplar
        """
        # Hata geçmişini güncelle (Adaptif PID için)
        self.error_history.append(error)
        if len(self.error_history) > self.history_max_length:
            self.error_history.pop(0)
        
        # PID parametrelerini güncelle
        self._update_pid_parameters()
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt < 0.001:
            dt = 0.001
        
        # Çok küçük hatalar için düzeltme yapma
        if abs(error) < 0.01:
            self.integral = 0  # İntegral birikimini sıfırla
            return 0
        
        # PID bileşenlerini hesapla
        p_term = self.kp * error
        
        # İntegral terimi - sınırlı integral
        self.integral = max(-30, min(30, self.integral + error * dt))
        i_term = self.ki * self.integral
        
        # Türev terimi - ani değişimleri yumuşat
        derivative = (error - self.previous_error) / dt
        derivative = max(-50, min(50, derivative))  # Türevi sınırla
        d_term = self.kd * derivative
        
        # Toplam PID çıkışı
        output = p_term + i_term + d_term
        
        # PID çıkışını sınırla
        output = max(-1, min(1, output))
        
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
        
        # Hız farkı varsa faktörleri güncelle - daha yumuşak düzeltme
        if abs(left_avg - right_avg) > 0.02:  # Daha hassas fark tespiti
            if left_avg > right_avg:
                # Sol motor daha hızlıysa, sol motoru azalt
                self.left_motor_factor = max(0.8, self.left_motor_factor * 0.95)
                if self._should_log():
                    logger.info(f"Sol motor faktörü azaltıldı: {self.left_motor_factor:.2f}")
            else:
                # Sağ motor daha hızlıysa, sağ motoru azalt
                self.right_motor_factor = max(0.8, self.right_motor_factor * 0.95)
                if self._should_log():
                    logger.info(f"Sağ motor faktörü azaltıldı: {self.right_motor_factor:.2f}")
        
            # Geçmişi temizle ve yeni değerlerle başla
            self.motor_speed_history['left'].clear()
            self.motor_speed_history['right'].clear()
            
            # Faktör değişimini logla
            logger.info(f"Motor faktörleri - Sol: {self.left_motor_factor:.2f}, Sağ: {self.right_motor_factor:.2f}")

    def _check_motor_health(self, motor_side):
        """Motor sağlığını kontrol et"""
        current_time = time.time()
        motor = self.motor_health[motor_side]
        
        # Hata sayacını sıfırla
        if current_time - motor['last_error_time'] > self.error_reset_time:
            motor['errors'] = 0
            
        return motor['errors'] < self.max_motor_errors
        
    def _check_overheating(self):
        """Aşırı ısınma kontrolü"""
        if not self.run_start_time:
            self.run_start_time = time.time()
            return False
            
        current_time = time.time()
        run_time = current_time - self.run_start_time
        
        if run_time > self.max_continuous_run_time:
            if current_time - self.last_cooldown_time < self.overheat_cooldown_time:
                return True
                
            self.last_cooldown_time = current_time
            self.run_start_time = current_time
            
        return False
        
    def _set_motor_speed(self, motor_side, direction, speed):
        """Güvenli motor hız kontrolü"""
        # Aşırı ısınma kontrolü
        if self._check_overheating():
            logger.warning(f"Aşırı ısınma koruması aktif. Motor: {motor_side}")
            return False
            
        # Motor sağlığı kontrolü
        if not self._check_motor_health(motor_side):
            logger.error(f"Motor sağlık kontrolü başarısız: {motor_side}")
            return False
            
        try:
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
            
            # Hız geçmişini güncelle (tüm hareketler için)
            if direction != 'stop':
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
                ramped_speed = 0
            
            # Önceki değerleri güncelle
            if motor_side == 'left':
                self.prev_left_speed = ramped_speed
                self.prev_left_direction = direction
            else:  # 'right'
                self.prev_right_speed = ramped_speed
                self.prev_right_direction = direction
            
            # Motor durumunu güncelle
            self.motor_states[motor_side].update({
                'running': direction != 'stop',
                'speed': speed,
                'direction': direction
            })
            
            return True
            
        except Exception as e:
            # Hata durumunu kaydet
            self.motor_health[motor_side]['errors'] += 1
            self.motor_health[motor_side]['last_error_time'] = time.time()
            logger.error(f"Motor kontrol hatası ({motor_side}): {e}")
            return False
    
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
        Şerit takibi için araç kontrolü - PID ile geliştirilmiş versiyon
        """
        if speed is None:
            speed = self.default_speed
            
        speed = min(speed, self.max_speed)
        
        if center_diff is None:
            self.lost_lane_counter += 1
            if self.lost_lane_counter > 10:
                if self._should_log():
                    logger.warning(f"Şerit kaybedildi ({self.lost_lane_counter}), kurtarma modu aktif...")
                
                # Şerit kaybedildiğinde son bilinen yöne devam et
                recovery_speed = self.default_speed * 0.8  # Kurtarma modunda daha düşük hız
                
                if self.previous_error > 20:  # Önemli sağa sapma
                    # Son bilinen sapma sağa ise, sola dön (daha agresif)
                    left_speed = self.min_motor_speed * 0.6  # Daha yavaş
                    right_speed = recovery_speed * 1.3       # Daha hızlı
                    logger.debug("Kurtarma: Sola dönüş (agresif)")
                elif self.previous_error < -20:  # Önemli sola sapma
                    # Son bilinen sapma sola ise, sağa dön (daha agresif)
                    left_speed = recovery_speed * 1.3        # Daha hızlı
                    right_speed = self.min_motor_speed * 0.6 # Daha yavaş
                    logger.debug("Kurtarma: Sağa dönüş (agresif)")
                elif self.previous_error > 5:  # Hafif sağa sapma
                    # Hafif sola dön
                    left_speed = self.min_motor_speed * 0.8
                    right_speed = recovery_speed * 1.1
                    logger.debug("Kurtarma: Sola dönüş (hafif)")
                elif self.previous_error < -5:  # Hafif sola sapma
                    # Hafif sağa dön
                    left_speed = recovery_speed * 1.1
                    right_speed = self.min_motor_speed * 0.8
                    logger.debug("Kurtarma: Sağa dönüş (hafif)")
                else:
                    # Sapma yoksa, zigzag arama modeli uygula
                    # Belirli aralıklarla yön değiştirerek şeridi bulmaya çalış
                    search_cycle = (self.lost_lane_counter - 10) % 30  # Daha kısa döngü
                    if search_cycle < 10:  # İlk 10 adım sola dön
                        left_speed = self.min_motor_speed * 0.7
                        right_speed = recovery_speed * 1.2
                        logger.debug("Kurtarma: Arama - sola dönüş")
                    elif search_cycle < 20:  # Sonraki 10 adım sağa dön
                        left_speed = recovery_speed * 1.2
                        right_speed = self.min_motor_speed * 0.7
                        logger.debug("Kurtarma: Arama - sağa dönüş")
                    else:  # Son 10 adım düz git
                        left_speed = recovery_speed * 0.9
                        right_speed = recovery_speed * 0.9
                        logger.debug("Kurtarma: Arama - düz git")
                
                # Uzun süre şerit bulunamazsa hızı kademeli olarak azalt
                if self.lost_lane_counter > 30:  # Daha kısa sürede yavaşla
                    slowdown_factor = max(0.5, 1.0 - (self.lost_lane_counter - 30) / 50)
                    left_speed *= slowdown_factor
                    right_speed *= slowdown_factor
                    
                    # Çok uzun süre şerit bulunamazsa dur (güvenlik önlemi)
                    if self.lost_lane_counter > 100:  # Daha kısa sürede dur
                        logger.warning("Şerit çok uzun süre bulunamadı, durduruluyor!")
                        self.stop()
                        return
                
                # Motorları çalıştır
                self._set_motor_speed('left', 'forward', left_speed)
                self._set_motor_speed('right', 'forward', right_speed)
                return
            return
        else:
            self.lost_lane_counter = 0
        
        # PID düzeltmesini hesapla
        pid_correction = self.calculate_pid(center_diff)
        
        # Temel hızları ayarla - sol motor daha düşük hız, sağ motor daha yüksek hız
        base_left_speed = speed * 0.85  # Sol motor daha düşük
        base_right_speed = speed * 0.9  # Sağ motor da hafif düşük
        
        # Virajlarda daha agresif dönüş için sapma miktarına göre ek düzeltme faktörü
        extra_correction = 0
        if abs(center_diff) > 50:  # Büyük sapmalar için
            extra_correction = 0.2  # Daha agresif
        elif abs(center_diff) > 30:  # Orta sapmalar için
            extra_correction = 0.15
        elif abs(center_diff) > 15:  # Küçük sapmalar için
            extra_correction = 0.1
        
        # PID düzeltmesini uygula
        if abs(center_diff) < self.center_deadzone:
            # Merkeze çok yakınsa sol motoru daha yavaş, sağ motoru daha hızlı tut
            left_speed = base_left_speed
            right_speed = base_right_speed
        else:
            # PID düzeltmesini hızlara uygula
            correction = (pid_correction * self.turn_speed_factor) + extra_correction
            
            if center_diff < 0:  # Sola dönüş gerekiyor
                left_speed = base_left_speed * (1 - abs(correction) * 1.5)  # Sol motor daha yavaş
                right_speed = base_right_speed * (1 + abs(correction) * 1.2)  # Sağ motor daha hızlı
            else:  # Sağa dönüş gerekiyor
                left_speed = base_left_speed * (1 + abs(correction) * 1.2)  # Sol motor daha hızlı
                right_speed = base_right_speed * (1 - abs(correction) * 1.5)  # Sağ motor daha yavaş
        
        # Hızları sınırla
        left_speed = max(min(left_speed, self.max_speed), self.min_motor_speed)
        right_speed = max(min(right_speed, self.max_speed), self.min_motor_speed)
        
        # Motor faktörlerini güncelle ve uygula
        self._update_motor_factors()
        
        # Motorları çalıştır
        self._set_motor_speed('left', 'forward', left_speed)
        self._set_motor_speed('right', 'forward', right_speed)
        
        # Debug loglarını sınırla
        self.debug_log_counter += 1
        if self.debug_log_counter >= self.log_threshold:
            self.debug_log_counter = 0
            logger.debug(f"Şerit takibi - Sapma: {center_diff:.2f}, PID: {pid_correction:.2f}, Sol: {left_speed:.2f}, Sağ: {right_speed:.2f}")
    
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