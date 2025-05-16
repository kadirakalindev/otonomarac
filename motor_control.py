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
                 max_speed=0.7,               # Maksimum hız değeri (0-1 arası)
                 default_speed=0.35,          # Varsayılan hız
                 use_board_pins=True,         # BOARD pin numaralandırmasını kullan
                 pwm_frequency=25):           # PWM frekansı (Hz) - aşırı ısınmayı önlemek için düşürüldü
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
            
            # PWM pinleri - düşük frekans ile motor ısınmasını azalt
            self.left_pwm = PWMOutputDevice(left_en_bcm, frequency=pwm_frequency, initial_value=0)
            self.right_pwm = PWMOutputDevice(right_en_bcm, frequency=pwm_frequency, initial_value=0)
            
            logger.info(f"Motor pinleri başarıyla başlatıldı (PWM Frekansı: {pwm_frequency} Hz)")
            
        except Exception as e:
            logger.error(f"Motor kontrol başlatma hatası: {e}")
            self.cleanup()
            raise
        
        # Dinamik PID parametreleri
        self.pid_params = {
            "kp": 0.5,  # Orantısal katsayı
            "ki": 0.02, # İntegral katsayı
            "kd": 0.1   # Türev katsayı
        }
        
        # Otomatik PID ayarlama parametreleri
        self.auto_pid_enabled = True  # Otomatik PID ayarlama etkin/devre dışı
        self.pid_adaptation_rate = 0.05  # Adaptasyon hızı (düşük = daha yavaş değişim)
        self.pid_error_history = []   # Hata değerleri geçmişi
        self.pid_history_size = 50    # Hata geçmişi boyutu
        self.pid_min_values = {"kp": 0.2, "ki": 0.0, "kd": 0.0}  # Minimum değerler
        self.pid_max_values = {"kp": 1.0, "ki": 0.1, "kd": 0.5}  # Maksimum değerler
        
        # PID sınır değerleri
        self.max_integral = 50.0
        
        # Hız limitleri ve ramping parametreleri
        self.min_motor_speed = 0.2    # Minimum motor hızı
        self.max_speed_change = 0.05  # Bir adımda maksimum hız değişimi (ramping için azaltıldı)
        self.ramp_rate = 0.01         # Ramping hızı - daha yumuşak hızlanma/yavaşlama
        self.prev_left_speed = 0      # Başlangıçta durgun
        self.prev_right_speed = 0
        self.prev_left_direction = 'stop'  # Son yön bilgisi
        self.prev_right_direction = 'stop'
        
        # Deadband (ölü bölge) parametresi - çok küçük hatalar için motoru gereksiz yere çalıştırma
        self.error_deadband = 0.05  # Merkez hattından sapma değeri (normalized)
        
        # PID hesaplaması için gerekli değişkenler
        self.previous_error = 0
        self.integral = 0
        
        # Son zamanlama (dt için)
        self.last_time = time.time()
        
        # Şerit kaybı için sayaç
        self.lost_lane_counter = 0
        
        # Motor performans izleme
        self.motor_stats = {
            "left_direction_changes": 0,  # Yön değişimleri
            "right_direction_changes": 0,
            "max_error_seen": 0.0,        # Görülen maksimum hata
            "avg_error": 0.0,             # Ortalama hata
            "error_samples": 0            # Toplam örnek sayısı
        }
        
        logger.info("Motor kontrol modülü başlatıldı.")
    
    def _apply_ramping(self, current_speed, target_speed, motor_side):
        """
        Hız değişimini sınırlayan ramping mekanizması.
        
        Args:
            current_speed (float): Mevcut hız
            target_speed (float): Hedef hız
            motor_side (str): Motor tarafı ('left' veya 'right')
            
        Returns:
            float: Ramping uygulanmış yeni hız
        """
        # Hız değişim miktarını hesapla
        speed_diff = target_speed - current_speed
        
        # Eğer değişim max_speed_change değerinden büyükse, sınırla
        if abs(speed_diff) > self.max_speed_change:
            # Değişim yönünü koru, ancak miktarı sınırla
            if speed_diff > 0:
                new_speed = current_speed + self.max_speed_change
            else:
                new_speed = current_speed - self.max_speed_change
        else:
            new_speed = target_speed
            
        # Hız değerini sınırlandır
        new_speed = min(max(0, new_speed), self.max_speed)
        
        # Debug bilgisi
        if abs(speed_diff) > self.max_speed_change:
            logger.debug(f"{motor_side} motor ramping: {current_speed:.2f} -> {new_speed:.2f} (hedef: {target_speed:.2f})")
            
        return new_speed
    
    def _set_motor_speed(self, motor_side, direction, speed):
        """
        Belirtilen motoru belirtilen yön ve hızda çalıştırır.
        Ramping mekanizması ile hız değişimleri yumuşatılır.
        
        Args:
            motor_side (str): 'left' veya 'right'
            direction (str): 'forward', 'backward' veya 'stop'
            speed (float): Hız değeri (0-1 arası)
        """
        # Hız değerini sınırlandır
        speed = min(max(0, speed), self.max_speed)
        
        if motor_side == 'left':
            # Yön değişimi takibi
            if self.prev_left_direction != direction and direction != 'stop' and self.prev_left_direction != 'stop':
                self.motor_stats["left_direction_changes"] += 1
                logger.debug(f"Sol motor yön değişimi: {self.prev_left_direction} -> {direction}")
            
            # Ramping mekanizması uygula
            if direction == 'stop':
                ramped_speed = 0
            else:
                ramped_speed = self._apply_ramping(self.prev_left_speed, speed, "sol")
                
            # Hız ve yön bilgisini güncelle
            self.prev_left_speed = ramped_speed
            self.prev_left_direction = direction
            
            motor_in1 = self.left_motor_in1
            motor_in2 = self.left_motor_in2
            motor_pwm = self.left_pwm
            
        else:  # 'right'
            # Yön değişimi takibi
            if self.prev_right_direction != direction and direction != 'stop' and self.prev_right_direction != 'stop':
                self.motor_stats["right_direction_changes"] += 1
                logger.debug(f"Sağ motor yön değişimi: {self.prev_right_direction} -> {direction}")
            
            # Ramping mekanizması uygula
            if direction == 'stop':
                ramped_speed = 0
            else:
                ramped_speed = self._apply_ramping(self.prev_right_speed, speed, "sağ")
                
            # Hız ve yön bilgisini güncelle
            self.prev_right_speed = ramped_speed
            self.prev_right_direction = direction
            
            motor_in1 = self.right_motor_in1
            motor_in2 = self.right_motor_in2
            motor_pwm = self.right_pwm
        
        # Yön kontrolü
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
    
    def _update_pid_parameters(self, error, output):
        """
        PID parametrelerini performansa göre otomatik olarak ayarlar.
        
        Args:
            error (float): Mevcut hata değeri
            output (float): PID kontrolcüsünün çıktısı
        """
        if not self.auto_pid_enabled:
            return
            
        # Hata geçmişini güncelle
        self.pid_error_history.append(abs(error))
        if len(self.pid_error_history) > self.pid_history_size:
            self.pid_error_history.pop(0)
            
        # Ortalama mutlak hata ve standart sapma hesapla
        if len(self.pid_error_history) > 10:  # En az 10 örnek olsun
            avg_error = np.mean(self.pid_error_history)
            error_std = np.std(self.pid_error_history)
            
            # Performans metriklerini güncelle
            if abs(error) > self.motor_stats["max_error_seen"]:
                self.motor_stats["max_error_seen"] = abs(error)
                
            self.motor_stats["avg_error"] = ((self.motor_stats["avg_error"] * self.motor_stats["error_samples"]) + 
                                          abs(error)) / (self.motor_stats["error_samples"] + 1)
            self.motor_stats["error_samples"] += 1
            
            # PID parametrelerini optimize et
            
            # 1. Yüksek salınım (osilatör) durumunu tespit et
            oscillating = error_std > 0.15 and len(self.pid_error_history) > 20
            
            # 2. Zayıf tepki durumunu tespit et (error düzelmiyor)
            slow_response = avg_error > 0.2 and error_std < 0.05
            
            # 3. Aşırı tepki durumunu tespit et
            overreacting = abs(output) > 0.8 and abs(error) < 0.2
            
            # Salınım durumunda: Kp ve Kd'yi azalt, Ki'yi artır
            if oscillating:
                logger.debug("Salınım tespit edildi, Kp ve Kd azaltılıyor, Ki artırılıyor")
                self.pid_params["kp"] = max(self.pid_min_values["kp"],
                                           self.pid_params["kp"] * (1 - self.pid_adaptation_rate))
                self.pid_params["kd"] = max(self.pid_min_values["kd"],
                                           self.pid_params["kd"] * (1 - self.pid_adaptation_rate))
                self.pid_params["ki"] = min(self.pid_max_values["ki"],
                                           self.pid_params["ki"] * (1 + self.pid_adaptation_rate))
                                           
            # Zayıf tepki durumunda: Kp ve Ki'yi artır
            elif slow_response:
                logger.debug("Zayıf tepki tespit edildi, Kp ve Ki artırılıyor")
                self.pid_params["kp"] = min(self.pid_max_values["kp"],
                                           self.pid_params["kp"] * (1 + self.pid_adaptation_rate))
                self.pid_params["ki"] = min(self.pid_max_values["ki"],
                                           self.pid_params["ki"] * (1 + self.pid_adaptation_rate))
                                           
            # Aşırı tepki durumunda: Kp ve Ki'yi azalt, Kd'yi artır
            elif overreacting:
                logger.debug("Aşırı tepki tespit edildi, Kp ve Ki azaltılıyor, Kd artırılıyor")
                self.pid_params["kp"] = max(self.pid_min_values["kp"],
                                           self.pid_params["kp"] * (1 - self.pid_adaptation_rate))
                self.pid_params["ki"] = max(self.pid_min_values["ki"],
                                           self.pid_params["ki"] * (1 - self.pid_adaptation_rate))
                self.pid_params["kd"] = min(self.pid_max_values["kd"],
                                           self.pid_params["kd"] * (1 + self.pid_adaptation_rate))
            
            # PID parametrelerini log'a yaz (sadece değişiklik olduğunda)
            logger.debug(f"PID Parametreleri - Kp: {self.pid_params['kp']:.3f}, "
                       f"Ki: {self.pid_params['ki']:.3f}, Kd: {self.pid_params['kd']:.3f}")
    
    def pid_control(self, error, dt=None):
        """
        PID kontrol algoritması ile hata değerine göre düzeltme hesaplar.
        Otomatik parametre ayarlama özelliği eklenmiştir.
        
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
        if abs(error) < self.error_deadband:
            error = 0
            self.integral = 0  # Ölü bölgedeyken integrali sıfırla
        
        # PID bileşenlerini hesapla
        proportional = error * self.pid_params["kp"]
        
        # İntegral hesabı - dt ile doğru entegrasyon
        self.integral += error * dt * self.pid_params["ki"]
        
        # İntegral sınırlaması (anti-windup)
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        
        # Türev hesabı - daha doğru (zamana göre değişim)
        if dt > 0:
            derivative = (error - self.previous_error) / dt * self.pid_params["kd"]
            # Türev filtreleme (ani değişimleri yumuşat)
            derivative = max(-1.0, min(derivative, 1.0))
        else:
            derivative = 0
            
        # PID çıktısını hesapla
        output = proportional + self.integral + derivative
        
        # Çıktıyı sınırlandır
        output = max(-1.0, min(output, 1.0))
        
        # PID parametrelerini güncelle
        self._update_pid_parameters(error, output)
        
        # Önceki hatayı güncelle
        self.previous_error = error
        
        # Debugging
        logger.debug(f"PID - Hata: {error:.4f}, P: {proportional:.4f}, I: {self.integral:.4f}, D: {derivative:.4f}, Çıktı: {output:.4f}")
        
        return output

    def follow_lane(self, center_diff, speed=None):
        """
        Şerit takibi için araç kontrolü.
        İyileştirilmiş PID kontrolü ve ramping mekanizması ile smoother hareket.
        
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
        
        # Normalize sapma değeri (piksel cinsinden -> -1 ile 1 arasında)
        # Görüntü genişliğinin yarısı maksimum sapma olarak kabul edilir
        normalized_diff = center_diff / (self.default_speed * 200)  # Daha yumuşak normalize etme
        normalized_diff = max(-1.0, min(normalized_diff, 1.0))  # -1 ile 1 arasına sınırla
            
        # PID kontrol çıktısını hesapla
        correction = self.pid_control(normalized_diff)
        
        # Daha yumuşak dönüşler için düzeltme faktörünü azalt (aşırı ısınmayı azaltır)
        steering_factor = 0.3
        
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
        logger.debug(f"Şerit takibi - Sapma: {center_diff:.4f}, Norm: {normalized_diff:.2f}, Düzeltme: {correction:.4f}, Sol: {left_speed:.2f}, Sağ: {right_speed:.2f}")
    
    def get_motor_stats(self):
        """
        Motor performans istatistiklerini döndürür
        
        Returns:
            dict: Motor istatistikleri
        """
        stats = self.motor_stats.copy()
        # Mevcut PID parametrelerini ekle
        stats["pid_params"] = self.pid_params.copy()
        return stats
    
    def stop(self):
        """
        Aracı durdurur.
        """
        self._set_motor_speed('left', 'stop', 0)
        self._set_motor_speed('right', 'stop', 0)
        
        logger.debug("Araç durduruldu.")
    
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