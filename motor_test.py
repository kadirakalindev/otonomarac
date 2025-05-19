#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Otonom Araç - Motor Test Programı
Bu program, motorların doğru çalışıp çalışmadığını test etmek için kullanılır.
Her motor sırayla ileri ve geri yönde çalıştırılır.
"""

import time
import sys
import argparse
import RPi.GPIO as GPIO
import signal

class MotorTest:
    def __init__(self, left_pins=(17, 27), right_pins=(22, 23), pwm_pins=(13, 12), frequency=100):
        """Motor test sınıfını başlatır"""
        self.left_forward_pin, self.left_backward_pin = left_pins
        self.right_forward_pin, self.right_backward_pin = right_pins
        self.left_pwm_pin, self.right_pwm_pin = pwm_pins
        self.frequency = frequency
        
        self.left_pwm = None
        self.right_pwm = None
        
        # GPIO'yu başlat
        self._initialize_gpio()
        
        # Ctrl+C sinyalini yakala
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Ctrl+C sinyalini yakalar ve temiz bir şekilde çıkış yapar"""
        print("\nProgram sonlandırılıyor...")
        self.cleanup()
        sys.exit(0)
    
    def _initialize_gpio(self):
        """GPIO pinlerini başlatır"""
        # GPIO modunu ayarla
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Pin modlarını ayarla
        GPIO.setup(self.left_forward_pin, GPIO.OUT)
        GPIO.setup(self.left_backward_pin, GPIO.OUT)
        GPIO.setup(self.right_forward_pin, GPIO.OUT)
        GPIO.setup(self.right_backward_pin, GPIO.OUT)
        GPIO.setup(self.left_pwm_pin, GPIO.OUT)
        GPIO.setup(self.right_pwm_pin, GPIO.OUT)
        
        # PWM nesnelerini oluştur
        self.left_pwm = GPIO.PWM(self.left_pwm_pin, self.frequency)
        self.right_pwm = GPIO.PWM(self.right_pwm_pin, self.frequency)
        
        # PWM'leri başlat (0% duty cycle ile)
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        # Tüm çıkışları sıfırla
        self._stop_all()
        
        print("GPIO başlatıldı.")
    
    def _stop_all(self):
        """Tüm motor çıkışlarını durdurur"""
        GPIO.output(self.left_forward_pin, GPIO.LOW)
        GPIO.output(self.left_backward_pin, GPIO.LOW)
        GPIO.output(self.right_forward_pin, GPIO.LOW)
        GPIO.output(self.right_backward_pin, GPIO.LOW)
        self.left_pwm.ChangeDutyCycle(0)
        self.right_pwm.ChangeDutyCycle(0)
    
    def _set_left_motor(self, direction, speed):
        """Sol motoru ayarlar
        
        Args:
            direction: Yön (1: ileri, -1: geri, 0: dur)
            speed: Hız (0-100)
        """
        if direction == 1:  # İleri
            GPIO.output(self.left_forward_pin, GPIO.HIGH)
            GPIO.output(self.left_backward_pin, GPIO.LOW)
        elif direction == -1:  # Geri
            GPIO.output(self.left_forward_pin, GPIO.LOW)
            GPIO.output(self.left_backward_pin, GPIO.HIGH)
        else:  # Dur
            GPIO.output(self.left_forward_pin, GPIO.LOW)
            GPIO.output(self.left_backward_pin, GPIO.LOW)
        
        self.left_pwm.ChangeDutyCycle(speed)
    
    def _set_right_motor(self, direction, speed):
        """Sağ motoru ayarlar
        
        Args:
            direction: Yön (1: ileri, -1: geri, 0: dur)
            speed: Hız (0-100)
        """
        if direction == 1:  # İleri
            GPIO.output(self.right_forward_pin, GPIO.HIGH)
            GPIO.output(self.right_backward_pin, GPIO.LOW)
        elif direction == -1:  # Geri
            GPIO.output(self.right_forward_pin, GPIO.LOW)
            GPIO.output(self.right_backward_pin, GPIO.HIGH)
        else:  # Dur
            GPIO.output(self.right_forward_pin, GPIO.LOW)
            GPIO.output(self.right_backward_pin, GPIO.LOW)
        
        self.right_pwm.ChangeDutyCycle(speed)
    
    def test_motors(self, speed=50, duration=2.0):
        """Motorları test eder
        
        Args:
            speed: Test hızı (0-100)
            duration: Her test adımının süresi (saniye)
        """
        try:
            print("\nMotor testi başlatılıyor...")
            print(f"Test hızı: %{speed}, Süre: {duration} saniye")
            
            # Sol motor ileri
            print("\n1. Sol motor ileri")
            self._set_left_motor(1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Sol motor geri
            print("2. Sol motor geri")
            self._set_left_motor(-1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Sağ motor ileri
            print("3. Sağ motor ileri")
            self._set_right_motor(1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Sağ motor geri
            print("4. Sağ motor geri")
            self._set_right_motor(-1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Her iki motor ileri
            print("5. Her iki motor ileri")
            self._set_left_motor(1, speed)
            self._set_right_motor(1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Her iki motor geri
            print("6. Her iki motor geri")
            self._set_left_motor(-1, speed)
            self._set_right_motor(-1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Sol ileri, sağ geri (yerinde dönüş)
            print("7. Sol ileri, sağ geri (yerinde sola dönüş)")
            self._set_left_motor(1, speed)
            self._set_right_motor(-1, speed)
            time.sleep(duration)
            self._stop_all()
            
            # Sol geri, sağ ileri (yerinde dönüş)
            print("8. Sol geri, sağ ileri (yerinde sağa dönüş)")
            self._set_left_motor(-1, speed)
            self._set_right_motor(1, speed)
            time.sleep(duration)
            self._stop_all()
            
            print("\nMotor testi tamamlandı.")
            
        except Exception as e:
            print(f"Hata: {e}")
        finally:
            self._stop_all()
    
    def interactive_test(self):
        """Etkileşimli motor testi"""
        print("\nEtkileşimli motor testi başlatılıyor...")
        print("Kontroller:")
        print("  w: İleri")
        print("  s: Geri")
        print("  a: Sol")
        print("  d: Sağ")
        print("  q: Sol yerinde dönüş")
        print("  e: Sağ yerinde dönüş")
        print("  x: Dur")
        print("  +: Hızı artır")
        print("  -: Hızı azalt")
        print("  0-9: Hızı ayarla (0-90%)")
        print("  ESC: Çıkış")
        
        import msvcrt  # Windows için tuş okuma
        
        speed = 50
        
        try:
            while True:
                print(f"\rHız: %{speed} | Komut bekliyor...", end="")
                
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    
                    if key == '\x1b':  # ESC
                        break
                    elif key == 'w':  # İleri
                        print(f"\rİleri (%{speed})           ")
                        self._set_left_motor(1, speed)
                        self._set_right_motor(1, speed)
                    elif key == 's':  # Geri
                        print(f"\rGeri (%{speed})            ")
                        self._set_left_motor(-1, speed)
                        self._set_right_motor(-1, speed)
                    elif key == 'a':  # Sol
                        print(f"\rSol (%{speed})             ")
                        self._set_left_motor(0, 0)
                        self._set_right_motor(1, speed)
                    elif key == 'd':  # Sağ
                        print(f"\rSağ (%{speed})             ")
                        self._set_left_motor(1, speed)
                        self._set_right_motor(0, 0)
                    elif key == 'q':  # Sol yerinde dönüş
                        print(f"\rSol yerinde dönüş (%{speed})")
                        self._set_left_motor(-1, speed)
                        self._set_right_motor(1, speed)
                    elif key == 'e':  # Sağ yerinde dönüş
                        print(f"\rSağ yerinde dönüş (%{speed})")
                        self._set_left_motor(1, speed)
                        self._set_right_motor(-1, speed)
                    elif key == 'x':  # Dur
                        print("\rDur                      ")
                        self._stop_all()
                    elif key == '+':  # Hızı artır
                        speed = min(speed + 10, 100)
                        print(f"\rHız: %{speed}              ")
                    elif key == '-':  # Hızı azalt
                        speed = max(speed - 10, 10)
                        print(f"\rHız: %{speed}              ")
                    elif key.isdigit():  # Hızı ayarla
                        speed = int(key) * 10
                        if speed == 0:
                            speed = 10
                        print(f"\rHız: %{speed}              ")
                
                time.sleep(0.1)
                
        except Exception as e:
            print(f"\nHata: {e}")
        finally:
            self._stop_all()
            print("\nEtkileşimli test sonlandırıldı.")
    
    def cleanup(self):
        """GPIO kaynaklarını temizler"""
        self._stop_all()
        if self.left_pwm:
            self.left_pwm.stop()
        if self.right_pwm:
            self.right_pwm.stop()
        GPIO.cleanup()
        print("GPIO temizlendi.")

def main():
    """Ana program"""
    parser = argparse.ArgumentParser(description="Otonom Araç Motor Test Programı")
    parser.add_argument("--speed", type=int, default=50, help="Motor test hızı (0-100)")
    parser.add_argument("--duration", type=float, default=2.0, help="Her test adımının süresi (saniye)")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli test modunu aktifleştir")
    parser.add_argument("--left-pins", type=int, nargs=2, default=[17, 27], help="Sol motor pin numaraları (ileri geri)")
    parser.add_argument("--right-pins", type=int, nargs=2, default=[22, 23], help="Sağ motor pin numaraları (ileri geri)")
    parser.add_argument("--pwm-pins", type=int, nargs=2, default=[13, 12], help="PWM pin numaraları (sol sağ)")
    args = parser.parse_args()
    
    try:
        # GPIO pinlerini ayarla
        tester = MotorTest(
            left_pins=tuple(args.left_pins),
            right_pins=tuple(args.right_pins),
            pwm_pins=tuple(args.pwm_pins)
        )
        
        # Test modunu seç
        if args.interactive:
            tester.interactive_test()
        else:
            tester.test_motors(speed=args.speed, duration=args.duration)
        
        # Temizlik
        tester.cleanup()
        
    except Exception as e:
        print(f"Hata: {e}")
        # GPIO'yu temizlemeye çalış
        try:
            GPIO.cleanup()
        except:
            pass

if __name__ == "__main__":
    print("\nOTONOM ARAÇ MOTOR TEST PROGRAMI")
    print("------------------------------")
    
    # Platformu kontrol et
    try:
        import RPi.GPIO
        print("Raspberry Pi GPIO modülü bulundu.")
    except ImportError:
        print("UYARI: RPi.GPIO modülü bulunamadı!")
        print("Bu program bir Raspberry Pi üzerinde çalıştırılmalıdır.")
        print("Simülasyon modunda devam edilecek.")
        
        # RPi.GPIO modülünü simüle et
        class GPIO:
            BCM = "BCM"
            OUT = "OUT"
            IN = "IN"
            HIGH = 1
            LOW = 0
            
            @staticmethod
            def setmode(mode):
                print(f"GPIO.setmode({mode})")
            
            @staticmethod
            def setwarnings(flag):
                print(f"GPIO.setwarnings({flag})")
            
            @staticmethod
            def setup(pin, mode):
                print(f"GPIO.setup({pin}, {mode})")
            
            @staticmethod
            def output(pin, value):
                print(f"GPIO.output({pin}, {value})")
            
            @staticmethod
            def cleanup():
                print("GPIO.cleanup()")
            
            class PWM:
                def __init__(self, pin, freq):
                    self.pin = pin
                    self.freq = freq
                    print(f"PWM({pin}, {freq})")
                
                def start(self, dc):
                    print(f"PWM.start({dc})")
                
                def ChangeDutyCycle(self, dc):
                    print(f"PWM.ChangeDutyCycle({dc})")
                
                def stop(self):
                    print("PWM.stop()")
        
        # GPIO modülünü değiştir
        sys.modules['RPi.GPIO'] = GPIO
    
    main() 