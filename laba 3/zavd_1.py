import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Завдання 1: Система керування кранами гарячої і холодної води

class WaterMixerFuzzySystem:
    def __init__(self):
        # Створення вхідних змінних
        self.temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
        self.pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'pressure')
        
        # Створення вихідних змінних
        self.hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')
        self.cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')
        
        self.setup_membership_functions()
        self.create_rules()
        self.create_control_system()
    
    def setup_membership_functions(self):
        """Налаштування функцій належності"""
        
        # Функції належності для температури
        self.temperature['cold'] = fuzz.trimf(self.temperature.universe, [0, 0, 30])
        self.temperature['cool'] = fuzz.trimf(self.temperature.universe, [10, 30, 50])
        self.temperature['warm'] = fuzz.trimf(self.temperature.universe, [30, 50, 70])
        self.temperature['not_very_hot'] = fuzz.trimf(self.temperature.universe, [50, 70, 90])
        self.temperature['hot'] = fuzz.trimf(self.temperature.universe, [70, 100, 100])
        
        # Функції належності для тиску
        self.pressure['weak'] = fuzz.trimf(self.pressure.universe, [0, 0, 40])
        self.pressure['not_very_strong'] = fuzz.trimf(self.pressure.universe, [20, 40, 60])
        self.pressure['strong'] = fuzz.trimf(self.pressure.universe, [40, 100, 100])
        
        # Функції належності для крану гарячої води
        self.hot_valve['big_left'] = fuzz.trimf(self.hot_valve.universe, [-90, -90, -45])
        self.hot_valve['medium_left'] = fuzz.trimf(self.hot_valve.universe, [-60, -30, 0])
        self.hot_valve['small_left'] = fuzz.trimf(self.hot_valve.universe, [-30, -15, 0])
        self.hot_valve['zero'] = fuzz.trimf(self.hot_valve.universe, [-5, 0, 5])
        self.hot_valve['small_right'] = fuzz.trimf(self.hot_valve.universe, [0, 15, 30])
        self.hot_valve['medium_right'] = fuzz.trimf(self.hot_valve.universe, [0, 30, 60])
        self.hot_valve['big_right'] = fuzz.trimf(self.hot_valve.universe, [45, 90, 90])
        
        # Функції належності для крану холодної води (аналогічні)
        self.cold_valve['big_left'] = fuzz.trimf(self.cold_valve.universe, [-90, -90, -45])
        self.cold_valve['medium_left'] = fuzz.trimf(self.cold_valve.universe, [-60, -30, 0])
        self.cold_valve['small_left'] = fuzz.trimf(self.cold_valve.universe, [-30, -15, 0])
        self.cold_valve['zero'] = fuzz.trimf(self.cold_valve.universe, [-5, 0, 5])
        self.cold_valve['small_right'] = fuzz.trimf(self.cold_valve.universe, [0, 15, 30])
        self.cold_valve['medium_right'] = fuzz.trimf(self.cold_valve.universe, [0, 30, 60])
        self.cold_valve['big_right'] = fuzz.trimf(self.cold_valve.universe, [45, 90, 90])
    
    def create_rules(self):
        """Створення правил нечіткої логіки"""
        
        self.rules = [
            # Правило 1: Якщо вода гаряча і її напір сильний
            ctrl.Rule(self.temperature['hot'] & self.pressure['strong'], 
                     [self.hot_valve['medium_left'], self.cold_valve['medium_right']]),
            
            # Правило 2: Якщо вода гаряча і її напір не дуже сильний
            ctrl.Rule(self.temperature['hot'] & self.pressure['not_very_strong'], 
                     [self.hot_valve['zero'], self.cold_valve['medium_right']]),
            
            # Правило 3: Якщо вода не дуже гаряча і її напір сильний
            ctrl.Rule(self.temperature['not_very_hot'] & self.pressure['strong'], 
                     [self.hot_valve['small_left'], self.cold_valve['zero']]),
            
            # Правило 4: Якщо вода не дуже гаряча і її напір слабий
            ctrl.Rule(self.temperature['not_very_hot'] & self.pressure['weak'], 
                     [self.hot_valve['small_right'], self.cold_valve['small_right']]),
            
            # Правило 5: Якщо вода тепла і її напір не дуже сильний
            ctrl.Rule(self.temperature['warm'] & self.pressure['not_very_strong'], 
                     [self.hot_valve['zero'], self.cold_valve['zero']]),
            
            # Правило 6: Якщо вода прохолодна і її напір сильний
            ctrl.Rule(self.temperature['cool'] & self.pressure['strong'], 
                     [self.hot_valve['medium_right'], self.cold_valve['medium_left']]),
            
            # Правило 7: Якщо вода прохолодна і її напір не дуже сильний
            ctrl.Rule(self.temperature['cool'] & self.pressure['not_very_strong'], 
                     [self.hot_valve['medium_right'], self.cold_valve['small_left']]),
            
            # Правило 8: Якщо вода холодна і її напір слабий
            ctrl.Rule(self.temperature['cold'] & self.pressure['weak'], 
                     [self.hot_valve['big_right'], self.cold_valve['zero']]),
            
            # Правило 9: Якщо вода холодна і її напір сильний
            ctrl.Rule(self.temperature['cold'] & self.pressure['strong'], 
                     [self.hot_valve['medium_left'], self.cold_valve['medium_right']]),
            
            # Правило 10: Якщо вода тепла і її напір сильний
            ctrl.Rule(self.temperature['warm'] & self.pressure['strong'], 
                     [self.hot_valve['small_left'], self.cold_valve['small_left']]),
            
            # Правило 11: Якщо вода тепла і її напір слабий
            ctrl.Rule(self.temperature['warm'] & self.pressure['weak'], 
                     [self.hot_valve['small_right'], self.cold_valve['small_right']])
        ]
    
    def create_control_system(self):
        """Створення системи керування"""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def evaluate(self, temp, press):
        """Обчислення виходу системи для заданих входів"""
        self.simulation.input['temperature'] = temp
        self.simulation.input['pressure'] = press
        self.simulation.compute()
        
        return {
            'hot_valve': self.simulation.output['hot_valve'],
            'cold_valve': self.simulation.output['cold_valve']
        }
    
    def plot_membership_functions(self):
        """Візуалізація функцій належності"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Температура
        self.temperature.view(ax=axes[0, 0])
        axes[0, 0].set_title('Функції належності температури')
        
        # Тиск
        self.pressure.view(ax=axes[0, 1])
        axes[0, 1].set_title('Функції належності тиску')
        
        # Кран гарячої води
        self.hot_valve.view(ax=axes[1, 0])
        axes[1, 0].set_title('Функції належності крану гарячої води')
        
        # Кран холодної води
        self.cold_valve.view(ax=axes[1, 1])
        axes[1, 1].set_title('Функції належності крану холодної води')
        
        plt.tight_layout()
        plt.show()
    
    def plot_control_surface(self):
        """Візуалізація поверхні керування"""
        # Створення сітки значень
        temp_range = np.arange(0, 101, 10)
        press_range = np.arange(0, 101, 10)
        temp_grid, press_grid = np.meshgrid(temp_range, press_range)
        
        hot_valve_output = np.zeros_like(temp_grid)
        cold_valve_output = np.zeros_like(temp_grid)
        
        # Обчислення виходів для кожної точки сітки
        for i in range(temp_grid.shape[0]):
            for j in range(temp_grid.shape[1]):
                try:
                    result = self.evaluate(temp_grid[i, j], press_grid[i, j])
                    hot_valve_output[i, j] = result['hot_valve']
                    cold_valve_output[i, j] = result['cold_valve']
                except:
                    hot_valve_output[i, j] = 0
                    cold_valve_output[i, j] = 0
        
        # Побудова 3D поверхонь
        fig = plt.figure(figsize=(15, 6))
        
        # Поверхня для крану гарячої води
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(temp_grid, press_grid, hot_valve_output, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Температура')
        ax1.set_ylabel('Тиск')
        ax1.set_zlabel('Кран гарячої води (градуси)')
        ax1.set_title('Поверхня керування краном гарячої води')
        
        # Поверхня для крану холодної води
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(temp_grid, press_grid, cold_valve_output, cmap='coolwarm', alpha=0.8)
        ax2.set_xlabel('Температура')
        ax2.set_ylabel('Тиск')
        ax2.set_zlabel('Кран холодної води (градуси)')
        ax2.set_title('Поверхня керування краном холодної води')
        
        plt.tight_layout()
        plt.show()
    
    def test_system(self):
        """Тестування системи з різними вхідними значеннями"""
        test_cases = [
            (80, 70, "Гаряча вода, сильний тиск"),
            (80, 30, "Гаряча вода, не дуже сильний тиск"),
            (60, 70, "Не дуже гаряча вода, сильний тиск"),
            (60, 20, "Не дуже гаряча вода, слабий тиск"),
            (50, 40, "Тепла вода, не дуже сильний тиск"),
            (30, 70, "Прохолодна вода, сильний тиск"),
            (30, 40, "Прохолодна вода, не дуже сильний тиск"),
            (15, 20, "Холодна вода, слабий тиск"),
            (15, 70, "Холодна вода, сильний тиск"),
            (50, 70, "Тепла вода, сильний тиск"),
            (50, 20, "Тепла вода, слабий тиск")
        ]
        
        print("=== Тестування системи керування змішувачем води ===\n")
        
        for temp, press, description in test_cases:
            result = self.evaluate(temp, press)
            print(f"Сценарій: {description}")
            print(f"Вхід: Температура={temp}°, Тиск={press}%")
            print(f"Вихід: Кран гарячої води={result['hot_valve']:.1f}°, "
                  f"Кран холодної води={result['cold_valve']:.1f}°")
            print("-" * 60)

# Основна функція для запуску
def main():
    # Створення системи
    water_mixer = WaterMixerFuzzySystem()
    
    # Тестування системи
    water_mixer.test_system()
    
    # Візуалізація функцій належності
    water_mixer.plot_membership_functions()
    
    # Візуалізація поверхні керування
    water_mixer.plot_control_surface()
    
    # Інтерактивне тестування
    print("\n=== Інтерактивне тестування ===")
    while True:
        try:
            temp = float(input("Введіть температуру (0-100) або 'q' для виходу: "))
            if temp < 0 or temp > 100:
                print("Температура має бути в діапазоні 0-100")
                continue
                
            press = float(input("Введіть тиск (0-100): "))
            if press < 0 or press > 100:
                print("Тиск має бути в діапазоні 0-100")
                continue
                
            result = water_mixer.evaluate(temp, press)
            print(f"Результат: Кран гарячої води={result['hot_valve']:.1f}°, "
                  f"Кран холодної води={result['cold_valve']:.1f}°\n")
                  
        except ValueError:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Для запуску потрібно встановити бібліотеки:
    # pip install scikit-fuzzy matplotlib numpy
    main()