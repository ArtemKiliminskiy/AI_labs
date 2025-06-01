import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AirConditionerFuzzySystem:
    def __init__(self):
        """Ініціалізація нечіткої системи керування кондиціонером"""
        self._create_variables()
        self._setup_membership_functions()
        self._create_rules()
        self._create_control_system()
    
    def _create_variables(self):
        """Створення вхідних та вихідних змінних"""
        # Вхідні змінні
        self.temperature = ctrl.Antecedent(np.arange(10, 36, 1), 'temperature')
        self.temp_change_rate = ctrl.Antecedent(np.arange(-5, 6, 1), 'temp_change_rate')
        
        # Вихідна змінна
        self.ac_control = ctrl.Consequent(np.arange(-90, 91, 1), 'ac_control')
    
    def _setup_membership_functions(self):
        """Налаштування функцій належності"""
        # Температура
        self.temperature['very_cold'] = fuzz.trimf(self.temperature.universe, [10, 10, 16])
        self.temperature['cold'] = fuzz.trimf(self.temperature.universe, [12, 18, 22])
        self.temperature['normal'] = fuzz.trimf(self.temperature.universe, [20, 22, 24])
        self.temperature['warm'] = fuzz.trimf(self.temperature.universe, [22, 26, 30])
        self.temperature['very_warm'] = fuzz.trimf(self.temperature.universe, [28, 35, 35])
        
        # Швидкість зміни температури
        self.temp_change_rate['negative'] = fuzz.trimf(self.temp_change_rate.universe, [-5, -5, -1])
        self.temp_change_rate['zero'] = fuzz.trimf(self.temp_change_rate.universe, [-0.5, 0, 0.5])
        self.temp_change_rate['positive'] = fuzz.trimf(self.temp_change_rate.universe, [1, 5, 5])
        
        # Керування кондиціонером
        self.ac_control['cold_big'] = fuzz.trimf(self.ac_control.universe, [-90, -90, -60])
        self.ac_control['cold_small'] = fuzz.trimf(self.ac_control.universe, [-45, -30, -15])
        self.ac_control['off'] = fuzz.trimf(self.ac_control.universe, [-5, 0, 5])
        self.ac_control['heat_small'] = fuzz.trimf(self.ac_control.universe, [15, 30, 45])
        self.ac_control['heat_big'] = fuzz.trimf(self.ac_control.universe, [60, 90, 90])
    
    def _create_rules(self):
        """Створення правил нечіткої логіки"""
        self.rules = [
            # Дуже теплі умови
            ctrl.Rule(self.temperature['very_warm'] & self.temp_change_rate['positive'], 
                     self.ac_control['cold_big']),
            ctrl.Rule(self.temperature['very_warm'] & self.temp_change_rate['negative'], 
                     self.ac_control['cold_small']),
            ctrl.Rule(self.temperature['very_warm'] & self.temp_change_rate['zero'], 
                     self.ac_control['cold_big']),
            
            # Теплі умови
            ctrl.Rule(self.temperature['warm'] & self.temp_change_rate['positive'], 
                     self.ac_control['cold_big']),
            ctrl.Rule(self.temperature['warm'] & self.temp_change_rate['negative'], 
                     self.ac_control['off']),
            ctrl.Rule(self.temperature['warm'] & self.temp_change_rate['zero'], 
                     self.ac_control['cold_small']),
            
            # Нормальні умови
            ctrl.Rule(self.temperature['normal'] & self.temp_change_rate['positive'], 
                     self.ac_control['cold_small']),
            ctrl.Rule(self.temperature['normal'] & self.temp_change_rate['negative'], 
                     self.ac_control['heat_small']),
            ctrl.Rule(self.temperature['normal'] & self.temp_change_rate['zero'], 
                     self.ac_control['off']),
            
            # Холодні умови
            ctrl.Rule(self.temperature['cold'] & self.temp_change_rate['positive'], 
                     self.ac_control['off']),
            ctrl.Rule(self.temperature['cold'] & self.temp_change_rate['negative'], 
                     self.ac_control['heat_big']),
            ctrl.Rule(self.temperature['cold'] & self.temp_change_rate['zero'], 
                     self.ac_control['heat_small']),
            
            # Дуже холодні умови
            ctrl.Rule(self.temperature['very_cold'] & self.temp_change_rate['positive'], 
                     self.ac_control['heat_small']),
            ctrl.Rule(self.temperature['very_cold'] & self.temp_change_rate['negative'], 
                     self.ac_control['heat_big']),
            ctrl.Rule(self.temperature['very_cold'] & self.temp_change_rate['zero'], 
                     self.ac_control['heat_big'])
        ]
    
    def _create_control_system(self):
        """Створення системи керування"""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def evaluate(self, temp, rate):
        """Обчислення виходу системи для заданих вхідних значень"""
        self.simulation.input['temperature'] = temp
        self.simulation.input['temp_change_rate'] = rate
        
        try:
            self.simulation.compute()
            return self.simulation.output['ac_control']
        except:
            return 0
    
    def interpret_control_value(self, value):
        """Інтерпретація числового значення керування"""
        if value < -60:
            return "Велике охолодження"
        elif -60 <= value < -15:
            return "Малe охолодження"
        elif -5 <= value <= 5:
            return "Виключений"
        elif 15 <= value < 45:
            return "Малe нагрівання"
        elif value >= 60:
            return "Велике нагрівання"
        else:
            return "Перехідний режим"
    
    def visualize_membership(self):
        """Візуалізація функцій належності"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        self.temperature.view(ax=ax1)
        ax1.set_title('Температура (°C)')
        
        self.temp_change_rate.view(ax=ax2)
        ax2.set_title('Швидкість зміни (°C/хв)')
        
        self.ac_control.view(ax=ax3)
        ax3.set_title('Керування кондиціонером')
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_control_surface(self):
        """Візуалізація 3D поверхні керування"""
        temp_range = np.linspace(10, 35, 25)
        rate_range = np.linspace(-5, 5, 20)
        temp_grid, rate_grid = np.meshgrid(temp_range, rate_range)
        
        control_output = np.zeros_like(temp_grid)
        
        for i in range(len(temp_range)):
            for j in range(len(rate_range)):
                control_output[j, i] = self.evaluate(temp_range[i], rate_range[j])
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(temp_grid, rate_grid, control_output,
                                cmap='coolwarm', edgecolor='none')
        
        ax.set_xlabel('Температура (°C)')
        ax.set_ylabel('Швидкість зміни (°C/хв)')
        ax.set_zlabel('Керування')
        ax.set_title('Поверхня керування кондиціонером')
        
        fig.colorbar(surface, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()
    
    def run_test_scenarios(self):
        """Запуск тестових сценаріїв"""
        test_cases = [
            (32, 2, "Дуже тепла температура, швидкість росте"),
            (32, -1, "Дуже тепла температура, швидкість падає"),
            (26, 3, "Тепла температура, швидкість росте"),
            (26, -2, "Тепла температура, швидкість падає"),
            (22, 0.1, "Нормальна температура, стабільно"),
            (22, 2, "Нормальна температура, швидкість росте"),
            (22, -1.5, "Нормальна температура, швидкість падає"),
            (18, 0, "Холодна температура, стабільно"),
            (18, 3, "Холодна температура, швидкість росте"),
            (18, -2, "Холодна температура, швидкість падає"),
            (14, 0, "Дуже холодна температура, стабільно"),
            (14, 1, "Дуже холодна температура, швидкість росте"),
            (14, -3, "Дуже холодна температура, швидкість падає")
        ]
        
        print("Результати тестування системи:")
        print("="*60)
        for temp, rate, desc in test_cases:
            control = self.evaluate(temp, rate)
            interpretation = self.interpret_control_value(control)
            print(f"\nСценарій: {desc}")
            print(f"Вхідні дані: Температура={temp}°C, Швидкість зміни={rate}°C/хв")
            print(f"Результат: {control:.1f} ({interpretation})")
        print("="*60)


def interactive_mode(system):
    """Інтерактивний режим тестування"""
    print("\nІнтерактивний режим (введіть 'q' для виходу)")
    while True:
        try:
            temp_input = input("Введіть температуру (10-35°C): ")
            if temp_input.lower() == 'q':
                break
            
            temp = float(temp_input)
            if not 10 <= temp <= 35:
                print("Температура повинна бути між 10 і 35")
                continue
                
            rate = float(input("Введіть швидкість зміни (-5 до 5°C/хв): "))
            if not -5 <= rate <= 5:
                print("Швидкість повинна бути між -5 і 5")
                continue
                
            control = system.evaluate(temp, rate)
            interpretation = system.interpret_control_value(control)
            print(f"\nРезультат: {control:.1f} - {interpretation}\n")
        
        except ValueError:
            print("Будь ласка, введіть числове значення")
        except KeyboardInterrupt:
            break


def main():
    """Головна функція програми"""
    ac_system = AirConditionerFuzzySystem()
    
    # Візуалізація
    ac_system.visualize_membership()
    ac_system.plot_3d_control_surface()
    
    # Тестування
    ac_system.run_test_scenarios()
    
    # Інтерактивний режим
    interactive_mode(ac_system)


if __name__ == "__main__":
    main()