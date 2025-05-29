import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Налаштування для українського тексту в графіках
plt.rcParams['font.family'] = 'DejaVu Sans'

class BayesianAnalyzer:
    """
    Клас для байєсівського аналізу даних про ціни на квитки
    іспанських високошвидкісних залізниць
    """
    
    def __init__(self):
        self.data = None
        self.label_encoders = {}
        self.naive_bayes_model = None
        
    def load_data(self, url=None):
        """Завантаження та генерація даних про квитки"""
        if url:
            try:
                self.data = pd.read_csv(url)
                print("Дані успішно завантажені з URL")
            except:
                print("Помилка завантаження з URL, генеруємо тестові дані")
                self._generate_sample_data()
        else:
            self._generate_sample_data()
            
    def _generate_sample_data(self):
        """Генерація тестових даних для демонстрації"""
        np.random.seed(42)
        
        train_types = ['AVE', 'ALVIA', 'AVANT', 'Regional']
        routes = ['Madrid-Barcelona', 'Madrid-Sevilla', 'Barcelona-Valencia', 'Madrid-Valencia']
        fare_types = ['Turista', 'Turista Plus', 'Preferente', 'Business']
        
        n_samples = 1000
        data = []
        
        for i in range(n_samples):
            train_type = np.random.choice(train_types)
            route = np.random.choice(routes)
            fare_type = np.random.choice(fare_types)
            
            # Базова ціна залежно від типу поїзда
            base_price = 50
            if train_type == 'AVE':
                base_price = 100
            elif train_type == 'ALVIA':
                base_price = 80
            elif train_type == 'AVANT':
                base_price = 60
                
            # Коригування ціни залежно від тарифу
            if fare_type == 'Preferente':
                base_price *= 1.8
            elif fare_type == 'Business':
                base_price *= 2.5
            elif fare_type == 'Turista Plus':
                base_price *= 1.3
                
            # Додавання випадкового шуму
            price = base_price + np.random.normal(0, 15)
            price = max(20, price)  # Мінімальна ціна
            
            # Відстань
            distance = {
                'Madrid-Barcelona': 621,
                'Madrid-Sevilla': 538,
                'Barcelona-Valencia': 349,
                'Madrid-Valencia': 391
            }[route]
            
            # Тривалість поїздки
            avg_speed = 200 if train_type == 'AVE' else 150
            duration = distance / avg_speed * 60  # хвилини
            
            # Категорія ціни
            if price < 60:
                price_category = 'Low'
            elif price < 120:
                price_category = 'Medium'
            else:
                price_category = 'High'
                
            data.append({
                'train_type': train_type,
                'route': route,
                'fare_type': fare_type,
                'price': round(price, 2),
                'distance': distance,
                'duration': round(duration),
                'price_category': price_category
            })
            
        self.data = pd.DataFrame(data)
        print(f"Згенеровано {len(self.data)} зразків даних")
        
    def explore_data(self):
        """Дослідницький аналіз даних"""
        print("=== ДОСЛІДНИЦЬКИЙ АНАЛІЗ ДАНИХ ===")
        print(f"Розмір датасету: {self.data.shape}")
        print(f"\nПерші 5 рядків:")
        print(self.data.head())
        
        print(f"\nОписова статистика:")
        print(self.data.describe())
        
        print(f"\nРозподіл за категоріями цін:")
        print(self.data['price_category'].value_counts())
        
        # Візуалізація
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Розподіл цін
        axes[0,0].hist(self.data['price'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Розподіл цін на квитки')
        axes[0,0].set_xlabel('Ціна (€)')
        axes[0,0].set_ylabel('Частота')
        
        # Ціни за типом поїзда
        self.data.boxplot(column='price', by='train_type', ax=axes[0,1])
        axes[0,1].set_title('Ціни за типом поїзда')
        axes[0,1].set_xlabel('Тип поїзда')
        axes[0,1].set_ylabel('Ціна (€)')
        
        # Розподіл за категоріями
        category_counts = self.data['price_category'].value_counts()
        axes[1,0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Розподіл за категоріями цін')
        
        # Кореляційна матриця
        numeric_data = self.data.select_dtypes(include=[np.number])
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Кореляційна матриця')
        
        plt.tight_layout()
        plt.show()
        
    def calculate_bayes_theorem(self):
        """Демонстрація теореми Байєса"""
        print("\n=== ТЕОРЕМА БАЙЄСА ===")
        print("P(H|E) = P(E|H) * P(H) / P(E)")
        print("де H - гіпотеза (категорія ціни), E - докази (характеристики)")
        
        # Розрахунок апріорних ймовірностей P(H)
        prior_probs = self.data['price_category'].value_counts(normalize=True)
        print(f"\nАпріорні ймовірності P(H):")
        for category, prob in prior_probs.items():
            print(f"P({category}) = {prob:.3f}")
            
        # Приклад розрахунку для конкретного випадку
        print(f"\nПриклад: P(High|AVE, Business)")
        
        # P(High)
        p_high = prior_probs['High']
        
        # P(AVE|High)
        high_data = self.data[self.data['price_category'] == 'High']
        p_ave_given_high = (high_data['train_type'] == 'AVE').mean()
        
        # P(Business|High)
        p_business_given_high = (high_data['fare_type'] == 'Business').mean()
        
        # P(AVE, Business)
        p_ave_business = ((self.data['train_type'] == 'AVE') & 
                         (self.data['fare_type'] == 'Business')).mean()
        
        # Наївне припущення незалежності
        posterior = (p_ave_given_high * p_business_given_high * p_high) / p_ave_business
        
        print(f"P(AVE|High) = {p_ave_given_high:.3f}")
        print(f"P(Business|High) = {p_business_given_high:.3f}")
        print(f"P(High) = {p_high:.3f}")
        print(f"P(AVE, Business) = {p_ave_business:.3f}")
        print(f"P(High|AVE, Business) ≈ {posterior:.3f}")
        
    def naive_bayes_classification(self):
        """Реалізація наївного байєсівського класифікатора"""
        print("\n=== НАЇВНИЙ БАЙЄСІВСЬКИЙ КЛАСИФІКАТОР ===")
        
        # Підготовка даних
        features = ['train_type', 'route', 'fare_type', 'distance', 'duration']
        X = self.data[features].copy()
        y = self.data['price_category']
        
        # Кодування категоріальних змінних
        for col in ['train_type', 'route', 'fare_type']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            
        # Розділення на тренувальну та тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Тренувальна вибірка: {X_train.shape[0]} зразків")
        print(f"Тестова вибірка: {X_test.shape[0]} зразків")
        
        # Тренування різних типів наївного Байєса
        models = {
            'Gaussian NB': GaussianNB(),
            'Categorical NB': CategoricalNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Точність: {accuracy:.3f}")
            print(f"\nЗвіт класифікації:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
        # Матриця помилок для кращої моделі
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['High', 'Low', 'Medium'],
                   yticklabels=['High', 'Low', 'Medium'])
        plt.title(f'Матриця помилок - {best_model_name}')
        plt.ylabel('Справжні значення')
        plt.xlabel('Передбачені значення')
        plt.show()
        
        self.naive_bayes_model = results[best_model_name]['model']
        
    def demonstrate_classifier_types(self):
        """Демонстрація різних типів байєсівських класифікаторів"""
        print("\n=== ТИПИ БАЙЄСІВСЬКИХ КЛАСИФІКАТОРІВ ===")
        
        print("1. ПОЛІНОМІАЛЬНИЙ НАЇВНИЙ БАЙЄС:")
        print("   - Використовується для дискретних ознак")
        print("   - Ідеально підходить для класифікації тексту")
        print("   - Припускає мультиноміальний розподіл")
        
        print("\n2. ГАУСІВСЬКИЙ НАЇВНИЙ БАЙЄС:")
        print("   - Для неперервних ознак")
        print("   - Припускає нормальний розподіл")
        print("   - Підходить для числових даних")
        
        print("\n3. БЕРНУЛЛІ НАЇВНИЙ БАЙЄС:")
        print("   - Для бінарних/булевих ознак")
        print("   - Використовується в аналізі тексту")
        print("   - Враховує наявність/відсутність ознак")
        
        print("\n4. КАТЕГОРІАЛЬНИЙ НАЇВНИЙ БАЙЄС:")
        print("   - Для категоріальних ознак")
        print("   - Не потребує припущень про розподіл")
        print("   - Підходить для змішаних типів даних")
        
    def predict_example(self):
        """Приклад передбачення для нового квитка"""
        if self.naive_bayes_model is None:
            print("Спочатку потрібно натренувати модель!")
            return
            
        print("\n=== ПРИКЛАД ПЕРЕДБАЧЕННЯ ===")
        
        # Новий приклад
        example = {
            'train_type': 'AVE',
            'route': 'Madrid-Barcelona', 
            'fare_type': 'Business',
            'distance': 621,
            'duration': 186
        }
        
        print(f"Характеристики квитка:")
        for key, value in example.items():
            print(f"  {key}: {value}")
            
        # Підготовка для передбачення
        example_encoded = example.copy()
        for col in ['train_type', 'route', 'fare_type']:
            example_encoded[col] = self.label_encoders[col].transform([example[col]])[0]
            
        X_example = np.array([[
            example_encoded['train_type'],
            example_encoded['route'], 
            example_encoded['fare_type'],
            example_encoded['distance'],
            example_encoded['duration']
        ]])
        
        # Передбачення
        prediction = self.naive_bayes_model.predict(X_example)[0]
        probabilities = self.naive_bayes_model.predict_proba(X_example)[0]
        
        print(f"\nПередбачена категорія: {prediction}")
        print(f"Ймовірності:")
        classes = self.naive_bayes_model.classes_
        for cls, prob in zip(classes, probabilities):
            print(f"  P({cls}) = {prob:.3f}")
            
    def analyze_feature_importance(self):
        """Аналіз важливості ознак"""
        print("\n=== АНАЛІЗ ОЗНАК ===")
        
        # Аналіз розподілу ознак за класами
        categorical_features = ['train_type', 'route', 'fare_type']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, feature in enumerate(categorical_features):
            crosstab = pd.crosstab(self.data[feature], self.data['price_category'], 
                                 normalize='index')
            crosstab.plot(kind='bar', ax=axes[i], stacked=True)
            axes[i].set_title(f'Розподіл {feature} за категоріями')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Частка')
            axes[i].legend(title='Категорія ціни')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def answer_control_questions(self):
        """Відповіді на контрольні питання"""
        print("\n=== ВІДПОВІДІ НА КОНТРОЛЬНІ ПИТАННЯ ===")
        
        print("1. ДЕ ЗАСТОСОВУЄТЬСЯ НАЇВНИЙ БАЙЄС?")
        print("   • Класифікація електронної пошти (спам/не спам)")
        print("   • Аналіз настроїв у текстах")
        print("   • Медична діагностика")
        print("   • Рекомендаційні системи")
        print("   • Категоризація документів")
        print("   • Розпізнавання зображень")
        
        print("\n2. ПОЯСНІТЬ ТЕОРЕМУ БАЙЄСА:")
        print("   Теорема Байєса описує ймовірність гіпотези H при наявності")
        print("   доказів E. Формула: P(H|E) = P(E|H) × P(H) / P(E)")
        print("   • P(H|E) - апостеріорна ймовірність (шукана)")
        print("   • P(E|H) - правдоподібність (likelihood)")
        print("   • P(H) - апріорна ймовірність")
        print("   • P(E) - повна ймовірність доказів")
        
        print("\n3. ТИПИ НАЇВНОГО БАЙЄСІВСЬКОГО КЛАСИФІКАТОРА:")
        print("   • Поліноміальний - для дискретних ознак і підрахунку")
        print("   • Гаусівський - для неперервних ознак з нормальним розподілом")
        print("   • Бернуллі - для бінарних ознак (0/1, True/False)")
        print("   • Категоріальний - для категоріальних ознак")
        print("   • Комплементарний - для незбалансованих наборів даних")

def main():
    """Головна функція для запуску аналізу"""
    print("БАЙЄСІВСЬКИЙ АНАЛІЗ ДАНИХ ПРО ЦІНИ НА КВИТКИ")
    print("Іспанські високошвидкісні залізниці")
    print("=" * 50)
    
    # Створення аналізатора
    analyzer = BayesianAnalyzer()
    
    # URL датасету (можна замінити на реальний)
    data_url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
    
    # Завантаження даних
    analyzer.load_data()  # Використовуємо згенеровані дані
    
    # Дослідницький аналіз
    analyzer.explore_data()
    
    # Теорема Байєса
    analyzer.calculate_bayes_theorem()
    
    # Наївний байєсівський класифікатор
    analyzer.naive_bayes_classification()
    
    # Типи класифікаторів
    analyzer.demonstrate_classifier_types()
    
    # Приклад передбачення
    analyzer.predict_example()
    
    # Аналіз ознак
    analyzer.analyze_feature_importance()
    
    # Контрольні питання
    analyzer.answer_control_questions()
    
    print("\nАналіз завершено!")

if __name__ == "__main__":
    main()