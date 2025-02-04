import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.combine import SMOTETomek
from joblib import Parallel, delayed
from binance.client import Client
from datetime import datetime, timedelta
import logging
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import joblib
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.metrics import f1_score
import sys
from sklearn.model_selection import GridSearchCV
from filterpy.kalman import KalmanFilter
from sklearn.metrics import f1_score, precision_score, recall_score
import shutil


# Логирование
logging.basicConfig(
    level=logging.INFO,  # Вывод всех сообщений уровня INFO и выше
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_log_flat_ensemble.log"),  # Лог-файл
        logging.StreamHandler()  # Вывод в консоль
    ]
)



# Имя файла для сохранения модели
market_type = "flat"

ensemble_model_filename = 'flat_stacked_ensemble_model.pkl'

checkpoint_base_dir = f"checkpoints/{market_type}"

ensemble_checkpoint_path = os.path.join(checkpoint_base_dir, f"{market_type}_ensemble_checkpoint.pkl")


def ensure_directory(path):
    """Создает директорию, если она не существует."""
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_logs_to_file(message):
    """
    Сохраняет логи в файл.
    """
    with open("trading_logs.txt", "a") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - {message}\n")
        

def calculate_cross_coin_features(data_dict):
    """
    Рассчитывает межмонетные признаки для всех пар.
    
    Args:
        data_dict (dict): Словарь DataFrame'ов по каждой монете
    Returns:
        dict: Словарь DataFrame'ов с добавленными признаками
    """
    btc_data = data_dict['BTCUSDT']
    
    for symbol, df in data_dict.items():
        # Корреляции с BTC
        df['btc_corr'] = df['close'].rolling(30).corr(btc_data['close'])
        
        # Относительная сила к BTC
        df['rel_strength_btc'] = (df['close'].pct_change() - 
                                btc_data['close'].pct_change())
        
        # Бета к BTC
        df['beta_btc'] = (df['close'].pct_change().rolling(30).cov(
            btc_data['close'].pct_change()) / 
            btc_data['close'].pct_change().rolling(30).var())
        
        # Опережение/следование за BTC
        df['lead_lag_btc'] = df['close'].pct_change().shift(-1).rolling(10).corr(
            btc_data['close'].pct_change())
            
        data_dict[symbol] = df
        
    return data_dict

def detect_anomalies(data):
    """
    Детектирует и фильтрует аномальные свечи для флэтового рынка.
    """
    # Рассчитываем z-score с меньшими окнами для большей чувствительности
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(50).mean()) / 
                            data['volume'].rolling(50).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(50).mean()) / 
                           data['close'].rolling(50).std())
    data['range_zscore'] = (((data['high'] - data['low']) - 
                            (data['high'] - data['low']).rolling(50).mean()) / 
                           (data['high'] - data['low']).rolling(50).std())
    
    # Более строгие пороги для флэтового рынка
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 3) |  # Уменьшен порог
                         (abs(data['price_zscore']) > 3) |    # Уменьшен порог
                         (abs(data['range_zscore']) > 3))     # Уменьшен порог
    
    # Дополнительные признаки для флэтового рынка
    data['price_stability'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
    data['range_stability'] = (data['high'] - data['low']).rolling(20).std() / data['close'].rolling(20).mean()
    
    # Флаги резких движений
    data['price_jump'] = (abs(data['close'].pct_change()) > 
                         2 * data['close'].pct_change().rolling(50).std()).astype(int)
    
    return data



def validate_volume_confirmation(data):
    """
    Добавляет признаки подтверждения движений объемом для флэтового рынка.
    """
    # Объемное подтверждение тренда - более чувствительное к малым движениям
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(5).mean()),  # Уменьшенное окно для большей чувствительности
        0.5,  # Уменьшенный вес для роста
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(5).mean()),
            -0.5,  # Уменьшенный вес для падения
            0
        )
    )
    
    # Сила объемного подтверждения с меньшим окном
    data['volume_strength'] = (data['volume'] / 
                             data['volume'].rolling(10).mean() *  # Меньшее окно
                             data['volume_trend_conf'])
    
    # Накопление объема с коротким окном
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(3).sum()  # Короткое окно
    
    # Добавляем признаки стабильности объема
    data['volume_stability'] = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()
    data['volume_trend_stability'] = data['volume_trend_conf'].rolling(10).std()
    
    # Флаги аномальных объемов
    volume_mean = data['volume'].rolling(20).mean()
    volume_std = data['volume'].rolling(20).std()
    data['volume_spike'] = (data['volume'] > (volume_mean + 2 * volume_std)).astype(int)
    
    return data


def remove_noise(data):
    """
    Улучшенная фильтрация шума с использованием фильтра Калмана.
    """
    # Kalman filter для сглаживания цены
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5  # Настройка шума измерений
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # Настройка шума процесса
    
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    # Используем сглаженные цены Калмана вместо простого скользящего среднего
    data['smoothed_close'] = smoothed_prices
    
    # Дополнительные фильтры для флэтового рынка
    data['price_volatility'] = data['close'].rolling(20).std()
    data['is_significant_move'] = (data['close'].pct_change().abs() > 
                                 data['price_volatility'] * 2)
    
    # Вычисляем "чистые" движения с учетом значимости
    data['clean_returns'] = np.where(
        data['is_significant_move'] & (data['is_anomaly'] == 0),
        data['smoothed_close'].pct_change(),
        0
    )
    
    return data


def preprocess_market_data(data_dict):
    """
    Комплексная предобработка данных с учетом межмонетных взаимосвязей.
    """
    # Добавляем межмонетные признаки
    data_dict = calculate_cross_coin_features(data_dict)
    
    for symbol, df in data_dict.items():
        # Детектируем аномалии
        df = detect_anomalies(df)
        
        # Добавляем подтверждение объемом
        df = validate_volume_confirmation(df)
        
        # Фильтруем шум
        df = remove_noise(df)
        
        data_dict[symbol] = df
    
    return data_dict


def get_checkpoint_path(model_name, market_type):
    """
    Создает уникальный путь к чекпоинтам для каждой модели.
    
    Args:
        model_name (str): Название модели ('rf', 'xgb', 'lgbm', etc.)
        market_type (str): Тип рынка ('bullish', 'bearish', 'flat')
    
    Returns:
        str: Путь к директории чекпоинтов
    """
    checkpoint_path = os.path.join("checkpoints", market_type, model_name)
    ensure_directory(checkpoint_path)
    return checkpoint_path


# GradientBoosting: сохранение после каждой итерации
class CheckpointGradientBoosting(GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, 
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         random_state=random_state, subsample=subsample, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf)
        self.checkpoint_dir = get_checkpoint_path("gradient_boosting", market_type)
        
    def fit(self, X, y):
        logging.info("[GradientBoosting] Начало обучения с чекпоинтами")
        
        # Инициализация
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        
        # Проверяем существующие чекпоинты
        existing_stages = []
        for i in range(self.n_estimators):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"gradient_boosting_checkpoint_{i + 1}.joblib")
            if os.path.exists(checkpoint_path):
                try:
                    stage = joblib.load(checkpoint_path)
                    if stage.n_features_ == n_features:
                        existing_stages.append(stage)
                        logging.info(f"[GradientBoosting] Загружена итерация {i + 1} из чекпоинта")
                except:
                    logging.warning(f"[GradientBoosting] Не удалось загрузить чекпоинт {i + 1}")

        # Если чекпоинты не подходят, очищаем директорию
        if not existing_stages:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            logging.info("[GradientBoosting] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_stages
            remaining_stages = self.n_estimators - len(existing_stages)
            if remaining_stages > 0:
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining_stages
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(self.estimators_)
                self.n_estimators = len(self.estimators_)

        # Сохраняем чекпоинты
        for i, stage in enumerate(self.estimators_):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"gradient_boosting_checkpoint_{i + 1}.joblib")
            if not os.path.exists(checkpoint_path):
                joblib.dump(stage, checkpoint_path)
                logging.info(f"[GradientBoosting] Сохранен чекпоинт {i + 1}")

        return self

# XGBoost: сохранение каждые 3 итерации
class CheckpointXGBoost(XGBClassifier):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 random_state=None, objective=None, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective=objective,
            **kwargs
        )
        self.checkpoint_dir = get_checkpoint_path("xgboost", market_type)

    def fit(self, X, y, **kwargs):
        logging.info("[XGBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self.checkpoint_dir, "xgboost_checkpoint")
        
        # Проверяем существующий чекпоинт
        final_checkpoint = f"{model_path}_final.joblib"
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if saved_model.n_features_ == X.shape[1]:
                    logging.info("[XGBoost] Загружена модель из чекпоинта")
                    return saved_model
            except:
                logging.warning("[XGBoost] Не удалось загрузить существующий чекпоинт")
        
        # Если чекпоинт не подходит, очищаем и начинаем заново
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        ensure_directory(self.checkpoint_dir)
        
        super().fit(X, y)
        joblib.dump(self, final_checkpoint)
        logging.info("[XGBoost] Сохранен новый чекпоинт")
        return self


# LightGBM: сохранение каждые 3 итерации
class CheckpointLightGBM(LGBMClassifier):
    def __init__(self, n_estimators=100, num_leaves=31, learning_rate=0.1,
                 min_data_in_leaf=20, max_depth=-1, random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            max_depth=max_depth,
            random_state=random_state
        )
        self._checkpoint_path = get_checkpoint_path("lightgbm", market_type)

    def fit(self, X, y, **kwargs):
        logging.info("[LightGBM] Начало обучения с чекпоинтами")
        model_path = os.path.join(self._checkpoint_path, "lightgbm_checkpoint")
        final_checkpoint = f"{model_path}_final.joblib"
        
        # Проверяем существующий чекпоинт
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, '_n_features') and saved_model._n_features == X.shape[1]:
                    logging.info("[LightGBM] Загружена модель из чекпоинта")
                    # Копируем атрибуты из сохраненной модели
                    self.__dict__.update(saved_model.__dict__)
                    # Проверяем, что модель обучена
                    _ = self.predict(X[:1])
                    return self
            except:
                logging.warning("[LightGBM] Не удалось загрузить существующий чекпоинт")
        
        # Если чекпоинт не подходит, очищаем и начинаем заново
        if os.path.exists(self._checkpoint_path):
            shutil.rmtree(self._checkpoint_path)
        ensure_directory(self._checkpoint_path)
        
        # Обучаем модель
        super().fit(X, y, **kwargs)
        self._n_features = X.shape[1]  # Сохраняем количество признаков
        joblib.dump(self, final_checkpoint)
        logging.info("[LightGBM] Сохранен новый чекпоинт")
        return self
    
    
# CatBoost: сохранение каждые 3 итерации
class CheckpointCatBoost(CatBoostClassifier):
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1,
                 random_state=None, **kwargs):
        # Удаляем save_snapshot из kwargs если он там есть
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
            
        super().__init__(
            iterations=iterations, 
            depth=depth, 
            learning_rate=learning_rate, 
            random_state=random_state,
            save_snapshot=False,  # Устанавливаем save_snapshot только здесь
            **kwargs
        )
        self.checkpoint_dir = get_checkpoint_path("catboost", market_type)

    def fit(self, X, y, **kwargs):
        logging.info("[CatBoost] Начало обучения с чекпоинтами")
        model_path = os.path.join(self.checkpoint_dir, "catboost_checkpoint")
        final_checkpoint = f"{model_path}_final.joblib"
        
        if os.path.exists(final_checkpoint):
            try:
                saved_model = joblib.load(final_checkpoint)
                if hasattr(saved_model, 'feature_count_') and saved_model.feature_count_ == X.shape[1]:
                    logging.info("[CatBoost] Загружена модель из чекпоинта")
                    return saved_model
            except:
                logging.warning("[CatBoost] Не удалось загрузить существующий чекпоинт")
        
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        ensure_directory(self.checkpoint_dir)
        
        # Удаляем save_snapshot из kwargs при вызове fit если он там есть
        if 'save_snapshot' in kwargs:
            del kwargs['save_snapshot']
        
        super().fit(X, y, **kwargs)
        joblib.dump(self, final_checkpoint)
        logging.info("[CatBoost] Сохранен финальный чекпоинт")
        
        return self
    
    
# RandomForest: сохранение после каждого дерева
class CheckpointRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, 
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, random_state=random_state)
        self.checkpoint_dir = get_checkpoint_path("random_forest", market_type)

    def fit(self, X, y):
        logging.info("[RandomForest] Начало обучения с чекпоинтами")
        
        # Сначала выполняем базовую инициализацию
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        
        # Проверяем существующие чекпоинты
        existing_trees = []
        for i in range(self.n_estimators):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"random_forest_tree_{i + 1}.joblib")
            if os.path.exists(checkpoint_path):
                try:
                    tree = joblib.load(checkpoint_path)
                    # Проверяем, соответствует ли дерево текущим данным
                    if tree.tree_.n_features == n_features:
                        existing_trees.append(tree)
                        logging.info(f"[RandomForest] Загружено дерево {i + 1} из чекпоинта")
                except:
                    logging.warning(f"[RandomForest] Не удалось загрузить чекпоинт {i + 1}, будет создано новое дерево")
        
        # Если чекпоинты не подходят, очищаем директорию
        if not existing_trees:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            ensure_directory(self.checkpoint_dir)
            logging.info("[RandomForest] Начинаем обучение с нуля")
            super().fit(X, y)
        else:
            self.estimators_ = existing_trees
            remaining_trees = self.n_estimators - len(existing_trees)
            if remaining_trees > 0:
                logging.info(f"[RandomForest] Продолжаем обучение: осталось {remaining_trees} деревьев")
                orig_n_classes = self.n_classes_
                self.n_estimators = remaining_trees
                super().fit(X, y)
                self.n_classes_ = orig_n_classes
                self.estimators_.extend(self.estimators_)
                self.n_estimators = len(self.estimators_)
        
        # Сохраняем чекпоинты для всех деревьев
        for i, estimator in enumerate(self.estimators_):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"random_forest_tree_{i + 1}.joblib")
            if not os.path.exists(checkpoint_path):
                joblib.dump(estimator, checkpoint_path)
                logging.info(f"[RandomForest] Создан чекпоинт для нового дерева {i + 1}")
        
        return self

        
def save_ensemble_checkpoint(ensemble_model, checkpoint_path):
    """Сохраняет общий чекпоинт ансамбля."""
    ensure_directory(os.path.dirname(checkpoint_path))
    joblib.dump(ensemble_model, checkpoint_path)
    logging.info(f"[Ensemble] Сохранен чекпоинт ансамбля: {checkpoint_path}")



def load_ensemble_checkpoint(checkpoint_path):
    """Загружает общий чекпоинт ансамбля."""
    if os.path.exists(checkpoint_path):
        logging.info(f"[Ensemble] Загрузка чекпоинта ансамбля: {checkpoint_path}")
        return joblib.load(checkpoint_path)
    logging.info(f"[Ensemble] Чекпоинт не найден: {checkpoint_path}")
    return None
        
# Функция загрузки данных с использованием многопоточности
def load_all_data(symbols, start_date, end_date, interval):
    all_data = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(get_historical_data, symbol, interval, start_date, end_date): symbol for symbol in symbols}
        for future in futures:
            symbol = futures[future]
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
            except Exception as e:
                logging.error(f"Ошибка при загрузке данных для {symbol}: {e}")
                save_logs_to_file(f"Ошибка при загрузке данных для {symbol}: {e}")

    if not all_data:
        logging.error("Не удалось получить данные ни для одного символа")
        save_logs_to_file("Не удалось получить данные ни для одного символа")
        raise ValueError("Не удалось получить данные ни для одного символа")

    data = pd.concat(all_data)
    logging.info(f"Всего загружено {len(data)} строк данных")
    save_logs_to_file(f"Всего загружено {len(data)} строк данных")
    return data

# Получение исторических данных

def get_historical_data(symbol, interval, start_date, end_date):
    logging.info(f"Начало загрузки данных для {symbol}, период: с {start_date} по {end_date}, интервал: {interval}")
    data = pd.DataFrame()
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=30), end_date)
        try:
            logging.info(f"Загрузка данных для {symbol}: с {current_date} по {next_date}")
            klines = client.get_historical_klines(
                symbol, interval,
                current_date.strftime('%d %b %Y %H:%M:%S'),
                next_date.strftime('%d %b %Y %H:%M:%S'),
                limit=1000
            )
            if not klines:
                logging.warning(f"Нет данных для {symbol} за период с {current_date} по {next_date}")
                continue

            # Преобразуем данные в DataFrame
            temp_data = pd.DataFrame(
                klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
            )

            # Преобразуем `timestamp`
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'], unit='ms')
            temp_data.set_index('timestamp', inplace=True)

            # Оставляем только нужные колонки
            temp_data = temp_data[['open', 'high', 'low', 'close', 'volume']].astype(float)

            if 'timestamp' not in temp_data.columns and not isinstance(temp_data.index, pd.DatetimeIndex):
                raise ValueError(f"Колонка 'timestamp' отсутствует в данных для {symbol} за период {current_date} - {next_date}")

            data = pd.concat([data, temp_data])

            logging.info(f"Данные успешно добавлены для {symbol}, текущий размер: {len(data)} строк")

        except Exception as e:
            logging.error(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")
            save_logs_to_file(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")

        current_date = next_date

    if data.empty:
        logging.warning(f"Все данные для {symbol} пусты.")
    return data


# Функция загрузки данных с использованием многопоточности

def load_data_for_periods(symbols, periods, interval="1m"):
    all_data = {}  # Изменено на словарь
    logging.info(f"Начало загрузки данных за заданные периоды для символов: {symbols}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for period in periods:
            start_date = datetime.strptime(period["start"], "%Y-%m-%d")
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            for symbol in symbols:
                futures.append((symbol, executor.submit(get_historical_data, symbol, interval, start_date, end_date)))

        for symbol, future in futures:
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    if symbol not in all_data:
                        all_data[symbol] = []
                    all_data[symbol].append(symbol_data)
                    logging.info(f"Данные добавлены для {symbol}. Текущий размер: {len(symbol_data)} строк.")
            except Exception as e:
                logging.error(f"Ошибка загрузки данных для {symbol}: {e}")

    # Объединяем данные для каждого символа
    for symbol in all_data:
        if all_data[symbol]:
            all_data[symbol] = pd.concat(all_data[symbol])
            
    if all_data:
        return all_data  # Возвращаем словарь с данными по каждой монете
    else:
        logging.error("Не удалось загрузить данные.")
        raise ValueError("Не удалось загрузить данные.")
    
    
'''def aggregate_to_2min(data):
    """
    Агрегация данных с интервала 1 минута до 2 минут.
    """
    logging.info("Агрегация данных с интервала 1 минута до 2 минут")

    # Проверка, установлен ли временной индекс
    if not isinstance(data.index, pd.DatetimeIndex):
        # Проверка и установка временного индекса
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Преобразуем в datetime
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

        # Убедитесь, что индекс является DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Индекс данных не является DatetimeIndex даже после преобразования.")

    # Агрегация данных
    data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logging.info(f"Агрегация завершена, размер данных: {len(data)} строк")
    logging.info(f"После агрегации данных на 2 минуты: NaN = {data.isna().sum().sum()}")
    return data'''


# Извлечение признаков
def extract_features(data):
    logging.info("Извлечение признаков для флэтового рынка")
    data = data.copy()
    
    # 1. Базовые метрики
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # 2. Метрики диапазона для флэта
    data['range_width'] = data['high'] - data['low']
    data['range_stability'] = data['range_width'].rolling(10).std()
    data['range_ratio'] = data['range_width'] / data['range_width'].rolling(20).mean()
    data['price_in_range'] = (data['close'] - data['low']) / data['range_width']
    
    # 3. Быстрые индикаторы для HFT
    data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
    data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema_8'] = data['close'].ewm(span=8, adjust=False).mean()
    
    # 4. Короткие осцилляторы
    data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()
    
    # 5. Волатильность малых периодов
    bb = BollingerBands(data['close'], window=10)
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data['atr_5'] = AverageTrueRange(data['high'], data['low'], data['close'], window=5).average_true_range()
    
    # 6. Объемные показатели
    data['volume_ma'] = data['volume'].rolling(10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['volume_stability'] = data['volume'].rolling(10).std() / data['volume_ma']
    
    # 7. Индикаторы пробоя
    data['breakout_intensity'] = abs(data['close'] - data['close'].shift(1)) / data['range_width']
    data['false_breakout'] = (data['high'] > data['high'].shift(1)) & (data['close'] < data['close'].shift(1))
    
    # 8. Микро-паттерны
    data['micro_trend'] = np.where(
        data['close'] > data['close'].shift(1), 1,
        np.where(data['close'] < data['close'].shift(1), -1, 0)
    )
    data['micro_trend_change'] = (data['micro_trend'] != data['micro_trend'].shift(1)).astype(int)
    
    # 9. Целевая переменная для эффективного прогнозирования минутных движений
    future_return = (data['close'].shift(-1) - data['close']) / data['close']
    
    data['target'] = np.where(
        future_return > 0, 1,  # Рост цены
        0                      # Падение цены
    )
    
    # Удаляем последнюю строку, так как для неё нет будущей цены
    data = data[:-1]
    
    return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def clean_data(X, y):
    """
    Очистка данных: удаление пропущенных значений и дубликатов, синхронизация индексов.
    """
    # Удаление строк с пропущенными значениями
    mask = X.notnull().all(axis=1)
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]  # Синхронизация индексов
    logging.debug(f"После удаления пропусков: X = {X_clean.shape}, y = {y_clean.shape}")

    # Удаление дубликатов в X
    duplicated_indices = X_clean.index.duplicated(keep='first')
    X_clean = X_clean.loc[~duplicated_indices]
    y_clean = y_clean.loc[~duplicated_indices]  # Синхронизация индексов
    logging.debug(f"После удаления дубликатов: X = {X_clean.shape}, y = {y_clean.shape}")

    # Проверка индексов
    if not X_clean.index.equals(y_clean.index):
        raise ValueError("Индексы X и y не совпадают после очистки данных")

    return X_clean, y_clean


# Удаление выбросов
def remove_outliers(data):
    numeric_data = data.select_dtypes(include=[np.number])  # Только числовые столбцы
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered_data

# Кластеризация для выделения рыночных сегментов
def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_10', 'sma_30', 'ema_50', 'ema_200',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    available_features = [feature for feature in features_for_clustering if feature in data.columns]
    missing_features = [feature for feature in features_for_clustering if feature not in data.columns]

    if missing_features:
        logging.warning(f"Пропущенные признаки для кластеризации: {missing_features}")

    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[available_features])
    return data


# Аугментация данных (добавление шума)
def augment_data(data):
    noise = np.random.normal(0, 0.01, data.shape)
    augmented_data = data + noise
    augmented_data.index = data.index  # Восстанавливаем исходные индексы
    return augmented_data

# Функции для SMOTETomek
def smote_process(X_chunk, y_chunk, chunk_id):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_chunk, y_chunk)
    return X_resampled, y_resampled

def parallel_smote(X, y, n_chunks=4):
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        logging.warning("В данных недостаточно классов для SMOTETomek. Добавляем аугментацию.")
        
        # Аугментация для создания второго класса
        minority_class = y.value_counts().idxmin()
        X_minority = pd.DataFrame(X[y == minority_class], columns=X.columns) if isinstance(X, pd.DataFrame) else pd.DataFrame(X[y == minority_class])
        X_augmented = X_minority + np.random.normal(0, 0.01, X_minority.shape)
        y_augmented = pd.Series([minority_class] * len(X_augmented))
        
        X = pd.concat([pd.DataFrame(X), X_augmented])
        y = pd.concat([pd.Series(y), y_augmented])

    X_chunks = np.array_split(X, n_chunks)
    y_chunks = np.array_split(y, n_chunks)
    results = Parallel(n_jobs=n_chunks)(
        delayed(smote_process)(X_chunk, y_chunk, idx)
        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks))
    )
    X_resampled = np.vstack([res[0] for res in results])
    y_resampled = np.hstack([res[1] for res in results])
    return X_resampled, y_resampled


# Подготовка данных для модели
def prepare_data(data):
    logging.info("Начало подготовки данных")

    # Проверка на пустые данные
    if data.empty:
        raise ValueError("Входные данные пусты")

    logging.info(f"Исходная форма данных: {data.shape}")

    # Убедимся, что временной индекс установлен
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.")
        
     # Детектируем аномалии
    data = detect_anomalies(data)
    logging.info("Аномалии обнаружены и помечены")

    # Применяем фильтрацию шума
    data = remove_noise(data)
    logging.info("Шум отфильтрован")

    # Добавляем подтверждение объемом
    data = validate_volume_confirmation(data)
    logging.info("Добавлены признаки подтверждения объемом")

    # Данные содержат несколько монет - добавляем межмонетные признаки
    if isinstance(data, dict):
        data = calculate_cross_coin_features(data)
        logging.info("Добавлены межмонетные признаки")

    # Извлечение признаков
    data = extract_features(data)
    logging.info(f"После извлечения признаков: {data.shape}")

    # Удаление выбросов
    data = remove_outliers(data)
    logging.info(f"После удаления выбросов: {data.shape}")

    # Добавление кластеризационного признака
    data = add_clustering_feature(data)
    logging.info(f"После кластеризации: {data.shape}")

    # Список признаков
    features = [col for col in data.columns if col != 'target']
    logging.info(f"Количество признаков: {len(features)}")
    logging.info(f"Список признаков: {features}")
    logging.info(f"Распределение target:\n{data['target'].value_counts()}")

    return data, features


def get_checkpoint_path(model_name, market_type):
    """
    Создает уникальный путь к чекпоинтам для каждой модели.
    
    Args:
        model_name (str): Название модели ('rf', 'xgb', 'lgbm', etc.)
        market_type (str): Тип рынка ('bullish', 'bearish', 'flat')
    
    Returns:
        str: Путь к директории чекпоинтов
    """
    checkpoint_path = os.path.join("checkpoints", market_type, model_name)
    ensure_directory(checkpoint_path)
    return checkpoint_path

def check_class_balance(y):
    """Проверка баланса классов"""
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    
    logging.info("Распределение классов:")
    for class_label, count in class_counts.items():
        percentage = (count / total) * 100
        logging.info(f"Класс {class_label}: {count} примеров ({percentage:.2f}%)")
    
    # Проверяем дисбаланс
    if class_counts.max() / class_counts.min() > 10:
        logging.warning("Сильный дисбаланс классов (>10:1)")
        

def check_feature_quality(X, y):
    """Проверка качества признаков"""
    # Проверяем дисперсию
    zero_var_features = []
    for col in X.columns:
        if X[col].std() == 0:
            zero_var_features.append(col)
    if zero_var_features:
        logging.warning(f"Признаки с нулевой дисперсией: {zero_var_features}")
    
    # Проверяем корреляции
    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    if high_corr_pairs:
        logging.warning(f"Сильно коррелирующие признаки: {high_corr_pairs}")
    
    # Проверяем важность признаков
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    logging.info("Топ-10 важных признаков:")
    logging.info(feature_scores.head(10))



# Обучение модели
def train_model(model, X_train, y_train, name):
    logging.info(f"Начало обучения модели {name}")
    model.fit(X_train, y_train)
    return model

def load_progress(base_learners, meta_model, checkpoint_path):
    """
    Загружает прогресс для базовых моделей и мета-модели из контрольных точек.
    """
    for i, (name, model) in enumerate(base_learners):
        intermediate_path = f"{checkpoint_path}_{name}.pkl"
        if os.path.exists(intermediate_path):
            logging.info(f"Загрузка прогресса модели {name} из {intermediate_path}")
            base_learners[i] = (name, joblib.load(intermediate_path))
        else:
            logging.info(f"Контрольная точка для {name} не найдена. Начало обучения с нуля.")
    
    meta_model_path = f"{checkpoint_path}_meta.pkl"
    if not os.path.exists(os.path.dirname(meta_model_path)):
        os.makedirs(os.path.dirname(meta_model_path))
    if os.path.exists(meta_model_path):
        logging.info(f"Загрузка прогресса мета-модели из {meta_model_path}")
        meta_model = joblib.load(meta_model_path)
    else:
        logging.info("Контрольная точка для мета-модели не найдена. Начало обучения с нуля.")
    
    return base_learners, meta_model


# Обучение ансамбля моделей
def train_ensemble_model(data, selected_features, model_filename='flat_stacked_ensemble_model.pkl'):
    """
    Обучает ансамбль моделей для флэтового рынка
    """
    logging.info("Начало обучения ансамбля моделей для флэтового рынка")
    
    # Проверки входных данных
    if data.empty:
        raise ValueError("Входные данные пусты")
        
    if not isinstance(selected_features, list):
        raise TypeError("selected_features должен быть списком")
    
    # Проверяем, что target не в списке признаков
    assert all(feat != 'target' for feat in selected_features), "target не должен быть в списке признаков"
        
    logging.info(f"Входные данные shape: {data.shape}")
    logging.info(f"Входные колонки: {data.columns.tolist()}")

    # Проверка наличия целевой переменной и её распределения
    if 'target' not in data.columns:
        raise KeyError("Отсутствует колонка 'target' во входных данных")
    
    target_dist = data['target'].value_counts()
    if len(target_dist) < 2:
        raise ValueError(f"Недостаточно классов в target: {target_dist}")
    
    # Сначала сохраняем target, затем создаём X из признаков
    y = data['target'].copy()
    X = data[selected_features].copy()
    
    logging.info(f"Распределение классов до обучения:\n{y.value_counts()}")
    logging.info(f"Размеры данных: X = {X.shape}, y = {y.shape}")
    
    # Очистка данных
    X_clean, y_clean = clean_data(X, y)
    logging.info(f"После clean_data: X = {X_clean.shape}, y = {y_clean.shape}")
    
    # Проверка распределения классов после очистки
    if len(pd.unique(y_clean)) < 2:
        raise ValueError(f"После очистки остался только один класс: {pd.value_counts(y_clean)}")
    
    # Удаление выбросов с адаптацией для флэтового рынка
    combined_data = pd.concat([X_clean, y_clean], axis=1)
    combined_data_cleaned = remove_outliers(combined_data)
    X_clean = combined_data_cleaned.drop(columns=['target'])
    y_clean = combined_data_cleaned['target']
    logging.info(f"После remove_outliers: X = {X_clean.shape}, y = {y_clean.shape}")
    assert X_clean.index.equals(y_clean.index), "Индексы X и y не совпадают после удаления выбросов!"
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    logging.info(f"Train size: X = {X_train.shape}, y = {y_train.shape}")
    logging.info(f"Test size: X = {X_test.shape}, y = {y_test.shape}")
    
    # Масштабирование с использованием RobustScaler для устойчивости к выбросам
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Аугментация данных с меньшим шумом для флэтового рынка
    X_augmented = augment_data(pd.DataFrame(X_train_scaled, columns=X_train.columns))
    logging.info(f"После аугментации: X = {X_augmented.shape}")

    # SMOTETomek с адаптированными параметрами для флэтового рынка
    logging.info("Применение SMOTETomek")
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_augmented, y_train)
    logging.info(f"После SMOTETomek: X = {X_resampled.shape}, y = {y_resampled.shape}")
    logging.info(f"Распределение классов после SMOTETomek:\n{pd.Series(y_resampled).value_counts()}")
    
    check_class_balance(y_resampled)
    check_feature_quality(X_resampled, y_resampled)
    
    # Попытка загрузки ансамбля
    if os.path.exists(ensemble_checkpoint_path):
        logging.info(f"[Ensemble] Загрузка ансамбля из {ensemble_checkpoint_path}")
        saved_data = joblib.load(ensemble_checkpoint_path)
        return saved_data["ensemble_model"], saved_data["scaler"], saved_data["features"]

    # Инициализация базовых моделей с параметрами для флэтового рынка
    rf_model = CheckpointRandomForest(
        n_estimators=150,  # Увеличено для лучшего обнаружения паттернов
        max_depth=5,       # Уменьшено для предотвращения переобучения
        min_samples_leaf=8 # Увеличено для стабильности
    )

    gb_model = CheckpointGradientBoosting(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,  # Уменьшено для более плавного обучения
        subsample=0.85
    )

    xgb_model = CheckpointXGBoost(
        n_estimators=120,
        max_depth=4,
        subsample=0.85,
        min_child_weight=6,
        learning_rate=0.008  # Уменьшено для флэтового рынка
    )

    lgbm_model = CheckpointLightGBM(
        n_estimators=120,
        num_leaves=20,
        learning_rate=0.05,
        min_data_in_leaf=8,
        random_state=42
    )

    catboost_model = CheckpointCatBoost(
        iterations=250,
        depth=5,
        learning_rate=0.05,
        min_data_in_leaf=8,
        save_snapshot=False
    )
    
    # Оптимизированные веса для флэтового рынка
    meta_weights = {
        'rf': 0.25,     # Увеличен вес для стабильности
        'gb': 0.2,      # Средний вес
        'xgb': 0.15,    # Уменьшен вес из-за склонности к переобучению
        'lgbm': 0.2,    # Средний вес
        'catboost': 0.2 # Средний вес
    }

    # Список базовых моделей
    base_learners = [
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('catboost', catboost_model)
    ]

    # Обучение базовых моделей
    for name, model in base_learners:
        checkpoint_path = get_checkpoint_path(name, market_type)
        final_checkpoint = os.path.join(checkpoint_path, f"{name}_final.joblib")
        
        logging.info(f"[Ensemble] Обучение модели {name}")
        model.fit(X_resampled, y_resampled)
        joblib.dump(model, final_checkpoint)
        logging.info(f"[Ensemble] Модель {name} сохранена в {final_checkpoint}")

    # Обучение мета-модели с адаптированными параметрами для флэтового рынка
    logging.info("[Ensemble] Обучение стекинг-ансамбля")
    meta_model = LogisticRegression(
        C=0.08,              
        class_weight='balanced',
        max_iter=30000,     
        tol=1e-8,          
        solver='saga',       
        n_jobs=-1,          
        random_state=42 
    )
    
    # Очистка старых чекпоинтов
    for name, _ in base_learners:
        checkpoint_dir = os.path.join(checkpoint_base_dir, name)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            ensure_directory(checkpoint_dir)

    X_resampled_scaled = RobustScaler().fit_transform(X_resampled)
    
    # Создание и обучение финального ансамбля
    ensemble_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        passthrough=True,
        cv=5,
        n_jobs=1  # Уменьшаем параллелизм для избежания конфликтов при сохранении
    )
    ensemble_model.fit(X_resampled_scaled, y_resampled)

    # Оценка метрики на тестовом наборе
    y_pred = ensemble_model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logging.info(f"F1-Score: {f1}, Precision: {precision}, Recall: {recall}")

    # Сохранение ансамбля, скейлера и признаков
    save_data = {
        "ensemble_model": ensemble_model,
        "scaler": scaler,
        "features": selected_features
    }
    ensure_directory(os.path.dirname(ensemble_checkpoint_path))
    joblib.dump(save_data, ensemble_checkpoint_path)
    logging.info(f"[Ensemble] Ансамбль сохранен в {ensemble_checkpoint_path}")
    
    return ensemble_model, scaler, selected_features


# Основной запуск
if __name__ == "__main__":
    
    # Инициализация клиента Binance
    client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT','XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'TONUSDT', 'NEARUSDT']

    # Периоды флэтового рынка
    flat_periods = [
        {"start": "2019-01-01", "end": "2019-12-31"},  # Полный период консолидации 2019
        {"start": "2020-09-01", "end": "2020-12-31"}   # Предбычий флэт конца 2020
    ]


    try:
        data_dict = load_data_for_periods(symbols, bearish_periods, interval="1m")
        logging.info("Данные успешно загружены")
    
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")

    data, selected_features = prepare_data(data_dict)
    logging.debug(f"Доступные столбцы после подготовки данных: {data.columns.tolist()}")
    logging.debug(f"Выбранные признаки: {selected_features}")
        

    try:
        prepared_data, selected_features = prepare_data(data)
        ensemble_model, scaler, features = train_ensemble_model(prepared_data, selected_features, ensemble_model_filename)
        logging.info("Обучение ансамбля моделей завершено!")
        save_logs_to_file("Обучение ансамбля моделей завершено!")
    except Exception as e:
        logging.error(f"Ошибка при обучении ансамбля: {e}")
        sys.exit(1)

    logging.info("Программа завершена успешно.")
    sys.exit(0)