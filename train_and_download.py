import os
import subprocess
import sys
import time
import requests

# === 1️⃣ УСТАНАВЛИВАЕМ API-КЛЮЧ (Перед запуском в Jupyter Notebook) ===
#os.environ["RUNPOD_API_KEY"] = "твой_ключ"  # ⚠️ НЕ ХРАНИ API-КЛЮЧ В ФАЙЛЕ ПУБЛИЧНО!

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    print("❌ API-ключ не найден! Установите его перед запуском.")
    sys.exit(1)

# === 2️⃣ ЛОКАЛЬНЫЕ ПАПКИ ДЛЯ СОХРАНЕНИЯ ===
BASE_LOCAL_DIR = r"C:\Users\Kroha\Documents\Auto-Blogging SaaS\devenv\Trading Bot\New Logic\Divided\3 models with a switcher"
NEURAL_NETWORKS_DIR = os.path.join(BASE_LOCAL_DIR, "Neural_Networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_LOCAL_DIR, "Ensemble_Models")

os.makedirs(NEURAL_NETWORKS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODELS_DIR, exist_ok=True)

# === 3️⃣ УСТАНОВКА ЗАВИСИМОСТЕЙ ===
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy", "tensorflow==2.11.0", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm", "joblib",
    "ta", "pandas-ta", "python-binance", "filterpy", "requests"
]

def install_packages():
    print("✅ Проверяем установку библиотек...")
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠ Ошибка установки пакета: {package}")

install_packages()

# === 4️⃣ ПРОВЕРКА GPU ===
def check_gpu():
    print("\n🔍 Проверяем доступность GPU...")
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if output.returncode == 0:
            print("✅ GPU доступен!")
            print(output.stdout)
        else:
            print("⚠ GPU не найден, обучение будет идти на CPU.")
    except FileNotFoundError:
        print("⚠ nvidia-smi не найден, вероятно, GPU недоступен.")

check_gpu()

# === 5️⃣ ЗАПУСК ОБУЧЕНИЯ И СКАЧИВАНИЕ ПОСЛЕ КАЖДОЙ МОДЕЛИ ===
MODELS_DIR = "/workspace/trading-models/neural_networks"
OUTPUT_DIR = "/workspace/trading-models/output/neural_networks"

MODELS = {
    "market_condition_classifier.py": "Market_Classifier",
    "bullish_neural_network.py": "Neural_Bullish",
    "bullish_ensemble.py": "Ensemble_Bullish",
    "flat_neural_network.py": "Neural_Flat",
    "flat_ensemble.py": "Ensemble_Flat",
    "bearish_neural_network.py": "Neural_Bearish",
    "bearish_ensemble.py": "Ensemble_Bearish"
}

def train_models():
    print("\n🚀 Запускаем обучение моделей...")
    for model_file, model_name in MODELS.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_path):
            print(f"🟢 Обучение модели: {model_file}")
            try:
                subprocess.run(["python3", model_path], check=True)

                # ✅ ПОСЛЕ ОБУЧЕНИЯ СРАЗУ СКАЧИВАЕМ МОДЕЛЬ
                print(f"📥 Копируем обученную модель {model_name} в локальную папку...")

                trained_model_path = os.path.join(OUTPUT_DIR, f"{model_name}.h5")
                
                if "Ensemble" in model_name:
                    local_model_path = os.path.join(ENSEMBLE_MODELS_DIR, f"{model_name}.h5")
                else:
                    local_model_path = os.path.join(NEURAL_NETWORKS_DIR, f"{model_name}.h5")

                if os.path.exists(trained_model_path):
                    subprocess.run(["cp", trained_model_path, local_model_path])
                    print(f"✅ Модель {model_name} успешно сохранена в {local_model_path}!")
                else:
                    print(f"⚠ Модель {model_name} не найдена!")

            except subprocess.CalledProcessError:
                print(f"⚠ Ошибка при обучении модели: {model_file}")
        else:
            print(f"⚠ Файл модели не найден: {model_path}")

train_models()

# === 6️⃣ ОСТАНОВКА ПОДА В RUNPOD ПОСЛЕ ОБУЧЕНИЯ ===
if RUNPOD_API_KEY:
    print("\n🔧 Работаем с RunPod...")

    # Получаем список подов
    try:
        response = requests.get(
            "https://api.runpod.io/v2/pod/list",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        pods = response.json()
        if "pods" in pods and pods["pods"]:
            POD_ID = pods["pods"][0]["id"]
            print(f"✅ Найден под с ID: {POD_ID}")

            # Завершаем под
            requests.post(
                f"https://api.runpod.io/v2/pod/{POD_ID}/stop",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            print(f"✅ Под {POD_ID} остановлен.")

            # Удаляем под
            requests.delete(
                f"https://api.runpod.io/v2/pod/{POD_ID}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            print(f"✅ Под {POD_ID} удалён.")
        else:
            print("⚠ Подов не найдено.")
    except Exception as e:
        print(f"⚠ Ошибка при управлении RunPod: {e}")

print("\n🎉 Обучение завершено, все модели скачаны, под остановлен!")
