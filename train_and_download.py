import os
import argparse
import subprocess
import sys
import time
import requests

# === 1️⃣ Получаем API-ключ RunPod ===
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    print("❌ API-ключ не найден! Передайте его через `export RUNPOD_API_KEY=...` перед запуском.")
    sys.exit(1)

# === 2️⃣ Получаем POD_ID (из аргумента или переменной окружения) ===
parser = argparse.ArgumentParser()
parser.add_argument("--pod_id", type=str, help="ID пода RunPod (если не передан, берётся из переменной окружения)")
args = parser.parse_args()

POD_ID = args.pod_id or os.getenv("POD_ID")

if not POD_ID:
    print("❌ POD_ID не задан! Передайте его через `--pod_id` или экспортируйте `export POD_ID=...`")
    sys.exit(1)

print(f"✅ Используется POD_ID: {POD_ID}")

# === 3️⃣ Локальные папки для сохранения моделей ===
BASE_LOCAL_DIR = r"C:\Users\Kroha\Documents\Auto-Blogging SaaS\devenv\Trading Bot\New Logic\Divided\3 models with a switcher"
NEURAL_NETWORKS_DIR = os.path.join(BASE_LOCAL_DIR, "Neural_Networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_LOCAL_DIR, "Ensemble_Models")

os.makedirs(NEURAL_NETWORKS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODELS_DIR, exist_ok=True)

# === 4️⃣ Установка зависимостей ===
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm", "joblib",
    "ta", "pandas-ta", "python-binance", "filterpy", "requests"
]

def install_packages():
    print("✅ Устанавливаем зависимости...")

    print("✅ Удаляем несовместимый numpy (если есть)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], check=False)

    print("✅ Устанавливаем numpy 1.23.5 (совместим с TensorFlow 2.12.0)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==1.23.5"], check=True)

    print("✅ Устанавливаем TensorFlow 2.12.0...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tensorflow==2.12.0"], check=True)

    # Устанавливаем остальные зависимости
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", package], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠ Ошибка установки пакета: {package}")


install_packages()

# === 5️⃣ Проверка доступности GPU ===
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

# === 6️⃣ Запуск обучения и скачивание моделей ===
MODELS_DIR = "/trading-models/neural_networks"
OUTPUT_DIR = "/trading-models/output/neural_networks"

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

                # ✅ Скачиваем модель после обучения
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

# === 7️⃣ Остановка пода в RunPod ===
if RUNPOD_API_KEY:
    print("\n🔧 Работаем с RunPod...")

    # Проверяем, установлен ли runpod
    try:
        subprocess.run(["pip", "install", "runpod"], check=True)
    except subprocess.CalledProcessError:
        print("⚠ Ошибка установки runpod CLI, под не будет остановлен.")
        sys.exit(1)

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
