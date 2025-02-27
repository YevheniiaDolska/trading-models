import os
import subprocess
import sys
import argparse
import requests

# === 1️⃣ Переменные окружения ===
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
GIT_REPO = "git@github.com:YevheniiaDolska/tr.git"  # Твой приватный репозиторий
BASE_DIR = "/workspace/tr"  # Папка, в которую клонируется код

if not RUNPOD_API_KEY:
    print("❌ API-ключ RunPod не найден! Установите его перед запуском.")
    sys.exit(1)

# === 2️⃣ Получаем POD_ID ===
parser = argparse.ArgumentParser()
parser.add_argument("--pod_id", type=str, help="ID пода RunPod (если не передан, берётся из переменной окружения)")
args = parser.parse_args()
POD_ID = args.pod_id or os.getenv("POD_ID")

if not POD_ID:
    print("❌ POD_ID не задан! Передайте его через `--pod_id` или экспортируйте `export POD_ID=...`")
    sys.exit(1)

print(f"✅ Используется POD_ID: {POD_ID}")

# === 3️⃣ Клонируем или обновляем репозиторий ===
if not os.path.exists(BASE_DIR):
    print("🚀 Клонируем приватный репозиторий...")
    subprocess.run(["git", "clone", GIT_REPO, BASE_DIR], check=True)
else:
    print("🔄 Обновляем код из репозитория...")
    subprocess.run(["git", "-C", BASE_DIR, "pull"], check=True)

# === 4️⃣ Пути к моделям ===
NEURAL_NETWORKS_DIR = os.path.join(BASE_DIR, "neural_networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_DIR, "ensemble_models")

# === 5️⃣ Установка зависимостей ===
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm", "joblib",
    "ta", "pandas-ta", "python-binance", "filterpy", "requests"
]

def install_packages():
    print("✅ Устанавливаем зависимости...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + REQUIRED_PACKAGES, check=True)

install_packages()

# === 6️⃣ Проверяем GPU ===
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

# === 7️⃣ Запускаем обучение ===
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
        if "ensemble" in model_file:
            model_path = os.path.join(ENSEMBLE_MODELS_DIR, model_file)
        else:
            model_path = os.path.join(NEURAL_NETWORKS_DIR, model_file)
        
        if os.path.exists(model_path):
            print(f"🟢 Обучение модели: {model_file}")
            try:
                subprocess.run(["python3", model_path], check=True)
            except subprocess.CalledProcessError:
                print(f"⚠ Ошибка при обучении модели: {model_file}")
        else:
            print(f"⚠ Файл модели не найден: {model_path}")

train_models()

# === 8️⃣ Остановка пода после завершения ===
if RUNPOD_API_KEY:
    print("\n🔧 Работаем с RunPod API...")
    try:
        response = requests.post(
            f"https://api.runpod.io/v2/pod/{POD_ID}/stop",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        if response.status_code == 200:
            print(f"✅ Под {POD_ID} остановлен.")

        response = requests.delete(
            f"https://api.runpod.io/v2/pod/{POD_ID}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        if response.status_code == 200:
            print(f"✅ Под {POD_ID} удалён.")
    except Exception as e:
        print(f"⚠ Ошибка при управлении RunPod: {e}")

print("\n🎉 Обучение завершено, под остановлен!")
