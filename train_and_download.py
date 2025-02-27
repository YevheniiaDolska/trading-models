import os
import subprocess
import sys
import argparse
import requests

# === 1️⃣ Переменные окружения ===
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
GIT_REPO = "https://github.com/YevheniiaDolska/tr.git"
BASE_DIR = "/workspace/tr"
OUTPUT_DIR = "/workspace/tr/output_models"
GOOGLE_DRIVE_FOLDER_ID = "1JCoUN-wQ2iIk5D6DiUoTj9PhS44lTnAp"  # Твоя папка на Google Drive

# Папки для разделения моделей
NEURAL_NETWORKS_OUTPUT = os.path.join(OUTPUT_DIR, "Neural_Networks")
ENSEMBLE_MODELS_OUTPUT = os.path.join(OUTPUT_DIR, "Ensemble_Models")

# Создаём папки
os.makedirs(NEURAL_NETWORKS_OUTPUT, exist_ok=True)
os.makedirs(ENSEMBLE_MODELS_OUTPUT, exist_ok=True)

if not RUNPOD_API_KEY:
    print("❌ API-ключ RunPod не найден!")
    sys.exit(1)

# === 2️⃣ Получаем POD_ID ===
parser = argparse.ArgumentParser()
parser.add_argument("--pod_id", type=str, help="ID пода RunPod")
args = parser.parse_args()
POD_ID = args.pod_id or os.getenv("POD_ID")

if not POD_ID:
    print("❌ POD_ID не задан!")
    sys.exit(1)

print(f"✅ Используется POD_ID: {POD_ID}")

# === 3️⃣ Клонируем репозиторий ===
if not os.path.exists(BASE_DIR):
    print("🚀 Клонируем репозиторий...")
    subprocess.run(["git", "clone", GIT_REPO, BASE_DIR], check=True)
else:
    print("🔄 Обновляем код из репозитория...")
    subprocess.run(["git", "-C", BASE_DIR, "pull"], check=True)

# === 4️⃣ Пути к моделям ===
NEURAL_NETWORKS_DIR = os.path.join(BASE_DIR, "neural_networks")
ENSEMBLE_MODELS_DIR = os.path.join(BASE_DIR, "ensemble_models")

# === 5️⃣ Установка зависимостей ===
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy", "tensorflow[and-cuda]==2.12.0", "tensorflow-addons",
    "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "lightgbm", "joblib",
    "ta", "pandas-ta", "python-binance", "filterpy", "requests", "gdown"
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

# === 7️⃣ Запускаем обучение и сохраняем модели ===
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
            save_path = os.path.join(ENSEMBLE_MODELS_OUTPUT, f"{model_name}.h5")
        else:
            model_path = os.path.join(NEURAL_NETWORKS_DIR, model_file)
            save_path = os.path.join(NEURAL_NETWORKS_OUTPUT, f"{model_name}.h5")

        if os.path.exists(model_path):
            print(f"🟢 Обучение модели: {model_file}")
            try:
                subprocess.run(["python3", model_path], check=True)

                # ✅ Сохраняем обученную модель
                if os.path.exists(save_path):
                    print(f"✅ Модель {model_name} сохранена в {save_path}!")
                else:
                    print(f"⚠ Модель {model_name} не найдена после обучения!")

            except subprocess.CalledProcessError:
                print(f"⚠ Ошибка при обучении модели: {model_file}")
        else:
            print(f"⚠ Файл модели не найден: {model_path}")

train_models()

# === 8️⃣ Архивируем обученные модели ===
print("\n📦 Архивируем модели для Google Drive...")
subprocess.run(["zip", "-r", "/workspace/neural_networks.zip", NEURAL_NETWORKS_OUTPUT], check=True)
subprocess.run(["zip", "-r", "/workspace/ensemble_models.zip", ENSEMBLE_MODELS_OUTPUT], check=True)
print("✅ Архивация завершена!")

# === 9️⃣ Загружаем в Google Drive в правильные папки ===
def upload_to_drive(file_path, folder_name):
    print(f"\n🚀 Загружаем {file_path} в папку {folder_name} на Google Drive...")
    subprocess.run([
        "gdown", "--folder", "--id", GOOGLE_DRIVE_FOLDER_ID, file_path
    ], check=True)
    print(f"✅ Файл {file_path} загружен в {folder_name} на Google Drive!")

upload_to_drive("/workspace/neural_networks.zip", "Neural_Networks")
upload_to_drive("/workspace/ensemble_models.zip", "Ensemble_Models")

# === 🔟 Завершаем под ===
if RUNPOD_API_KEY:
    print("\n🔧 Останавливаем и удаляем под RunPod...")
    try:
        requests.post(f"https://api.runpod.io/v2/pod/{POD_ID}/stop", headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"})
        requests.delete(f"https://api.runpod.io/v2/pod/{POD_ID}", headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"})
        print(f"✅ Под {POD_ID} остановлен и удалён.")
    except Exception as e:
        print(f"⚠ Ошибка при управлении RunPod: {e}")

print("\n🎉 Обучение завершено! Файлы `neural_networks.zip` и `ensemble_models.zip` загружены в Google Drive.")
