"""
Скрипт для предварительного скачивания моделей
"""
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, cache_dir=None):
    """
    Скачивает модель и токенизатор с HuggingFace
    
    Args:
        model_name: имя модели на HuggingFace
        cache_dir: директория для сохранения (если None, используется ./models/)
    """
    # Определяем директорию для сохранения
    if cache_dir is None:
        cache_dir = os.path.join("models", model_name.replace("/", "_"))
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"СКАЧИВАНИЕ МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    print(f"Директория: {os.path.abspath(cache_dir)}\n")
    
    # Скачивание токенизатора
    print("Скачивание токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(cache_dir)
    print("✓ Токенизатор скачан")
    
    # Скачивание модели
    print("\nСкачивание модели (это может занять время)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.save_pretrained(cache_dir)
    print("✓ Модель скачана")
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"МОДЕЛЬ УСПЕШНО СКАЧАНА")
    print(f"{'='*60}")
    print(f"Параметры: {total_params/1e6:.1f}M")
    print(f"Путь: {os.path.abspath(cache_dir)}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Скачивание моделей с HuggingFace")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Имя модели на HuggingFace (например: microsoft/DialoGPT-small)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Директория для сохранения модели (по умолчанию: ./models/)"
    )
    
    args = parser.parse_args()
    
    try:
        download_model(args.model_name, args.cache_dir)
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}\n")
        print("Проверьте:")
        print("1. Правильность имени модели")
        print("2. Подключение к интернету")
        print("3. Доступность модели на HuggingFace")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
