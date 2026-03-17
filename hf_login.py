"""
Скрипт для авторизации в Hugging Face Hub
"""
from huggingface_hub import login, HfApi
import sys

def main():
    print("="*60)
    print("Hugging Face Hub - Авторизация")
    print("="*60)
    print("\nДля доступа к приватным моделям и загрузки моделей")
    print("вам нужен токен доступа от Hugging Face.")
    print("\nПолучить токен можно здесь:")
    print("https://huggingface.co/settings/tokens")
    print("\n" + "="*60)
    
    token = input("\nВведите ваш токен (или нажмите Enter для пропуска): ").strip()
    
    if not token:
        print("\n⚠ Авторизация пропущена.")
        print("Вы сможете использовать только публичные модели.")
        return
    
    try:
        login(token=token, add_to_git_credential=True)
        print("\n✓ Успешная авторизация!")
        
        # Проверка
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Вы вошли как: {user_info['name']}")
        
    except Exception as e:
        print(f"\n✗ Ошибка авторизации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
