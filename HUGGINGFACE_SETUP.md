# Настройка Hugging Face Hub

## Что такое Hugging Face Hub?

Hugging Face Hub - это платформа для хранения и обмена моделями машинного обучения. Большинство моделей доступны публично без авторизации.

## Когда нужна авторизация?

Авторизация требуется только для:
- Доступа к приватным моделям
- Загрузки своих моделей на Hub
- Использования некоторых закрытых моделей (например, Llama 2)

## Как авторизоваться?

### Способ 1: Через Python скрипт (рекомендуется)

```bash
python hf_login.py
```

Скрипт попросит ввести токен доступа.

### Способ 2: Через Python код

```python
from huggingface_hub import login

# Введите ваш токен
login(token="your_token_here")
```

### Способ 3: Через переменную окружения

```bash
# Windows
set HF_TOKEN=your_token_here

# Linux/Mac
export HF_TOKEN=your_token_here
```

## Получение токена

1. Зарегистрируйтесь на https://huggingface.co/
2. Перейдите в настройки: https://huggingface.co/settings/tokens
3. Создайте новый токен (Access Token)
4. Скопируйте токен

## Проверка авторизации

```python
from huggingface_hub import HfApi

api = HfApi()
user_info = api.whoami()
print(f"Вы вошли как: {user_info['name']}")
```

## Использование без авторизации

Для большинства публичных моделей авторизация не требуется:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Работает без авторизации
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
```

## Кэширование моделей

Модели автоматически сохраняются локально:
- **В проекте**: `models/` (настроено в скриптах)
- **Системный кэш**: `~/.cache/huggingface/` (по умолчанию)

Повторная загрузка использует локальную копию.

## Полезные команды

### Проверка установки

```bash
python -c "from huggingface_hub import HfApi; print('Hugging Face Hub установлен')"
```

### Очистка кэша

```python
from huggingface_hub import scan_cache_dir

# Посмотреть размер кэша
cache_info = scan_cache_dir()
print(f"Размер кэша: {cache_info.size_on_disk / 1024**3:.2f} GB")

# Очистить кэш (осторожно!)
# cache_info.delete_revisions(*cache_info.repos).execute()
```

## Troubleshooting

### Ошибка: "Repository not found"
- Проверьте правильность имени модели
- Убедитесь, что модель публична или вы авторизованы

### Ошибка: "Token is invalid"
- Проверьте, что токен скопирован полностью
- Создайте новый токен на сайте Hugging Face

### Медленная загрузка
- Используйте локальное кэширование (уже настроено в проекте)
- Проверьте интернет-соединение

## Дополнительная информация

- [Документация Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
- [Список моделей](https://huggingface.co/models)
- [Документация по токенам](https://huggingface.co/docs/hub/security-tokens)
