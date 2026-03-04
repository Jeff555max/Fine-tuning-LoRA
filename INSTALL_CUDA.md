# Установка PyTorch с CUDA для RTX 5060

## Проблема
RTX 5060 имеет CUDA capability sm_120 (Blackwell архитектура), которая требует PyTorch 2.5+ или nightly build.

Если вы видите ошибку:
- `CUDA capability sm_120 is not compatible with the current PyTorch installation`
- `no kernel image is available for execution on the device`

Это означает, что нужна более новая версия PyTorch.

## Решение

### 1. Проверьте текущую версию PyTorch
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')"
```

### 2. Удалите текущий PyTorch и bitsandbytes
```bash
pip uninstall torch torchvision torchaudio bitsandbytes
```

### 3. Установите PyTorch Nightly с CUDA 12.4+ (для RTX 5060 / sm_120)
```bash
# Nightly build с поддержкой sm_120
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### 4. Или установите PyTorch 2.5+ (если доступен)
```bash
pip install torch>=2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 5. Установите bitsandbytes заново (важно!)
```bash
pip install bitsandbytes
```

### 6. Если bitsandbytes не работает, попробуйте без quantization
Запустите обучение БЕЗ флага `--use_4bit`:
```bash
python train.py --model_name "microsoft/DialoGPT-small" --dataset_path "../example_dataset.json"
```

### 5. Проверьте установку
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda}'); cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; print(f'Compute Capability: sm_{cap[0]}{cap[1]}' if cap else 'N/A'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
- `CUDA доступна: True`
- `CUDA версия: 12.4` (или выше)
- `Compute Capability: sm_120` (для RTX 5060)
- `GPU: NVIDIA GeForce RTX 5060`

### 6. Если quantization не работает (ошибка "no kernel image")

RTX 5060 (sm_120) может не поддерживаться bitsandbytes. В этом случае:

**Вариант A: Запустите БЕЗ quantization**
```bash
cd fine_tuning
python train.py --model_name "microsoft/DialoGPT-small" --dataset_path "../example_dataset.json"
# БЕЗ флага --use_4bit
```

**Вариант B: Дождитесь обновления bitsandbytes**
Следите за обновлениями: https://github.com/TimDettmers/bitsandbytes

### 7. Запустите обучение
```bash
cd fine_tuning
# С quantization (если поддерживается)
python train.py --model_name "microsoft/DialoGPT-small" --dataset_path "../example_dataset.json" --use_4bit

# ИЛИ без quantization (гарантированно работает)
python train.py --model_name "microsoft/DialoGPT-small" --dataset_path "../example_dataset.json"
```

## Примечания

- **RTX 5060 (sm_120)** требует PyTorch 2.5+ или nightly build
- Убедитесь, что у вас установлены драйверы NVIDIA последней версии (560+)
- bitsandbytes может не поддерживать sm_120 - используйте обучение без quantization
- Без quantization обучение будет медленнее, но займет больше памяти GPU

