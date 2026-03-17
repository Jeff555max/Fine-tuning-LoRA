# Установка PyTorch с CUDA для RTX 5060

Этот гайд поможет установить PyTorch с поддержкой CUDA для работы с RTX 5060.

## Проверка текущей установки

Сначала проверьте, установлен ли PyTorch с CUDA:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Если `CUDA доступна: False`, следуйте инструкциям ниже.

## Для RTX 5060 (Архитектура Blackwell, sm_120)

RTX 5060 использует новую архитектуру Blackwell (compute capability 12.0), которая требует:
- PyTorch 2.5+ или nightly build
- CUDA 12.4+

### Вариант 1: PyTorch Nightly (Рекомендуется)

```bash
# Удалите текущую установку
pip uninstall torch torchvision torchaudio bitsandbytes -y

# Установите PyTorch Nightly с CUDA 12.4
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Установите bitsandbytes для quantization
pip install bitsandbytes
```

### Вариант 2: PyTorch 2.5+ (когда выйдет стабильная версия)

```bash
# Удалите текущую установку
pip uninstall torch torchvision torchaudio bitsandbytes -y

# Установите PyTorch 2.5+ с CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установите bitsandbytes
pip install bitsandbytes
```

## Для других GPU (RTX 30xx, RTX 40xx и т.д.)

### CUDA 12.1

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Проверка установки

После установки проверьте:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести:
```
PyTorch: 2.5.0+cu124 (или выше)
CUDA: True
GPU: NVIDIA GeForce RTX 5060
Compute Capability: (12, 0)
```

## Решение проблем

### Ошибка "no kernel image is available for execution"

Это означает, что PyTorch не поддерживает вашу GPU архитектуру. Решение:
1. Установите PyTorch Nightly (см. Вариант 1 выше)
2. Или запускайте обучение БЕЗ `--use_4bit` флага

### Ошибка при использовании bitsandbytes

Если bitsandbytes не работает с RTX 5060:
1. Обновите bitsandbytes: `pip install --upgrade bitsandbytes`
2. Или запускайте БЕЗ quantization (уберите `--use_4bit`)

### CUDA не обнаружена после установки

1. Проверьте, установлены ли драйверы NVIDIA: `nvidia-smi`
2. Убедитесь, что установлена правильная версия PyTorch
3. Перезагрузите компьютер

## Дополнительные ресурсы

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/#start-locally)
