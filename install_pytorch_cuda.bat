@echo off
echo ============================================
echo Installing PyTorch with CUDA support
echo ============================================
echo.

echo Uninstalling CPU version of PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 12.4 support (torch 2.6+)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ============================================
echo Verifying installation...
echo ============================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
echo Installation complete!
pause
