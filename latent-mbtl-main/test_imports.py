# test_imports.py
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib not installed")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ Scikit-learn not installed")

try:
    import jupyter
    print(f"✓ Jupyter installed")
except ImportError:
    print("✗ Jupyter not installed")

print("\nAll required packages are installed!" if all([
    'torch' in dir(), 'numpy' in dir(), 'matplotlib' in dir(), 
    'sklearn' in dir(), 'jupyter' in dir()
]) else "\nSome packages are missing.")