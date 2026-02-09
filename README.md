# RRTx (C++ core + Python visualizer)

Repo này chạy thuật toán RRTx với lõi C++ (pybind11) để tăng tốc, còn phần hiển thị, logic cao cấp ở Python. Thư mục `oldRRTx/` chứa toàn bộ code Python cũ chạy được nhưng chậm, nên hiện tại chuyển sang core C++ để chạy nhanh hơn.

---

## Yêu cầu

- Python 3.12+
- CMake 3.10+
- C++ compiler (MSVC trên Windows, hoặc GCC/Clang trên Linux/macOS)

Các thư viện Python chính:

- numpy
- pygame
- opencv-python
- torch
- torchvision
- pillow
- tqdm

---

## Tạo môi trường ảo

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### Cài thư viện Python
```bash
pip install numpy pygame opencv-python torch torchvision pillow tqdm
```

---

## Build C++ core (pybind11)

### CMake build (cross-platform)
```bash
cmake -S . -B build
cmake --build build --config Release
```

Sau khi build, copy module `rrtx_cpp` về thư mục gốc của repo:

- Linux/macOS: copy file dạng `rrtx_cpp*.so`
- Windows: copy file dạng `rrtx_cpp*.pyd`

Ví dụ (Linux/macOS):
```bash
cp build/rrtx_cpp*.so .
```

Ví dụ (Windows PowerShell):
```powershell
Copy-Item build\Release\rrtx_cpp*.pyd .
```

---

## Chạy demo

### Single-thread
```bash
chmod +x single_thread_run.sh
./single_thread_run.sh
```

### Multi-thread
```bash
chmod +x run.sh
./run.sh
```

### Pure python
```bash
python ./
```

---

## Ghi chú

- `oldRRTx/` là bản Python thuần chạy toàn bộ thuật toán chậm.
- Bản hiện tại tách phần core sang C++ để tăng tốc, Python chỉ giữ phần hiển thị + logic điều khiển.