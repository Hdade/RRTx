# Lệnh chạy single thread trên CPU:
```
cd build
make
cp rrtx_cpp.cpython-*.so ..
cd ..
python3 single_thread_main.py
```

# Hoặc:
```
chmod +x single_thread_run.sh
./single_thread_run.sh
```

---

# Lệnh chạy multi-thread trên CPU:
```
cd build
make
cp rrtx_cpp.cpython-*.so ..
cd ..
python3 main.py
```

# Hoặc:
```
chmod +x run.sh
./run.sh
```