set -e

echo "Building C++..."

cd build

make -j4

echo "📦 Copying libraries to out..."
cp rrtx_cpp.cpython-*.so ..

cd ..

echo "🚀 Running Visualizer..."
python3 main.py