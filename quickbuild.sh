rm -r build
mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build . -j 16

