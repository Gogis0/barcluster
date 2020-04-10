ldtw:
	@ echo "Compiling ldtw.main ..."
	c++ -O3 -W -Wall -g -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` dtw.cpp computematrix.cpp dtwmodule.cpp -o ldtw`python3-config --extension-suffix`
	mv ldtw`python3-config --extension-suffix` ../

clean:
	@ echo "Removing the binaries ..."
	rm ldtw
