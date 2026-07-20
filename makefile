all:
	nvcc -O3 -std=c++17 -Xcompiler -fPIC -shared grand_decoder.cu -o libgranddecoder.so

clean:
	rm -f libgranddecoder.so