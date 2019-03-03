ALL = matmul sor lstm

all: $(ALL)

%: %.cpp
	g++ -Wall -g -DMATRIX_TYPE=float -march=native -O3 -fopenmp -ftree-vectorize --std=c++11 $< -o $@

clean:
	rm -f $(ALL)

submission.zip: lstm.cpp matmul.cpp sor.cpp writeup.pdf
	zip -r $@ $^

submission: submission.zip
