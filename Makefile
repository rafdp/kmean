
.DEFAULT_GOAL := main

CU = nvcc
CUFLAGS = -Wno-deprecated-gpu-targets -std=c++11 --expt-extended-lambda 

main: DatasetGeneration.o main.o HelperFunctions.o kmean.o
	$(CU) $(CUFLAGS) -o $@ $^ -lcuda

DatasetGeneration.o: 
	$(CU) $(CUFLAGS) -c DatasetGeneration.cu

main.o: 
	$(CU) $(CUFLAGS) -c main.cu

HelperFunctions.o: 
	$(CU) $(CUFLAGS) -c HelperFunctions.cu

kmean.o:
	$(CU) $(CUFLAGS) -c kmean.cu

r: main
	@echo
	@echo "====== Execution begins: "
	@./main
	@echo "====== Execution ended: "

c: 
	rm -rf main *.o

test: main
	./main 1000 100 20
	gnuplot plot.p
	./giffer.sh


