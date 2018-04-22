
.DEFAULT_GOAL := main

CU = nvcc
CUFLAGS = -Wno-deprecated-gpu-targets -std=c++11 --expt-extended-lambda -I. -rdc=true -arch=sm_61 

main: DatasetGeneration.o main.o HelperFunctions.o kmean.o KOptimizer.o
	$(CU) $(CUFLAGS) -o $@ $^ -lcuda -lcudadevrt

DatasetGeneration.o: DatasetGeneration.cu
	$(CU) $(CUFLAGS) -c DatasetGeneration.cu

main.o: main.cu 
	$(CU) $(CUFLAGS) -c main.cu

HelperFunctions.o: 
	$(CU) $(CUFLAGS) -c HelperFunctions.cu

kmean.o: kmean.cu
	$(CU) $(CUFLAGS) -c kmean.cu

KOptimizer.o: KOptimizer.cu
	$(CU) $(CUFLAGS) -c KOptimizer.cu

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


