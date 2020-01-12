################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../lib/libviennacl/src/backend.cpp \
../lib/libviennacl/src/blas1.cpp \
../lib/libviennacl/src/blas1_host.cpp \
../lib/libviennacl/src/blas1_opencl.cpp \
../lib/libviennacl/src/blas2.cpp \
../lib/libviennacl/src/blas2_host.cpp \
../lib/libviennacl/src/blas2_opencl.cpp \
../lib/libviennacl/src/blas3.cpp \
../lib/libviennacl/src/blas3_host.cpp \
../lib/libviennacl/src/blas3_opencl.cpp 

OBJS += \
./lib/libviennacl/src/backend.o \
./lib/libviennacl/src/blas1.o \
./lib/libviennacl/src/blas1_host.o \
./lib/libviennacl/src/blas1_opencl.o \
./lib/libviennacl/src/blas2.o \
./lib/libviennacl/src/blas2_host.o \
./lib/libviennacl/src/blas2_opencl.o \
./lib/libviennacl/src/blas3.o \
./lib/libviennacl/src/blas3_host.o \
./lib/libviennacl/src/blas3_opencl.o 

CPP_DEPS += \
./lib/libviennacl/src/backend.d \
./lib/libviennacl/src/blas1.d \
./lib/libviennacl/src/blas1_host.d \
./lib/libviennacl/src/blas1_opencl.d \
./lib/libviennacl/src/blas2.d \
./lib/libviennacl/src/blas2_host.d \
./lib/libviennacl/src/blas2_opencl.d \
./lib/libviennacl/src/blas3.d \
./lib/libviennacl/src/blas3_host.d \
./lib/libviennacl/src/blas3_opencl.d 


# Each subdirectory must supply rules for building sources it contributes
lib/libviennacl/src/%.o: ../lib/libviennacl/src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/sunkg/workspace/GPUSR/lib" -I/home/sunkg/workspace/GPUSR/lib/libviennacl/include -I/home/sunkg/workspace/GPUSR/lib/viennacl/linalg -I/usr/include/ -I/usr/include/c++/5/ -I/home/sunkg/workspace/GPUSR/ -I/usr/include/opencv -I/home/sunkg/workspace/GPUSR/lib/OpenCV -I/usr/include/mpi -I/usr/include/eigen3 -include/home/sunkg/workspace/GPUSR/lib/viennacl/ocl/backend.hpp -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


