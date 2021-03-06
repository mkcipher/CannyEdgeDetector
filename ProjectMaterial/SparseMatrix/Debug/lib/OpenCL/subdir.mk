################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../lib/OpenCL/Device.cpp \
../lib/OpenCL/Error.cpp \
../lib/OpenCL/Event.cpp \
../lib/OpenCL/GetError.cpp \
../lib/OpenCL/Program.cpp 

OBJS += \
./lib/OpenCL/Device.o \
./lib/OpenCL/Error.o \
./lib/OpenCL/Event.o \
./lib/OpenCL/GetError.o \
./lib/OpenCL/Program.o 

CPP_DEPS += \
./lib/OpenCL/Device.d \
./lib/OpenCL/Error.d \
./lib/OpenCL/Event.d \
./lib/OpenCL/GetError.d \
./lib/OpenCL/Program.d 


# Each subdirectory must supply rules for building sources it contributes
lib/OpenCL/%.o: ../lib/OpenCL/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_EIGEN -I"/home/sunkg/workspace/GPUSR/lib" -I/home/sunkg/workspace/GPUSR/src -I/home/sunkg/workspace/GPUSR/lib/viennacl/linalg -I/usr/include/eigen3 -I/home/sunkg/workspace/GPUSR/lib/OpenCV -I/usr/lib/gcc/x86_64-linux-gnu/5/include -I/usr/include -I/usr/include/x86_64-linux-gpu -I/usr/include/opencv -I/usr/include/mpi -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


