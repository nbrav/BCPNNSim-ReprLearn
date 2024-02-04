# ml PDC/22.06 rocm/5.0.2 craype-accel-amd-gfx90a anaconda3

SRC_DIR		= ./src_hip
APP_DIR         = ./apps

DEPS 		= $(SRC_DIR)/Pop.h $(SRC_DIR)/Prj.h $(SRC_DIR)/Globals.h $(SRC_DIR)/Pats.h $(SRC_DIR)/Parseparam.h $(SRC_DIR)/Logger.h
OBJS 		= $(SRC_DIR)/Pop.o $(SRC_DIR)/Prj.o $(SRC_DIR)/Globals.o $(SRC_DIR)/Pats.o $(SRC_DIR)/Parseparam.o $(SRC_DIR)/Logger.o

CXX		= hipcc
MPICXX		= hipcc # mpic++ ?

INCLUDE		= -I$(SRC_DIR) # for header files

FLAGS		= -O3
HIP_FLAGS	= -munsafe-fp-atomics --offload-arch=gfx90a -I/opt/rocm-5.0.2/include/rocrand/ -I/opt/rocm-5.0.2/include/hiprand/ -L/opt/rocm-5.0.2/lib/ -munsafe-fp-atomics -I${MPICH_DIR}/include -L${MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a}
MPIXX_FLAGS	= $(HIP_FLAGS) -lmpi -lhipblas -lhiprand ${PE_MPICH_GTL_LIBS_amd_gfx90a}

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(INCLUDE) $(FLAGS) $(HIP_FLAGS)

reprlearnmain: $(APP_DIR)/reprlearn/reprlearnmain.o $(OBJS)
	$(MPICXX) -o $(APP_DIR)/reprlearn/reprlearnmain $^ $(INCLUDE) $(FLAGS) $(MPIXX_FLAGS)

all: reprlearnmain

.PHONY: clean
clean : 
	rm -f *.o *.bin *.log *.png *.gif *.out out.txt err.txt *~ core reprlearnmain
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.bin $(SRC_DIR)/*~
	rm -f $(APP_DIR)/reprlearn/*.o $(APP_DIR)/reprlearn/*~ $(APP_DIR)/reprlearn/reprlearnmain