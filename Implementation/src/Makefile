# Makefile for the FloorSet-Lite SA floorplanner.

CXX      ?= g++
CXXSTD   ?= -std=c++17
OPT      ?= -O3 -DNDEBUG
WARN     ?= -Wall -Wextra -Wpedantic -Wno-unused-parameter
INCLUDES  = -Iinclude
CXXFLAGS  = $(CXXSTD) $(OPT) $(WARN) $(INCLUDES) -pthread

SRC = src/btree.cpp src/packer.cpp src/cost.cpp src/moves.cpp \
      src/sa.cpp src/parallel.cpp src/parser.cpp src/main.cpp

OBJ = $(SRC:.cpp=.o)
DEP = $(SRC:.cpp=.d)

BIN = floorplanner

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) -pthread

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEP)

debug: CXXFLAGS := $(CXXSTD) -O0 -g -DDEBUG $(WARN) $(INCLUDES) -pthread
debug: clean $(BIN)

clean:
	rm -f $(BIN) $(OBJ) $(DEP)

# A tiny smoke test (uses the bundled benchmark file)
check: $(BIN)
	./$(BIN) benchmarks/toy.txt benchmarks/toy.sol --time 5 --threads 4 --verbose

.PHONY: all clean debug check
