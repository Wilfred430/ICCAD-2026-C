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

# A statically-linked binary for maximum portability across Linux distros.
# Useful when the contest evaluation server may have a different glibc /
# libstdc++ than your build machine. Slightly larger (~3-5 MB) but no
# runtime dependencies beyond the kernel.
static: clean
	$(MAKE) all CXX="$(CXX)" CXXFLAGS="$(CXXSTD) $(OPT) $(WARN) $(INCLUDES) -pthread -static -static-libgcc -static-libstdc++"
	@echo
	@file $(BIN) | tee /dev/stderr | grep -q 'statically linked' \
		&& echo "[OK] $(BIN) is statically linked." \
		|| echo "[WARN] $(BIN) is NOT fully static; check toolchain."

# Bundle exactly the two files the contest framework needs
# (my_optimizer.py + the binary) into submit/floorplanner_submission.zip.
# Drop this folder's contents into FloorSet/iccad2026contest/ on the
# contest machine.
submit: $(BIN) my_optimizer.py
	@rm -rf submit/floorplanner_submission
	@mkdir -p submit/floorplanner_submission
	@cp my_optimizer.py submit/floorplanner_submission/
	@cp $(BIN)         submit/floorplanner_submission/
	@chmod +x          submit/floorplanner_submission/$(BIN)
	@cd submit && zip -qr floorplanner_submission.zip floorplanner_submission/
	@echo "[OK] wrote submit/floorplanner_submission.zip"
	@echo "Drop the unzipped contents into FloorSet/iccad2026contest/ then run:"
	@echo "    python iccad2026_evaluate.py --validate my_optimizer.py"

.PHONY: all clean debug check static submit
