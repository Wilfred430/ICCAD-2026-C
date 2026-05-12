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

# Integration with Official Framework
OFFICIAL_DIR = ../ICCAD-C-FloorSet-official
CONTEST_DIR  = $(OFFICIAL_DIR)/iccad2026contest
# 使用官方目錄下的虛擬環境 Python (如果存在)，否則退回系統 python3
OFFICIAL_PYTHON = conda run --no-capture-output -n iccad2026_c python

# Sync: Copy binary and wrapper to official contest folder
sync: $(BIN) my_optimizer.py
	@mkdir -p $(CONTEST_DIR)
	@cp my_optimizer.py $(CONTEST_DIR)/
	@cp $(BIN)          $(CONTEST_DIR)/
	@chmod +x           $(CONTEST_DIR)/$(BIN)
	@echo "[OK] Files synced to $(CONTEST_DIR)"

# Evaluate: Run official evaluation (full 100 cases)
eval: sync
	@rm -rf eval_results && mkdir -p eval_results
	@export FLOORPLANNER_KEEP=1 && \
		cd $(CONTEST_DIR) && $(OFFICIAL_PYTHON) iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
	@echo "[eval] Copying intermediate files to eval_results/ ..."
	@cp -r $$(ls -dt /tmp/my_optimizer_*/ 2>/dev/null | head -1)* eval_results/ 2>/dev/null || true
	@echo "[OK] Eval complete. Results saved to eval_results/"

# Get the system temp directory (works on both Linux and Windows)
TEMP_DIR = $(shell $(OFFICIAL_PYTHON) -c "import tempfile; print(tempfile.gettempdir().replace('\\\\', '/'))")

# Evaluate quick: Run 25 sampled cases (every 4th case: 0,4,8,...,96)
eval-quick: sync
	@rm -rf eval_results && mkdir -p eval_results
	@export FLOORPLANNER_KEEP=1 && cd $(CONTEST_DIR) && \
		for i in 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96; do \
			echo "--- case $$i ---"; \
			$(OFFICIAL_PYTHON) iccad2026_evaluate.py --evaluate my_optimizer.py --test-id $$i --save-solutions 2>&1 | tail -3; \
			PAD_ID=$$(printf "%03d" $$i); \
			LATEST_DIR=$$(ls -dt $(TEMP_DIR)/my_optimizer_*/ 2>/dev/null | head -1); \
			cp "$${LATEST_DIR}/case_000.txt" "$(CURDIR)/eval_results/case_$${PAD_ID}.txt" 2>/dev/null || true; \
			cp "$${LATEST_DIR}/case_000.sol" "$(CURDIR)/eval_results/case_$${PAD_ID}.sol" 2>/dev/null || true; \
			cp "my_optimizer_results.json" "$(CURDIR)/eval_results/results_$${PAD_ID}.json" 2>/dev/null || true; \
		done
	@echo "[OK] Quick eval (25 cases) complete. Results saved to eval_results/"

# Evaluate mini: Just cases 0, 4, 8 for quick visualization check
eval-mini: sync
	@rm -rf eval_results && mkdir -p eval_results
	@export FLOORPLANNER_KEEP=1 && cd $(CONTEST_DIR) && \
		for i in 0 4 8; do \
			echo "--- case $$i ---"; \
			$(OFFICIAL_PYTHON) iccad2026_evaluate.py --evaluate my_optimizer.py --test-id $$i --save-solutions 2>&1 | tail -3; \
			PAD_ID=$$(printf "%03d" $$i); \
			LATEST_DIR=$$(ls -dt $(TEMP_DIR)/my_optimizer_*/ 2>/dev/null | head -1); \
			cp "$${LATEST_DIR}/case_000.txt" "$(CURDIR)/eval_results/case_$${PAD_ID}.txt" 2>/dev/null || true; \
			cp "$${LATEST_DIR}/case_000.sol" "$(CURDIR)/eval_results/case_$${PAD_ID}.sol" 2>/dev/null || true; \
			cp "my_optimizer_results.json" "$(CURDIR)/eval_results/results_$${PAD_ID}.json" 2>/dev/null || true; \
		done
	@echo "[OK] Mini eval complete."
	@bash tools/remind-me/remind.sh "快速評估 (25 cases) 已完成！"

# Validate: Check if my_optimizer.py adheres to the contest interface
validate: sync
	@cd $(CONTEST_DIR) && $(OFFICIAL_PYTHON) iccad2026_evaluate.py --validate my_optimizer.py

# Visualize: Generate floorplan layout images from eval_results/
# Falls back to toy benchmark if eval_results/ does not exist.
viz:
	@rm -rf plots && mkdir -p plots
	@if [ -d "eval_results" ] && ls eval_results/case_*.txt 1>/dev/null 2>&1; then \
		echo "[viz] Reading cases from eval_results/"; \
		$(OFFICIAL_PYTHON) tools/visualize_floorplan.py --batch eval_results/ -o plots/ --max-cases 100; \
	else \
		echo "[viz] No eval_results/ found, using toy benchmark as preview"; \
		echo "[tip] Run 'make eval' first to generate all 100 cases"; \
		$(OFFICIAL_PYTHON) tools/visualize_floorplan.py benchmarks/toy.txt benchmarks/toy.sol -o plots/toy.png; \
	fi
	@echo "[OK] Floorplan images saved to Implementation/plots/"

# Report: Analyze costs and calculate weighted score
report:
	@$(OFFICIAL_PYTHON) tools/report_score.py

.PHONY: all clean debug check static submit sync eval eval-quick validate viz viz-case report
