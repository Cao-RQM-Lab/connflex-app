# ConnFlex build helper. Run `make help` for the targets.
#
# Works with GNU make on macOS, Linux, and Windows (Git Bash / MSYS2). Override PYTHON or
# VENV on the command line, e.g. `make venv PYTHON=python3.12`.

PYTHON ?= python3
VENV   ?= venv
NAME   := ConnFlex

ifeq ($(OS),Windows_NT)
  BIN      := $(VENV)/Scripts
  EXE      := dist/$(NAME)/$(NAME).exe
  PLATFORM := windows-x64
else
  BIN := $(VENV)/bin
  UNAME_S := $(shell uname -s)
  UNAME_M := $(shell uname -m)
  ifeq ($(UNAME_S),Darwin)
    EXE      := dist/$(NAME).app/Contents/MacOS/$(NAME)
    PLATFORM := macos-$(if $(filter arm64,$(UNAME_M)),arm64,x64)
  else
    EXE      := dist/$(NAME)/$(NAME)
    PLATFORM := linux-$(if $(filter x86_64,$(UNAME_M)),x64,$(UNAME_M))
  endif
endif

PY      := $(BIN)/python
STAMP   := $(VENV)/.installed
VERSION  = $(shell $(PY) -c "import connflex_app; print(connflex_app.__version__)")
ARCHIVE  = $(NAME)-$(VERSION)-$(PLATFORM)

ACT_IMAGE ?= catthehacker/ubuntu:act-22.04
ACT_FLAGS ?= -P ubuntu-22.04=$(ACT_IMAGE) --container-architecture linux/amd64 --rm

.DEFAULT_GOAL := help
.PHONY: help venv install lint test check run demo icons build verify package dist clean distclean act-ci act-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

$(STAMP): pyproject.toml
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,build]"
	@touch $(STAMP)

venv: $(STAMP) ## Create the virtual environment and install the app with dev and build extras
install: venv ## Alias for venv

lint: venv ## Lint with ruff
	$(BIN)/ruff check .

test: venv ## Run the headless test suite
	QT_QPA_PLATFORM=offscreen $(PY) -m pytest -q

check: lint test ## Lint, test, and run the headless self-test
	$(PY) -m connflex_app --self-test

run: venv ## Launch the app from the virtual environment (pass a layout with ARGS=path.gds)
	$(PY) -m connflex_app $(ARGS)

demo: venv ## Regenerate the demo layouts in examples/ (ARRAY=n adds an n x n array)
	$(PY) examples/make_demo_gds.py
	$(if $(ARRAY),$(PY) examples/make_demo_gds.py --array=$(ARRAY),)

icons: venv ## Regenerate the application icons in packaging/
	$(PY) packaging/make_icons.py

build: venv ## Build the self-contained executable with PyInstaller into dist/
	$(BIN)/pyinstaller packaging/connflex.spec --noconfirm --clean
	@echo "built $(EXE)"

verify: ## Run the bundle's headless self-test and GUI smoke test (needs a display)
	$(EXE) --self-test
	$(EXE) --smoke-test smoke-$(PLATFORM).png

package: ## Archive dist/ as the release artifact for this platform (DMG, zip, or tar.gz)
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -Command "Compress-Archive -Force -Path dist/$(NAME) -DestinationPath $(ARCHIVE).zip"
else ifeq ($(UNAME_S),Darwin)
	codesign --force --deep --sign - dist/$(NAME).app
	hdiutil create -volname $(NAME) -srcfolder dist/$(NAME).app -ov -format UDZO $(ARCHIVE).dmg
else
	tar -C dist -czf $(ARCHIVE).tar.gz $(NAME)
endif
	@ls -la $(ARCHIVE).*

dist: build verify package ## Build, verify, and package in one go

act-ci: ## Run the CI workflow locally with act (needs Docker; emulates the x64 GitHub runner)
	act push -W .github/workflows/ci.yml $(ACT_FLAGS)

act-release: ## Run the Linux job of the release workflow locally with act (slow: PyInstaller under emulation)
	act workflow_dispatch -W .github/workflows/release.yml --matrix os:ubuntu-22.04 --artifact-server-path $(CURDIR)/.act-artifacts $(ACT_FLAGS)

clean: ## Remove build outputs, caches, and test artifacts (keeps the virtual environment)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .act-artifacts smoke-*.png smoke-*.gds smoke_test.png $(NAME)-*.dmg $(NAME)-*.zip $(NAME)-*.tar.gz
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf

distclean: clean ## Also remove the virtual environment
	rm -rf $(VENV)
