.DEFAULT_GOAL = all
# Bash is needed for time, compgen, [[ and other builtin commands
SHELL := /bin/bash -o pipefail
RED := $(shell tput setaf 1)
GREEN := $(shell tput setaf 2)
NOCOLOR := $(shell tput sgr0)
PYTHON := /usr/bin/python3
VENVDIR := $(CURDIR)/venv
VENVPIP := $(VENVDIR)/bin/python -m pip
VENVPYTHON := $(VENVDIR)/bin/python

all: venv download test
	@echo "$(GREEN)The package is succesfully installed into the virtualenv ($(VENVDIR)) and all tests are OK!$(NOCOLOR)"

venv:
	@echo "Creating virtualenv in $(VENVDIR)...$(NOCOLOR)"
	@rm -rf $(VENVDIR)
	@$(PYTHON) -m venv $(VENVDIR)
	@which $(VENVPYTHON)
	@$(VENVPYTHON) --version
	@$(VENVPIP) install wheel
	@$(VENVPIP) install -r requirements.txt
	@echo "$(GREEN)Virtualenv is succesfully created!$(NOCOLOR)"
.PHONY: venv

download:
	@mkdir -pv models
	@$(VENVPYTHON) guessers.py
	@wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim -nc --directory-prefix models
	@wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.syn1neg.npy -nc --directory-prefix models
	@wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.wv.vectors.npy -nc \
           --directory-prefix models
	@echo "$(GREEN)Models are successfully downloaded and trie successfully build!$(NOCOLOR)"
	# We can download even larger wordlist here if we find it useful.
.PHONY: download

test:
	@$(VENVPYTHON) tests.py
	@echo "$(GREEN)Tests are OK!$(NOCOLOR)"
.PHONY: test

clean:
	@rm -rvf venv models
.PHONY: clean