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

corp:
	@echo "Preparing contextbank."
	@wget https://nessie.ilab.sztaki.hu/~levai/bert_guessinggame_resources/webcorp_2_freqs.tsv -nc --directory-prefix resources
	@cut -f1 resources/webcorp_2_freqs.tsv | head -n 3000000 > wordlist_3M_unfiltered.csv
	@$(VENVPYTHON) create_corpus/prepare_corp.py
	@echo "Contextbank is successfully created!"
.PHONY: corp

download:
	@mkdir -pv models
	@$(VENVPYTHON) bertfluff/guessers/bert_guesser.py
	@wget https://nessie.ilab.sztaki.hu/~levai/bert_guessinggame_resources/10M_pruned.bin -nc --directory-prefix models
	@wget https://nessie.ilab.sztaki.hu/~levai/bert_guessinggame_resources/hu_fasttext_100.gensim -nc --directory-prefix models
	@wget https://nessie.ilab.sztaki.hu/~levai/bert_guessinggame_resources/hu_fasttext_100.gensim.wv.vectors_ngrams.npy -nc --directory-prefix models
	@echo "$(GREEN)Models are successfully downloaded and trie successfully built!$(NOCOLOR)"
	# We can download even larger wordlist here if we find it useful.
.PHONY: download

kenlm:
	@echo "This will take around 10-30 minutes, and needs 8 GBs of RAM!"
	@sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
	@git clone https://github.com/kpu/kenlm
	@cd kenlm
	@mkdir -p build
	@cd build
	@cmake ..
	@make -j 4
	@bin/lmplz -o 5 --prune 0 4 9 16 25 <../10M.spl >10M_pruned.arpa
	@bin/build_binary 10M_pruned.arpa 10M_pruned.bin
	@mv 10M_pruned.bin ../../models/
.PHONY: kenlm

test:
	@$(VENVPYTHON) tests.py
	@echo "$(GREEN)Tests are OK!$(NOCOLOR)"
.PHONY: test

clean:
	@rm -rvf venv models
.PHONY: clean