##
# Project Title
#
# @file
# @version 0.1

.PRECIOUS: data/hypothesis_% data/hypothesis_%_qualities.pdf

HYPOTHESES = $(basename $(notdir $(wildcard hypotheses/*.py)))

all: $(HYPOTHESES)

hypothesis_%: hypotheses/hypothesis_%.py data/hypothesis_% data/hypothesis_%.csv data/hypothesis_%_qualities.pdf
	@echo 'Finished Generation & Evaluation of $@'

data/hypothesis_%: hypotheses/hypothesis_%.py
	@mkdir -p data
	python3 $^

data/hypothesis_%.csv: data/hypothesis_%
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

data/hypothesis_%_qualities.pdf: data/hypothesis_%.csv data/hypothesis_%_qualities.csv
	@mkdir -p plots
	python3 plot_lsh_hnsw.py $<

.PHONY: clean

clean:
	rm -rf data

# end
