##
# Project Title
#
# @file
# @version 0.1

.PRECIOUS: data/hypothesis_% data/hypothesis_%_qualities.pdf

HYPOTHESES = $(basename $(notdir $(wildcard hypotheses/*.py)))

all: $(HYPOTHESES)

hypothesis_%: generate_hypothesis_% test_hypothesis_% plot_hypothesis_%
	@echo 'Finished Generation & Evaluation of $@'

generate_hypothesis_%: hypotheses/hypothesis_%.py
	@mkdir -p data
	python3 $^

test_hypothesis_%: data/hypothesis_%.h5
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

plot_hypothesis_%:  data/hypothesis_%_results.h5
	@mkdir -p plots
	python3 plot_lsh_hnsw.py $<

.PHONY: clean

clean:
	rm -rf data

# end
