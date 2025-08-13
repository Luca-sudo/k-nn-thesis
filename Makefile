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

generate_%: hypotheses/%.py
	@mkdir -p data
	python3 $^

test_hypothesis_%: data/hypothesis_%.h5
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

test_sift1m: data/sift1m.h5
	python3 test_sift1m.py $<

plot_%:  data/%_results.h5
	@mkdir -p plots
	python3 plot_lsh_hnsw.py $<

stats_%: data/%.h5
	python3 compute_statistics.py $<

.PHONY: clean

clean:
	rm -rf data

# end
