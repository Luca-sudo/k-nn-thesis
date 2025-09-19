##
# Project Title
#
# @file
# @version 0.1

.PRECIOUS: data/hypothesis_% data/hypothesis_%_qualities.pdf

HYPOTHESES = $(basename $(notdir $(wildcard hypotheses/*.py)))

all: $(HYPOTHESES)

run: plot_hypothesis_5 generate_hypothesis_6 test_hypothesis_6 plot_hypothesis_6 generate_hypothesis_7 test_hypothesis_7 plot_hypothesis_7

hypothesis_%: generate_hypothesis_% test_hypothesis_% plot_hypothesis_%
	@echo 'Finished Generation & Evaluation of $@'

generate_%: hypotheses/%.py
	@mkdir -p data
	python3 $^

additional_test_%: stats/test_hypothesis_%_further.py
	python3 $<

additional_stats_%: stats/hypothesis_%_further.py
	python3 $<

test_hypothesis_%: data/hypothesis_%.h5
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

test_sift1m: data/sift1m.h5
	python3 test_lsh_hnsw.py $<

test_gist1m: data/gist1m.h5
	python3 test_sift1m.py $<

plot_hypothesis_%: plotting/plot_hypothesis_%.py data/hypothesis_%_results.h5
	python3 $^

stats_hypothesis_%: stats/hypothesis_%.py
	python3 $<

.PHONY: clean

clean:
	rm -rf data

# end
