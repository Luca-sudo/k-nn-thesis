##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean

hypothesis_%: hypothesis_%.py data/hypothesis_% data/hypothesis_%.csv data/hypothesis_%_qualities.pdf
	@echo 'Finished Generation & Evaluation of $@'

data/hypothesis_%: hypothesis_%.py
	@mkdir -p data
	python3 $^

data/hypothesis_%.csv: data/hypothesis_%
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

data/hypothesis_%_qualities.pdf: data/hypothesis_%.csv
	@mkdir -p plots
	python3 plot_lsh_hnsw.py $<

clean:
	rm -rf data

# end
