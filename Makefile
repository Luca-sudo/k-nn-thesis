##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean

result/%: data/%
	mkdir -p result
	python3 test_lsh_hnsw.py $^

hypothesis_%: data/hypothesis_% evaluate_hypothesis_%
	@echo 'Finished Generation & Evaluation of $@'

data/hypothesis_%: hypothesis_%.py
	@mkdir -p data
	python3 $^

evaluate_hypothesis_%: data/hypothesis_%
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

clean:
	rm -rf data

# end
