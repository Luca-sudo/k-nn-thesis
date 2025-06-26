##
# Project Title
#
# @file
# @version 0.1


result/%: data/%
	mkdir -p result
	python3 test_lsh_hnsw.py $^

hypothesis_%: generate_hypothesis_% evaluate_hypothesis_%
	@echo 'Finished Generation & Evaluation of $@'

generate_hypothesis_%: hypothesis_%.py
	@mkdir -p data
	python3 $^

evaluate_hypothesis_%: data/hypothesis_% test_hypothesis_%.py
	@mkdir -p evaluation
	python3 test_lsh_hnsw.py $<

# end
