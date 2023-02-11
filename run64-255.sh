#! /bin/bash

for i in {2..7}
do
	python3 find_occurrence.py $((i * 32)) $(((i + 1) * 32)) 
done
