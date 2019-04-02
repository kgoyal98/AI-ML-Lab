#!/bin/bash
if [ $# = 1 ]
then
	python3 deterministicMazeEncoder.py $1
else
	python3 stochasticMazeEncoder.py $1 $2
fi

