#!/bin/bash
if [ $# = 2 ]
then
	python3 deterministicMazeDecoder.py $1 $2
else
	python3 stochasticMazeDecoder.py $1 $2 $3
fi

