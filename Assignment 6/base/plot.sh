#!/bin/bash
data_file="data.csv"
p_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
n=1
# echo ${p_list[@]}
rm $data_file
touch $data_file
for p in ${p_list[@]}
do
	./encoder.sh $1 $p > mdpfile
	./valueiteration.sh mdpfile > value_and_policy_file
	echo -n $p "," >> $data_file
	y=0
	for (( i=0; i<n; i++ ));
	{
	   output=$(./decoder.sh $1 value_and_policy_file $p | wc -w)
		y=$(($y+$output))
	}
	y=$(($y/$n))
	echo $y >> $data_file
done
gnuplot graph.gnuplot
rm  mdpfile value_and_policy_file