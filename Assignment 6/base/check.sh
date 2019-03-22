./encoder.sh "data/maze/grid$1.txt" > mdpfile
./valueiteration.sh mdpfile > value_and_policy_file
./decoder.sh "data/maze/grid$1.txt" value_and_policy_file > out
diff "data/maze/solution$1.txt" out
