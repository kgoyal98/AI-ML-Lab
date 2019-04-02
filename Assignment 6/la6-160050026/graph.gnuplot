set term png
set datafile separator ","
set output "plot.png"
set xlabel "p"
set ylabel "Number of steps to complete the maze"
plot "data.csv" using 1:2 title "Stochastic MDP Maze" with lines
