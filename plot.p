set terminal png
set output "graph.png"

set xrange [0:1]
set yrange [0:1]

plot "test_data.txt" u 1:2:3 notitle lc variable, "test_centroids.txt" u 1:2:3 notitle lc variable lw 4
