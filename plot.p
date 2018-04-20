set terminal png
set output "graph.png"

set xrange [0:1]
set yrange [0:1]

plot "test_data.txt" u 1:2:3 notitle lc variable, "test_centroids.txt" u 1:2:3 notitle lc variable lw 4

do for[s=1:1000] {

do for [t=0:30] {
  outfile = sprintf('data/%d/graph_%02d.png', s ,t)
  set output outfile
  infile1 = sprintf('data/%d/data_%d.txt', s, t)
  infile2 = sprintf('data/%d/centr_%d.txt', s, t)
  ti = sprintf ('Iteration %d/30', t)
  set title ti
  plot infile1 u 1:2:3 notitle lc variable, infile2 u 1:2:3 notitle lc variable lw 4
}

}
