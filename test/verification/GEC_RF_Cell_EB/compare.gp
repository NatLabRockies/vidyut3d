set terminal pdf
set output "electronden_geccell.pdf"
set termoption font "Helvetica,20"
set key spacing 1.1
set xlabel "Axial distance (cm)"
set ylabel "Number density x 1e15 (#/m^3)"
set tics nomirror
set yrange [0.01:4]
set ytics 1.0
plot 'eden100mtorrOverzetHopkins_APL_63_1993' u 1:2 w p ps 1.5 pt 7 title "Experiments",\
     'plt01037.slice' u ($1*100.0):($2/1e15) w l lw 4 lc 8 title "Current work",\
     'eden100mtorrBouefPitchford_PhysRevE_51_2_1995' u 1:2 w lp ps 0.8 lw 1 pt 4 lc 7 title "Fluid Model"
