for i in ServerOutput/*
do
/home/homingla/Software/PyMOL-2.5.1_283-Linux-x86_64-py37/bin/pymol -cq "${i}/uploaded.pml"
done
