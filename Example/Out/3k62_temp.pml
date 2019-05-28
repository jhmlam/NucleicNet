
# """
# A Template in Visualising Surface by Pymol
# """


# To run the script: "pymol -cq Nucleic-Bind_VisualisePymol.pml"
# To make image: ray

# ========================
# Display Set up
# ========================
set cartoon_rect_width, 0.1
set cartoon_rect_length, 0.7
set cartoon_transparency, 0.2
set line_radius = 0.1
set sphere_scale = 0.35
set transparency_mode, 1
set gamma=1.5
set mesh_radius = 0.01 
set antialias = 1 
set stick_radius = 0.22
set dash_radius=0.07
set ribbon_radius =0.1 
#set direct =0.0
set cartoon_fancy_helices=1
bg_color white
set orthoscopic = on
#util.ray_shadows('none')
set transparency, 0.5
set solvent_radius, 0.5
set cartoon_highlight_color =grey50
set ray_trace_mode, 0
# ===========================
# Loading structures
# ===========================

load 3k62.pdb
color white, (/3k62/)

load 3k62_strong_Bootstrap.xyz
create C, (name Co & /3k62_strong_Bootstrap/)
create U, (name U & /3k62_strong_Bootstrap/)
create A, (name Ar & /3k62_strong_Bootstrap/)
create G, (name Ge & /3k62_strong_Bootstrap/)
create P, (name P & /3k62_strong_Bootstrap/)
create R, (name Re & /3k62_strong_Bootstrap/)
create Pur, (name Pu & /3k62_strong_Bootstrap/)
create Pyr, (name Y & /3k62_strong_Bootstrap/)



color red, (/C/)
color magenta, (/U/)
color blue, (/A/)
color cyan, (/G/)
color yellow, (/P/)
color green, (/R/)

color red, (/Pyr/)
color blue, (/Pur/)

show surface, (/A/)
show surface, (/G/)
show surface, (/U/)
show surface, (/C/)
show surface, (/P/)
show surface, (/R/)

hide spheres
hide lines

set transparency_mode, 2
set transparency, 0.8
set ray_transparency_contrast, 0.50
set ray_transparency_oblique, 1.0




ray

save 3k62_pymol.pse
