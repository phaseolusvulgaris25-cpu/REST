# This recalls functions from bio3d library.
library(bio3d)

# This parses your PDB.
pdb <- read.pdb("no_water_ion.pdb")
dcd <- read.dcd("no_water_ion.dcd") 

# Calpha selection
ca.inds <- atom.select(pdb, elety="C1") 
xyz <- fit.xyz(fixed=pdb$xyz, mobile=dcd,
               fixed.inds=ca.inds$xyz,
               mobile.inds=ca.inds$xyz) 

# Check
dim(xyz) == dim(dcd)
## If everything is okay, you will see "[1] TRUE TRUE" in the screen.

# This generates Dynamic Cross Correlation Map (DCCM).
cij<-dccm(xyz[,ca.inds$xyz])
plot(cij)

# This saves your DCCM as pgn.
dev.copy(png, "rest.png") 
dev.off() 

png(filename = "rest.png", width = 600, height = 600)
plot(cij, main = "")
dev.off()
