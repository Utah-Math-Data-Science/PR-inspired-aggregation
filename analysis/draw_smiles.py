# import os.path as osp

# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw


# raw_dir = '/root/workspace/data/peptides-functional/raw/'
# data_df = pd.read_csv(osp.join(raw_dir,'peptide_multi_class_dataset.csv.gz'))

# # SMILES string for caffeine
# smiles = data_df['smiles'][0]

# # Create a molecule object
# molecule = Chem.MolFromSmiles(smiles)

# # Save the molecule to a file
# Draw.MolToFile(molecule, "/root/workspace/out/smiles.png", size=(3000, 3000))
import os.path as osp
import pandas as pd
from PIL import Image
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

raw_dir = '/root/workspace/data/peptides-functional/raw/'
data_df = pd.read_csv(osp.join(raw_dir,'peptide_multi_class_dataset.csv.gz'))

# SMILES string for caffeine
smiles = data_df['smiles'][10]

# Create a molecule object
molecule = Chem.MolFromSmiles(smiles)

# Add hydrogens to the molecule
molecule = Chem.AddHs(molecule)

# Generate a 3D conformation of the molecule
AllChem.EmbedMolecule(molecule)

# Save the RDKit molecule to a mol file
Chem.MolToMolFile(molecule, '/root/workspace/out/molecule.mol')

# Call PyMOL from command line to generate a PNG file
subprocess.run(["pymol", "-qc", "-d", 
"""
load /root/workspace/out/molecule.mol
hide everything, all
show sticks, all
show spheres, all
set stick_radius, 0.1
set sphere_scale, 0.2
util.cbaw molecule
bg_color gainsboro
set ambient, 0.1
set specular, 0
set ray_opaque_background=0
set ray_trace_mode, 3
set ray_shadow, 1
center molecule
ray 3000, 3000
png /root/workspace/out/molecule.png
quit
"""
])

import cairosvg
import io


# Open the PNG image
# image = Image.open('/root/workspace/out/molecule.png')
# image = image.convert('RGB')

# Convert to PDF
# image.save('/root/workspace/out/molecule.pdf', 'PDF', resolution=300.0, quality=100, icc_profile=image.info.get('icc_profile'))

image = Image.open('/root/workspace/out/molecule.png')

# Create a BytesIO object to store the SVG data
svg_data = io.BytesIO()

# Convert PNG to SVG
image.save(svg_data, format='svg')

# Reset the file pointer to the beginning of the stream
svg_data.seek(0)

# Convert SVG to PDF using cairosvg
cairosvg.svg2pdf(file_obj=svg_data, write_to='/root/workspace/out/molecule.pdf')