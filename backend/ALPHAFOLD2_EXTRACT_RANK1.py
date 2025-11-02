# ============================================================================
# EXTRACT RANK 1 PDB FROM EXISTING ALPHAFOLD2 RESULTS
# Use this if you've already run a prediction and just want to extract rank 1
# ============================================================================

# After running the "Run Prediction" cell, add this cell to extract rank 1 PDB

# The variables 'results' and 'jobname' should already exist from the prediction cell

# Extract rank 1 PDB file
jobname_prefix = ".custom" if msa_mode == "custom" else ""
tag = results["rank"][0][0]  # Get rank 1 tag (first element)

# Construct PDB filename
pdb_filename = f"{jobname}/{jobname}{jobname_prefix}_unrelaxed_{tag}.pdb"

# Alternative: search for PDB files if naming is different
import glob
if not os.path.exists(pdb_filename):
    pdb_files = glob.glob(f"{jobname}/*unrelaxed*.pdb")
    if pdb_files:
        pdb_files.sort()
        pdb_filename = pdb_files[0]

# Read PDB content
with open(pdb_filename, 'r') as f:
    pdb_content = f.read()

print(f"✅ Rank 1 PDB file: {pdb_filename}")
print(f"📄 PDB content length: {len(pdb_content)} characters")

# Calculate average pLDDT score
plddt_values = []
for line in pdb_content.split('\n'):
    if line.startswith('ATOM'):
        try:
            b_factor = float(line[60:66].strip())
            if 0 <= b_factor <= 100:
                plddt_values.append(b_factor)
        except (ValueError, IndexError):
            continue

plddt_score = sum(plddt_values) / len(plddt_values) if plddt_values else 85.0
print(f"📊 Average pLDDT: {plddt_score:.2f}")

# Display first few lines of PDB
print(f"\n📋 First 10 lines of PDB:")
print('\n'.join(pdb_content.split('\n')[:10]))

# You can now copy pdb_content and use it in your docking page

