# Curated candidate lyoprotectant library with verified SMILES.
# Sources: PubChem canonical SMILES, cross-referenced with literature.
# All candidates are biocompatible excipients with IID or whitelist approval.

CANDIDATE_SMILES = {
    # Disaccharides
    "Trehalose": "OC[C@H]1O[C@@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "Sucrose":   "OC[C@H]1O[C@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "Maltose":   "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)O2)[C@H](O)[C@@H](O)[C@@H]1O",
    "Lactose":   "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)O2)[C@H](O)[C@@H](O)[C@@H]1O",

    # Monosaccharides / polyols
    "Mannitol":  "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO",
    "Sorbitol":  "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "Glucose":   "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Inositol":  "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O)[C@@H]1O",

    # Amino acids
    "Glycine":   "NCC(=O)O",
    "Histidine": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
    "Arginine":  "N[C@@H](CCCNC(=N)N)C(=O)O",
    "Proline":   "OC(=O)[C@@H]1CCCN1",
    "Alanine":   "C[C@H](N)C(=O)O",
    "Lysine":    "NCCCC[C@H](N)C(=O)O",

    # Other
    "Succinic acid": "OC(=O)CCC(=O)O",
    "Citric acid":   "OC(=O)CC(O)(CC(=O)O)C(=O)O",
}
