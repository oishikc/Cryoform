# Glass transition temperature (Tg) training dataset.
# Sources:
#   - ATHAS database (http://athas.prz.rzeszow.pl/)
#   - Hancock & Zografi (1997) Pharm Res 14:422 (sugars/polyols)
#   - Franks (1990) Cryo-Letters 11:93 (disaccharides)
#   - Carpenter et al. (1997) Pharm Res 14:969 (amino acids)
#   - Shamblin et al. (1999) J Phys Chem B 103:4113 (mixtures reference)
# Tg values are for amorphous (non-crystalline) form at 0% RH unless noted.
# Crystallizers (mannitol, glycine) are flagged — they don't form stable glasses.

TG_DATA = [
    # name, SMILES, Tg_C, crystallizer (True = unreliable Tg, will crystallize)
    ("Trehalose",    "OC[C@H]1O[C@@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O", 120.0, False),
    ("Sucrose",      "OC[C@H]1O[C@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",  70.0, False),
    ("Maltose",      "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)O2)[C@H](O)[C@@H](O)[C@@H]1O", 87.0, False),
    ("Lactose",      "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)O2)[C@H](O)[C@@H](O)[C@@H]1O",  98.0, False),
    ("Glucose",      "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  31.0, False),
    ("Mannitol",     "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO",  13.0, True),
    ("Sorbitol",     "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",   -2.0, False),
    ("Inositol",     "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O)[C@@H]1O", 225.0, False),
    ("Glycine",      "NCC(=O)O",                                  270.0, True),
    ("Histidine",    "N[C@@H](Cc1cnc[nH]1)C(=O)O",               250.0, False),
    ("Arginine",     "N[C@@H](CCCNC(=N)N)C(=O)O",                200.0, False),
    ("Proline",      "OC(=O)[C@@H]1CCCN1",                        -60.0, False),
    ("Alanine",      "C[C@H](N)C(=O)O",                          190.0, False),
    ("Lysine",       "NCCCC[C@H](N)C(=O)O",                      152.0, False),
    ("Succinic acid","OC(=O)CCC(=O)O",                            -10.0, False),
    ("Citric acid",  "OC(=O)CC(O)(CC(=O)O)C(=O)O",               11.0, False),
]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from rdkit import Chem

    print(f'Total entries: {len(TG_DATA)}')
    print(f'{"Name":<16} {"Tg (C)":>8}  {"Crystallizer":>12}  {"SMILES valid":>12}')
    print('-' * 56)
    for name, smiles, tg, cryst in TG_DATA:
        mol = Chem.MolFromSmiles(smiles)
        valid = 'YES' if mol else 'FAIL'
        print(f'{name:<16} {tg:>8.1f}  {str(cryst):>12}  {valid:>12}')
