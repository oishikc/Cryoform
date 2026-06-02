import pandas as pd
import os

IID_PATH = os.path.join(os.path.dirname(__file__), "IIR_OCOMM.csv")

WHITELIST = {
    "trehalose",
    "trehalose dihydrate",
}

# Known aliases: common name -> IID name
ALIASES = {
    "peg": "polyethylene glycol",
    "pva": "polyvinyl alcohol",
}

APPROVED_ROUTES = "INTRAVENOUS|INJECTION|SUBCUTANEOUS|INTRADERMAL"

def load_approved_ingredients() -> set:
    df = pd.read_csv(IID_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().upper() for c in df.columns]

    iv_routes = df[
        df["ROUTE"].str.upper().str.contains(APPROVED_ROUTES, na=False)
    ]

    approved = set(iv_routes["INGREDIENT_NAME"].str.strip().str.lower().unique())

    # Add aliases
    for common, iid_name in ALIASES.items():
        if iid_name in approved:
            approved.add(common)

    # Add whitelist
    approved.update(WHITELIST)

    return approved


def is_approved(name: str, approved_set: set) -> bool:
    name_lower = name.strip().lower()
    if name_lower in approved_set:
        return True
    alias_target = ALIASES.get(name_lower)
    if alias_target and alias_target in approved_set:
        return True
    return False


if __name__ == "__main__":
    approved = load_approved_ingredients()
    print(f"Total approved ingredients: {len(approved)}")

    test = [
        "trehalose", "sucrose", "mannitol", "histidine",
        "glycine", "arginine", "sorbitol", "lactose"
    ]
    print("\n=== Candidate Biocompatibility Check ===")
    for name in test:
        status = "PASS" if is_approved(name, approved) else "FAIL"
        print(f"  {status}  {name}")
