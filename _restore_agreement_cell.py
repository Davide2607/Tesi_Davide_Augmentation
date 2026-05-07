import json
from pathlib import Path

cur_path = Path(r"c:\Users\david\Desktop\Polito\Tesi_Davide\CONFIDENCE_MODEL.ipynb")
bak_path = Path(r"c:\Users\david\Desktop\Polito\Tesi_Davide\CONFIDENCE_MODEL_backup.ipynb")

cur = json.loads(cur_path.read_text(encoding="utf-8"))
bak = json.loads(bak_path.read_text(encoding="utf-8"))

needle = "wrong_list = wrong_models[idx]"

bak_cell = None
for cell in bak.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if needle in src:
        bak_cell = cell
        break

if bak_cell is None:
    raise SystemExit("Could not find agreement cell in backup")

bak_src = list(bak_cell.get("source", []))

# Patch only the example comment to be model-agnostic
patched_src = []
for line in bak_src:
    if needle in line:
        patched_src.append("                wrong_list = wrong_models[idx]  # lista di (model_name, predicted_class)\n")
    else:
        patched_src.append(line)

# Find overwritten cell in current notebook by the unique placeholder line we inserted
placeholder_needle = "# ... (resto cella invariato; edit minimo)"
cur_cell_index = None
for i, cell in enumerate(cur.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if placeholder_needle in src:
        cur_cell_index = i
        break

if cur_cell_index is None:
    raise SystemExit("Could not find overwritten cell in current notebook")

cur["cells"][cur_cell_index]["source"] = patched_src

cur_path.write_text(json.dumps(cur, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Restored cell at index", cur_cell_index, "from backup; comment updated.")
