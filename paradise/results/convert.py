#!/usr/bin/env python3
import ast
import csv
import re
import sys

# breakdown indices: [FW, BW, Rec, DP, MP, EP, CP, P2P, BBL]
I_FW, I_BW, I_REC, I_DP, I_MP, I_EP, I_CP, I_P2P, I_BBL = range(9)

# Quote unquoted dict keys like {EP: 8, PP: 1} -> {"EP": 8, "PP": 1}
KEYS = ["EP", "PP", "DP", "MP", "MB", "MBS", "CP", "P2P", "BBL", "FW", "BW", "Rec"]
KEY_RE = re.compile(r'(?<![\'"])\b(' + "|".join(map(re.escape, KEYS)) + r')\b\s*:')

def normalize(text: str) -> str:
    return KEY_RE.sub(r'"\1":', text)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.txt output.csv", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    raw = open(in_path, "r", encoding="utf-8").read().strip()
    raw = normalize(raw)

    data = ast.literal_eval(raw)  # now it's valid (keys are quoted)

    header = ["EP","PP","DP","MP","MB","MBS","time","comp","ep_wait","pp_wait","dp_wait","mp_wait"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for dims_dict, _mem_mb, t, parts in data:
            EP = int(dims_dict.get("EP", 0))
            PP = int(dims_dict.get("PP", 0))
            DP = int(dims_dict.get("DP", 0))
            MP = int(dims_dict.get("MP", 0))
            MB = int(dims_dict.get("MB", 0))
            MBS = int(dims_dict.get("MBS", 0))

            fw = float(parts[I_FW])
            bw = float(parts[I_BW])
            rec = float(parts[I_REC])
            dp_wait = float(parts[I_DP])
            mp_wait = float(parts[I_MP])
            ep_wait = float(parts[I_EP])
            bbl = float(parts[I_BBL])

            comp = fw + bw + rec
            pp_wait = bbl  # proxy (bubble) since no explicit PP bucket in your vector

            w.writerow([
                EP, PP, DP, MP, MB, MBS,
                f"{float(t):.12e}",
                f"{comp:.12e}",
                f"{ep_wait:.12e}",
                f"{pp_wait:.12e}",
                f"{dp_wait:.12e}",
                f"{mp_wait:.12e}",
            ])

if __name__ == "__main__":
    main()
