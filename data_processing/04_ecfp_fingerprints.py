#!/usr/bin/env python3
"""Compute 2,048-bit ECFP4 fingerprints (Morgan, radius 2) for the curated set.

Uses the RDKit-recommended MorganGenerator API. Output columns are bit_0 ..
bit_2047 plus a valid_smiles flag; the descriptor columns of the input are
carried through unchanged.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def compute_ecfp(input_csv: str,
                 output_csv: str,
                 smiles_column: str = 'canonical_smiles',
                 radius: int = 2,
                 n_bits: int = 2048,
                 include_chirality: bool = False):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )

    df = pd.read_csv(input_csv)
    if smiles_column not in df.columns:
        raise ValueError(f"column '{smiles_column}' not found. Available: {list(df.columns)}")

    print(f"Read {len(df)} compounds; computing ECFP{radius * 2} fingerprints (n_bits={n_bits})")

    fingerprints = []
    valid_flags = []
    for smiles in df[smiles_column]:
        if pd.isna(smiles) or str(smiles).strip() == '':
            fp = np.zeros(n_bits, dtype=np.int8)
            valid = 0
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fp = np.zeros(n_bits, dtype=np.int8)
                valid = 0
            else:
                fp_bitvect = morgan_gen.GetFingerprint(mol)
                fp = np.zeros((n_bits,), dtype=np.int8)
                for bit in fp_bitvect.GetOnBits():
                    fp[bit] = 1
                valid = 1
        fingerprints.append(fp)
        valid_flags.append(valid)

    fp_df = pd.DataFrame(fingerprints, columns=[f'bit_{i}' for i in range(n_bits)])
    fp_df['valid_smiles'] = valid_flags
    result_df = pd.concat([df, fp_df], axis=1)
    result_df.to_csv(output_csv, index=False)
    print(f"Fingerprints written to {output_csv}")


if __name__ == "__main__":
    compute_ecfp(
        input_csv='../data/PIK3CA_curated_descriptors.csv',
        output_csv='../data/PIK3CA_ecfp2048.csv',
        smiles_column='canonical_smiles',
        radius=2,
        n_bits=2048,
    )
