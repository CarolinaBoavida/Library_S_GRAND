# Examples

Usage examples for the GRAND Decoder public API. Each script loads
`libgranddecoder.so` (built via `make` in the repository root) and calls the
library through Python's `ctypes`.

| Script | Demonstrates | API function(s) used |
|---|---|---|
| `example_grand_shots_cc.py` | TODO: one-line description of the scenario (e.g. decoding a batch of CSS syndromes for a [[n,k,d]] code) | `grand_decode_cc` (and/or `grand_decode_prob_cc`, if used) |
| `example_grand_shots_cln.py` | TODO: one-line description of the scenario (e.g. decoding a batch of DEM detector shots and recovering logical outcomes) | `grand_decode_cln` (and/or `grand_decode_prob_cln` / `grand_decode_ML_cln`, if used) |
| `auxilary_functions.py` | Shared helper functions used by the examples above (e.g. loading matrices, building `GrandDecoderConfig`) | — |

## Running an example

```bash
# from the repository root
make
cd Examples
python3 example_grand_shots_cc.py
```

TODO: add any command-line arguments the scripts expect, and note which
sample data files (from `H_Matrix/`, `Syndromes/`, etc.) each script uses by
default.
