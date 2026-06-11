# Attribution

This directory is a vendored copy of **LLaVA-NeXT**
(https://github.com/LLaVA-VL/LLaVA-NeXT), licensed under **Apache-2.0**.

Some files in this tree have been **modified** for integration into SToP — in
particular, the visual encoders
(`llava/model/multimodal_encoder/clip_encoder.py`,
`llava/model/multimodal_encoder/siglip_encoder.py`) and related model code were
changed to support token-pruning wrappers (VisionZip, FastVID, HoliTom, FlashVID,
and SToP).

The upstream Apache-2.0 license is preserved verbatim in [`LICENSE`](LICENSE) in
this directory (LLaVA-NeXT ships no NOTICE file). See
[`../../THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md) for the complete
third-party inventory.
