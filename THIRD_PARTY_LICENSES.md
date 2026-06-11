# Third-Party Licenses

SToP integrates and adapts code from several open-source projects. This file
records the origin and license of each. Full license texts are kept under
[`licenses/`](licenses/). Individual adapted source files also carry an
attribution header pointing back to their upstream project.

SToP itself is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
All code distributed in this repository is under **Apache-2.0 or MIT**, which are
mutually compatible.

## Bundled components

| Project | Location in this repo | License | Upstream | License text |
|---|---|---|---|---|
| LLaVA-NeXT | [`pruning/LLaVA_NeXT/`](pruning/LLaVA_NeXT/) | Apache-2.0 | https://github.com/LLaVA-VL/LLaVA-NeXT | [pruning/LLaVA_NeXT/LICENSE](pruning/LLaVA_NeXT/LICENSE) |
| lmms-eval | [`pruning/lmms_eval/`](pruning/lmms_eval/) | Apache-2.0 + MIT* | https://github.com/EvolvingLMMs-Lab/lmms-eval | [pruning/lmms_eval/LICENSE](pruning/lmms_eval/LICENSE) |
| VisionZip | [`pruning/visionzip/`](pruning/visionzip/) | Apache-2.0 | https://github.com/dvlab-research/VisionZip | [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| HoliTom | [`pruning/holitom/`](pruning/holitom/) | Apache-2.0 | https://github.com/cokeshao/HoliTom | [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| FastVID | [`pruning/fastvid/`](pruning/fastvid/) | MIT | https://github.com/LunarShen/FastVID | [licenses/MIT-FastVID.txt](licenses/MIT-FastVID.txt) |
| FlashVID | [`pruning/flashvid/`](pruning/flashvid/) | MIT | https://github.com/Fanziyang-v/FlashVID | [licenses/MIT-FlashVID.txt](licenses/MIT-FlashVID.txt) |
| HuggingFace Transformers (Qwen2) | `pruning/*/modeling_qwen2.py` | Apache-2.0 | https://github.com/huggingface/transformers | [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |

Notes:

- \*lmms-eval is dual-licensed by its authors: the main pipeline code (inherited
  from lm-evaluation-harness) is **MIT**, and the `tasks/` and `models/` folders
  are **Apache-2.0**. Both are compatible with this repository. Its upstream
  LICENSE is preserved verbatim at [`pruning/lmms_eval/LICENSE`](pruning/lmms_eval/LICENSE).
- The vendored LLaVA-NeXT and lmms-eval trees keep their upstream `LICENSE` file
  at the directory root, plus an `ATTRIBUTION.md` noting they were modified.
  Neither upstream ships a NOTICE file.
- Several files (e.g. each method's `llava_arch.py`, `modeling_qwen2.py`) are
  **modified** copies of upstream code. Per the Apache-2.0 redistribution terms,
  modified files carry a prominent notice stating that they were changed; see the
  attribution header at the top of each adapted file.
- The Apache-2.0 license text is identical across all Apache-2.0 dependencies, so a
  single copy is provided at [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).
- The two MIT dependencies carry distinct copyright lines and are provided
  separately.

## Separately obtained component (NOT bundled)

| Project | License | Upstream | How to obtain |
|---|---|---|---|
| PruneVid | **CC BY-NC-SA 4.0 (NonCommercial)** | https://github.com/Visual-AI/PruneVid | `bash pruning/download_prunevid.sh` |

**PruneVid is not distributed with SToP.** Its license, CC BY-NC-SA 4.0, is a
NonCommercial copyleft license that is incompatible with the Apache-2.0 / MIT
licensing of this repository. To keep SToP cleanly Apache-2.0 + MIT, all
PruneVid-derived code has been removed from the repository.

If you want the PruneVid reference implementation, run:

```bash
bash pruning/download_prunevid.sh
```

This clones the upstream PruneVid repository (including its CC BY-NC-SA 4.0
LICENSE) into `pruning/prunevid_upstream/`, which is git-ignored. Anything you
use or derive from it is bound by CC BY-NC-SA 4.0: **attribution required,
ShareAlike, and NonCommercial use only.** Note that upstream PruneVid is
PLLaVA-based, whereas SToP's harness is LLaVA-NeXT/Qwen2-based, so the upstream
code is provided for reference and is not a drop-in SToP baseline.
