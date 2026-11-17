# Third-party notices

This repository is MIT-licensed as a whole, but parts of it are copies or derivations of other
permissively licensed work. Their notices are reproduced here, as those licenses require.

## Vendored source code

The modules under `src/mmdmr/mmd/` are **not** all original to this project. The following files are
copies (with local modifications) from [mmd2gltf-gui](https://github.com/masaka1024/mmd2gltf-gui), used so
that PMX bone flags, VMD interpolation and the deform pipeline are read exactly the way the glTF exporter
reads them:

| this repository | upstream |
|---|---|
| `src/mmdmr/mmd/pmx.py` | `mmd2gltf/pmx.py` |
| `src/mmdmr/mmd/vmd.py` | `mmd2gltf/vmd.py` |
| `src/mmdmr/mmd/animation.py` | `mmd2gltf/animation.py` |
| `src/mmdmr/mmd/mathutil.py` | `mmd2gltf/mathutil.py` |

The remaining modules (`basis.py`, `bake.py`, `constraints.py`, `mesh.py`, `mjcf.py`, `msh.py`, `vpd.py`)
are original to this project.

```
MIT License

Copyright (c) 2026 mmd2gltf-gui Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Referenced

[mmd-anim](https://github.com/yohawing/mmd-anim) informed how the MMD format is handled. No file in this
repository is a verbatim copy of it; it is listed here for attribution, and its notice is reproduced in
case a later change starts carrying code from it.

```
MIT License

Copyright (c) 2026 Yohawing
```

## Runtime dependencies

These are installed, not redistributed by this repository:

| package | license |
|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| [numpy](https://numpy.org), [scipy](https://scipy.org) | BSD-3-Clause |
| [Pillow](https://python-pillow.org) | HPND / MIT-CMU |
| [GMR](https://github.com/Yanjieze/GMR) (separate checkout, MIT) | MIT |

GMR is **not** vendored: the retarget step imports it from its own checkout and environment
(`--gmr-root`). Its license therefore does not extend to this repository, and this repository's license
does not extend to it.

## Third-party assets — NOT covered by this project's license

The MMD character models (`.pmx`) and motion files (`.vmd`) used for development are third-party works
with their own terms. They are deliberately excluded from version control (see `.gitignore`), and the
applicable terms are kept in `license/` in this repository:

- motion data: modification and redistribution allowed, **commercial use prohibited**, attribution to
  MIHOYO / o狐妖o / Ponx迫奈熏 (see `license/read me.txt`), and to 翡水 (see `license/れどめ.txt`);
- character models: the terms of their respective authors apply.

The robot descriptions under `assets/` carry their own `LICENSE` files (Unitree G1, Booster T1); those
also stay outside this project's MIT grant.
