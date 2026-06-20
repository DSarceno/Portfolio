"""Convert remaining inline $...$ LaTeX in the article markdown to readable Unicode,
so Medium (which renders neither LaTeX nor raw $-syntax) displays it cleanly.
Block equations have already been replaced by images. Run from articles/.
"""
from __future__ import annotations

import glob
import re

GREEK = {
    r"\lambda": "λ", r"\mu": "μ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\xi": "ξ", r"\rho": "ρ", r"\tau": "τ", r"\Delta": "Δ", r"\sigma": "σ",
    r"\pi": "π", r"\theta": "θ", r"\phi": "φ", r"\eta": "η",
}
OPS = {
    r"\leq": "≤", r"\geq": "≥", r"\le": "≤", r"\ge": "≥", r"\approx": "≈",
    r"\neq": "≠", r"\cdot": "·", r"\times": "×", r"\sum": "Σ", r"\in": "∈",
    r"\to": "→", r"\leftarrow": "←", r"\mapsto": "↦", r"\sim": "~",
    r"\pm": "±", r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅", r"\star": "*",
    r"\ll": "≪", r"\gg": "≫", r"\propto": "∝",
}
FUNCS = ["exp", "tanh", "log", "ln", "sin", "cos", "max", "min", "arg"]
MATHBB = {r"\mathbb{N}": "ℕ", r"\mathbb{Z}": "ℤ", r"\mathbb{R}": "ℝ",
          r"\mathbb{E}": "E", r"\mathbb{1}": "1"}


def convert(s: str) -> str:
    # fractions a/b
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    # hat -> combining circumflex on the inner content's last char
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "̂", s)
    # blackboard specials before generic brace strippers
    for k, v in MATHBB.items():
        s = s.replace(k, v)
    # text-like wrappers -> inner
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathbb|operatorname)\{([^{}]*)\}", r"\1", s)
    # greek and operators (longest keys first to avoid prefix clashes)
    for k in sorted({**GREEK, **OPS}, key=len, reverse=True):
        v = {**GREEK, **OPS}[k]
        s = s.replace(k, v)
    # named functions: \exp -> exp
    for fn in FUNCS:
        s = s.replace("\\" + fn, fn)
    # spacing macros -> single space
    s = re.sub(r"\\[,;:!> ]", " ", s)
    s = s.replace(r"\quad", " ").replace(r"\qquad", "  ")
    # super/sub scripts with braces -> parens
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", r"_(\1)", s)
    # leftover braces
    s = s.replace("{", "").replace("}", "")
    # tidy spaces
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s


INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def main():
    for path in sorted(glob.glob("0[1-5]_*.md")):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new = INLINE.sub(lambda m: convert(m.group(1)), text)
        if new != text:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
            print("converted", path)
        else:
            print("no change", path)
        leftover = new.count("$")
        if leftover:
            print(f"   WARNING: {leftover} '$' remain in {path}")


if __name__ == "__main__":
    main()
