"""Convert a `# %%` cell-marked script into a validated .ipynb."""
import re
import sys
import pathlib
import nbformat as nbf


def convert(src_path, out_path):
    src = pathlib.Path(src_path).read_text(encoding="utf-8")
    cells, cur, kind = [], [], None

    def flush():
        if not cur:
            return
        body = "\n".join(cur).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.split("\n"))
            cells.append(nbf.v4.new_markdown_cell(body))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in src.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); cur, kind = [], "markdown"; continue
        if line.startswith("# %%"):
            flush(); cur, kind = [], "code"; continue
        cur.append(line)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.write(nb, str(out_path))
    nbf.validate(nbf.read(str(out_path), as_version=4))
    md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    code = sum(1 for c in nb.cells if c.cell_type == "code")
    return md, code


if __name__ == "__main__":
    md, code = convert(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}: {md} markdown + {code} code cells")
