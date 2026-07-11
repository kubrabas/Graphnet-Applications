#!/usr/bin/env python3
"""Execute a simple notebook in-place without nbformat/nbclient.

This is intentionally small and dependency-light for cluster environments where
Jupyter is partially installed. It executes code cells sequentially in one
Python namespace and writes stdout plus the final expression representation back
to the notebook JSON.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import traceback
from pathlib import Path


def _text_result(result):
    if result is None:
        return []
    data = {"text/plain": repr(result)}
    html = getattr(result, "_repr_html_", lambda: None)()
    if html is not None:
        data["text/html"] = html
    return [
        {
            "output_type": "execute_result",
            "execution_count": None,
            "metadata": {},
            "data": data,
        }
    ]


def _figure_outputs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outputs = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii")
                },
            }
        )
        plt.close(fig)
    return outputs


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__", "__file__": str(path)}
    execution_count = 1

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        cell["execution_count"] = execution_count
        execution_count += 1
        outputs = []

        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                tree = ast.parse(source, filename=str(path), mode="exec")
                last_expr = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None

                if last_expr is not None:
                    body = tree.body[:-1]
                    if body:
                        exec(
                            compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
                            namespace,
                        )
                    result = eval(
                        compile(ast.Expression(body=last_expr.value), str(path), "eval"),
                        namespace,
                    )
                else:
                    exec(compile(tree, str(path), "exec"), namespace)
                    result = None

            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
            outputs.extend(_figure_outputs())
            outputs.extend(_text_result(result))

        except Exception as exc:
            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                }
            )
            cell["outputs"] = outputs
            path.write_text(json.dumps(notebook, indent=1) + "\n")
            raise

        cell["outputs"] = outputs

    path.write_text(json.dumps(notebook, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    execute_notebook(args.notebook)
    print(f"executed notebook in-place: {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
