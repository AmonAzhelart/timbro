#!/usr/bin/env python3
"""Controlli statici che replicano gli errori di avvio di FastAPI/Pydantic.

Il container scarica ~8 GB di immagine prima di dirti che una rotta è malformata:
questo script dà la stessa risposta in mezzo secondo, senza dipendenze.

    python3 backend/tools/preflight.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

# `frontend/` sta accanto a `backend/` nel repo ma accanto ad `app/` nel
# container: lo cerchiamo risalendo, così lo script gira in entrambi i casi.
FRONTEND = next(
    (p / "frontend" for p in APP.parents if (p / "frontend" / "index.html").exists()),
    None,
)

problems: list[str] = []
checks_run = 0


def fail(file: str, line: int, msg: str) -> None:
    problems.append(f"{file}:{line}  {msg}")


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


MODULES = {p.stem: parse(p) for p in sorted(APP.glob("*.py"))}
SOURCES = {p.stem: p for p in sorted(APP.glob("*.py"))}


# ---------------------------------------------------------------------------
# 1. Rotte FastAPI
# ---------------------------------------------------------------------------
def check_routes() -> None:
    """Riproduce gli assert di `APIRoute.__init__`."""
    global checks_run
    tree = MODULES["main"]
    name = "app/main.py"
    seen: dict[tuple[str, str], int] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if not (isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app"):
                continue
            method = dec.func.attr
            if method not in ("get", "post", "put", "delete", "patch", "head", "options"):
                continue

            checks_run += 1
            path = ast.literal_eval(dec.args[0])
            kw = {k.arg: k.value for k in dec.keywords}
            status = ast.literal_eval(kw["status_code"]) if "status_code" in kw else 200

            # (a) 204/304 non ammettono corpo: `-> None` diventa NoneType e
            #     fa esplodere l'assert, a meno di response_model=None esplicito.
            if status in (204, 304) and node.returns is not None:
                if "response_model" not in kw:
                    fail(name, node.lineno,
                         f"{method.upper()} {path}: status {status} con annotazione "
                         f"di ritorno '{ast.unparse(node.returns)}' e senza "
                         f"response_model=None → AssertionError all'avvio")

            # (b) i segmenti {param} devono esistere fra gli argomenti
            declared = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            for raw in re.findall(r"\{([^}]+)\}", path):
                param = raw.split(":")[0]
                if param not in declared:
                    fail(name, node.lineno,
                         f"{method.upper()} {path}: il path contiene '{{{param}}}' "
                         f"ma la funzione non ha quel parametro")

            # (c) stessa coppia metodo+path dichiarata due volte
            key = (method, re.sub(r"\{[^}]+\}", "{}", path))
            if key in seen:
                fail(name, node.lineno,
                     f"{method.upper()} {path}: già dichiarata alla riga {seen[key]}")
            seen[key] = node.lineno


# ---------------------------------------------------------------------------
# 2. Riferimenti fra moduli
# ---------------------------------------------------------------------------
def _module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def check_cross_module() -> None:
    """`store.get_job(...)` deve corrispondere a qualcosa di definito in store.py."""
    global checks_run
    exported = {mod: _module_names(tree) for mod, tree in MODULES.items()}

    for mod, tree in MODULES.items():
        # alias locali dei moduli fratelli: `from . import store, pipeline`
        local: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
                for alias in node.names:
                    if alias.name in MODULES:
                        local.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
                continue
            target = node.value.id
            if target not in local or target not in exported:
                continue
            checks_run += 1
            if node.attr not in exported[target]:
                fail(f"app/{mod}.py", node.lineno,
                     f"'{target}.{node.attr}' non esiste in app/{target}.py")


# ---------------------------------------------------------------------------
# 3. Import interni
# ---------------------------------------------------------------------------
def check_imports() -> None:
    global checks_run
    for mod, tree in MODULES.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level == 1):
                continue
            checks_run += 1
            if node.module is None:  # from . import x, y
                for alias in node.names:
                    if alias.name not in MODULES:
                        fail(f"app/{mod}.py", node.lineno,
                             f"modulo '{alias.name}' inesistente")
            elif node.module in MODULES:  # from .config import settings
                available = _module_names(MODULES[node.module])
                for alias in node.names:
                    if alias.name not in available:
                        fail(f"app/{mod}.py", node.lineno,
                             f"'{alias.name}' non definito in app/{node.module}.py")


# ---------------------------------------------------------------------------
# 4. Modelli Pydantic
# ---------------------------------------------------------------------------
RESERVED = {
    "model_config", "model_fields", "model_dump", "model_dump_json",
    "model_validate", "model_copy", "model_construct", "schema", "json", "dict", "copy",
}


def check_pydantic() -> None:
    global checks_run
    for mod, tree in MODULES.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {ast.unparse(b) for b in node.bases}
            if not bases & {"BaseModel", "BaseSettings"}:
                continue
            for stmt in node.body:
                if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                    continue
                checks_run += 1
                field = stmt.target.id
                if field in RESERVED:
                    fail(f"app/{mod}.py", stmt.lineno,
                         f"{node.name}.{field} sovrascrive un membro di BaseModel")
                if field.startswith("model_") and field != "model_config":
                    fail(f"app/{mod}.py", stmt.lineno,
                         f"{node.name}.{field} usa il namespace protetto 'model_'")


# ---------------------------------------------------------------------------
# 5. Coerenza con l'interfaccia
# ---------------------------------------------------------------------------
def check_frontend() -> None:
    global checks_run
    if FRONTEND is None:
        print("… frontend/index.html non trovato: salto i controlli sull'interfaccia")
        return
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    js = html.split("<script>")[1]

    declared: set[tuple[str, str]] = set()
    for m in re.finditer(r'@app\.(get|post|put|delete|patch)\("(/api[^"]*)"', MODULES and
                         (APP / "main.py").read_text(encoding="utf-8")):
        declared.add((m.group(1).upper(), re.sub(r"\{[^}]+\}", "{}", m.group(2)).rstrip("/")))

    for m in re.finditer(r"api\('(/[^']*)'(?:,\s*\{[^}]*method:\s*'(\w+)')?", js):
        checks_run += 1
        method = (m.group(2) or "GET").upper()
        path = re.sub(r"\$\{[^}]+\}", "{}", "/api" + m.group(1)).split("?")[0].rstrip("/")
        if (method, path) not in declared:
            fail("frontend/index.html", 0, f"la UI chiama {method} {path} ma la rotta non esiste")

    # ogni id usato dal JS deve esistere nel markup
    ids = set(re.findall(r'id="([^"]+)"', html))
    for used in set(re.findall(r"\$\('#([A-Za-z0-9_]+)'\)", js)):
        checks_run += 1
        if used not in ids:
            fail("frontend/index.html", 0, f"il JS usa #{used} ma l'elemento non esiste")

    # anche i selettori raccolti nella mappa `const S = {...}`
    table = re.search(r"const S = \{(.*?)\n\};", js, re.S)
    if table:
        for key, sel in re.findall(r"(\w+)\s*:\s*'#([A-Za-z0-9_]+)'", table.group(1)):
            checks_run += 1
            if sel not in ids:
                fail("frontend/index.html", 0,
                     f"S.{key} punta a #{sel} ma l'elemento non esiste")

    # i campi inviati nel PUT /api/settings devono essere accettati da SettingsPatch
    patch_body = re.search(r"const patch = \{(.*?)\n  \};", js, re.S)
    if patch_body:
        allowed = set()
        for node in ast.walk(MODULES["models"]):
            if isinstance(node, ast.ClassDef) and node.name == "SettingsPatch":
                allowed = {
                    s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                }
        for field in re.findall(r"^\s*(\w+)\s*:", patch_body.group(1), re.M):
            checks_run += 1
            if allowed and field not in allowed:
                fail("frontend/index.html", 0,
                     f"la UI invia '{field}', rifiutato da SettingsPatch (extra='forbid')")


# ---------------------------------------------------------------------------
def main() -> int:
    for check in (check_imports, check_cross_module, check_routes,
                  check_pydantic, check_frontend):
        check()

    if problems:
        print(f"\n✗ {len(problems)} problemi ({checks_run} controlli eseguiti):\n")
        for p in problems:
            print("  " + p)
        return 1

    print(f"✓ Nessun problema rilevato ({checks_run} controlli eseguiti).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
