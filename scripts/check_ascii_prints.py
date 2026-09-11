"""Guard: ningun print() de codigo ejecutable puede llevar caracteres no-ASCII.

Existe porque el mismo bug reaparecio tres veces. En una consola Windows con
cp1252 —el entorno por defecto de este proyecto— un print con un caracter fuera
de cp1252 lanza UnicodeEncodeError y aborta el proceso. Ocurrio con emoji en
app/main.py (paraba el arranque del backend), con un caracter de caja en
scripts/seed_db.py (dejaba al operador sin las credenciales recien creadas) y una
tercera vez durante la propia correccion de la tercera auditoria.

Se analiza el AST y no el texto plano: una primera version buscaba `print(` por
regex y marcaba como fallo un COMENTARIO que mencionaba la palabra print. Los
comentarios y los docstrings pueden llevar acentos sin ningun riesgo — solo
importa lo que llega a stdout.

Uso:  python scripts/check_ascii_prints.py     (exit 1 si encuentra alguno)
"""
import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIRS = ("ml", "app", "scripts", "alembic")


def _literales(nodo: ast.AST):
    """Todas las cadenas literales que se evaluan dentro de la llamada."""
    for hijo in ast.walk(nodo):
        if isinstance(hijo, ast.Constant) and isinstance(hijo.value, str):
            yield hijo.value


def revisar(ruta: Path) -> list[str]:
    try:
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{ruta.relative_to(RAIZ)}: no parsea ({exc})"]
    fallos = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Call)
                and isinstance(nodo.func, ast.Name)
                and nodo.func.id == "print"):
            continue
        malos = sorted({c for txt in _literales(nodo) for c in txt if ord(c) > 127})
        if malos:
            fallos.append(
                f"{ruta.relative_to(RAIZ)}:{nodo.lineno}: {''.join(malos)!r}"
            )
    return fallos


def main() -> int:
    fallos = []
    for d in DIRS:
        for f in sorted((RAIZ / d).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            fallos += revisar(f)
    if fallos:
        print("FALLO: print() con caracteres no-ASCII (rompen en consolas cp1252):")
        for x in fallos:
            print("  " + x)
        return 1
    print(f"OK: ningun print() con caracteres no-ASCII en {', '.join(DIRS)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
