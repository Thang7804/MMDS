"""
Khoi tao schema PostgreSQL cho do an male voice retrieval.

Chay:
    pip install psycopg[binary]
    python db/init_postgres.py

Bien moi truong ho tro:
    PGHOST
    PGPORT
    PGDATABASE
    PGUSER
    PGPASSWORD
"""

from pathlib import Path
import os

import psycopg


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_connection_string() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "male_voice_db")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD")

    if not password:
        raise RuntimeError(
            "Khong tim thay PGPASSWORD. Hay set bien moi truong hoac tao file "
            "'db/.env.postgres' (hoac 'db/.env.postgres.example') voi thong tin ket noi."
        )

    return (
        f"host={host} "
        f"port={port} "
        f"dbname={dbname} "
        f"user={user} "
        f"password={password}"
    )


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    schema_path = root_dir / "schema.sql"
    env_path = root_dir / ".env.postgres"
    example_env_path = root_dir / ".env.postgres.example"

    load_env_file(env_path)
    load_env_file(example_env_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Khong tim thay file schema: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")
    conn_str = get_connection_string()

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()

    print("[INFO] Da khoi tao schema PostgreSQL thanh cong.")
    print(
        "[INFO] Da ket noi toi "
        f"db='{os.getenv('PGDATABASE', 'male_voice_db')}' "
        f"tren {os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')} "
        f"voi user='{os.getenv('PGUSER', 'postgres')}'."
    )
    print("[INFO] Database san sang cho cac bang audio_files, audio_features, retrieval_sessions, retrieval_results.")


if __name__ == "__main__":
    main()
