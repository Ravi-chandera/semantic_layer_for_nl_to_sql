import sqlite3
import json

def extract_schema_with_composite_keys(db_path, k=3, output_file="../data/schema.json"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    schema_info = {"tables": []}

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row['name'] for row in cursor.fetchall()]

        for table in tables:
            table_data = {
                "table_name": table,
                "columns": [],
                "primary_keys": [], # New field for composite/single keys
                "foreign_keys": []
            }

            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()

            # We'll use a temporary list to sort composite keys by their order
            pk_map = {}

            for col in columns:
                col_name = col['name']
                pk_rank = col['pk']

                # Capture PK rank if it's part of a primary key
                if pk_rank > 0:
                    pk_map[pk_rank] = col_name

                # Get sample values
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT {k};")
                samples = [row[0] for row in cursor.fetchall()]

                table_data["columns"].append({
                    "name": col_name,
                    "type": col['type'],
                    "is_part_of_pk": pk_rank > 0,
                    "sample_values": samples
                })

            # Sort PKs by their rank (order in the composite key)
            table_data["primary_keys"] = [pk_map[rank] for rank in sorted(pk_map.keys())]

            # Get Foreign Keys (same as before)
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            for fk in cursor.fetchall():
                table_data["foreign_keys"].append({
                    "column": fk['from'],
                    "references_table": fk['table'],
                    "references_column": fk['to']
                })

            schema_info["tables"].append(table_data)

        with open(output_file, 'w') as f:
            json.dump(schema_info, f, indent=4)
        
        print(f"Schema (with composite keys) saved to {output_file}")

    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    extract_schema_with_composite_keys("../data/assignment.db")