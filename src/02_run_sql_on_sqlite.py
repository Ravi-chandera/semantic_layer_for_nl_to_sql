import sqlite3

def run_query(query, db_name="../data/assignment.db"):
    try:
        conn = sqlite3.connect(db_name)
        # This row_factory makes results look like dictionaries (easier to read)
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert rows to a list of dicts for clean output
        results = [dict(row) for row in rows]
        return results

    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        if conn:
            conn.close()

# Usage Example:
my_sql = "SELECT * FROM po_line_items LIMIT 5;" # Your generated SQL goes here
data = run_query(my_sql)

for entry in data:
    print(entry)