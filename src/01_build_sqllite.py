import sqlite3

def build_database(sql_file_path, db_name="../data/assignment.db"):
    conn = None
    try:
        # 1. Connect to (or create) the database file
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 2. Read the .sql file
        with open(sql_file_path, 'r') as sql_file:
            sql_script = sql_file.read()

        # 3. Execute the script (this handles multiple commands)
        cursor.executescript(sql_script)
        
        conn.commit()
        print(f"Success! '{db_name}' has been created and populated.")
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

# Usage
if __name__ == "__main__":
    build_database("../data/cashflo_sample_schema_and_data.sql")