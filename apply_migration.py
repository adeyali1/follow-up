import psycopg2
import os

# Connection details
DB_HOST = "supabase.kawkab.dev"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "Wh3TAoO2ZNQTA9t6Psjy8Uba0Ge3Rugg"

def run_migration():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Connected to database!")
        
        # Read migration file
        with open("supabase/migrations/20240115000000_initial_schema.sql", "r") as f:
            sql = f.read()
            
        # Execute migration
        cursor.execute(sql)
        print("Migration applied successfully!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: If connection failed, ensure port 5432 is exposed and accessible.")

if __name__ == "__main__":
    run_migration()
