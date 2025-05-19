import sqlite3

with open("1a.sql", "r") as file:
    sql_script = file.read()

conn = sqlite3.connect("college.db")
cursor = conn.cursor()

# Run the full SQL script
cursor.executescript(sql_script)

# Now run a SELECT query to fetch data
cursor.execute("SELECT * FROM students")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.commit()
conn.close()
