import sqlite3

conn = sqlite3.connect("data/hate_speech_detector.db")
c = conn.cursor()
c.execute("DELETE FROM experiments WHERE id = '2a6f1a0d-9375-49bf-899f-fd5067d921b9'")
conn.commit()

c.execute("SELECT exp_code, count(*) FROM experiments GROUP BY exp_code")
print("Experiments grouped:", c.fetchall())

c.execute("SELECT count(*) FROM experiments")
print("Total experiments:", c.fetchone()[0])
conn.close()
