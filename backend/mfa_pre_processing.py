import sqlite3, os

DB = 'recordings.db'
ALIGNMENT_FOLDER = './db/alignments'
IN_PATH = './db/post/voiced/'
TEXTS_PATH = ALIGNMENT_FOLDER + '/texts'

def cycle_db(db_path, in_path, texts_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Perform create alignment operations for all entries in the database recordings using the method create_alignment
    cursor.execute("SELECT uuid, text, voiced_file_path_flac FROM recordings")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} recordings in the database.")
    for row in rows:
        uuid, text, voiced_file_path_flac = row
        create_alignment(in_path, texts_path, uuid, text, voiced_file_path_flac)
    
    cursor.close()
    conn.close()


def create_alignment(in_path, texts_path, uuid, text, voiced_file_path_flac):
    # Create file with the name uuid + '.txt' in the texts_path containing the text
    text_file_path = os.path.join(texts_path, f"{uuid}.txt")
    if os.path.exists(text_file_path):
        print(f"Text file for UUID: {uuid} already exists. Skipping.")
        return
    with open(text_file_path, 'x') as f:
        f.write(text)  
    print(f"Created text file for UUID: {uuid}")
    # Copy the voiced_file_path_flac to in_path with the name uuid + '.flac'
    flac_file_path = os.path.join(in_path, f"{uuid}.flac")
    with open(voiced_file_path_flac, 'rb') as src_file:
        with open(flac_file_path, 'xb') as dest_file:
            dest_file.write(src_file.read())
    print(f"Copied FLAC file for UUID: {uuid}")


if __name__ == "__main__":
    cycle_db(DB, ALIGNMENT_FOLDER, ALIGNMENT_FOLDER)