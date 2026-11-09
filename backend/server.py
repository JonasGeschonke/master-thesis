from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import sys
import json
import uuid as uuid_lib
import threading
import sqlite3
from typing import Optional
from recording import Recording

app = FastAPI(title='SSI Gatherer Backend')

# ─── CORS ──────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ─── Directory setup ───────────────────────────────────────────────────────

directory = './db/pre'
os.makedirs(directory, exist_ok=True)

voiced_flac_dir = './db/post/voiced'
unvoiced_flac_dir = './db/post/unvoiced'
os.makedirs(voiced_flac_dir, exist_ok=True)
os.makedirs(unvoiced_flac_dir, exist_ok=True)

alignment_directory = './db/alignments/out'
model_directory = './db/models'

UPLOAD_DIRS = {
    'models': './db/models',
    'images': './db/images',
}
for dir_path in UPLOAD_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Upload session state (tracked in-memory for chunk reassembly)
upload_sessions: dict[str, dict] = {}
upload_lock = threading.Lock()


# ─── Health ────────────────────────────────────────────────────────────────

async def _health():
    return Response(status_code=200)

app.add_api_route('/health', _health, methods=['GET', 'POST', 'OPTIONS'])


# ─── Audio endpoints ──────────────────────────────────────────────────────

@app.get('/audio/unvoiced/{uuid}')
async def get_audio_unvoiced_file(uuid: str):
    recording = get_recording(uuid)
    if recording is None:
        return Response(status_code=404)
    if not recording.unvoiced_file_path_flac:
        return Response(status_code=404)
    filename = f'unvoiced_{uuid}.flac'
    return FileResponse(
        recording.unvoiced_file_path_flac,
        media_type='audio/flac',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get('/audio/voiced/{uuid}')
async def get_audio_voiced_file(uuid: str):
    recording = get_recording(uuid)
    if recording is None:
        return Response(status_code=404)
    if not recording.voiced_file_path_flac:
        return Response(status_code=404)
    filename = f'voiced_{uuid}.flac'
    return FileResponse(
        recording.voiced_file_path_flac,
        media_type='audio/flac',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get('/audio/uuids')
async def get_uuids():
    recordings = get_recordings()
    uuids = [
        r.uuid for r in recordings
        if (r.voiced_file_path_flac is None and r.unvoiced_file_path_flac is None)
        or (r.voiced_file_path_flac == '' and r.unvoiced_file_path_flac == '')
    ]
    return uuids


# ─── Experiment / model-file upload (multi-part & octet-stream) ────────────

@app.post('/experiment/{folder_name:path}')
async def post_experiment(folder_name: str, request: Request):
    """Receives multiple files and saves them under a folder named `folder_name`."""
    try:
        folder_path = os.path.join(model_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        form = await request.form()
        for key, value in form.items():
            if isinstance(value, UploadFile):
                filename = key or datetime.datetime.now().isoformat()
                file_path = os.path.join(folder_path, filename)
                content = await value.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
        return Response(status_code=200)
    except Exception as e:
        print(f"Error in post_experiment: {e}", file=sys.stderr)
        return Response(status_code=500)


@app.post('/experiment/{folder_name:path}/{filename:path}')
async def upload_chunk(folder_name: str, filename: str, request: Request):
    """
    Upload a file in chunks using octet-stream.
    Bypasses body-size limits by streaming directly to disk.
    """
    try:
        folder_path = os.path.join(model_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'wb') as f:
            async for chunk in request.stream():
                f.write(chunk)

        return Response(status_code=200)
    except Exception as e:
        print(f"Error in upload_chunk: {e}", file=sys.stderr)
        return Response(status_code=500)


# ─── TextGrid alignment files ──────────────────────────────────────────────

@app.get('/textgrids/{uuid}')
async def get_textgrid(uuid: str):
    if not os.path.exists(alignment_directory):
        print(f"Alignment directory {alignment_directory} does not exist.")
        return []
    if not os.listdir(alignment_directory):
        print(f"Alignment directory {alignment_directory} is empty.")
        return []
    filepath = os.path.join(alignment_directory, f'{uuid}.TextGrid')
    if not os.path.exists(filepath):
        print(f"TextGrid file {filepath} does not exist.")
        return []
    return FileResponse(
        filepath,
        media_type='text/plain',
        headers={'Content-Disposition': f'attachment; filename="{uuid}.TextGrid"'},
    )


# ─── Recordings CRUD ───────────────────────────────────────────────────────

@app.get('/recordings')
async def get_recordings_api():
    recordings = get_recordings()
    return [r.to_dict() for r in recordings]


@app.post('/recordings')
async def save_recording_api(
    uuid: str = Form(...),
    text: str = Form(...),
    timestamp: str = Form(''),
    voiced: UploadFile = File(...),
    unvoiced: UploadFile = File(...),
):
    voiced_path = os.path.join(directory, f'voiced_{uuid}.webm')
    unvoiced_path = os.path.join(directory, f'unvoiced_{uuid}.webm')

    voiced_content = await voiced.read()
    with open(voiced_path, 'wb') as f:
        f.write(voiced_content)

    unvoiced_content = await unvoiced.read()
    with open(unvoiced_path, 'wb') as f:
        f.write(unvoiced_content)

    recording = Recording(uuid, voiced_path, unvoiced_path, text=text, timestamp=timestamp)
    save_recording(recording)

    # WebM → FLAC conversion is intentionally NOT done here anymore.
    # Run `python updated_processing.py` to convert pending recordings.
    return Response(status_code=200)


@app.post('/recordings/{uuid}')
async def patch_recording(
    uuid: str,
    unvoiced_flac_path: str = Form(...),
    voiced_flac_path: str = Form(...),
):
    recording = get_recording(uuid)
    if recording is None:
        return Response(status_code=404)
    recording.unvoiced_file_path_flac = unvoiced_flac_path
    recording.voiced_file_path_flac = voiced_flac_path
    update_recording(recording)
    return Response(status_code=200)


# ─── Text storage ─────────────────────────────────────────────────────────

@app.get('/text')
async def get_text_api():
    return get_texts()


@app.post('/text')
async def save_text_api(data: dict):
    text = data.get('text', '')
    save_text(text)
    return Response(status_code=200)


# ─── Models ────────────────────────────────────────────────────────────────

@app.get('/models')
async def get_models_api():
    return get_models_from_db()


@app.get('/models/{filename:path}')
async def get_model_file(filename: str):
    safe_name = os.path.basename(filename)
    models_dir = UPLOAD_DIRS['models']
    file_path = os.path.join(models_dir, safe_name)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    return FileResponse(
        file_path,
        media_type='application/octet-stream',
        headers={'Content-Disposition': f'attachment; filename="{safe_name}"'},
    )


@app.delete('/models')
async def delete_model_api(data: dict):
    filename = data.get('filename', '')
    safe_name = os.path.basename(filename)
    models_dir = UPLOAD_DIRS['models']
    file_path = os.path.join(models_dir, safe_name)

    conn, cursor = connect_db()
    cursor.execute('DELETE FROM models WHERE filename = ?', (safe_name,))
    conn.commit()
    close_db(conn, cursor)

    if os.path.exists(file_path):
        os.remove(file_path)
        return Response(status_code=200)
    return Response(status_code=404)

# ─── Images ────────────────────────────────────────────────────────────────

@app.get('/images/{filename:path}')
async def get_image_file(filename: str):
    safe_name = os.path.basename(filename)
    img_dir = UPLOAD_DIRS['images']
    file_path = os.path.join(img_dir, safe_name)
    if not os.path.exists(file_path):
        return Response(status_code=404)

    ext = os.path.splitext(safe_name)[1].lower()
    mime_map = {
        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
    }
    mime = mime_map.get(ext, 'application/octet-stream')
    return FileResponse(file_path, media_type=mime)

# ─── WebSocket: chunked large-file upload ─────────────────────────────────

@app.websocket('/ws/upload')
async def upload_websocket(websocket: WebSocket):
    await websocket.accept()
    sid = None  # upload session id
    try:
        # ── Step 1: receive metadata ──
        raw = await websocket.receive_text()
        meta = json.loads(raw)
        if meta.get('action') != 'upload_start':
            await websocket.send_json({
                "status": "error",
                "message": "First message must be upload_start",
            })
            return

        filename = meta.get('filename', 'unnamed')
        file_size = meta.get('fileSize', 0)
        destination = meta.get('destination', 'models')

        if destination not in UPLOAD_DIRS:
            await websocket.send_json({
                "status": "error",
                "message": f"Invalid destination '{destination}'. "
                           f"Must be one of: {list(UPLOAD_DIRS.keys())}",
            })
            return

        # Normalise path separators and strip leading slash
        safe_name = filename.replace('\\', '/').lstrip('/')
        # Prevent directory traversal
        parts = safe_name.split('/')
        if any(p in ('..', '.') or p == '' for p in parts):
            await websocket.send_json({
                'status': 'error',
                'message': 'Filename must not contain .. or . path components',
            })
            return
        if not safe_name:
            safe_name = 'unnamed'

        target_dir = UPLOAD_DIRS[destination]
        sid = str(uuid_lib.uuid4())
        tmp_path = os.path.join(target_dir, f'.{sid}.tmp')

        # Resolve final path (allow subdirectory structure), avoid name collisions
        final_path = os.path.join(target_dir, safe_name)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        counter = 1
        name_stem, ext = os.path.splitext(os.path.basename(safe_name))
        dir_part = os.path.dirname(safe_name)
        while os.path.exists(final_path):
            stem = f'{name_stem}_{counter}{ext}'
            final_path = os.path.join(target_dir, dir_part, stem) if dir_part else os.path.join(target_dir, stem)
            counter += 1

        # ── Step 2: register session ──
        with upload_lock:
            upload_sessions[sid] = {
                'filename': os.path.basename(final_path),
                'final_path': final_path,
                'tmp_path': tmp_path,
                'total_size': file_size,
                'received': 0,
                'tmp_file': open(tmp_path, 'wb'),
                'destination': destination,
            }

        MAX_CHUNK = 256 * 1024  # 256 KB per chunk

        await websocket.send_json({
            "status": "ready",
            "uploadId": sid,
            "maxChunkSize": MAX_CHUNK,
        })

        # ── Steps 3-5: receive chunks ──
        while True:
            raw = await websocket.receive()
            if raw is None:
                break  # client disconnected

            # Control message (JSON text)
            if raw['type'] == 'websocket.receive':
                msg = raw.get('text')
                if msg is not None:
                    ctrl = json.loads(msg)
                    act = ctrl.get('action')

                    if act == 'upload_complete' and ctrl.get('uploadId') == sid:
                        # ── Step 6: finalise ──
                        with upload_lock:
                            sess = upload_sessions.get(sid)
                            if sess:
                                sess['tmp_file'].close()
                                os.rename(sess['tmp_path'], sess['final_path'])

                                if sess['destination'] == 'models':
                                    save_model_to_db(
                                        sess['filename'],
                                        sess['final_path'],
                                        sess['total_size'],
                                    )

                                await websocket.send_json({
                                    "status": "complete",
                                    "uploadId": sid,
                                    "filePath": sess['final_path'],
                                    "fileName": sess['filename'],
                                    "fileSize": sess['total_size'],
                                })
                                del upload_sessions[sid]
                        break

                    elif act == 'upload_cancel' and ctrl.get('uploadId') == sid:
                        with upload_lock:
                            sess = upload_sessions.pop(sid, None)
                            if sess:
                                sess['tmp_file'].close()
                                if os.path.exists(sess['tmp_path']):
                                    os.remove(sess['tmp_path'])
                        await websocket.send_json({
                            "status": "cancelled",
                            "uploadId": sid,
                        })
                        break

                    else:
                        await websocket.send_json({
                            "status": "error",
                            "message": f"Unexpected control message: {act}",
                        })
                    continue

                # Binary chunk
                bytes_data = raw.get('bytes')
                if bytes_data is not None:
                    with upload_lock:
                        sess = upload_sessions.get(sid)
                        if not sess:
                            await websocket.send_json({
                                "status": "error",
                                "message": "Session not found",
                            })
                            break
                        sess['tmp_file'].write(bytes_data)
                        sess['received'] += len(bytes_data)
                        recv, total = sess['received'], sess['total_size']

                    pct = round((recv / total * 100), 1) if total > 0 else 0
                    await websocket.send_json({
                        "status": "progress",
                        "uploadId": sid,
                        "received": recv,
                        "total": total,
                        "percent": pct,
                    })

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError:
        try:
            await websocket.send_json({
                "status": "error",
                "message": "Invalid JSON in control message",
            })
        except Exception:
            pass
    except Exception as exc:
        print(f"[ws/upload] Error: {exc}")
        try:
            await websocket.send_json({
                "status": "error",
                "uploadId": sid or "unknown",
                "message": str(exc),
            })
        except Exception:
            pass
    finally:
        # Cleanup on unexpected disconnect
        if sid:
            with upload_lock:
                sess = upload_sessions.pop(sid, None)
                if sess:
                    sess['tmp_file'].close()
                    if os.path.exists(sess['tmp_path']):
                        os.remove(sess['tmp_path'])

# ══════════════════════════════════════════════════════════════════════════
#  DB FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def setup_db():
    conn = sqlite3.connect('recordings.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            name TEXT,
            text TEXT NOT NULL,
            voiced_file_path TEXT NOT NULL,
            unvoiced_file_path TEXT NOT NULL,
            voiced_file_path_flac TEXT,
            unvoiced_file_path_flac TEXT
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            upload_time TEXT NOT NULL,
            description TEXT DEFAULT ''
        );
    ''')
    try:
        cursor.execute('ALTER TABLE recordings ADD COLUMN timestamp TEXT;')
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()


def connect_db():
    conn = sqlite3.connect('recordings.db')
    cursor = conn.cursor()
    return conn, cursor


def close_db(conn, cursor):
    cursor.close()
    conn.close()


# ─── FETCH FUNCTIONS ───────────────────────────────────────────────────────

def get_recordings():
    conn, cursor = connect_db()
    cursor.execute('SELECT * FROM recordings')
    fetched = cursor.fetchall()
    close_db(conn, cursor)
    return [
        Recording(
            uuid=row[1],
            voiced_file_path=row[4],
            unvoiced_file_path=row[5],
            voiced_file_path_flac=row[6],
            unvoiced_file_path_flac=row[7],
            text=row[3],
            timestamp=row[8] if len(row) > 8 else '',
        )
        for row in fetched
    ]


def get_recording(uuid: str) -> Optional[Recording]:
    conn, cursor = connect_db()
    cursor.execute('SELECT * FROM recordings WHERE uuid LIKE ?', (uuid,))
    fetch = cursor.fetchone()
    close_db(conn, cursor)
    if fetch is None:
        return None
    return Recording(
        uuid=fetch[1],
        voiced_file_path=fetch[4],
        unvoiced_file_path=fetch[5],
        voiced_file_path_flac=fetch[6],
        unvoiced_file_path_flac=fetch[7],
        text=fetch[3],
        timestamp=fetch[8] if len(fetch) > 8 else '',
    )


def get_uuids_for_processed_recordings():
    conn, cursor = connect_db()
    cursor.execute(
        'SELECT * FROM recordings WHERE '
        '(voiced_file_path_flac IS NOT NULL AND voiced_file_path_flac != "") AND '
        '(unvoiced_file_path_flac IS NOT NULL AND unvoiced_file_path_flac != "")'
    )
    fetched = cursor.fetchall()
    close_db(conn, cursor)
    return [row[1] for row in fetched]


def update_recording(recording: Recording):
    conn, cursor = connect_db()
    cursor.execute(
        'UPDATE recordings SET voiced_file_path_flac = ?, unvoiced_file_path_flac = ? WHERE uuid = ?',
        (recording.voiced_file_path_flac, recording.unvoiced_file_path_flac, recording.uuid),
    )
    conn.commit()
    close_db(conn, cursor)


def get_texts():
    conn, cursor = connect_db()
    cursor.execute('SELECT * FROM text')
    texts = cursor.fetchall()
    close_db(conn, cursor)
    return texts


def save_text(text: str):
    conn, cursor = connect_db()
    cursor.execute('INSERT INTO text (text) VALUES (?)', (text,))
    conn.commit()
    close_db(conn, cursor)


def save_recording(recording: Recording):
    conn, cursor = connect_db()
    cursor.execute(
        'INSERT OR REPLACE INTO recordings '
        '(uuid, voiced_file_path, unvoiced_file_path, text, timestamp, voiced_file_path_flac, unvoiced_file_path_flac) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (recording.uuid, recording.voiced_file_path, recording.unvoiced_file_path,
         recording.text, recording.timestamp,
         recording.voiced_file_path_flac, recording.unvoiced_file_path_flac),
    )
    conn.commit()
    close_db(conn, cursor)


# ─── Model DB helpers ──────────────────────────────────────────────────────

def save_model_to_db(filename: str, file_path: str, file_size: int, description: str = ''):
    conn, cursor = connect_db()
    now = datetime.datetime.now().isoformat()
    cursor.execute(
        'INSERT INTO models (filename, file_path, file_size, upload_time, description) VALUES (?, ?, ?, ?, ?)',
        (filename, file_path, file_size, now, description),
    )
    conn.commit()
    close_db(conn, cursor)


def get_models_from_db() -> list[dict]:
    conn, cursor = connect_db()
    cursor.execute('SELECT * FROM models ORDER BY upload_time DESC')
    rows = cursor.fetchall()
    close_db(conn, cursor)
    return [
        {
            'id': r[0],
            'filename': r[1],
            'filePath': r[2],
            'fileSize': r[3],
            'uploadTime': r[4],
            'description': r[5],
        }
        for r in rows
    ]


# ─── App factory (for gunicorn / uvicorn) ─────────────────────────────────

def create_app():
    """Factory function – ensures DB is initialised, then returns the app."""
    if not os.path.exists('recordings.db'):
        print('Creating recordings.db')
        open('recordings.db', 'a').close()
        setup_db()

    setup_db()

    return app


# ─── Direct entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    create_app()
    uvicorn.run('server:app', host='127.0.0.1', port=14080, reload=False)
