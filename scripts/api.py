"""
Task 3: CRUD API Endpoints for Air Quality Data
Serves both SQL (SQLite) and MongoDB (mongomock) databases.
Run with: python api.py
"""

from flask import Flask, request, jsonify
import sqlite3
import mongomock
import os
import json
from datetime import datetime

app = Flask(__name__)

# ──────────────────────────────────────────────────
# Database Paths & Config
# ──────────────────────────────────────────────────
# scripts/ is one level below project root; DB lives in data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SQL_DB_PATH = os.path.join(ROOT_DIR, "data", "airquality.db")

# MongoDB (in-memory mock)
mongo_client = mongomock.MongoClient()
mongo_db = mongo_client["airquality_db"]
mongo_collection = mongo_db["air_quality_readings"]


def get_sql_conn():
    """Get a new SQLite connection."""
    conn = sqlite3.connect(SQL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_mongo_from_sql():
    """Populate MongoDB from SQLite if empty."""
    if mongo_collection.count_documents({}) > 0:
        return
    conn = get_sql_conn()
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT r.reading_id, r.datetime, r.co_gt, r.co_sensor, r.benzene_gt,
               r.nmhc_sensor, r.nox_gt, r.nox_sensor, r.no2_gt, r.no2_sensor, r.o3_sensor,
               m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r
        LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        ORDER BY r.datetime
    """).fetchall()
    
    docs = []
    for row in rows:
        doc = {
            "reading_id": row["reading_id"],
            "station": {"station_id": 1, "name": "UCI Air Quality Station",
                        "city": "Italian City", "country": "Italy"},
            "datetime": datetime.fromisoformat(row["datetime"]),
            "pollutants": {
                "co_gt": row["co_gt"], "co_sensor": row["co_sensor"],
                "benzene_gt": row["benzene_gt"], "nmhc_sensor": row["nmhc_sensor"],
                "nox_gt": row["nox_gt"], "nox_sensor": row["nox_sensor"],
                "no2_gt": row["no2_gt"], "no2_sensor": row["no2_sensor"],
                "o3_sensor": row["o3_sensor"]
            },
            "meteorology": {
                "temperature": row["temperature"], "rel_humidity": row["rel_humidity"],
                "abs_humidity": row["abs_humidity"]
            }
        }
        docs.append(doc)
    
    if docs:
        mongo_collection.insert_many(docs)
    conn.close()
    print(f"Loaded {len(docs)} documents into MongoDB")


# ══════════════════════════════════════════════════
# SQL CRUD ENDPOINTS
# ══════════════════════════════════════════════════

@app.route("/api/sql/readings", methods=["POST"])
def sql_create_reading():
    """POST — Create a new reading in SQL database."""
    data = request.get_json(force=True)
    conn = get_sql_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO readings (station_id, datetime, co_gt, co_sensor, benzene_gt,
                                  nmhc_sensor, nox_gt, nox_sensor, no2_gt, no2_sensor, o3_sensor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data.get("station_id", 1), data["datetime"],
              data.get("co_gt"), data.get("co_sensor"), data.get("benzene_gt"),
              data.get("nmhc_sensor"), data.get("nox_gt"), data.get("nox_sensor"),
              data.get("no2_gt"), data.get("no2_sensor"), data.get("o3_sensor")))
        
        reading_id = cursor.lastrowid
        
        # Also insert meteorology
        cursor.execute("""
            INSERT INTO meteorology (reading_id, temperature, rel_humidity, abs_humidity)
            VALUES (?, ?, ?, ?)
        """, (reading_id, data.get("temperature"), data.get("rel_humidity"),
              data.get("abs_humidity")))
        
        conn.commit()
        return jsonify({"message": "Reading created", "reading_id": reading_id}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()


@app.route("/api/sql/readings", methods=["GET"])
def sql_get_readings():
    """GET — Retrieve readings. Supports ?start=...&end=... for date range."""
    conn = get_sql_conn()
    start = request.args.get("start")
    end = request.args.get("end")
    limit = request.args.get("limit", 100, type=int)
    
    query = """
        SELECT r.reading_id, r.datetime, r.co_gt, r.co_sensor, r.benzene_gt,
               r.nmhc_sensor, r.nox_gt, r.nox_sensor, r.no2_gt, r.no2_sensor, r.o3_sensor,
               m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r
        LEFT JOIN meteorology m ON r.reading_id = m.reading_id
    """
    params = []
    
    if start and end:
        query += " WHERE r.datetime BETWEEN ? AND ?"
        params = [start, end]
    
    query += " ORDER BY r.datetime DESC LIMIT ?"
    params.append(limit)
    
    rows = conn.execute(query, params).fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return jsonify(results)


@app.route("/api/sql/readings/<int:reading_id>", methods=["GET"])
def sql_get_reading(reading_id):
    """GET — Retrieve a single reading by ID."""
    conn = get_sql_conn()
    row = conn.execute("""
        SELECT r.*, m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r
        LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        WHERE r.reading_id = ?
    """, (reading_id,)).fetchone()
    conn.close()
    
    if row:
        return jsonify(dict(row))
    return jsonify({"error": "Reading not found"}), 404


@app.route("/api/sql/readings/<int:reading_id>", methods=["PUT"])
def sql_update_reading(reading_id):
    """PUT — Update an existing reading."""
    data = request.get_json(force=True)
    conn = get_sql_conn()
    try:
        conn.execute("""
            UPDATE readings SET co_gt=?, co_sensor=?, benzene_gt=?, nmhc_sensor=?,
                   nox_gt=?, nox_sensor=?, no2_gt=?, no2_sensor=?, o3_sensor=?
            WHERE reading_id=?
        """, (data.get("co_gt"), data.get("co_sensor"), data.get("benzene_gt"),
              data.get("nmhc_sensor"), data.get("nox_gt"), data.get("nox_sensor"),
              data.get("no2_gt"), data.get("no2_sensor"), data.get("o3_sensor"),
              reading_id))
        
        conn.execute("""
            UPDATE meteorology SET temperature=?, rel_humidity=?, abs_humidity=?
            WHERE reading_id=?
        """, (data.get("temperature"), data.get("rel_humidity"),
              data.get("abs_humidity"), reading_id))
        
        conn.commit()
        return jsonify({"message": f"Reading {reading_id} updated"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()


@app.route("/api/sql/readings/<int:reading_id>", methods=["DELETE"])
def sql_delete_reading(reading_id):
    """DELETE — Remove a reading."""
    conn = get_sql_conn()
    conn.execute("DELETE FROM meteorology WHERE reading_id=?", (reading_id,))
    conn.execute("DELETE FROM readings WHERE reading_id=?", (reading_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": f"Reading {reading_id} deleted"})


@app.route("/api/sql/readings/latest", methods=["GET"])
def sql_latest_reading():
    """GET — Retrieve the most recent reading."""
    conn = get_sql_conn()
    row = conn.execute("""
        SELECT r.*, m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r
        LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        ORDER BY r.datetime DESC LIMIT 1
    """).fetchone()
    conn.close()
    if row:
        return jsonify(dict(row))
    return jsonify({"error": "No readings found"}), 404


@app.route("/api/sql/readings/daterange", methods=["GET"])
def sql_daterange():
    """GET — Retrieve readings within a date range. ?start=YYYY-MM-DD&end=YYYY-MM-DD"""
    start = request.args.get("start")
    end = request.args.get("end")
    if not start or not end:
        return jsonify({"error": "Both 'start' and 'end' parameters required"}), 400
    
    conn = get_sql_conn()
    rows = conn.execute("""
        SELECT r.reading_id, r.datetime, r.co_gt, r.co_sensor, r.benzene_gt,
               r.nmhc_sensor, r.nox_gt, r.nox_sensor, r.no2_gt, r.no2_sensor,
               r.o3_sensor, m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r
        LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        WHERE r.datetime BETWEEN ? AND ?
        ORDER BY r.datetime
    """, (start, end)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════════
# MongoDB CRUD ENDPOINTS
# ══════════════════════════════════════════════════

def mongo_doc_to_json(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("datetime"), datetime):
        doc["datetime"] = doc["datetime"].isoformat()
    return doc


@app.route("/api/mongo/readings", methods=["POST"])
def mongo_create_reading():
    """POST — Create a new reading in MongoDB."""
    data = request.get_json(force=True)
    doc = {
        "station": data.get("station", {"station_id": 1, "name": "UCI Air Quality Station",
                                         "city": "Italian City", "country": "Italy"}),
        "datetime": datetime.fromisoformat(data["datetime"]),
        "pollutants": data.get("pollutants", {}),
        "meteorology": data.get("meteorology", {})
    }
    result = mongo_collection.insert_one(doc)
    return jsonify({"message": "Reading created", "id": str(result.inserted_id)}), 201


@app.route("/api/mongo/readings", methods=["GET"])
def mongo_get_readings():
    """GET — Retrieve readings. Supports ?limit=N"""
    limit = request.args.get("limit", 100, type=int)
    docs = list(mongo_collection.find().sort("datetime", -1).limit(limit))
    return jsonify([mongo_doc_to_json(d) for d in docs])


@app.route("/api/mongo/readings/<reading_id>", methods=["GET"])
def mongo_get_reading(reading_id):
    """GET — Retrieve a single reading by reading_id field."""
    doc = mongo_collection.find_one({"reading_id": int(reading_id)})
    if doc:
        return jsonify(mongo_doc_to_json(doc))
    return jsonify({"error": "Reading not found"}), 404


@app.route("/api/mongo/readings/<reading_id>", methods=["PUT"])
def mongo_update_reading(reading_id):
    """PUT — Update an existing reading in MongoDB."""
    data = request.get_json(force=True)
    update_fields = {}
    if "pollutants" in data:
        for k, v in data["pollutants"].items():
            update_fields[f"pollutants.{k}"] = v
    if "meteorology" in data:
        for k, v in data["meteorology"].items():
            update_fields[f"meteorology.{k}"] = v
    
    result = mongo_collection.update_one(
        {"reading_id": int(reading_id)},
        {"$set": update_fields}
    )
    if result.modified_count > 0:
        return jsonify({"message": f"Reading {reading_id} updated"})
    return jsonify({"error": "Reading not found or no changes made"}), 404


@app.route("/api/mongo/readings/<reading_id>", methods=["DELETE"])
def mongo_delete_reading(reading_id):
    """DELETE — Remove a reading from MongoDB."""
    result = mongo_collection.delete_one({"reading_id": int(reading_id)})
    if result.deleted_count > 0:
        return jsonify({"message": f"Reading {reading_id} deleted"})
    return jsonify({"error": "Reading not found"}), 404


@app.route("/api/mongo/readings/latest", methods=["GET"])
def mongo_latest_reading():
    """GET — Retrieve the most recent reading from MongoDB."""
    doc = mongo_collection.find_one(sort=[("datetime", -1)])
    if doc:
        return jsonify(mongo_doc_to_json(doc))
    return jsonify({"error": "No readings found"}), 404


@app.route("/api/mongo/readings/daterange", methods=["GET"])
def mongo_daterange():
    """GET — Retrieve readings within a date range. ?start=YYYY-MM-DD&end=YYYY-MM-DD"""
    start = request.args.get("start")
    end = request.args.get("end")
    if not start or not end:
        return jsonify({"error": "Both 'start' and 'end' parameters required"}), 400
    
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end + "T23:59:59") if "T" not in end else datetime.fromisoformat(end)
    
    docs = list(mongo_collection.find(
        {"datetime": {"$gte": start_dt, "$lte": end_dt}}
    ).sort("datetime", 1))
    
    return jsonify([mongo_doc_to_json(d) for d in docs])


# ══════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "sql_db": SQL_DB_PATH,
                    "mongo_docs": mongo_collection.count_documents({})})


if __name__ == "__main__":
    init_mongo_from_sql()
    print(f"\n Air Quality API starting...")
    print(f"   SQL DB: {SQL_DB_PATH}")
    print(f"   MongoDB documents: {mongo_collection.count_documents({})}")
    print(f"\nEndpoints:")
    print(f"  SQL:   POST/GET/PUT/DELETE /api/sql/readings[/<id>]")
    print(f"         GET /api/sql/readings/latest")
    print(f"         GET /api/sql/readings/daterange?start=...&end=...")
    print(f"  Mongo: POST/GET/PUT/DELETE /api/mongo/readings[/<id>]")
    print(f"         GET /api/mongo/readings/latest")
    print(f"         GET /api/mongo/readings/daterange?start=...&end=...")
    print(f"  Health: GET /api/health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
