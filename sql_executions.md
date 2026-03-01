# Rrooster — Query Explanations

---

### 1. Create Sessions Table

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    project_id          TEXT NOT NULL,
    name                TEXT NOT NULL,
    framework           TEXT NOT NULL,
    project_name        TEXT NOT NULL,
    project_description TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
```
So this reates the single static table that set's up every training session in the database. 

If the table already exists on startup, it is left untouched, making it safe to run every time the backend restarts, which I think saves people.

---

### 2. Insert Session Row

```sql
INSERT OR IGNORE INTO sessions
(session_id, project_id, name, framework, project_name, project_description, created_at)
VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
```
Writes one row representing the newly registered training session, using `datetime('now')` to stamp the exact moment of registration. `OR IGNORE` 
ensures that if a client reconnects and sends a duplicate `Register` message, the existing session row is silently preserved rather than causing an error.

---

### 3. Create Metric Table (Dynamic)

```sql
CREATE TABLE IF NOT EXISTS metric_{name} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL REFERENCES sessions(session_id),
    step       INTEGER NOT NULL,
    timestamp  TEXT    NOT NULL,
    value      TEXT    NOT NULL
)
```
Dynamically creates one table per user-declared metric (e.g. `metric_loss`, `metric_accuracy`), run inside the same transaction as the session insert so both are atomic. 
The `REFERENCES sessions(session_id)` foreign key ensures no child metric rows can exist without a parent session.

---

### 4. Insert Log Entry

```sql
INSERT INTO metric_{name} (session_id, step, timestamp, value)
VALUES (?, ?, ?, ?)
```
Appends a single training step's data to the appropriate metric table, where `value` is a JSON-serialized blob of everything in `Log.data` for that step. 
Because a single SQLite `INSERT` is inherently atomic, I didn't put any explicit transaction is needed here.

---

### 5. List All Sessions

```sql
SELECT session_id, project_id, name, framework,
       project_name, project_description, created_at
FROM sessions
ORDER BY created_at DESC
```
Returns all registered sessions sorted by most recent first, giving the CLI's session picker an always up-to-date list to display. No filtering is applied every session ever recorded on this device is returned, which
respects the original goal of robust, local based tracking.

---

### 6. Get Single Session

```sql
SELECT session_id, project_id, name, framework,
       project_name, project_description, created_at
FROM sessions
WHERE session_id = ?
```
Fetches one specific session by its UUID, returning `None` if it doesn't exist. Since UUID is standard, choose it over automated `usize` assignments!
Used by the CLI when the user selects a session to monitor or review.

---

### 7. Discover Metric Tables

```sql
SELECT name FROM sqlite_master
WHERE type = 'table' AND name LIKE 'metric_%'
ORDER BY name
```
Queries SQLite's internal schema catalog to discover all dynamically created metric tables, since their names aren't known at compile time. This is the read-side mirror of the dynamic `CREATE TABLE` pattern used during session registration.

---


## AI disclosure - AI was used for grammer checks
