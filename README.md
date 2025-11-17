# 🚀 Projekt-Setup: RAG/Evaluation mit Neo4j und Ollama

Dieses Projekt verbindet **mehrere Wissensgraphen (Neo4j)** mit einem **lokalen Large Language Model (Ollama)** und stellt darauf ein **RAG-** sowie **Evaluationssetup** bereit.

**Die Komponenten müssen in dieser Reihenfolge eingerichtet werden:**

1.  **Neo4j installieren & vorbereiten**
2.  **Ollama installieren & Modell bereitstellen**
3.  **Adapter-System (KG ↔ LLM) konfigurieren**
4.  **Evaluationssystem (Gold-Eval) starten**

---

## 1. 💾 Neo4j installieren & vorbereiten

### Installation und Setup

1.  **Neo4j Desktop herunterladen:**
    * [https://neo4j.com/download](https://neo4j.com/download)
2.  **Instanz und Datenbanken anlegen:**
    * Erstellen Sie in Neo4j Desktop eine neue Instanz/DBMS und legen Sie ein **Passwort** fest.
    * Erstellen Sie in dieser Instanz die drei Datenbanken: `kgraphdata` (für LPD Daten), `mtfdata`, `naveldata`.
3.  **Plugins installieren:**
    * In der Instanz unter **Plugins** **APOC** aktivieren.

#### 🛠️ Detail: n10s Plugin (neosemantics) installieren

1.  **Neosemantics-JAR herunterladen:**
    * Laden Sie die aktuelle Version von der Releases-Seite herunter:  
      [https://github.com/neo4j-labs/neosemantics/releases](https://github.com/neo4j-labs/neosemantics/releases)
    * Speichern Sie die `.jar`-Datei lokal ab.

2.  **JAR in den Plugins-Ordner der Instanz kopieren:**
    * In Neo4j Desktop bei Ihrer **DBMS-Instanz** auf die **drei Punkte (`…`)** klicken.
    * **Open Folder → Plugins** wählen.
    * Die heruntergeladene `neosemantics-…​.jar` in diesen `plugins`-Ordner kopieren.

3.  **Neo4j-Konfiguration (neo4j.conf) für APOC und n10s anpassen:**
    * Erneut bei der gleichen Instanz auf die **drei Punkte (`…`)** klicken.
    * **Open → neo4j.conf** (bzw. Konfiguration öffnen) wählen.
    * Sicherstellen, dass folgende Einträge vorhanden sind (ggf. ergänzen oder anpassen):

      ```properties
      dbms.security.procedures.unrestricted=apoc.*,n10s.*
      dbms.security.procedures.allowlist=apoc.*,n10s.*
      ```

4.  **DBMS-Instanz neu starten**, damit APOC und n10s aktiv sind.


### Konfiguration und Datenimport

**Führen Sie alle folgenden Cypher-Befehle nacheinander im integrierten Browser für jede der drei Datenbanken (`kgraphdata`, `mtfdata`, `naveldata`) aus.**

1.  **Constraint setzen:**
    ```cypher
    CREATE CONSTRAINT n10s_unique_uri
    FOR (r:Resource)
    REQUIRE r.uri IS UNIQUE;
    ```

2.  **n10s konfigurieren:**
    ```cypher
    CALL n10s.graphconfig.init({
      handleVocabUris : "IGNORE",
      applyNeo4jLabels : true,
      addResourceLabels: false,
      handleMultival  : "ARRAY"
    });
    ```

3.  **TTL importieren:**
    * **Achtung: Pfad anpassen!**
    ```cypher
    CALL n10s.rdf.import.fetch(
      "file:///Users/.../PATH/TO/FILE.ttl",
      "Turtle",
      { commitSize: 1000 }
    );
    ```

---

## 2. 🤖 Ollama installieren & Modell bereitstellen

### Installation und Start

1.  **Ollama herunterladen:**
    * [https://ollama.com/download](https://ollama.com/download)
2.  **Ollama-Server starten:**
    * Öffnen Sie ein Terminal und führen Sie aus:
        ```bash
        ollama serve
        ```
    * Der Dienst läuft auf `http://127.0.0.1:11434` und muss **im Hintergrund aktiv** sein.

### Modell herunterladen und testen

1.  **Modell herunterladen:**
    * In einem zweiten Terminal:
        ```bash
        ollama pull llama3.2:3b-instruct-q4_0
        ```
2.  **Funktionstest:**
    * Starten Sie das Modell kurz:
        ```bash
        ollama run llama3.2:3b-instruct-q4_0
        ```

---

## 3. 🐍 Adapter-System (KG ↔ LLM)

**Voraussetzung:** Neo4j läuft **und** `ollama serve` läuft.

1.  **Neo4j-Verbindungsdaten anpassen:**
    * Öffnen Sie die Datei **`kgadapterv2.py`** und passen Sie `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` sowie die Datenbank-Namen an Ihre Installation an.

2.  **Python-Umgebung vorbereiten (empfohlen im Ordner `llm-integration/`):**
    * Installieren Sie **Python 3.13**.
    * Wechseln Sie in den Projektordner:
      ```bash
      cd llm-integration
      ```

    * Erstellen und aktivieren Sie optional eine virtuelle Umgebung:

      **macOS / Linux:**
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```

      **Windows (CMD):**
      ```bat
      python -m venv venv
      venv\Scripts\activate
      ```

      *(Hinweis: Unter Windows PowerShell kann die Aktivierung z.B. mit `.\venv\Scripts\Activate.ps1` erfolgen.)*

3.  **Pakete installieren:**
    ```bash
    pip install neo4j
    pip install requests
    ```

4.  **Adapter testen:**
    * Führen Sie das Script aus, um die Verbindung zu prüfen:
        ```bash
        python kgadapterv2.py "TESTFRAGE ?"
        ```

---

## 4. 📊 Evaluationssystem (Gold-Eval)

**Voraussetzung:** Alle vorherigen Schritte (Neo4j, Ollama, Adapter) funktionieren.

1.  **Ausführung starten:**
    * Das System nimmt Fragen aus `100questions.jsonl` und schreibt die Ergebnisse als JSONL-Datei.
    ```bash
    python run_gold_eval.py --out-jsonl "100questionsOUT.jsonl"
    ```

---

## 5. ℹ️ Hinweis zu Modi-Bezeichnungen

In diesem Projekt werden die Modi in Code, Logs und Abbildungen teilweise mit leicht abweichenden Kurzbezeichnungen geführt. Die Zuordnung ist wie folgt:

- `llm-base` entspricht `llm`
- `llm-a` entspricht `llm-aug`
- `rag` entspricht `kg-r`
- `rag-aug` entspricht `kg-rag`

Bitte berücksichtigen Sie diese Mapping-Tabelle beim Vergleich von README, Code, Evaluationsskripten und Abbildungen.
