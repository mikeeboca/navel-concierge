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
    * `https://neo4j.com/download`
2.  **Instanz und Datenbanken anlegen:**
    * Erstellen Sie in Neo4j Desktop eine neue Instanz/DBMS und legen Sie ein **Passwort** fest.
    * Erstellen Sie in dieser Instanz die drei Datenbanken: `kgraphdata`, `mtfdata`, `naveldata`.
3.  **Plugins installieren:**
    * In der Instanz unter **Plugins** **APOC** aktivieren.

#### 🛠️ Detail: n10s Plugin (neosemantics) installieren

1.  **Im Plugins-Tab** Ihrer Instanz das Plugin **"neosemantics (n10s)"** suchen und installieren.
2.  *Alternativ:* Laden Sie die kompatible `.jar` Datei von der n10s-Seite herunter und kopieren Sie diese in den **`plugins`**-Ordner Ihrer Instanz.
3.  Starten Sie die DBMS-Instanz neu.

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
    * `https://ollama.com/download`
2.  **Ollama-Server starten:**
    * Öffnen Sie ein Terminal und führen Sie aus:
        ```
        ollama serve
        ```
    * Der Dienst läuft auf `http://127.0.0.1:11434` und muss **im Hintergrund aktiv** sein.

### Modell herunterladen und testen

1.  **Modell herunterladen:**
    * In einem zweiten Terminal:
        ```
        ollama pull llama3.2:3b-instruct-q4_0
        ```
2.  **Funktionstest:**
    * Starten Sie das Modell kurz:
        ```
        ollama run llama3.2:3b-instruct-q4_0
        ```

---

## 3. 🐍 Adapter-System (KG ↔ LLM)

**Voraussetzung:** Neo4j läuft **und** `ollama serve` läuft.

1.  **Neo4j-Verbindungsdaten anpassen:**
    * Öffnen Sie die Datei **`kgadapterv2.py`** und passen Sie `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` sowie die Datenbank-Namen an Ihre Installation an.
2.  **Python-Umgebung vorbereiten:**
    * Installieren Sie **Python 3.13.7**.
    * Erstellen und aktivieren Sie optional eine virtuelle Umgebung:
        ```
        python3 -m venv venv
        source venv/bin/activate
        ```
3.  **Pakete installieren:**
    ```
    pip install neo4j
    pip install requests
    ```
4.  **Adapter testen:**
    * Führen Sie das Script aus, um die Verbindung zu prüfen:
        ```
        python kgadapterv2.py "TESTFRAGE ?"
        ```

---

## 4. 📊 Evaluationssystem (Gold-Eval)

**Voraussetzung:** Alle vorherigen Schritte (Neo4j, Ollama, Adapter) funktionieren.

1.  **Ausführung starten:**
    * Das System nimmt Fragen aus `100questions.jsonl` und schreibt die Ergebnisse als JSONL-Datei.
    ```
    python3 run_gold_eval.py --out-jsonl "100questionsOUT.jsonl"
    ```
    
---

## 5. ℹ️ Hinweis zu Modi-Bezeichnungen

In diesem Projekt werden die Modi in Code, Logs und Abbildungen teilweise mit leicht abweichenden Kurzbezeichnungen geführt. Die Zuordnung ist wie folgt:

- `llm-base` entspricht `llm`
- `llm-a` entspricht `llm-aug`
- `rag` entspricht `kg-r`
- `rag-aug` entspricht `kg-rag`

Bitte berücksichtigen Sie diese Mapping-Tabelle beim Vergleich von README, Code, Evaluationsskripten und Abbildungen.
