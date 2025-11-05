import csv
import hashlib
import re

def _slug(text):
    if not text:
        return ''
    text = re.sub(r'[\s\W]+', '_', text).strip('_')
    return text.lower()

def _create_uri(s_id, base_prefix, slug=None):
    if not s_id:
        return None
    if slug:
        return f'{base_prefix}:{_slug(slug)}'
    return f'{base_prefix}:n{s_id}'

def _read_csv_rows(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
    return rows

def _clean_ttl_string(text):
    """
    Maskiert Anführungszeichen und ersetzt Zeilenumbrüche, um TTL-Syntaxfehler zu vermeiden.
    """
    if not isinstance(text, str):
        return ""
    # Maskiert doppelte Anführungszeichen und Backslashes
    cleaned_text = text.replace('\\', '\\\\').replace('"', '\\"')
    # Ersetzt Zeilenumbrüche durch Leerzeichen
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    return cleaned_text

def _split_labels(labels_string):
    """
    Teilt einen String mit Labels und gibt den ersten als Hauptlabel und den Rest zurück.
    """
    if not labels_string:
        return None, []
    labels = labels_string.split(';')
    main_label = labels[0].strip()
    other_labels = [l.strip() for l in labels[1:] if l.strip()]
    return main_label, other_labels

def write_ttl(node_file, edge_file, output_file):
    base_prefix = "ex"
    schema_prefix = "schema"
    rdfs_prefix = "rdfs"

    parts = [
        "@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .",
        f"@prefix {schema_prefix}: <https://schema.org/> .",
        f"@prefix {base_prefix}: <http://example.org/navelinfo#> ."
    ]

    # Statusanzeige: Knoten verarbeiten
    print("Verarbeite Knoten...")
    node_rows = _read_csv_rows(node_file)
    id2uri = {r.get('id'): _create_uri(r.get('id'), base_prefix, r.get('title')) for r in node_rows}
    for r in node_rows:
        s = (r.get('id') or '').strip()
        u = id2uri.get(s)
        if not u:
            continue

        # Extrahiert den Typ und weitere Labels
        main_type, additional_labels = _split_labels(r.get('labels'))
        if main_type:
            parts.append(f'{u} a {base_prefix}:{main_type} .')

        # Fügt zusätzliche Labels als rdfs:label hinzu
        for label in additional_labels:
            parts.append(f'{u} {rdfs_prefix}:label "{label}" .')

        for k, v in r.items():
            if k in ['id', 'label', 'type'] or not v:
                continue
            pred = f'{schema_prefix}:{k}'
            parts.append(f'{u} {pred} "{_clean_ttl_string(v)}" .')

    # Statusanzeige: Kanten verarbeiten
    print("Verarbeite Kanten...")
    edge_rows = _read_csv_rows(edge_file)
    for i, r in enumerate(edge_rows):
        s = (r.get('source') or r.get('src') or r.get('src_id') or r.get('from') or r.get('quelle') or '').strip()
        t = (r.get('target') or r.get('dst') or r.get('dst_id') or r.get('to')   or r.get('ziel')   or '').strip()
        p = (r.get('relation') or r.get('rel') or r.get('label') or r.get('kante') or 'relatedTo').strip()

        su = id2uri.get(s, f'{base_prefix}:n{s}')
        tu = id2uri.get(t, f'{base_prefix}:n{t}')
        pred = p if (':' in p or p.startswith('http')) else f'{base_prefix}:{_slug(p)}'

        parts.append(f'{su} {pred} {tu} .')

        for k, v in r.items():
            if k in ['source', 'target', 'src', 'dst', 'src_id', 'dst_id', 'from', 'to', 'relation', 'rel', 'label', 'kante'] or not v:
                continue
            pred = f'{schema_prefix}:{k}'

            if k == 'confidence' and re.match(r'^\d+(\.\d+)?$', v):
                parts.append(f'{su} {pred} {v} .')
            else:
                parts.append(f'{su} {pred} "{_clean_ttl_string(v)}" .')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))

    print("Vollständig abgeschlossen!")
