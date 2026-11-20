"""
RenalGuard AI — Aplikasi web Flask untuk prediksi penyakit ginjal kronis (CKD)
menggunakan model Random Forest yang dilatih di notebook.

Jalankan:  python app.py   ->  buka http://127.0.0.1:5000
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_ckd.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "app_metadata.json")
DATA_PATH  = os.path.join(BASE_DIR, "data",  "raw_CKD_dataset.csv")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model & metadata sekali saat startup
# ---------------------------------------------------------------------------
pipeline = joblib.load(MODEL_PATH)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

FEATURES      = meta["feature_names"]          # 35 fitur, urutan sesuai training
STATS         = meta["feature_stats"]
NUM_FEATURES  = meta["numeric_features"]
CAT_FEATURES  = meta["categorical_features"]
CLASS_NAMES   = meta["target_classes"]         # urutan dari model.classes_
TOP10         = meta["top10_feature_importance"]
ACCURACY      = meta["accuracy"]
N_ESTIMATORS  = meta["n_estimators"]
MAX_DEPTH     = meta["max_depth"]

# ---------------------------------------------------------------------------
# Deskripsi fitur untuk form
# ---------------------------------------------------------------------------
FEATURE_META = {
    # Demografi & Antropometri
    "Age":                    {"label": "Usia",                   "hint": "Tahun (20–84)",                       "unit": "tahun",  "group": "Demografi"},
    "Gender":                 {"label": "Jenis Kelamin",           "hint": "0 = Perempuan, 1 = Laki-laki",        "unit": "",       "group": "Demografi", "type": "numeric_int"},
    "BMI":                    {"label": "BMI",                     "hint": "Indeks Massa Tubuh",                  "unit": "kg/m²",  "group": "Demografi"},
    # Kardiovaskular
    "Systolic_BP":            {"label": "Tekanan Darah Sistolik",  "hint": "Tekanan darah atas",                  "unit": "mmHg",   "group": "Kardiovaskular"},
    "Diastolic_BP":           {"label": "Tekanan Darah Diastolik", "hint": "Tekanan darah bawah",                 "unit": "mmHg",   "group": "Kardiovaskular"},
    "Heart_Rate":             {"label": "Detak Jantung",           "hint": "Denyut per menit",                    "unit": "bpm",    "group": "Kardiovaskular"},
    # Fungsi Ginjal
    "Serum_Creatinine":       {"label": "Kreatinin Serum",         "hint": "Produk sisa metabolisme otot",        "unit": "mg/dL",  "group": "Fungsi Ginjal"},
    "Blood_Urea_Nitrogen":    {"label": "Blood Urea Nitrogen",     "hint": "Kadar urea nitrogen dalam darah",     "unit": "mg/dL",  "group": "Fungsi Ginjal"},
    "eGFR":                   {"label": "eGFR",                    "hint": "Laju filtrasi glomerulus estimasi",   "unit": "mL/min", "group": "Fungsi Ginjal"},
    "Urine_Albumin":          {"label": "Albumin Urin",            "hint": "Albumin dalam urin",                  "unit": "mg/L",   "group": "Fungsi Ginjal"},
    "Urine_Protein":          {"label": "Protein Urin",            "hint": "Protein total dalam urin",            "unit": "g/dL",   "group": "Fungsi Ginjal"},
    "Albumin_Creatinine_Ratio": {"label": "Rasio Albumin/Kreatinin","hint": "Indikator kerusakan ginjal dini",    "unit": "mg/g",   "group": "Fungsi Ginjal"},
    "Urine_Specific_Gravity": {"label": "Berat Jenis Urin",        "hint": "Konsentrasi partikel dalam urin",     "unit": "",       "group": "Fungsi Ginjal"},
    # Elektrolit
    "Sodium":                 {"label": "Natrium (Na)",            "hint": "Konsentrasi natrium serum",           "unit": "mEq/L",  "group": "Elektrolit"},
    "Potassium":              {"label": "Kalium (K)",              "hint": "Konsentrasi kalium serum",            "unit": "mEq/L",  "group": "Elektrolit"},
    "Calcium":                {"label": "Kalsium (Ca)",            "hint": "Konsentrasi kalsium serum",           "unit": "mg/dL",  "group": "Elektrolit"},
    "Phosphorus":             {"label": "Fosfor (P)",              "hint": "Konsentrasi fosfor serum",            "unit": "mg/dL",  "group": "Elektrolit"},
    "Chloride":               {"label": "Klorida (Cl)",            "hint": "Konsentrasi klorida serum",           "unit": "mEq/L",  "group": "Elektrolit"},
    "Bicarbonate":            {"label": "Bikarbonat",              "hint": "Cadangan alkali darah",               "unit": "mEq/L",  "group": "Elektrolit"},
    # Hematologi
    "Hemoglobin":             {"label": "Hemoglobin",              "hint": "Protein pembawa oksigen dalam darah", "unit": "g/dL",   "group": "Hematologi"},
    "RBC_Count":              {"label": "Jumlah Sel Darah Merah",  "hint": "Eritrosit per mm³",                   "unit": "juta/µL","group": "Hematologi"},
    "WBC_Count":              {"label": "Jumlah Sel Darah Putih",  "hint": "Leukosit per mm³",                    "unit": "sel/µL", "group": "Hematologi"},
    "Platelet_Count":         {"label": "Jumlah Trombosit",        "hint": "Trombosit per mm³",                   "unit": "ribu/µL","group": "Hematologi"},
    "Packed_Cell_Volume":     {"label": "Hematokrit (PCV)",        "hint": "Persentase sel darah merah",          "unit": "%",      "group": "Hematologi"},
    # Metabolik
    "Blood_Glucose_Random":   {"label": "Glukosa Darah Acak",      "hint": "Gula darah sewaktu",                  "unit": "mg/dL",  "group": "Metabolik"},
    "Fasting_Glucose":        {"label": "Glukosa Puasa",           "hint": "Gula darah setelah puasa",            "unit": "mg/dL",  "group": "Metabolik"},
    "HbA1c":                  {"label": "HbA1c",                   "hint": "Rata-rata gula darah 2–3 bulan",      "unit": "%",      "group": "Metabolik"},
    "Cholesterol":            {"label": "Kolesterol Total",         "hint": "Kolesterol total serum",              "unit": "mg/dL",  "group": "Metabolik"},
    "Triglycerides":          {"label": "Trigliserida",             "hint": "Lemak darah jenis trigliserida",      "unit": "mg/dL",  "group": "Metabolik"},
    "Serum_Albumin":          {"label": "Albumin Serum",            "hint": "Protein utama plasma darah",          "unit": "g/dL",   "group": "Metabolik"},
    "Total_Protein":          {"label": "Protein Total",            "hint": "Total protein plasma",                "unit": "g/dL",   "group": "Metabolik"},
    # Riwayat Kesehatan
    "Diabetes":               {"label": "Diabetes",                "hint": "Riwayat diabetes mellitus",           "unit": "",       "group": "Riwayat", "type": "categorical"},
    "Hypertension":           {"label": "Hipertensi",              "hint": "Riwayat tekanan darah tinggi",        "unit": "",       "group": "Riwayat", "type": "categorical"},
    "Smoking_Status":         {"label": "Status Merokok",          "hint": "Perokok aktif atau mantan",           "unit": "",       "group": "Riwayat", "type": "categorical"},
    "Family_History_Kidney":  {"label": "Riwayat Keluarga CKD",   "hint": "Anggota keluarga menderita CKD",      "unit": "",       "group": "Riwayat", "type": "categorical"},
}

# Urutan grup
GROUP_ORDER = ["Demografi", "Kardiovaskular", "Fungsi Ginjal", "Elektrolit",
               "Hematologi", "Metabolik", "Riwayat"]

# Warna per kelas CKD (untuk tampilan badge)
CLASS_COLORS = {
    "Healthy Kidney":           {"css": "healthy",  "icon": "fa-heart-pulse",         "id": 0},
    "Mild CKD (Stage 1–2)":     {"css": "mild",     "icon": "fa-circle-exclamation",  "id": 1},
    "Moderate CKD (Stage 3)":   {"css": "moderate", "icon": "fa-triangle-exclamation","id": 2},
    "Severe CKD (Stage 4)":     {"css": "severe",   "icon": "fa-skull-crossbones",    "id": 3},
    "Kidney Failure (Stage 5)": {"css": "critical", "icon": "fa-circle-xmark",        "id": 4},
}

# ---------------------------------------------------------------------------
# Build form groups
# ---------------------------------------------------------------------------
def build_form_groups():
    groups = {g: [] for g in GROUP_ORDER}
    for feat in FEATURES:
        fm = FEATURE_META.get(feat, {})
        g  = fm.get("group", "Lainnya")
        st = STATS[feat]
        item = {
            "name":  feat,
            "label": fm.get("label", feat),
            "hint":  fm.get("hint", ""),
            "unit":  fm.get("unit", ""),
            "ftype": st.get("type", "numeric"),
        }
        if item["ftype"] == "numeric":
            item.update({"min": st["min"], "max": st["max"], "median": st["median"]})
        else:
            item["choices"] = st["unique"]
        groups.setdefault(g, []).append(item)
    return [(g, groups[g]) for g in GROUP_ORDER if groups.get(g)]

FORM_GROUPS = build_form_groups()


def load_examples():
    try:
        ex = meta.get("examples", {})
        # Mapping class name -> slug untuk key
        slug_map = {
            "Healthy Kidney":           "sehat",
            "Mild CKD (Stage 1–2)":     "ringan",
            "Moderate CKD (Stage 3)":   "sedang",
            "Severe CKD (Stage 4)":     "berat",
            "Kidney Failure (Stage 5)": "gagal",
        }
        return {slug_map[k]: v for k, v in ex.items() if k in slug_map}
    except Exception:
        return {}


EXAMPLES = load_examples()


def importance_list():
    items = []
    for name, val in TOP10.items():
        fm = FEATURE_META.get(name, {})
        items.append({
            "name":  fm.get("label", name),
            "raw":   name,
            "value": round(val * 100, 1),
        })
    return sorted(items, key=lambda x: x["value"], reverse=True)

IMPORTANCE = importance_list()


# ---------------------------------------------------------------------------
# Helper: parse inputs
# ---------------------------------------------------------------------------
def parse_inputs(source):
    values, errors = {}, []
    for feat in FEATURES:
        st = STATS[feat]
        raw = source.get(feat)
        if raw is None or str(raw).strip() == "":
            errors.append(f"'{FEATURE_META.get(feat,{}).get('label', feat)}' belum diisi.")
            continue
        if st["type"] == "categorical":
            choices = st["unique"]
            if str(raw).strip() not in choices:
                errors.append(f"'{FEATURE_META.get(feat,{}).get('label', feat)}' harus salah satu dari {choices}.")
                continue
            values[feat] = str(raw).strip()
        else:
            try:
                val = float(str(raw).replace(",", "."))
            except ValueError:
                errors.append(f"'{FEATURE_META.get(feat,{}).get('label', feat)}' harus berupa angka.")
                continue
            values[feat] = val
    return values, errors


def out_of_range(values):
    warns = []
    for feat, v in values.items():
        st = STATS[feat]
        if st["type"] != "numeric":
            continue
        if v < st["min"] or v > st["max"]:
            warns.append(
                f"{FEATURE_META.get(feat,{}).get('label', feat)} = {v:g} "
                f"(rentang data latih {st['min']:g} – {st['max']:g})"
            )
    return warns


def predict(values):
    df_in = pd.DataFrame([values], columns=FEATURES)
    kelas  = str(pipeline.predict(df_in)[0])
    proba  = pipeline.predict_proba(df_in)[0]
    prob_dict = {c: round(float(p) * 100, 1) for c, p in zip(CLASS_NAMES, proba)}
    cc = CLASS_COLORS.get(kelas, {"css": "mild", "icon": "fa-circle-exclamation", "id": 0})
    return {
        "label":     kelas,
        "css":       cc["css"],
        "icon":      cc["icon"],
        "probs":     prob_dict,
        "top_prob":  round(max(prob_dict.values()), 1),
        "warnings":  out_of_range(values),
    }


def page_context(**extra):
    ctx = dict(
        form_groups=FORM_GROUPS,
        examples=EXAMPLES,
        importance=IMPORTANCE,
        accuracy=ACCURACY,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        result=None,
        errors=[],
        form={},
    )
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **page_context())


@app.route("/predict", methods=["POST"])
def predict_form():
    values, errors = parse_inputs(request.form)
    if errors:
        return render_template("index.html", **page_context(errors=errors, form=request.form)), 400
    return render_template("index.html", **page_context(result=predict(values), form=request.form))


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """Endpoint JSON. Body: {fitur: nilai, ...} (35 fitur)."""
    payload = request.get_json(silent=True) or {}
    values, errors = parse_inputs(payload)
    if errors:
        return jsonify({"error": errors}), 400
    hasil = predict(values)
    return jsonify({
        "prediction": hasil["label"],
        "probabilities": hasil["probs"],
        "warnings": hasil["warnings"],
    })


if __name__ == "__main__":
    app.run(debug=True)
