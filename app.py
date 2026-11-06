"""MediGuard v2 - hybrid AI system for medicine expiry intelligence & safe disposal.
pip install streamlit google-genai pandas scikit-learn plotly
python -m streamlit run app.py"""
import calendar, html, json, os, re, time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="MediGuard", page_icon="💊", layout="wide")
CSV = "inventory.csv"
COLS = ["name", "active_ingredient", "form", "category", "ml_category", "ml_conf", "batch_no", "expiry",
        "days_left", "status", "flags", "source"]
CATEGORIES = ["antibiotic", "controlled", "sharps", "liquid", "tablet_capsule", "topical", "inhaler", "other"]
ICONS = dict(zip(CATEGORIES, ["🦠", "🔒", "💉", "🧴", "💊", "🧪", "🌬️", "📦"]))
RANK = dict(antibiotic=3, controlled=3, sharps=3, inhaler=2, liquid=2, tablet_capsule=2, other=2, topical=1)
COLORS = {"Expired": "#ef4444", "Expiring <=30d": "#f97316", "Expiring <=90d": "#eab308", "OK": "#22c55e", "Unknown": "#94a3b8"}
RISKC = {3: "#ef4444", 2: "#f59e0b", 1: "#22c55e"}
RISKN = {3: "High", 2: "Medium", 1: "Low"}
DISPOSAL = {
    "antibiotic": "Never flush or bin loose - residue in water drives antibiotic resistance. Return to a pharmacy/hospital collection point; if none, seal with used tea leaves or sawdust in a pouch and put in general waste.",
    "controlled": "Do not throw or flush. Hand over to a pharmacist, hospital or doctor for supervised disposal.",
    "sharps": "Put needles/syringes in a puncture-proof container (hard plastic bottle with lid) and give to a hospital or biomedical-waste collector. Never in household bins.",
    "liquid": "Do not pour down the drain. Mix with sand or used coffee grounds in a sealed container, then general waste. Rinse and recycle the clean bottle.",
    "tablet_capsule": "Remove from the strip, mix with an unappealing substance (tea leaves, soil), seal in a bag, then general waste. Recycle the empty carton as paper.",
    "topical": "Seal the tube/jar and put in general waste. Empty, clean tubes can go with plastic recycling.",
    "inhaler": "Pressurised canisters must not be burned or crushed. Return to a pharmacy or hazardous-waste drop-off.",
    "other": "Ask your pharmacist for the safest route. When unsure, do not flush.",
}
TRAIN = {  # small labelled name+ingredient+form dataset for the category classifier
    "antibiotic": ["augmentin amoxicillin clavulanate tablet", "azithral azithromycin tablet", "cifran ciprofloxacin tablet", "taxim-o cefixime tablet", "doxycycline capsule", "metrogyl metronidazole tablet", "levoflox levofloxacin tablet", "mox amoxicillin capsule"],
    "controlled": ["alprazolam tablet", "tramadol tablet", "clonazepam tablet", "diazepam tablet", "zolpidem tablet", "codeine phosphate syrup", "morphine sulphate", "lorazepam tablet"],
    "sharps": ["insulin pen injection", "lovenox enoxaparin injection syringe", "heparin injection vial", "insulin glargine cartridge", "diclofenac injection ampoule", "vitamin b12 injection", "lancet needle", "prefilled syringe"],
    "liquid": ["benadryl cough syrup", "crocin paracetamol suspension", "ascoril expectorant syrup", "zincovit drops", "ors oral rehydration solution", "cyclopam suspension", "digene antacid liquid", "calpol suspension"],
    "topical": ["betadine povidone iodine ointment", "volini pain relief gel", "soframycin framycetin cream", "candid clotrimazole cream", "moov pain relief cream", "burnol cream", "iodex ointment", "fucidin fusidic acid cream"],
    "inhaler": ["asthalin salbutamol inhaler", "foracort formoterol budesonide inhaler", "seroflo salmeterol fluticasone inhaler", "duolin ipratropium inhaler", "budecort respules inhaler", "levolin levosalbutamol inhaler", "tiotropium rotacap inhaler", "pulmicort inhaler"],
    "tablet_capsule": ["dolo 650 paracetamol tablet", "cetirizine tablet", "pantoprazole tablet", "metformin tablet", "atorvastatin tablet", "amlodipine tablet", "becosules vitamin b capsule", "shelcal calcium tablet"],
    "other": ["nasal spray", "eye drops", "ayurvedic churna powder", "glucose powder", "hand sanitizer", "herbal tonic", "bandage dressing", "thermometer strip"],
}
PROMPT = f"""You read medicine packaging (strips, boxes, bottles, tubes). Return ONLY JSON with keys:
name, active_ingredient, form, category (one of {CATEGORIES}), batch_no, mfg_date, expiry_date, confidence (0-1).
Dates as YYYY-MM or YYYY-MM-DD. Use null for anything unreadable. Never guess. 'EXP 08/27' means 2027-08."""

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.block-container{padding-top:1.5rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0f766e,#4338ca);border-radius:20px;padding:26px 32px;color:#fff;margin-bottom:16px}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff;padding:0}.hero p{margin:6px 0 0;opacity:.92}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:6px 0 14px}
.kpi{background:#fff;border-radius:16px;padding:14px 16px;box-shadow:0 2px 10px rgba(0,0,0,.08);border-top:5px solid var(--c)}
.kpi b{display:block;font-size:1.8rem;color:#0f172a}.kpi span{color:#64748b;font-size:.8rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:8px}
.card{background:#fff;border-radius:16px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.08);border-left:6px solid var(--c);color:#0f172a}
.card h4{margin:8px 0 2px;font-size:1.05rem;color:#0f172a;padding:0}.muted{color:#64748b;font-size:.82rem}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;font-weight:600;color:#fff;background:var(--c)}
.bar{height:6px;background:#e2e8f0;border-radius:9px;margin-top:10px;overflow:hidden}.bar i{display:block;height:100%;background:var(--c)}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 16px}
.step{background:#eef2ff;color:#3730a3;border-radius:12px;padding:8px 12px;font-weight:600;font-size:.85rem}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:1rem}.stButton>button{border-radius:12px;font-weight:600}
@media(max-width:800px){.kpis{grid-template-columns:repeat(2,1fr)}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def train():
    X = [t for v in TRAIN.values() for t in v]
    y = [k for k, v in TRAIN.items() for _ in v]
    mk = lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), LogisticRegression(max_iter=2000, C=10))
    pred = cross_val_predict(mk(), X, y, cv=StratifiedKFold(4, shuffle=True, random_state=7))
    return mk().fit(X, y), accuracy_score(y, pred), confusion_matrix(y, pred, labels=CATEGORIES), len(X)


MODEL, CV_ACC, CM, N_TRAIN = train()


def classify(text, llm_cat):
    """Safety-first ensemble: ML classifier + vision-LLM label. On disagreement keep the higher-risk class."""
    p = MODEL.predict_proba([text.lower()])[0]
    probs = dict(zip(MODEL.classes_, p))
    ml, conf = max(probs, key=probs.get), float(p.max())
    if llm_cat not in CATEGORIES or llm_cat == ml:
        return ml if llm_cat not in CATEGORIES else llm_cat, ml, conf, probs, False
    if RANK[llm_cat] != RANK[ml]:
        return (llm_cat if RANK[llm_cat] > RANK[ml] else ml), ml, conf, probs, True
    return (ml if conf >= 0.5 else llm_cat), ml, conf, probs, True


def parse_date(s):
    m = re.match(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", str(s or ""))
    if not m:
        return pd.NaT
    y, mo = int(m[1]), int(m[2])
    try:  # month-only dates expire on the last day of the month
        return pd.Timestamp(y, mo, int(m[3]) if m[3] else calendar.monthrange(y, mo)[1])
    except ValueError:
        return pd.NaT


def validate(d, exp):
    fl, mfg, c = [], parse_date(d.get("mfg_date")), d.get("confidence")
    if pd.isna(exp):
        fl.append("expiry unreadable")
    elif not 2015 <= exp.year <= date.today().year + 10:
        fl.append("implausible expiry year")
    if not pd.isna(mfg) and not pd.isna(exp) and mfg > exp:
        fl.append("mfg after expiry")
    if isinstance(c, (int, float)) and c < 0.6:
        fl.append("low OCR confidence")
    return fl


def refresh(df):
    exp = pd.to_datetime(df["expiry"], errors="coerce")
    left = (exp - pd.Timestamp.today().normalize()).dt.days
    df["expiry"], df["days_left"] = exp, left
    df["status"] = ["Unknown" if pd.isna(d) else "Expired" if d < 0 else "Expiring <=30d" if d <= 30
                    else "Expiring <=90d" if d <= 90 else "OK" for d in left]
    return df


def scan(client, model, f):
    last = None
    for m in [model, "gemini-flash-latest", "gemini-flash-lite-latest"]:
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=m, contents=[types.Part.from_bytes(data=f.getvalue(), mime_type=f.type or "image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                d = json.loads(r.text)
                return d[0] if isinstance(d, list) else d
            except Exception as e:
                last = e
                if "503" in str(e) or "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                else:
                    break
    raise last


def analyse(client, model, f):
    d = scan(client, model, f)
    exp = parse_date(d.get("expiry_date"))
    text = f'{d.get("name") or ""} {d.get("active_ingredient") or ""} {d.get("form") or ""}'
    cat, ml, conf, probs, clash = classify(text, d.get("category"))
    fl = validate(d, exp) + (["ML and vision-LLM category differ - verify"] if clash else [])
    row = dict(name=d.get("name"), active_ingredient=d.get("active_ingredient"), form=d.get("form"), category=cat,
               ml_category=ml, ml_conf=round(conf, 2), batch_no=d.get("batch_no"), expiry=exp, flags="; ".join(fl), source="scan")
    return row, dict(name=d.get("name"), raw=d, probs=probs, ml=ml, conf=conf, llm=d.get("category"), final=cat)


def demo():
    t = date.today()
    items = [("Augmentin 625", "amoxicillin clavulanate", "tablet", "antibiotic", -20), ("Dolo 650", "paracetamol", "tablet", "tablet_capsule", 25),
             ("Asthalin", "salbutamol", "inhaler", "inhaler", 60), ("Benadryl", "diphenhydramine", "syrup", "liquid", -90),
             ("Betadine", "povidone iodine", "ointment", "topical", 200), ("Alprazolam 0.25", "alprazolam", "tablet", "controlled", 15),
             ("Cetirizine", "cetirizine", "tablet", "tablet_capsule", 400), ("Insulin pen", "insulin glargine", "injection", "sharps", -5)]
    rows = []
    for i, (n, a, f, c, dd) in enumerate(items):
        cat, ml, conf, _, _ = classify(f"{n} {a} {f}", c)
        rows.append(dict(name=n, active_ingredient=a, form=f, category=cat, ml_category=ml, ml_conf=round(conf, 2),
                         batch_no=f"B{100 + i}", expiry=pd.Timestamp(t + timedelta(days=dd)), flags="", source="demo"))
    return refresh(pd.DataFrame(rows).reindex(columns=COLS))


def make_ics(df):
    L = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MediGuard//EN"]
    for _, r in df.dropna(subset=["expiry"]).iterrows():
        for lab, dt in (("expires", r["expiry"]), ("expires in 30 days", r["expiry"] - pd.Timedelta(days=30))):
            if dt.date() >= date.today():
                L += ["BEGIN:VEVENT", f"UID:{abs(hash((str(r['name']), str(dt), lab)))}@mediguard", f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                      f"DTSTART;VALUE=DATE:{dt:%Y%m%d}", f"SUMMARY:{r['name']} {lab}", "END:VEVENT"]
    return "\r\n".join(L + ["END:VCALENDAR"])


def esc(x):
    return html.escape("-" if pd.isna(x) else str(x))


def card(r):
    c, d = COLORS[r["status"]], r["days_left"]
    when = "Expiry unknown" if pd.isna(d) else f"{abs(int(d))} days ago" if d < 0 else f"{int(d)} days left"
    pct = 0 if pd.isna(d) else max(0, min(100, d / 365 * 100))
    exp = "-" if pd.isna(r["expiry"]) else r["expiry"].strftime("%d %b %Y")
    fl = "" if pd.isna(r["flags"]) or not r["flags"] else f'<div class="muted" style="color:#b45309;margin-top:4px">⚠ {esc(r["flags"])}</div>'
    return (f'<div class="card" style="--c:{c}"><span class="pill">{r["status"]}</span>'
            f'<h4>{ICONS.get(r["category"], "📦")} {esc(r["name"])}</h4>'
            f'<div class="muted">{esc(r["active_ingredient"])} · {esc(r["form"])}</div>'
            f'<div style="margin-top:8px"><b>{exp}</b> <span class="muted">· {when}</span></div>'
            f'<div class="bar"><i style="width:{pct}%"></i></div>'
            f'<div class="muted" style="margin-top:6px">Batch {esc(r["batch_no"])} · class: {r["category"]}</div>{fl}</div>')


def grid(df):
    st.markdown('<div class="grid">' + "".join(card(r) for _, r in df.iterrows()) + "</div>", unsafe_allow_html=True)


def fig_style(f, h=320):
    f.update_layout(margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", height=h, template="plotly_white")
    return f


def save():
    st.session_state.inv.to_csv(CSV, index=False)


if "inv" not in st.session_state:
    st.session_state.inv = refresh(pd.read_csv(CSV).reindex(columns=COLS)) if os.path.exists(CSV) else pd.DataFrame(columns=COLS)
    st.session_state.last, st.session_state.trace = pd.DataFrame(columns=COLS), []

with st.sidebar:
    st.title("💊 MediGuard")
    key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    model = st.text_input("Model", value="gemini-3.8-flash")
    if st.button("📥 Load demo data", use_container_width=True):
        st.session_state.inv = pd.concat([st.session_state.inv, demo()], ignore_index=True); save(); st.rerun()
    if st.button("🗑️ Clear inventory", use_container_width=True):
        st.session_state.inv = pd.DataFrame(columns=COLS); save(); st.rerun()
    st.caption("Informational only. Not medical advice - never judge whether a medicine is safe to use from this app.")

def hero_style():
    import base64
    for ext, mime in (("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png")):
        if os.path.exists(f"banner.{ext}"):
            b64 = base64.b64encode(open(f"banner.{ext}", "rb").read()).decode()
            return ("min-height:180px;background:linear-gradient(120deg,rgba(15,118,110,.8),rgba(67,56,202,.8)),"
                    f"url(data:image/{mime};base64,{b64}) center/cover;")
    return ""
st.markdown(f'<div class="hero" style="{hero_style()}"><h1>💊 MediGuard</h1><p>Hybrid AI system for medicine expiry intelligence and safe disposal</p></div>', unsafe_allow_html=True)
tabs = st.tabs(["📊 Dashboard", "📷 Scan", "📦 Inventory", "♻️ Disposal", "🧠 ML Insights", "💬 Assistant"])
inv = st.session_state.inv

with tabs[0]:
    n = lambda s: int((inv.status == s).sum())
    flagged = int(inv["flags"].fillna("").ne("").sum())
    hi = int(((inv.status == "Expired") & (inv.category.map(RANK) == 3)).sum())
    kp = [("Total medicines", len(inv), "#4338ca"), ("Expired", n("Expired"), COLORS["Expired"]), ("Expiring in 30 days", n("Expiring <=30d"), COLORS["Expiring <=30d"]),
          ("Expired high-risk (take-back)", hi, "#be123c"), ("Need manual review", flagged, "#0ea5e9")]
    st.markdown('<div class="kpis">' + "".join(f'<div class="kpi" style="--c:{c}"><b>{v}</b><span>{t}</span></div>' for t, v, c in kp) + "</div>", unsafe_allow_html=True)
    if inv.empty:
        st.info("No data yet. Scan a medicine, or click **Load demo data** in the sidebar.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(fig_style(px.pie(inv, names="status", hole=.55, color="status", color_discrete_map=COLORS, title="Expiry status")), use_container_width=True)
        b.plotly_chart(fig_style(px.histogram(inv, x="category", color="status", color_discrete_map=COLORS, title="Medicines by disposal class")), use_container_width=True)
        t = inv.dropna(subset=["expiry"])
        if not t.empty:
            f = px.scatter(t, x="expiry", y="name", color="status", color_discrete_map=COLORS, title="Expiry timeline")
            f.update_traces(marker=dict(size=14))
            f.add_shape(type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1, yref="paper", line=dict(dash="dash", color="#475569"))
            st.plotly_chart(fig_style(f, 340), use_container_width=True)
        st.markdown("#### 🚨 Action needed")
        act = inv[inv.status.isin(["Expired", "Expiring <=30d"])]
        grid(act.sort_values("days_left")) if len(act) else st.success("Nothing expiring in the next 30 days.")

with tabs[1]:
    l, r = st.columns(2, gap="large")
    with l:
        src = st.radio("Source", ["Upload photos", "Camera"], horizontal=True)
        files = st.file_uploader("Medicine photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True) if src == "Upload photos" \
            else [x for x in [st.camera_input("Take a photo")] if x]
        if files and st.button("🔍 Analyse", type="primary", use_container_width=True):
            if not key:
                st.error("Add your Gemini API key in the sidebar.")
            else:
                client, rows, traces = genai.Client(api_key=key), [], []
                for f in files:
                    with st.spinner(f"Reading {f.name}..."):
                        try:
                            row, tr = analyse(client, model, f)
                            rows.append(row); traces.append(tr)
                        except Exception as e:
                            st.warning(f"{f.name}: could not read ({e})")
                if rows:
                    new = refresh(pd.DataFrame(rows).reindex(columns=COLS))
                    st.session_state.last, st.session_state.trace = new, traces
                    st.session_state.inv = pd.concat([st.session_state.inv, new], ignore_index=True); save()
    with r:
        st.markdown("##### Results")
        if st.session_state.last.empty:
            st.markdown('<div class="muted">Results and the pipeline trace appear here.</div>', unsafe_allow_html=True)
        else:
            grid(st.session_state.last)
            for tr in st.session_state.trace:
                with st.expander(f"🔬 Pipeline trace - {tr['name']}"):
                    st.caption(f"Vision LLM: **{tr['llm']}** · ML classifier: **{tr['ml']}** ({tr['conf']:.0%}) · Final (safety-first): **{tr['final']}**")
                    pr = pd.Series(tr["probs"]).sort_values()
                    st.plotly_chart(fig_style(px.bar(x=pr.values, y=pr.index, orientation="h", labels={"x": "probability", "y": ""}, title="Classifier probabilities"), 260), use_container_width=True)
                    st.json(tr["raw"], expanded=False)

with tabs[2]:
    if inv.empty:
        st.info("Nothing here yet.")
    else:
        show = st.selectbox("Filter", ["All"] + list(COLORS))
        view = inv if show == "All" else inv[inv.status == show]
        grid(view.assign(p=view.status.map({"Expired": 0, "Expiring <=30d": 1, "Expiring <=90d": 2, "OK": 3, "Unknown": 2}) * -10 + (-view.category.map(RANK).fillna(2)))
             .sort_values(["p", "days_left"]).drop(columns="p"))
        with st.expander("✏️ Edit data / fix OCR mistakes"):
            with st.form("edit"):
                ed = st.data_editor(inv, num_rows="dynamic", use_container_width=True, disabled=["days_left", "status"],
                                    column_config={"expiry": st.column_config.DateColumn("expiry"), "category": st.column_config.SelectboxColumn(options=CATEGORIES)})
                if st.form_submit_button("💾 Save changes"):
                    st.session_state.inv = refresh(ed); save(); st.rerun()
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("⬇️ CSV", inv.to_csv(index=False), "medicines.csv", use_container_width=True)
        c2.download_button("📅 Calendar reminders (.ics)", make_ics(inv), "expiry_reminders.ics", use_container_width=True)

with tabs[3]:
    exp = inv[inv.status == "Expired"]
    if exp.empty:
        st.info("No expired medicines.")
    else:
        cards = ""
        for _, r in exp.sort_values("category", key=lambda s: -s.map(RANK)).iterrows():
            k = RANK.get(r["category"], 2)
            cards += (f'<div class="card" style="--c:{RISKC[k]}"><span class="pill">{RISKN[k]} risk</span><h4>{ICONS.get(r["category"], "📦")} {esc(r["name"])}</h4>'
                      f'<div class="muted">Class: {r["category"]}</div><div style="margin-top:6px;font-size:.9rem">{DISPOSAL.get(r["category"], DISPOSAL["other"])}</div></div>')
        st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
    st.caption("Sealed, unexpired medicines can sometimes be donated through NGOs or hospital programmes - ask your pharmacist.")

with tabs[4]:
    st.markdown("#### System pipeline")
    steps = ["📷 Image", "🧠 Vision-LLM extraction", "✅ Rule-based validation", "🤖 TF-IDF + LogReg classifier", "⚖️ Safety-first ensemble", "♻️ Disposal engine", "📅 Alerts & analytics"]
    st.markdown('<div class="flow">' + " → ".join(f'<span class="step">{s}</span>' for s in steps) + "</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Classifier CV accuracy", f"{CV_ACC:.0%}"); m2.metric("Training samples", N_TRAIN); m3.metric("Classes", len(CATEGORIES))
    st.caption("4-fold stratified cross-validation on a small hand-labelled dataset (character n-gram TF-IDF + logistic regression). Treat accuracy as indicative; scaling the dataset is future work.")
    st.plotly_chart(fig_style(px.imshow(CM, x=CATEGORIES, y=CATEGORIES, text_auto=True, color_continuous_scale="Teal", labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (cross-validated)"), 460), use_container_width=True)
    q = st.text_input("Try the classifier", placeholder="e.g. cefpodoxime tablet / volini spray / insulin cartridge")
    if q:
        pr = pd.Series(dict(zip(MODEL.classes_, MODEL.predict_proba([q.lower()])[0]))).sort_values()
        st.plotly_chart(fig_style(px.bar(x=pr.values, y=pr.index, orientation="h", labels={"x": "probability", "y": ""}, title=f"Prediction: {pr.index[-1]}"), 300), use_container_width=True)


def chat_reply(client, model, history, system):
    while history and history[0]["role"] != "user":
        history = history[1:]
    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])]) for m in history]
    last = None
    for mdl in [model, "gemini-flash-latest", "gemini-flash-lite-latest"]:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=mdl, contents=contents,
                                                      config=types.GenerateContentConfig(system_instruction=system, temperature=0.3)).text
            except Exception as e:
                last = e
                if "503" in str(e) or "429" in str(e):
                    time.sleep(2)
                else:
                    break
    return f"Sorry, the AI is busy right now ({last}). Please try again in a minute."


with tabs[5]:
    st.markdown("#### 💬 MediGuard Assistant")
    st.caption("Ask about your medicines: what expires soon, what to take to the pharmacy first, how to dispose of something. Answers are grounded in your inventory.")
    ctx = inv[["name", "category", "expiry", "days_left", "status"]].assign(expiry=lambda d: d.expiry.dt.strftime("%Y-%m-%d")).to_csv(index=False) if len(inv) else "(inventory is empty)"
    SYSTEM = f"""You are MediGuard Assistant inside a medicine expiry and safe-disposal app. Today is {date.today()}.
User inventory (CSV):
{ctx}
Disposal guidance by class: {json.dumps(DISPOSAL)}
Rules: answer briefly and practically using the inventory and the disposal guidance. Prioritise high-risk classes (antibiotic, controlled, sharps).
NEVER say an expired medicine is safe to use. Never give dosing or diagnosis; for questions about taking a medicine, tell the user to ask a pharmacist or doctor.
If the question is outside medicine storage, expiry or disposal, politely redirect. If the inventory lacks the needed info, say so."""
    if "chat" not in st.session_state:
        st.session_state.chat = []
    qs = ["What expires in the next 30 days?", "Which expired items are high-risk?", "How do I dispose of my expired medicines?"]
    clicks = [c.button(q, use_container_width=True) for c, q in zip(st.columns(len(qs)), qs)]
    pending = next((q for q, k in zip(qs, clicks) if k), None)
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt = st.chat_input("Ask about your medicines...") or pending
    if prompt:
        if not key:
            st.error("Add your Gemini API key in the sidebar.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans = chat_reply(genai.Client(api_key=key), model, st.session_state.chat[-12:], SYSTEM)
                st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})
    if st.session_state.chat and st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.rerun()
