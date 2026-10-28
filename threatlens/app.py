import html, json, time
import pandas as pd, streamlit as st
import threatlens as ps

st.set_page_config(page_title="ThreatLens", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root{--bg:#0b0b14;--pn:rgba(255,255,255,.035);--ln:rgba(255,255,255,.09);--tx:#f1f1f8;--mu:#8b8ba3;--ac:#7c6cf0;--ac2:#6d5dfc;--gr:#34d399;--am:#f5b13d;--rd:#f26b6b}
.stApp{background:radial-gradient(700px 420px at 12% 88%,rgba(109,93,252,.22),transparent 60%),radial-gradient(800px 400px at 80% 0,rgba(109,93,252,.14),transparent 60%),#0b0b14}
header[data-testid="stHeader"]{background:transparent}.block-container{padding:.9rem 1.6rem 2rem;max-width:1450px}
section[data-testid="stSidebar"]{width:64px!important;min-width:64px!important;background:#0d0d18;border-right:1px solid var(--ln)}
section[data-testid="stSidebar"]>div{width:64px!important;padding:14px 6px}[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarHeader"]{display:none}
.logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#6d5dfc,#8b7cf8);display:grid;place-items:center;font-weight:700;margin:0 auto 18px}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px}[data-testid="stSidebar"] div[role="radiogroup"] label{justify-content:center;border-radius:9px;padding:8px 0;width:100%;color:var(--mu);font-size:18px}
[data-testid="stSidebar"] div[role="radiogroup"] label>div:first-child{display:none}[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:rgba(124,108,240,.2);color:#fff}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--pn);border:1px solid var(--ln)!important;border-radius:18px;backdrop-filter:blur(8px)}
.top{font-weight:600;font-size:15px;padding-top:6px}
.pill{display:inline-block;padding:5px 14px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--gr);background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.35)}
.av{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#6d5dfc,#8b7cf8);display:grid;place-items:center;font-size:11px;font-weight:700}
[data-testid="stTextInput"] input,[data-baseweb="select"]>div{background:#0f0f1b!important;border:1px solid var(--ln)!important;border-radius:99px!important;color:var(--tx)}
.eye{font-size:10.5px;letter-spacing:.06em;color:var(--mu);font-weight:600}
.hero h2{font-size:25px;margin:6px 0 8px;font-weight:700;line-height:1.2}.hero p{color:#b9b9cf;font-size:13.5px;max-width:560px;line-height:1.5}
.stButton>button{background:linear-gradient(135deg,#6d5dfc,#8b7cf8);color:#fff;border:0;border-radius:99px;font-weight:600;padding:.45rem 1.4rem}
.big{font-size:26px;font-weight:700;margin:4px 0}.sub{font-size:11.5px;color:var(--mu)}.g{color:var(--gr)}
.prog{height:6px;border-radius:4px;background:rgba(255,255,255,.1);margin:12px 0 8px}.prog i{display:block;height:100%;border-radius:4px;background:linear-gradient(90deg,#6d5dfc,#8b7cf8)}
.ct{font-weight:600;font-size:14px;margin-bottom:6px}
.chat{min-height:250px;display:flex;flex-direction:column;gap:10px;font-size:12.5px}
.b1{align-self:flex-start;max-width:88%;background:rgba(255,255,255,.06);border:1px solid var(--ln);border-radius:12px;padding:9px 12px}
.b2{align-self:flex-end;max-width:92%;background:rgba(109,93,252,.18);border:1px solid rgba(124,108,240,.4);border-radius:12px;padding:9px 12px}
.b2 b{display:block;margin-bottom:3px;font-size:11.5px}
.auto{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;background:rgba(255,255,255,.04);border:1px solid var(--ln);border-radius:12px;margin-bottom:8px;font-size:12.5px}
.dot{width:8px;height:8px;border-radius:50%}
.tb{width:100%;border-collapse:collapse;font-size:12.5px}.tb th{color:var(--mu);font-weight:500;text-align:left;padding:6px}.tb td{padding:7px 6px;border-top:1px solid var(--ln)}
.ok{color:var(--gr)}.wn{color:var(--am)}.er{color:var(--rd)}.inf{color:#8b9bff}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_models(): return ps.load_models()
@st.cache_data
def get_results():
    p = ps.ROOT / "outputs/results.json"
    if not p.exists(): ps.run_pipeline(log=False)
    return json.loads(p.read_text())
models, R = get_models(), get_results()
ss = st.session_state
for k, v in dict(events=[], last=None, model="TF-IDF + LogReg", cmd="", n=0).items(): ss.setdefault(k, v)

SAMPLES = {"AT-01 · Legitimate password reset": "Hello, we received a request to reset your password. If you did not request this, ignore this message. Visit the portal directly from your bookmarks. No action is needed otherwise.",
 "AT-02 · Phishing, unfamiliar wording": "Hi, the finance side flagged a hiccup with your pay details; please tidy it up here: hxxp://bit[.]ly/58 Thanks, really appreciate it.",
 "Obvious phishing": "URGENT!!! Your account will be suspended within 24 hours. Verify your password immediately at hxxp://192[.]0[.]2[.]9/login. Dear customer, final notice!",
 "Legitimate urgent payroll notice": "Reminder: payroll closes Friday. Confirm any bank detail changes in the HR system; do not click links in email."}

def scan(t):
    t = (t or "").strip()
    if not t: return
    mdl = models[ss.model]; p, d = ps.score_message(t, "email", mdl); cues, terms = ps.explain(t, models["TF-IDF + LogReg"]); ss.n += 1
    ss.last = dict(text=t, p=p, d=d, cues=cues, terms=terms, model=ss.model)
    ss.events.insert(0, dict(ts=time.strftime("%H:%M:%S"), id=f"M-{ss.n:03d}", ty={1: "ERR", 0: "INF", -1: "WRN"}[d], msg=f"{ps.LABEL[d]} ({p:.0%}) via {ss.model}: {t[:40]}..."))
def on_cmd(): scan(ss.cmd); ss.cmd = ""
def on_pick():
    if ss.pick: scan(SAMPLES[ss.pick])
def run_tests():
    for t in list(SAMPLES.values())[:3]: scan(t)

def smooth(p):
    d = f"M{p[0][0]:.1f},{p[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(p, p[1:]): m = (x0 + x1) / 2; d += f" C{m:.1f},{y0:.1f} {m:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    return d
def line_chart(labels, series, ymax, w=700, h=230):
    L, T, B = 20, 12, 26; pw, ph = w - 2 * L, h - T - B; xs = [L + pw * i / (len(labels) - 1) for i in range(len(labels))]; g = ""
    for x, lb in zip(xs, labels): g += f'<text x="{x}" y="{h-8}" text-anchor="middle" fill="#8b8ba3" font-size="10">{lb}</text>'
    for vals, c, dash in series:
        pts = [(x, T + ph * (1 - v / ymax)) for x, v in zip(xs, vals)]
        g += f'<path d="{smooth(pts)}" fill="none" stroke="{c}" stroke-width="2.8" stroke-linecap="round"' + (' stroke-dasharray="5 5"' if dash else "") + "/>"
        if not dash: g += "".join(f'<circle cx="{x}" cy="{y}" r="2.6" fill="{c}"/>' for x, y in pts)
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{g}</svg>'
def spark(vals, w=210, h=46):
    mx = max(vals) or 1; xs = [w * i / (len(vals) - 1) for i in range(len(vals))]
    return f'<svg viewBox="0 0 {w} {h}" width="100%"><path d="{smooth([(x, h-4-(h-8)*v/mx) for x, v in zip(xs, vals)])}" fill="none" stroke="#34d399" stroke-width="2.2"/></svg>'
def table(head, rows): return '<table class="tb"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
tyc = {"INF": "inf", "WRN": "wn", "ERR": "er"}
def seed_events():
    t = time.strftime("%H:%M:%S", time.localtime((ps.ROOT / "outputs/results.json").stat().st_mtime)); d = R["dedup"]
    ev = [dict(ts=t, id="PIPE", ty="INF", msg=f"Dedup removed {d['removed']} near-duplicates ({d['before']} to {d['after']})"), dict(ts=t, id="EVAL", ty="INF", msg=f"Evaluated {len(R['split']['holdout_campaigns'])} unseen campaigns, {R['split']['holdout']} messages")]
    for k, v in R["models"].items():
        if v["macro_f1"] < .5: ev.append(dict(ts=t, id="MODEL", ty="ERR", msg=f"{k} macro-F1 {v['macro_f1']:.2f} on unseen campaigns"))
        if v["abstain_rate"] > .3: ev.append(dict(ts=t, id="MODEL", ty="WRN", msg=f"{k} abstains on {v['abstain_rate']:.0%} of holdout"))
    return ev

with st.sidebar:
    st.markdown('<div class="logo">✕</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Models", "Logs", "Settings"], format_func=lambda x: {"Overview": "▣", "Models": "◉", "Logs": "≡", "Settings": "⚙"}[x], label_visibility="collapsed")
M = R["models"][ss.model]

a, b, c, d = st.columns([1.1, 2.2, 0.9, 0.35], vertical_alignment="center")
a.markdown('<div class="top">ThreatLens</div>', unsafe_allow_html=True)
b.selectbox("Load a test case", list(SAMPLES), index=None, placeholder="Load a test case to scan…", key="pick", on_change=on_pick, label_visibility="collapsed")
c.markdown('<div style="text-align:right"><span class="pill">MODEL LIVE</span></div>', unsafe_allow_html=True)
d.markdown('<div class="av">ML</div>', unsafe_allow_html=True)

if page == "Overview":
    left, right = st.columns([2.6, 1])
    with left:
        with st.container(border=True):
            sw = '<svg viewBox="0 0 300 90" width="300" style="opacity:.9"><path d="M0,70 C60,70 80,10 150,45 C210,75 240,20 300,30" fill="none" stroke="#7c6cf0" stroke-width="2"/><circle cx="150" cy="45" r="5" fill="#cfc8ff"/><circle cx="150" cy="45" r="12" fill="#7c6cf0" opacity=".3"/></svg>'
            st.markdown(f'<div class="hero" style="display:flex;justify-content:space-between;align-items:center;gap:12px"><div><div class="eye">SECURITY ANALYST OVERVIEW</div><h2>Holdout macro-F1 is {M["macro_f1"]:.2f} on unseen campaigns</h2><p>Random split shows {R["random_split_contrast"]["macro_f1"]:.2f} because templates leak across splits. {html.escape(ss.model)} abstains on {M["abstain_rate"]:.0%} of held-out messages and sends them to a human.</p></div><div class="hide-s">{sw}</div></div>', unsafe_allow_html=True)
            st.button("Run acceptance tests", on_click=run_tests)
        c1, c2, c3 = st.columns(3)
        ph = M["hist"]["phishing"]
        with c1, st.container(border=True):
            st.markdown(f'<div class="eye">DETECTION</div><div class="big">{M["macro_f1"]:.2f} macro-F1</div>{spark(ph)}<div class="sub g">↑ recall {M["recall"]:.2f} · precision {M["precision"]:.2f}</div>', unsafe_allow_html=True)
        kept = R["dedup"]["after"] / R["dedup"]["before"]
        with c2, st.container(border=True):
            st.markdown(f'<div class="eye">CORPUS</div><div class="big">{R["dedup"]["after"]} Messages</div><div class="prog"><i style="width:{kept*100:.0f}%"></i></div><div class="sub">{kept:.0%} kept after dedup ({R["dedup"]["removed"]} removed)</div>', unsafe_allow_html=True)
        f = M["fpr_legit_urgent"]
        with c3, st.container(border=True):
            st.markdown(f'<div class="eye">LEGIT URGENT MAIL</div><div class="big">{"OPTIMAL" if f == 0 else f"{f:.0%} FPR"}</div><div class="sub" style="margin-top:14px"><span class="dot" style="display:inline-block;background:var(--{"gr" if f == 0 else "am"});margin-right:6px"></span>{"No legitimate urgent message flagged high risk" if f == 0 else "Some legitimate urgent mail flagged high risk"}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            h = M["hist"]; sp, sl = max(sum(h["phishing"]), 1), max(sum(h["legitimate"]), 1)
            P, Lg = [v / sp * 100 for v in h["phishing"]], [v / sl * 100 for v in h["legitimate"]]
            ym = max(20, -(-max(P + Lg) // 20) * 20)
            st.markdown('<div class="ct">Risk Score Distribution on Held-out Campaigns</div>' + line_chart([f"{i*10}%" for i in range(10)], [(P, "#7c6cf0", False), (Lg, "#8b8ba3", True)], ym) + '<div class="sub">Solid: phishing · dashed: legitimate. Share of messages per risk bucket.</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            L = ss.last
            if L:
                c = {1: "var(--rd)", 0: "var(--gr)", -1: "var(--am)"}[L["d"]]
                why = "; ".join(html.escape(x) for x in L["cues"]) or "no rule cues matched"; tm = f" Model terms: {html.escape(', '.join(L['terms']))}." if L["terms"] else ""
                q, ans = html.escape(L["text"][:150]) + ("…" if len(L["text"]) > 150 else ""), f'<b style="color:{c}">Risk {L["p"]:.0%}: {ps.LABEL[L["d"]]}</b>Why: {why}.{tm}'
            else:
                q, ans = "What is the holdout macro-F1 vs the random split?", f'<b>AI Response:</b>Holdout {M["macro_f1"]:.2f} vs random split {R["random_split_contrast"]["macro_f1"]:.2f}. Template leakage inflates the random split.'
            st.markdown(f'<div class="ct"><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#7c6cf0;margin-right:8px;vertical-align:middle"></span>Analyst Assistant</div><div class="chat"><div class="b1">{q}</div><div class="b2">{ans}</div></div>', unsafe_allow_html=True)
            st.text_input("Type a message to scan", key="cmd", on_change=on_cmd, placeholder="Paste a message and press Enter…", label_visibility="collapsed")
            st.caption(ps.DISCLAIMER)
        with st.container(border=True):
            ab = "wn" if M["abstain_rate"] > .3 else "ok"; mx = max(v["decision_flip_rate"] for v in M["drift"].values())
            items = [("Dedup → Campaign split", "gr"), ("Score → Abstain gate", "am" if ab == "wn" else "gr"), ("Perturb → Drift check", "am" if mx >= .1 else "gr"), ("Links → Inert text only", "gr")]
            st.markdown('<div class="ct">Active Checks</div>' + "".join(f'<div class="auto"><span>{n}</span><span class="dot" style="background:var(--{c})"></span></div>' for n, c in items), unsafe_allow_html=True)

elif page == "Models":
    cm = M["confusion"]; at = R["acceptance"]; x, y = st.columns(2)
    with x, st.container(border=True):
        st.markdown(f'<div class="ct">Holdout confusion · {html.escape(ss.model)}</div>' + table(["", "Pred low", "Pred high"], [("Legitimate", cm["tn"], cm["fp"]), ("Phishing", cm["fn"], cm["tp"])]) + f'<div class="sub" style="margin-top:8px">Abstained: {cm["abstain"]}</div>', unsafe_allow_html=True)
    with y, st.container(border=True):
        st.markdown('<div class="ct">All models, held-out campaigns</div>' + table(["Model", "F1", "Abstain", "FPR-LU"], [(k, f'{v["macro_f1"]:.2f}', f'{v["abstain_rate"]:.0%}', f'{v["fpr_legit_urgent"]:.0%}') for k, v in R["models"].items()]), unsafe_allow_html=True)
    x, y = st.columns(2)
    with x, st.container(border=True):
        st.markdown('<div class="ct">Mean risk per held-out campaign (LogReg)</div>' + table(["Campaign", "Label", "n", "Mean risk"], [(k, "phishing" if v["label"] else "legit", v["n"], f'{v["mean_risk"]:.0%}') for k, v in R["per_campaign"].items()]), unsafe_allow_html=True)
    with y, st.container(border=True):
        st.markdown(f'<div class="ct">Robustness drift · {html.escape(ss.model)}</div>' + table(["Perturbation", "Mean |shift|", "Flips"], [(k, f'{v["mean_abs_shift"]:.3f}', f'{v["decision_flip_rate"]:.0%}') for k, v in M["drift"].items()]), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="ct">Acceptance tests</div>' + table(["Case", "Condition", "Risk", "Decision"], [("AT-01", at["AT-01"]["case"], f'{at["AT-01"]["risk"]:.0%}', at["AT-01"]["decision"]), ("AT-02", at["AT-02"]["case"], f'{at["AT-02"]["risk"]:.0%}', at["AT-02"]["decision"]), ("AT-03", at["AT-03"]["case"], f'{at["AT-03"]["base_risk"]:.0%}', "see drift table / perturb below")]), unsafe_allow_html=True)
        L = ss.last
        if L and st.button("Perturb last scanned message (AT-03)"):
            m = models[L["model"]]; base = m.predict_proba([L["text"] + " urlfeat_none"])[0, 1]
            rows = [("Original", base)] + [(k, m.predict_proba([fn(L["text"]) + " urlfeat_none"])[0, 1]) for k, fn in ps.PERTURB.items()]
            st.markdown(table(["Variant", "Risk", "Shift"], [(k, f"{p:.0%}", "" if k == "Original" else f"{(p-base)*100:+.0f}") for k, p in rows]) + '<div class="sub">Large shifts show the model limitation.</div>', unsafe_allow_html=True)

elif page == "Logs":
    with st.container(border=True):
        ev = ss.events + seed_events()
        st.markdown('<div class="ct">Activity logs</div>' + table(["Timestamp", "ID", "Type", "Message"], [(e["ts"], e["id"], f'<span class="{tyc[e["ty"]]}">{e["ty"]}</span>', html.escape(e["msg"])) for e in ev]), unsafe_allow_html=True)

else:
    with st.container(border=True):
        st.selectbox("Active model", list(models), key="model"); st.json(R["settings"]); st.json({"holdout_campaigns": R["split"]["holdout_campaigns"], "dedup": R["dedup"]}); st.caption(ps.DISCLAIMER)
