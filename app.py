import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:      #0a0a0f;
    --bg-card:      #111118;
    --bg-input:     #1a1a24;
    --accent:       #c9a84c;
    --accent-dim:   #a07830;
    --accent-glow:  rgba(201,168,76,0.18);
    --text-primary: #f0ead6;
    --text-muted:   #7a7a8a;
    --border:       rgba(201,168,76,0.25);
    --border-dim:   rgba(255,255,255,0.07);
    --radius:       16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-dim); border-radius: 99px; }

/* HERO */
.hero { text-align: center; padding: 52px 20px 36px; }
.hero-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 5px;
    text-transform: uppercase; color: var(--accent); margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(48px, 7vw, 88px); letter-spacing: 6px;
    color: var(--text-primary); line-height: 1; margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-line {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin-top: 20px; opacity: 0.4;
}
.hero-line::before, .hero-line::after {
    content: ''; height: 1px; width: 80px;
    background: linear-gradient(90deg, transparent, var(--accent));
}
.hero-line::after { background: linear-gradient(90deg, var(--accent), transparent); }
.hero-dot { width: 5px; height: 5px; background: var(--accent); border-radius: 50%; }

/* CARD */
.card {
    background: var(--bg-card); border: 1px solid var(--border-dim);
    border-radius: var(--radius); padding: 32px 28px;
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.card-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 22px;
    letter-spacing: 3px; color: var(--accent); margin-bottom: 6px;
}
.card-subtitle { font-size: 12px; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 24px; }

/* SPEC GRID */
.spec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 24px; }
.spec-item {
    background: var(--bg-input); border: 1px solid var(--border-dim);
    border-radius: 10px; padding: 10px 14px;
    display: flex; justify-content: space-between; align-items: center;
}
.spec-label { font-size: 11px; color: var(--text-muted); letter-spacing: 0.5px; }
.spec-value { font-size: 11px; font-weight: 600; color: var(--accent); }

/* INPUTS */
[data-testid="stNumberInput"] > div,
[data-testid="stSelectbox"]   > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important; transition: border-color 0.2s;
}
[data-testid="stNumberInput"] > div:focus-within,
[data-testid="stSelectbox"]   > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: #0a0a0f !important;
    -webkit-text-fill-color: #0a0a0f !important;
}
[data-testid="stNumberInput"] input::placeholder {
    color: #888 !important;
    -webkit-text-fill-color: #888 !important;
}
[data-testid="stWidgetLabel"] p {
    color: var(--text-muted) !important; font-size: 12px !important;
    letter-spacing: 0.5px !important; font-family: 'DM Sans', sans-serif !important;
}

/* BUTTON */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), var(--accent-dim)) !important;
    color: #0a0a0f !important; font-family: 'Bebas Neue', sans-serif !important;
    font-size: 18px !important; letter-spacing: 3px !important;
    border: none !important; border-radius: 10px !important;
    padding: 14px 0 !important; cursor: pointer !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 24px rgba(201,168,76,0.3) !important; margin-top: 8px !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201,168,76,0.45) !important;
}

/* CAR IMAGE BANNER */
.car-banner {
    position: relative; border-radius: 14px; overflow: hidden;
    margin-bottom: 6px; border: 1px solid var(--border-dim); background: #0d0d14;
}
.car-banner img {
    width: 100%; height: 180px; object-fit: cover; display: block; opacity: 0.88;
}
.car-banner-overlay {
    position: absolute; bottom: 0; left: 0; right: 0; padding: 10px 14px;
    background: linear-gradient(0deg, rgba(10,10,15,0.92) 0%, transparent 100%);
    display: flex; justify-content: space-between; align-items: flex-end;
}
.car-type-badge {
    font-family: 'Bebas Neue', sans-serif; font-size: 13px; letter-spacing: 2px;
    color: var(--accent); background: rgba(201,168,76,0.12);
    border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px;
}
.car-segment-text { font-size: 10px; letter-spacing: 1px; color: var(--text-muted); }

/* SECTION LABEL */
.section-label {
    font-size: 10px; letter-spacing: 2.5px; color: var(--text-muted);
    text-transform: uppercase; margin-bottom: 8px; margin-top: 14px;
}

/* MOBIL LIST dari dataset */
.car-list-wrap {
    background: var(--bg-input);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.car-list-title {
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 10px;
}
.car-list-grid {
    display: flex; flex-wrap: wrap; gap: 6px;
}
.car-pill {
    font-size: 11px; font-weight: 500;
    padding: 4px 10px; border-radius: 99px;
    border: 1px solid var(--border-dim);
    color: var(--text-primary);
    background: rgba(255,255,255,0.04);
    white-space: nowrap;
}
.car-pill-eco  { border-color: rgba(76,175,130,0.3);  color: #4caf82; background: rgba(76,175,130,0.07); }
.car-pill-mid  { border-color: rgba(201,168,76,0.3);  color: var(--accent); background: rgba(201,168,76,0.07); }
.car-pill-pre  { border-color: rgba(229,115,115,0.3); color: #e57373; background: rgba(229,115,115,0.07); }

/* RESULT */
.result-wrap { text-align: center; padding: 20px 20px 16px; }
.result-label { font-size: 11px; letter-spacing: 4px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; }
.result-price {
    font-family: 'Bebas Neue', sans-serif; font-size: clamp(48px, 7vw, 72px);
    letter-spacing: 4px;
    background: linear-gradient(135deg, #f5d87a, var(--accent), #a07830);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
    filter: drop-shadow(0 0 20px rgba(201,168,76,0.4));
}
.result-sub { font-size: 13px; color: var(--text-muted); margin-top: 6px; }

.segment-tag {
    display: inline-block; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 600; padding: 5px 14px;
    border-radius: 99px; margin-top: 10px;
}
.seg-ekonomis { background: rgba(76,175,130,0.15); color: #4caf82; border: 1px solid rgba(76,175,130,0.3); }
.seg-menengah { background: rgba(201,168,76,0.15); color: var(--accent); border: 1px solid var(--border); }
.seg-premium  { background: rgba(229,115,115,0.15); color: #e57373; border: 1px solid rgba(229,115,115,0.3); }

.result-divider { height: 1px; background: var(--border-dim); margin: 18px 0; }

.detail-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid var(--border-dim);
}
.detail-row:last-child { border-bottom: none; }
.detail-key { font-size: 12px; color: var(--text-muted); letter-spacing: 0.5px; }
.detail-val { font-size: 13px; font-weight: 600; color: var(--text-primary); }

/* PLACEHOLDER */
.placeholder { text-align: center; padding: 30px 20px; opacity: 0.35; }
.placeholder-icon { font-size: 44px; margin-bottom: 12px; }
.placeholder-text { font-size: 13px; color: var(--text-muted); letter-spacing: 1px; }

/* FOOTER */
.footer-card {
    background: linear-gradient(135deg, #1a1a28, #12121e);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 22px; text-align: center; margin-top: 18px;
    position: relative; overflow: hidden;
}
.footer-card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, var(--accent-glow), transparent 60%);
    pointer-events: none;
}
.footer-eyebrow { font-size: 10px; letter-spacing: 3px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; }
.footer-name { font-family: 'Bebas Neue', sans-serif; font-size: 22px; letter-spacing: 3px; color: var(--accent); }
.footer-npm { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

[data-testid="stHorizontalBlock"] { gap: 20px !important; }

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Car_sales.xlsx")

# =========================
# PREPROCESSING
# =========================
df.dropna(subset=['Price_in_thousands'], inplace=True)
df['Car_Name'] = df['Manufacturer'].astype(str) + ' ' + df['Model'].astype(str)

fitur_numerik = ['Engine_size','Horsepower','Wheelbase','Width','Curb_weight','Fuel_capacity','Fuel_efficiency']
for col in fitur_numerik:
    df[col] = df[col].fillna(df[col].median())

# =========================
# ENCODING
# =========================
le = LabelEncoder()
df['Vehicle_type_enc'] = le.fit_transform(df['Vehicle_type'])

# =========================
# FEATURE & TARGET
# =========================
fitur = ['Engine_size','Horsepower','Wheelbase','Width','Curb_weight','Fuel_capacity','Fuel_efficiency','Vehicle_type_enc']
X = df[fitur]
y = df['Price_in_thousands']

# =========================
# TRAINING MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# DAFTAR MOBIL PER SEGMEN (dari dataset)
# =========================
mobil_ekonomis = df[df['Price_in_thousands'] < 25]['Car_Name'].tolist()
mobil_menengah = df[(df['Price_in_thousands'] >= 25) & (df['Price_in_thousands'] < 45)]['Car_Name'].tolist()
mobil_premium  = df[df['Price_in_thousands'] >= 45]['Car_Name'].tolist()

def car_pills_html(car_list, pill_class, max_show=10):
    """Buat HTML pill untuk daftar mobil"""
    pills = ""
    for name in car_list[:max_show]:
        pills += f'<span class="car-pill {pill_class}">{name}</span>'
    if len(car_list) > max_show:
        sisa = len(car_list) - max_show
        pills += f'<span class="car-pill" style="opacity:0.5;">+{sisa} lainnya</span>'
    return pills

# =========================
# GAMBAR MAPPING
# =========================
VEHICLE_IMAGES = {
    "Car": {
        "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&q=80",
        "label": "SEDAN / CAR",
        "desc": "Passenger Car"
    },
    "Passenger": {
        "url": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=800&q=80",
        "label": "SUV / PASSENGER",
        "desc": "Passenger Vehicle"
    }
}

def get_price_segment(harga):
    if harga < 25:
        return {
            "url":       "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=800&q=80",
            "segment":   "EKONOMIS",
            "css_class": "seg-ekonomis",
            "pill_class":"car-pill-eco",
            "range":     "< $25K",
            "desc":      "Segmen terjangkau",
            "car_list":  mobil_ekonomis
        }
    elif harga < 45:
        return {
            "url":       "https://images.unsplash.com/photo-1555215695-3004980ad54e?w=800&q=80",
            "segment":   "MENENGAH",
            "css_class": "seg-menengah",
            "pill_class":"car-pill-mid",
            "range":     "$25K – $45K",
            "desc":      "Segmen menengah",
            "car_list":  mobil_menengah
        }
    else:
        return {
            "url":       "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800&q=80",
            "segment":   "PREMIUM",
            "css_class": "seg-premium",
            "pill_class":"car-pill-pre",
            "range":     "> $45K",
            "desc":      "Segmen premium",
            "car_list":  mobil_premium
        }

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◆ Final Project · Sains Data · CRISP-DM ◆</div>
    <h1 class="hero-title">PREDIKSI <span>HARGA</span> MOBIL</h1>
    <div class="hero-line"><div class="hero-dot"></div></div>
</div>
""", unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1], gap="medium")

# =========================
# INPUT FORM
# =========================
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">INPUT SPESIFIKASI</div>
        <div class="card-subtitle">MASUKKAN DATA KENDARAAN</div>
        <div class="spec-grid">
            <div class="spec-item"><span class="spec-label">Engine Size</span><span class="spec-value">1 – 8 L</span></div>
            <div class="spec-item"><span class="spec-label">Horsepower</span><span class="spec-value">50 – 500 HP</span></div>
            <div class="spec-item"><span class="spec-label">Wheelbase</span><span class="spec-value">80 – 140 in</span></div>
            <div class="spec-item"><span class="spec-label">Width</span><span class="spec-value">60 – 90 in</span></div>
            <div class="spec-item"><span class="spec-label">Curb Weight</span><span class="spec-value">1.5 – 6 t</span></div>
            <div class="spec-item"><span class="spec-label">Fuel Capacity</span><span class="spec-value">5 – 40 gal</span></div>
            <div class="spec-item"><span class="spec-label">Fuel Efficiency</span><span class="spec-value">10 – 60 mpg</span></div>
            <div class="spec-item"><span class="spec-label">Vehicle Type</span><span class="spec-value">Car / Pass.</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    engine_size     = st.number_input("Variable 1 — Engine Size (Liter)",    min_value=1.0,  max_value=8.0,   value=3.0,   step=0.1)
    horsepower      = st.number_input("Variable 2 — Horsepower (HP)",        min_value=50,   max_value=500,   value=150)
    wheelbase       = st.number_input("Variable 3 — Wheelbase (inch)",       min_value=80.0, max_value=140.0, value=100.0, step=0.1)
    width           = st.number_input("Variable 4 — Width (inch)",           min_value=60.0, max_value=90.0,  value=70.0,  step=0.1)
    curb_weight     = st.number_input("Variable 5 — Curb Weight (ton)",      min_value=1.5,  max_value=6.0,   value=3.0,   step=0.1)
    fuel_capacity   = st.number_input("Variable 6 — Fuel Capacity (gallon)", min_value=5.0,  max_value=40.0,  value=15.0,  step=0.1)
    fuel_efficiency = st.number_input("Variable 7 — Fuel Efficiency (mpg)",  min_value=10.0, max_value=60.0,  value=25.0,  step=0.1)
    vehicle_type    = st.selectbox("Variable 8 — Vehicle Type", df['Vehicle_type'].unique())

    tombol = st.button("🚗  HITUNG HARGA MOBIL")

# =========================
# HASIL PREDIKSI
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        <div class="card-title">PERKIRAAN HARGA</div>
        <div class="card-subtitle">HASIL PREDIKSI MODEL</div>
    """, unsafe_allow_html=True)

    vtype_key  = vehicle_type if vehicle_type in VEHICLE_IMAGES else "Car"
    vtype_info = VEHICLE_IMAGES[vtype_key]

    if tombol:
        # Prediksi
        vehicle_type_enc = le.transform([vehicle_type])[0]
        input_data = pd.DataFrame({
            'Engine_size':      [engine_size],
            'Horsepower':       [horsepower],
            'Wheelbase':        [wheelbase],
            'Width':            [width],
            'Curb_weight':      [curb_weight],
            'Fuel_capacity':    [fuel_capacity],
            'Fuel_efficiency':  [fuel_efficiency],
            'Vehicle_type_enc': [vehicle_type_enc]
        })
        hasil = model.predict(input_data)[0]
        if hasil < 0:
            hasil = 0
        harga_usd   = hasil * 1000
        price_info  = get_price_segment(hasil)
        pills_html  = car_pills_html(price_info['car_list'], price_info['pill_class'])

        # Gambar 1: Tipe kendaraan
        st.markdown('<div class="section-label">◈ Tipe Kendaraan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="car-banner">
            <img src="{vtype_info['url']}" alt="{vtype_info['label']}"/>
            <div class="car-banner-overlay">
                <span class="car-type-badge">{vtype_info['label']}</span>
                <span class="car-segment-text">{vtype_info['desc']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gambar 2: Segmen harga + daftar mobil dari dataset
        st.markdown('<div class="section-label">◈ Segmen Harga</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="car-banner">
            <img src="{price_info['url']}" alt="{price_info['segment']}"/>
            <div class="car-banner-overlay">
                <span class="car-type-badge">{price_info['segment']}</span>
                <span class="car-segment-text">{price_info['range']} · {price_info['desc']}</span>
            </div>
        </div>
        <div class="car-list-wrap">
            <div class="car-list-title">◈ Contoh mobil dalam dataset di segmen ini</div>
            <div class="car-list-grid">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Harga
        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-label">Estimated Price (USD Thousands)</div>
            <div class="result-price">${hasil:,.2f}K</div>
            <div class="result-sub">≈ ${harga_usd:,.0f} USD</div>
            <span class="segment-tag {price_info['css_class']}">{price_info['segment']} · {price_info['range']}</span>
        </div>
        <div class="result-divider"></div>
        <div class="detail-row"><span class="detail-key">Engine Size</span><span class="detail-val">{engine_size} L</span></div>
        <div class="detail-row"><span class="detail-key">Horsepower</span><span class="detail-val">{horsepower} HP</span></div>
        <div class="detail-row"><span class="detail-key">Wheelbase</span><span class="detail-val">{wheelbase} in</span></div>
        <div class="detail-row"><span class="detail-key">Width</span><span class="detail-val">{width} in</span></div>
        <div class="detail-row"><span class="detail-key">Curb Weight</span><span class="detail-val">{curb_weight} t</span></div>
        <div class="detail-row"><span class="detail-key">Fuel Capacity</span><span class="detail-val">{fuel_capacity} gal</span></div>
        <div class="detail-row"><span class="detail-key">Fuel Efficiency</span><span class="detail-val">{fuel_efficiency} mpg</span></div>
        <div class="detail-row"><span class="detail-key">Vehicle Type</span><span class="detail-val">{vehicle_type}</span></div>
        """, unsafe_allow_html=True)

    else:
        # Preview sebelum prediksi
        st.markdown('<div class="section-label">◈ Tipe Kendaraan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="car-banner">
            <img src="{vtype_info['url']}" alt="preview"/>
            <div class="car-banner-overlay">
                <span class="car-type-badge">{vtype_info['label']}</span>
                <span class="car-segment-text">Preview · {vtype_info['desc']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🔍</div>
            <div class="placeholder-text">Masukkan spesifikasi kendaraan<br>lalu klik tombol prediksi</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-card">
        <div class="footer-eyebrow">Sistem ini dibuat oleh</div>
        <div class="footer-name">Rosi Saraswati Alrasid</div>
        <div class="footer-npm">NPM · 237006087</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
