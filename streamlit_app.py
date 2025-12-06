import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Opsional: geo analiz üçün
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

import pydeck as pdk
import matplotlib.pyplot as plt


# ================== KONFİQURASİYA ==================

MODEL_PATH = r"C:\Users\MSI GF75\Desktop\AzerCosmos\fire_ignition_xgb_model.joblib"
DATA_PATH = r"C:\Users\MSI GF75\Desktop\AzerCosmos\azer_forest_sensors_20k_regions.csv"

BEST_THRESHOLD = 0.7
ROW_INTERVAL_SEC = 20   # hər neçə saniyədən bir yeni row işlənir
REFRESH_SEC = 1        # səhifə nə qədər tez-tez yenilənsin


# ================== MODEL FUNKSİYASI ==================

def predict_risk(model, sensor_reading: dict, threshold: float):
    """
    Sensor oxunuşuna əsasən yanğın ehtimalı hesablayır.
    """
    feature_cols = [
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
        "solar_rad_wm2", "rain_last_24h_mm", "vpd_kpa",
        "co_ppm", "co2_ppm", "tvoc_ppb",
        "h2_idx", "ethanol_idx", "hydrocarbon_idx",
        "lat", "lon"
    ]

    x = pd.DataFrame([sensor_reading])[feature_cols]
    prob = model.predict_proba(x)[0, 1]
    alert = prob >= threshold
    return prob, alert


# ================== CACHED YARDIMÇI FUNKSİYALAR ==================

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # datetime string-dir, sonra ayrıca da parse edirik
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_osm_graph_for_sensors(df: pd.DataFrame):
    """
    Sensorların lat/lon koordinatlarına görə bounding box hesablayıb
    osmnx ilə həmin ərazinin yol şəbəkəsini yükləyir.
    """
    if not HAS_OSMNX:
        return None, None

    if not {"lat", "lon"}.issubset(df.columns):
        return None, None

    min_lat, max_lat = df["lat"].min(), df["lat"].max()
    min_lon, max_lon = df["lon"].min(), df["lon"].max()

    lat_pad = (max_lat - min_lat) * 0.1 if max_lat != min_lat else 0.05
    lon_pad = (max_lon - min_lon) * 0.1 if max_lon != min_lon else 0.05

    north = max_lat + lat_pad
    south = min_lat - lat_pad
    east = max_lon + lon_pad
    west = min_lon - lon_pad

    try:
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="drive"
        )
        fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0)
        return G, fig
    except Exception:
        return None, None


# ================== MAIN APP ==================

def main():
    st.set_page_config(page_title="Wildfire AI – Advanced Monitoring Dashboard", layout="wide")
    st.title("🔥 Wildfire AI – Real-time Fire Risk Monitoring Dashboard")

    # -------- Sidebar – konfiqurasiya --------
    st.sidebar.header("ML Model & Dataset")
    st.sidebar.write(f"Model faylı: `{MODEL_PATH}`")
    st.sidebar.write(f"Dataset faylı: `{DATA_PATH}`")

    current_threshold = st.sidebar.slider(
        "Risk Threshold (P(ignite))",
        min_value=0.1,
        max_value=0.95,
        value=float(BEST_THRESHOLD),
        step=0.05,
        help="Bu həddən yuxarı olan ehtimallar ALERT kimi qeyd olunur.",
    )

    st.sidebar.write(f"Hər {ROW_INTERVAL_SEC} saniyədən bir yeni row işlənir.")
    st.sidebar.write(f"Səhifə hər {REFRESH_SEC} saniyədən bir yenilənir.")

    # -------- Model və data --------
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success("ML model uğurla yükləndi ✅")
    except Exception as e:
        st.sidebar.error(f"ML model yüklənmədi: {e}")
        st.stop()

    try:
        df_data = load_data(DATA_PATH)
        st.sidebar.success(f"Dataset yükləndi. Sətir sayı: {len(df_data)}")
    except Exception as e:
        st.sidebar.error(f"Dataset yüklənmədi: {e}")
        st.stop()

    if len(df_data) == 0:
        st.error("Dataset boşdur.")
        st.stop()

    # -------- Session state --------
    if "pred_buffer" not in st.session_state:
        st.session_state.pred_buffer = []
    if "alert_buffer" not in st.session_state:
        st.session_state.alert_buffer = []
    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0
    if "last_update" not in st.session_state:
        st.session_state.last_update = 0.0

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Predictions", "🚨 Alerts", "🌍 Geo & Map", "📈 Analytics"]
    )

    # ================== DATASET STREAMING ==================
    now = time.time()
    if now - st.session_state.last_update >= ROW_INTERVAL_SEC:
        idx = st.session_state.row_idx % len(df_data)
        row = df_data.iloc[idx].to_dict()

        # region / datetime yoxdursa, defolt
        row.setdefault("region", "Unknown")
        row.setdefault("datetime", time.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            prob, is_alert = predict_risk(model, row, current_threshold)
            event = {
                **row,  # bütün CSV sütunları buradan gəlir
                "p_ignite_percent": float(prob * 100.0),
                "alert": bool(is_alert),
            }

            st.session_state.pred_buffer.append(event)

            if is_alert:
                st.sidebar.warning(
                    f"🔥 MODEL ALERT: {row.get('region', 'Unknown')} | "
                    f"P(fire)={prob*100:.1f}% ≥ threshold {current_threshold*100:.1f}%"
                )
                st.session_state.alert_buffer.append(event)
            else:
                st.sidebar.write(
                    f"ℹ️ No alert. prob={prob:.3f} ({prob*100:.1f}%) < {current_threshold*100:.1f}%"
                )

            st.session_state.row_idx += 1
            st.session_state.last_update = now

        except Exception as e:
            st.error(f"Prediction error: {e}")

    df_pred = pd.DataFrame(st.session_state.pred_buffer[-500:])   # daha çox history
    df_alerts = pd.DataFrame(st.session_state.alert_buffer[-500:])

    # datetime parsinq
    if "datetime" in df_pred.columns:
        df_pred["datetime_parsed"] = pd.to_datetime(
            df_pred["datetime"], errors="coerce"
        )
    if "datetime" in df_alerts.columns:
        df_alerts["datetime_parsed"] = pd.to_datetime(
            df_alerts["datetime"], errors="coerce"
        )

    # ================== TAB 1: PREDICTIONS ==================
    with tab1:
        st.subheader("📡 Live Fire Risk Predictions (dataset simulasiya)")

        if df_pred.empty:
            st.info("Hələlik prediction yoxdur. 20 saniyədən bir yeni sətir işlənəcək.")
        else:
            cols_to_show = [
                "datetime", "unix_timestamp", "region", "lat", "lon",
                "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
                "solar_rad_wm2", "rain_last_24h_mm", "vpd_kpa",
                "co_ppm", "co2_ppm", "tvoc_ppb",
                "h2_idx", "ethanol_idx", "hydrocarbon_idx",
                "ignite_next_5min", "risk_score_noisy",
                "p_ignite_percent", "alert"
            ]
            existing_cols = [c for c in cols_to_show if c in df_pred.columns]

            left, right = st.columns([3, 2])

            with left:
                st.write("Son proqnozlar:")
                st.dataframe(
                    df_pred[existing_cols].sort_values(
                        "datetime", ascending=False
                    ),
                    use_container_width=True,
                )

            with right:
                if "p_ignite_percent" in df_pred.columns:
                    st.write("Risk metrikləri:")
                    st.metric(
                        "Avg P(ignite)",
                        f"{df_pred['p_ignite_percent'].mean():.2f}%"
                    )
                    st.metric(
                        "Max P(ignite)",
                        f"{df_pred['p_ignite_percent'].max():.2f}%"
                    )

                if {"temp_c", "humidity_pct"}.issubset(df_pred.columns):
                    st.write("Orta Temp / Humidity:")
                    st.metric(
                        "Avg Temp (°C)",
                        f"{df_pred['temp_c'].mean():.2f}"
                    )
                    st.metric(
                        "Avg Humidity (%)",
                        f"{df_pred['humidity_pct'].mean():.2f}"
                    )

                if {"lat", "lon"}.issubset(df_pred.columns):
                    st.write("Sensor locations (basic map):")
                    st.map(df_pred[["lat", "lon"]])

        st.caption(
            "Dataset simulasiya rejimi: müəyyən intervalda növbəti sətir işlənir, səhifə isə avto-refresh olur."
        )

    # ================== TAB 2: ALERTS ==================
    with tab2:
        st.subheader("🚨 Critical Fire Alerts")

        if df_alerts.empty:
            st.info("Hələ ki, kritik alert yoxdur ✅")
        else:
            last_alert = df_alerts.sort_values("datetime_parsed", ascending=False).iloc[0]
            st.markdown(
                f"""
                ### Son Kritiki Alert  
                **Region:** {last_alert.get('region', 'Unknown')}  
                **Tarix:** {last_alert.get('datetime', '')}  
                **P(fire):** {last_alert.get('p_ignite_percent', 0.0):.2f}%  
                **Label (ignite_next_5min):** {int(last_alert.get('ignite_next_5min', 0))}  
                **Koordinatlar:** {last_alert.get('lat', '?')}, {last_alert.get('lon', '?')}
                """
            )
            st.divider()

            for _, row in df_alerts.sort_values("datetime_parsed", ascending=False).iterrows():
                p_val = row.get("p_ignite_percent", 0.0)
                label = row.get("ignite_next_5min", 0)
                st.error(
                    f"🔥 **CRITICAL ALERT** – {row.get('region', 'Unknown')} | "
                    f"{row.get('datetime', '')} | "
                    f"P(fire)={p_val:.2f}% | Label={int(label)}\n\n"
                    f"Lat/Lon: {row.get('lat', '?')}, {row.get('lon', '?')}"
                )

    # ================== TAB 3: GEO & MAP ==================
    with tab3:
        st.subheader("🌍 Geo Intelligence & Map View")

        if df_pred.empty or not {"lat", "lon"}.issubset(df_pred.columns):
            st.info("Xəritə üçün kifayət qədər geo məlumat yoxdur (lat/lon sütunları).")
        else:
            st.markdown("**İnteraktiv xəritə (pydeck) – sensorlar və risk səviyyəsi**")

            df_map = df_pred.dropna(subset=["lat", "lon"]).tail(500).copy()
            df_map["radius"] = df_map["p_ignite_percent"].fillna(0) * 10 + 100

            midpoint = (
                np.average(df_map["lat"]),
                np.average(df_map["lon"]),
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_radius="radius",
                pickable=True,
                opacity=0.6,
            )

            tooltip = {
                "html": "<b>Region:</b> {region}<br/>"
                        "<b>P(fire):</b> {p_ignite_percent}%<br/>"
                        "<b>Temp:</b> {temp_c} °C<br/>"
                        "<b>Humidity:</b> {humidity_pct} %<br/>"
                        "<b>risk_score_noisy:</b> {risk_score_noisy}",
                "style": {"backgroundColor": "white", "color": "black"}
            }

            view_state = pdk.ViewState(
                latitude=midpoint[0],
                longitude=midpoint[1],
                zoom=7,
                pitch=40,
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )

            st.pydeck_chart(r)

            if HAS_OSMNX:
                st.markdown("**OSM Yol Şəbəkəsi (osmnx) – sensor bbox daxilində**")
                G, fig = load_osm_graph_for_sensors(df_data)
                if G is not None and fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("osmnx üçün yol şəbəkəsi yüklənmədi və ya lat/lon məlumatı yetərli deyil.")
            else:
                st.info("`osmnx` paketini quraşdırsan, yol şəbəkəsi vizualizasiyası da aktiv olacaq.")

    # ================== TAB 4: ANALYTICS ==================
    with tab4:
        st.subheader("📈 High-level Analytics")

        if df_pred.empty:
            st.info("Analytics üçün hələlik kifayət qədər data yoxdur.")
        else:
            # ---- Region filter ----
            regions = sorted(df_pred["region"].dropna().unique().tolist()) if "region" in df_pred.columns else []
            selected_regions = st.multiselect(
                "Region filter",
                options=regions,
                default=regions,
            )

            if selected_regions and "region" in df_pred.columns:
                df_ana = df_pred[df_pred["region"].isin(selected_regions)].copy()
            else:
                df_ana = df_pred.copy()

            # ---- Ümumi metriclər ----
            col1, col2, col3, col4 = st.columns(4)

            total_events = len(df_ana)
            total_alerts = int(df_ana["alert"].sum()) if "alert" in df_ana.columns else 0
            avg_risk = df_ana["p_ignite_percent"].mean() if "p_ignite_percent" in df_ana.columns else 0.0
            avg_label = df_ana["ignite_next_5min"].mean() if "ignite_next_5min" in df_ana.columns else 0.0

            with col1:
                st.metric("Event sayı", f"{total_events}")
            with col2:
                st.metric("Kritik alert sayı", f"{total_alerts}")
            with col3:
                st.metric("Orta P(ignite)", f"{avg_risk:.2f}%")
            with col4:
                st.metric("Label avg (ignite_next_5min)", f"{avg_label:.3f}")

            # ---- Zaman seriyası: Temp, Humidity, Risk ----
            st.markdown("### Zaman seriyası – Temp / Humidity / Risk")

            if "datetime_parsed" in df_ana.columns:
                ts_cols = {}
                if "temp_c" in df_ana.columns:
                    ts_cols["temp_c"] = df_ana["temp_c"]
                if "humidity_pct" in df_ana.columns:
                    ts_cols["humidity_pct"] = df_ana["humidity_pct"]
                if "p_ignite_percent" in df_ana.columns:
                    ts_cols["p_ignite_percent"] = df_ana["p_ignite_percent"]

                if ts_cols:
                    ts_df = pd.DataFrame(ts_cols)
                    ts_df.index = df_ana["datetime_parsed"]
                    st.line_chart(ts_df)
                else:
                    st.info("Zaman seriyası üçün uyğun sütun tapılmadı.")

            # ---- risk_score_noisy vs model risk ----
            st.markdown("### Model Risk vs risk_score_noisy")

            if {"risk_score_noisy", "p_ignite_percent"}.issubset(df_ana.columns):
                fig, ax = plt.subplots()
                ax.scatter(
                    df_ana["risk_score_noisy"],
                    df_ana["p_ignite_percent"],
                    alpha=0.6,
                )
                ax.set_xlabel("risk_score_noisy (dataset)")
                ax.set_ylabel("P(ignite) % (model)")
                ax.set_title("Model vs Noisy Risk Score")
                st.pyplot(fig)

            # ---- Label (ignite_next_5min) vs model ----
            st.markdown("### Label (ignite_next_5min) vs Model Risk")

            if {"ignite_next_5min", "p_ignite_percent"}.issubset(df_ana.columns):
                by_label = (
                    df_ana.groupby("ignite_next_5min")["p_ignite_percent"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                st.write("Label üzrə orta P(ignite) və nümunə sayı:")
                st.dataframe(by_label, use_container_width=True)

                fig2, ax2 = plt.subplots()
                ax2.bar(
                    by_label["ignite_next_5min"].astype(str),
                    by_label["mean"],
                )
                ax2.set_xlabel("ignite_next_5min (0/1)")
                ax2.set_ylabel("Avg P(ignite) %")
                ax2.set_title("Label qrupları üzrə orta model risk")
                st.pyplot(fig2)

            # ---- Correlation heatmap ----
            st.markdown("### Korrelasiya heatmap – bütün sensor & risk feature-lər")

            numeric_cols = [
                c for c in df_ana.columns
                if pd.api.types.is_numeric_dtype(df_ana[c])
            ]
            if len(numeric_cols) >= 2:
                corr = df_ana[numeric_cols].corr()

                fig3, ax3 = plt.subplots(figsize=(8, 6))
                im = ax3.imshow(corr.values, aspect="auto")
                ax3.set_xticks(range(len(numeric_cols)))
                ax3.set_yticks(range(len(numeric_cols)))
                ax3.set_xticklabels(numeric_cols, rotation=90)
                ax3.set_yticklabels(numeric_cols)
                ax3.set_title("Correlation heatmap")
                fig3.colorbar(im, ax=ax3)
                plt.tight_layout()
                st.pyplot(fig3)
            else:
                st.info("Korrelasiya üçün yetərli sayda rəqəmsal sütun yoxdur.")

    # Avtomatik yenidən run
    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
