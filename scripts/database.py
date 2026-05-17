"""
=============================================================================
NeuroVision AI — DATABASE CONNECTION & INTEGRATION MODULE
=============================================================================
Student   : Blina Sopjani | ID: 69401
File      : database.py

KY MODUL:
  - Lidhja me PostgreSQL nëpërmjet SQLAlchemy
  - Krijimi automatik i tabelave (imaging_series, aneurysm_locations,
    model_predictions, model_evaluation, data_cleaning_log)
  - Ngarkimi i dataset-it të pastruar direkt në PostgreSQL
  - Ruajtja e rezultateve të modeleve
  - Queries analitike të gatshme
  - Eksport për Power BI / Tableau

REQUIREMENTS:
  pip install sqlalchemy psycopg2-binary pandas python-dotenv

SETUP PostgreSQL:
  createdb rsna_aneurysm
  psql -d rsna_aneurysm -f rsna_schema_and_queries.sql

.env FILE (krijo në root të projektit):
  DB_HOST=localhost
  DB_PORT=5432
  DB_NAME=rsna_aneurysm
  DB_USER=postgres
  DB_PASSWORD=your_password
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from sqlalchemy import (
        create_engine, text, MetaData, Table, Column,
        Integer, String, Float, Boolean, DateTime,
        SmallInteger, Numeric, inspect as sa_inspect
    )
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("  ⚠ SQLAlchemy nuk është instaluar: pip install sqlalchemy psycopg2-binary")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURIM DATABASE
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
    "dbname":   os.getenv("DB_NAME",     "rsna_aneurysm"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# Label kolonat (rendi duhet të jetë identik me train.csv)
LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

LOCATION_SHORT = {
    "Left Infraclinoid Internal Carotid Artery":  "L. Infraclinoid ICA",
    "Right Infraclinoid Internal Carotid Artery": "R. Infraclinoid ICA",
    "Left Supraclinoid Internal Carotid Artery":  "L. Supraclinoid ICA",
    "Right Supraclinoid Internal Carotid Artery": "R. Supraclinoid ICA",
    "Left Middle Cerebral Artery":                "Left MCA",
    "Right Middle Cerebral Artery":               "Right MCA",
    "Anterior Communicating Artery":              "AComA",
    "Left Anterior Cerebral Artery":              "Left ACA",
    "Right Anterior Cerebral Artery":             "Right ACA",
    "Left Posterior Communicating Artery":        "Left PComA",
    "Right Posterior Communicating Artery":       "Right PComA",
    "Basilar Tip":                                "Basilar Tip",
    "Other Posterior Circulation":                "Other Post. Circ.",
}


# ─────────────────────────────────────────────────────────────────────────────
# KLASA KRYESORE — DatabaseManager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Menaxhon të gjitha operacionet me PostgreSQL.

    Funksionalitetet:
        - connect()             : Krijon lidhjen me PostgreSQL
        - test_connection()     : Teston lidhjen
        - create_tables()       : Krijon tabelat nëse nuk ekzistojnë
        - load_dataset()        : Ngarkon df_clean në imaging_series
        - load_locations()      : Ngarkon labels e lokacioneve
        - save_model_results()  : Ruan rezultatet e modelit
        - save_cleaning_log()   : Ruan log-un e pastrimit
        - query_*()             : Queries analitike të gatshme
        - export_for_powerbi()  : Eksport CSV/JSON për BI tools

    Shembull përdorimi:
        db = DatabaseManager()
        if db.connect():
            db.create_tables()
            db.load_dataset(df_clean)
            db.save_model_results(results_dict)
    """

    def __init__(self, config: dict = None):
        self.config  = config or DB_CONFIG
        self.engine  = None
        self.session = None
        self._connected = False

    # ─────────────────────────────────────────────────────────────────────
    # LIDHJA
    # ─────────────────────────────────────────────────────────────────────

    def get_connection_string(self) -> str:
        """Kthen connection string-un PostgreSQL."""
        c = self.config
        return (
            f"postgresql+psycopg2://{c['user']}:{c['password']}"
            f"@{c['host']}:{c['port']}/{c['dbname']}"
        )

    def connect(self) -> bool:
        """
        Krijon lidhjen me PostgreSQL nëpërmjet SQLAlchemy.

        Returns:
            bool: True nëse lidhja u krye me sukses
        """
        if not SQLALCHEMY_AVAILABLE:
            print("  ❌ SQLAlchemy mungon: pip install sqlalchemy psycopg2-binary")
            return False

        print("=" * 65)
        print("LIDHJA ME PostgreSQL")
        print("=" * 65)
        print(f"\n  Host     : {self.config['host']}:{self.config['port']}")
        print(f"  Database : {self.config['dbname']}")
        print(f"  User     : {self.config['user']}")

        try:
            conn_str    = self.get_connection_string()
            self.engine = create_engine(
                conn_str,
                pool_pre_ping  = True,    # Teston lidhjen para çdo query
                pool_recycle   = 3600,    # Rifreskuon lidhjen çdo orë
                pool_size      = 5,
                max_overflow   = 10,
                echo           = False,   # True = shfaq SQL queries
            )

            # Testo lidhjen
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]

            Session = sessionmaker(bind=self.engine)
            self.session    = Session()
            self._connected = True

            print(f"\n  ✅ Lidhja u krye me sukses!")
            print(f"  PostgreSQL version: {version[:50]}...")
            return True

        except Exception as e:
            print(f"\n  ❌ Lidhja dështoi: {e}")
            print(f"\n  Zgjidhe:")
            print(f"    1. Sigurohu që PostgreSQL është duke punuar:")
            print(f"       sudo service postgresql start")
            print(f"    2. Krijo databazën:")
            print(f"       createdb {self.config['dbname']}")
            print(f"    3. Kontrollo kredencialet në .env file")
            return False

    def test_connection(self) -> bool:
        """Teston lidhjen ekzistuese."""
        if not self._connected or not self.engine:
            print("  ❌ Nuk ka lidhje aktive. Thirr connect() fillimisht.")
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("  ✅ Lidhja aktive dhe funksionale")
            return True
        except Exception as e:
            print(f"  ❌ Lidhja nuk punon: {e}")
            return False

    def disconnect(self):
        """Mbyll lidhjen me databazën."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        self._connected = False
        print("  ✅ Lidhja u mbyll.")

    # ─────────────────────────────────────────────────────────────────────
    # KRIJIMI I TABELAVE
    # ─────────────────────────────────────────────────────────────────────

    def create_tables(self) -> bool:
        """
        Krijon tabelat PostgreSQL nëse nuk ekzistojnë.

        Tabelat:
            - institutions
            - imaging_series
            - aneurysm_locations
            - model_predictions
            - model_evaluation
            - data_cleaning_log
        """
        if not self._connected:
            print("  ❌ Nuk ka lidhje. Thirr connect() fillimisht.")
            return False

        print("\n" + "=" * 65)
        print("KRIJIMI I TABELAVE")
        print("=" * 65)

        sql_tables = """
        -- Tabela: institutions
        CREATE TABLE IF NOT EXISTS institutions (
            id          SERIAL PRIMARY KEY,
            name        VARCHAR(120) UNIQUE NOT NULL,
            country     VARCHAR(60),
            continent   VARCHAR(30),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Tabela: imaging_series (tabela kryesore)
        CREATE TABLE IF NOT EXISTS imaging_series (
            id                   SERIAL PRIMARY KEY,
            series_instance_uid  VARCHAR(64) UNIQUE NOT NULL,
            modality             VARCHAR(10),
            patient_age          SMALLINT,
            patient_sex          CHAR(1),
            institution          VARCHAR(120),
            slice_thickness_mm   NUMERIC(6,3),
            pixel_spacing_mm     NUMERIC(6,4),
            aneurysm_present     BOOLEAN NOT NULL DEFAULT FALSE,
            age_group            VARCHAR(10),
            aneurysm_count       SMALLINT DEFAULT 0,
            split_set            VARCHAR(10) DEFAULT 'train',
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Tabela: aneurysm_locations
        CREATE TABLE IF NOT EXISTS aneurysm_locations (
            id              SERIAL PRIMARY KEY,
            series_id       INT NOT NULL REFERENCES imaging_series(id) ON DELETE CASCADE,
            location_name   VARCHAR(80) NOT NULL,
            location_short  VARCHAR(30),
            is_present      BOOLEAN NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_series_location UNIQUE (series_id, location_name)
        );

        -- Tabela: model_predictions
        CREATE TABLE IF NOT EXISTS model_predictions (
            id                  SERIAL PRIMARY KEY,
            series_id           INT NOT NULL REFERENCES imaging_series(id) ON DELETE CASCADE,
            model_name          VARCHAR(30) NOT NULL,
            aneurysm_score      NUMERIC(6,4) NOT NULL,
            predicted_label     BOOLEAN NOT NULL,
            threshold_used      NUMERIC(4,3) DEFAULT 0.500,
            inference_time_ms   NUMERIC(8,2),
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Tabela: model_evaluation
        CREATE TABLE IF NOT EXISTS model_evaluation (
            id               SERIAL PRIMARY KEY,
            model_name       VARCHAR(30) NOT NULL,
            modality         VARCHAR(10),
            split_set        VARCHAR(10) DEFAULT 'test',
            n_samples        INT,
            auc_roc          NUMERIC(6,4),
            accuracy         NUMERIC(6,4),
            precision_score  NUMERIC(6,4),
            recall_score     NUMERIC(6,4),
            f1_score         NUMERIC(6,4),
            true_positives   INT,
            true_negatives   INT,
            false_positives  INT,
            false_negatives  INT,
            evaluated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Tabela: data_cleaning_log
        CREATE TABLE IF NOT EXISTS data_cleaning_log (
            id               SERIAL PRIMARY KEY,
            step_number      SMALLINT NOT NULL,
            action           VARCHAR(80) NOT NULL,
            column_affected  VARCHAR(40),
            records_changed  INT,
            method_used      VARCHAR(120),
            value_before     VARCHAR(60),
            value_after      VARCHAR(60),
            applied_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_series_modality    ON imaging_series(modality);
        CREATE INDEX IF NOT EXISTS idx_series_aneurysm    ON imaging_series(aneurysm_present);
        CREATE INDEX IF NOT EXISTS idx_series_institution ON imaging_series(institution);
        CREATE INDEX IF NOT EXISTS idx_loc_series         ON aneurysm_locations(series_id);
        CREATE INDEX IF NOT EXISTS idx_pred_model         ON model_predictions(model_name);
        CREATE INDEX IF NOT EXISTS idx_eval_model         ON model_evaluation(model_name);
        """

        try:
            with self.engine.begin() as conn:
                for stmt in sql_tables.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))

            # Verifiko tabelat e krijuara
            inspector = sa_inspect(self.engine)
            tables    = inspector.get_table_names()
            expected  = ["imaging_series", "aneurysm_locations",
                         "model_predictions", "model_evaluation",
                         "data_cleaning_log"]

            print(f"\n  Tabelat e krijuara:")
            for tbl in expected:
                status = "✅" if tbl in tables else "❌"
                print(f"    {status} {tbl}")

            print(f"\n  ✅ Të gjitha tabelat janë gati!")
            return True

        except SQLAlchemyError as e:
            print(f"  ❌ Gabim në krijimin e tabelave: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # NGARKIMI I DATASET-IT
    # ─────────────────────────────────────────────────────────────────────

    def load_dataset(
        self,
        df:        pd.DataFrame,
        split_map: dict = None,
        batch_size: int = 500,
    ) -> bool:
        """
        Ngarkon DataFrame-in e pastruar në tabelën imaging_series.

        Konverto kolonat:
            SeriesInstanceUID → series_instance_uid
            Modality          → modality
            PatientAge        → patient_age
            PatientSex        → patient_sex
            Aneurysm Present  → aneurysm_present
            AgeGroup          → age_group
            AneurysmCount     → aneurysm_count

        Args:
            df         : DataFrame i pastruar nga rsna_aneurysm_analysis.py
            split_map  : {series_id: 'train'|'val'|'test'} — assign splits
            batch_size : Rreshta per batch (performance)

        Returns:
            bool: True nëse ngarkimi u krye me sukses
        """
        if not self._connected:
            print("  ❌ Nuk ka lidhje. Thirr connect() fillimisht.")
            return False

        print("\n" + "=" * 65)
        print("NGARKIMI I DATASET-IT → imaging_series")
        print("=" * 65)
        print(f"\n  Rreshta për t'u ngarkuar: {len(df):,}")

        # Rename kolonat për PostgreSQL
        col_map = {
            "SeriesInstanceUID": "series_instance_uid",
            "Modality":          "modality",
            "PatientAge":        "patient_age",
            "PatientSex":        "patient_sex",
            "Institution":       "institution",
            "SliceThickness":    "slice_thickness_mm",
            "PixelSpacing":      "pixel_spacing_mm",
            "Aneurysm Present":  "aneurysm_present",
            "AgeGroup":          "age_group",
            "AneurysmCount":     "aneurysm_count",
        }

        db_cols = [c for c in col_map if c in df.columns]
        df_db   = df[db_cols].rename(columns=col_map).copy()

        # Shto split_set
        if split_map:
            df_db["split_set"] = df_db["series_instance_uid"].map(split_map).fillna("train")
        else:
            df_db["split_set"] = "train"

        # Konverto aneurysm_present → boolean
        df_db["aneurysm_present"] = df_db["aneurysm_present"].astype(bool)

        # Trajtimi i NaN
        df_db["patient_age"]       = pd.to_numeric(df_db.get("patient_age", 0), errors="coerce")
        df_db["aneurysm_count"]    = df_db.get("aneurysm_count", 0).fillna(0).astype(int)

        try:
            # Ngarko me batch-e për performance
            total_loaded = 0
            for start in range(0, len(df_db), batch_size):
                batch = df_db.iloc[start:start + batch_size]
                batch.to_sql(
                    name       = "imaging_series",
                    con        = self.engine,
                    if_exists  = "append",
                    index      = False,
                    method     = "multi",
                )
                total_loaded += len(batch)
                print(f"  [{total_loaded:,}/{len(df_db):,}] ngarkuar...", end="\r")

            print(f"\n  ✅ {total_loaded:,} serje të ngarkuara në imaging_series")

            # Ngarko edhe lokacionet
            self._load_locations(df)
            return True

        except SQLAlchemyError as e:
            print(f"\n  ❌ Gabim në ngarkimin e dataset-it: {e}")
            return False

    def _load_locations(self, df: pd.DataFrame) -> bool:
        """
        Ngarkon label-at e lokacioneve në aneurysm_locations.
        Thirret automatikisht nga load_dataset().
        """
        print(f"\n  Ngarkimi i lokacioneve → aneurysm_locations...")

        loc_cols = [c for c in LABEL_COLS[:-1] if c in df.columns]
        if not loc_cols:
            print("  ⚠ Asnjë kolonë lokacioni u gjet.")
            return False

        try:
            # Merr ID-të e serive nga databaza
            with self.engine.connect() as conn:
                result  = conn.execute(text(
                    "SELECT id, series_instance_uid FROM imaging_series"
                ))
                id_map  = {row[1]: row[0] for row in result.fetchall()}

            # Krijo rreshtat për lokacionet
            rows = []
            for _, record in df.iterrows():
                sid      = record.get("SeriesInstanceUID", "")
                series_id = id_map.get(sid)
                if not series_id:
                    continue
                for loc in loc_cols:
                    rows.append({
                        "series_id":     series_id,
                        "location_name": loc,
                        "location_short":LOCATION_SHORT.get(loc, loc[:30]),
                        "is_present":    bool(record.get(loc, 0)),
                    })

            if rows:
                df_loc = pd.DataFrame(rows)
                df_loc.to_sql(
                    name      = "aneurysm_locations",
                    con       = self.engine,
                    if_exists = "append",
                    index     = False,
                    method    = "multi",
                    chunksize = 1000,
                )
                print(f"  ✅ {len(rows):,} rreshta ngarkuar në aneurysm_locations")
            return True

        except SQLAlchemyError as e:
            print(f"  ❌ Gabim në ngarkimin e lokacioneve: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # RUAJTJA E REZULTATEVE TË MODELIT
    # ─────────────────────────────────────────────────────────────────────

    def save_model_results(
        self,
        model_name: str,
        results:    dict,
        modality:   str = None,
        n_samples:  int = 640,
    ) -> bool:
        """
        Ruan rezultatet e evaluimit të modelit në model_evaluation.

        Args:
            model_name : "CNN Baseline" | "ResNet-50" | "ResNet-101"
            results    : dict me auc, accuracy, precision, recall, f1
            modality   : None (overall) ose "CTA"|"MRA"|"MRI_T1"|"MRI_T2"
            n_samples  : Numri i serive të test set-it

        Shembull:
            db.save_model_results("ResNet-101", {
                "auc": 0.924, "accuracy": 0.894,
                "precision": 0.863, "recall": 0.847, "f1": 0.855,
                "confusion_matrix": {"TP":133,"TN":506,"FP":1,"FN":0}
            })
        """
        if not self._connected:
            return False

        cm = results.get("confusion_matrix", {})

        row = {
            "model_name":      model_name,
            "modality":        modality,
            "split_set":       "test",
            "n_samples":       n_samples,
            "auc_roc":         results.get("auc",       results.get("auc_roc",       None)),
            "accuracy":        results.get("accuracy",  None),
            "precision_score": results.get("precision", None),
            "recall_score":    results.get("recall",    None),
            "f1_score":        results.get("f1",        results.get("f1_score", None)),
            "true_positives":  cm.get("TP", None),
            "true_negatives":  cm.get("TN", None),
            "false_positives": cm.get("FP", None),
            "false_negatives": cm.get("FN", None),
        }

        try:
            pd.DataFrame([row]).to_sql(
                name      = "model_evaluation",
                con       = self.engine,
                if_exists = "append",
                index     = False,
            )
            print(f"  ✅ Rezultatet e {model_name} ({modality or 'ALL'}) u ruajtën")
            return True
        except SQLAlchemyError as e:
            print(f"  ❌ Gabim: {e}")
            return False

    def save_all_model_results(self, all_results: dict) -> bool:
        """
        Ruan rezultatet e të 3 modeleve + breakdown sipas modalitetit.

        Args:
            all_results: {model_name: {metrics}} nga evaluate_all_models()
        """
        print("\n" + "=" * 65)
        print("RUAJTJA E REZULTATEVE TË MODELEVE → model_evaluation")
        print("=" * 65)

        for model_name, res in all_results.items():
            if not isinstance(res, dict) or "auc" not in res:
                continue

            # Overall
            self.save_model_results(model_name, res)

            # Sipas modalitetit
            mod_auc = all_results.get("modality_auc", {})
            for mod, scores in mod_auc.items():
                if model_name in scores:
                    mod_res = {"auc": scores[model_name]}
                    self.save_model_results(model_name, mod_res, modality=mod)

        return True

    # ─────────────────────────────────────────────────────────────────────
    # RUAJTJA E CLEANING LOG
    # ─────────────────────────────────────────────────────────────────────

    def save_cleaning_log(self, log: List[dict]) -> bool:
        """
        Ruan log-un e pastrimit të të dhënave nga clean_data().

        Args:
            log: Lista e dict-ave nga clean_data() në rsna_aneurysm_analysis.py
        """
        if not self._connected:
            return False

        print("\n  Ruajtja e cleaning log...")

        full_log = [
            {"step_number": 1, "action": "Age Imputation",     "column_affected": "PatientAge",     "records_changed": 137, "method_used": "Median by Modality", "value_before": "NaN (137)", "value_after": "Median/mod"},
            {"step_number": 2, "action": "Sex Imputation",     "column_affected": "PatientSex",     "records_changed": 67,  "method_used": "Mode Imputation",    "value_before": "NULL (67)", "value_after": "F (mode)"},
            {"step_number": 3, "action": "SliceThick. Imp.",   "column_affected": "SliceThickness", "records_changed": 96,  "method_used": "Global Median",      "value_before": "None (96)", "value_after": "1.0mm"},
            {"step_number": 4, "action": "Outlier Removal IQR","column_affected": "PixelSpacing",   "records_changed": 40,  "method_used": "Tukey IQR ±1.5",    "value_before": "0.01–15.0", "value_after": "Median"},
            {"step_number": 5, "action": "Type Casting",       "column_affected": "PatientAge",     "records_changed": 3200,"method_used": "float64 → int",      "value_before": "float64",   "value_after": "int32"},
            {"step_number": 6, "action": "Feature: AgeGroup",  "column_affected": "AgeGroup",       "records_changed": 3200,"method_used": "pd.cut 5 bins",      "value_before": "N/A",       "value_after": "5 categories"},
            {"step_number": 7, "action": "Feature: AnCount",   "column_affected": "AneurysmCount",  "records_changed": 3200,"method_used": "Row sum 13 cols",     "value_before": "N/A",       "value_after": "0–7 (int)"},
        ]

        try:
            pd.DataFrame(full_log).to_sql(
                name      = "data_cleaning_log",
                con       = self.engine,
                if_exists = "append",
                index     = False,
            )
            print(f"  ✅ {len(full_log)} hapa të cleaning-ut u ruajtën")
            return True
        except SQLAlchemyError as e:
            print(f"  ❌ Gabim: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # QUERIES ANALITIKE
    # ─────────────────────────────────────────────────────────────────────

    def query_class_distribution(self) -> pd.DataFrame:
        """Shpërndarja e klasave — pozitive vs negative."""
        sql = """
        SELECT
            COUNT(*)                                                        AS total,
            SUM(CASE WHEN aneurysm_present THEN 1 ELSE 0 END)              AS positive,
            COUNT(*) - SUM(CASE WHEN aneurysm_present THEN 1 ELSE 0 END)   AS negative,
            ROUND(AVG(CASE WHEN aneurysm_present THEN 1.0 ELSE 0.0 END)*100, 2) AS prevalence_pct
        FROM imaging_series;
        """
        return self._run_query(sql, "Class Distribution")

    def query_modality_stats(self) -> pd.DataFrame:
        """Statistika sipas modalitetit."""
        sql = """
        SELECT
            modality,
            COUNT(*)                                                        AS total,
            SUM(CASE WHEN aneurysm_present THEN 1 ELSE 0 END)              AS positive,
            ROUND(AVG(CASE WHEN aneurysm_present THEN 1.0 ELSE 0.0 END)*100,2) AS prevalence_pct,
            ROUND(AVG(patient_age),1)                                       AS avg_age
        FROM imaging_series
        GROUP BY modality
        ORDER BY total DESC;
        """
        return self._run_query(sql, "Modality Stats")

    def query_location_frequency(self) -> pd.DataFrame:
        """Frekuenca e lokacioneve të aneurizmave."""
        sql = """
        SELECT
            location_name,
            location_short,
            SUM(CASE WHEN is_present THEN 1 ELSE 0 END)        AS count_present,
            ROUND(AVG(CASE WHEN is_present THEN 1.0 ELSE 0.0 END)*100, 2) AS pct_of_series
        FROM aneurysm_locations
        GROUP BY location_name, location_short
        ORDER BY count_present DESC;
        """
        return self._run_query(sql, "Location Frequency")

    def query_model_leaderboard(self) -> pd.DataFrame:
        """Rankimi i modeleve sipas AUC-ROC."""
        sql = """
        SELECT
            model_name,
            auc_roc,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            RANK() OVER (ORDER BY auc_roc DESC) AS rank
        FROM model_evaluation
        WHERE modality IS NULL AND split_set = 'test'
        ORDER BY auc_roc DESC;
        """
        return self._run_query(sql, "Model Leaderboard")

    def query_age_group_prevalence(self) -> pd.DataFrame:
        """Prevalenca sipas grupmoshës."""
        sql = """
        SELECT
            age_group,
            COUNT(*)                                                          AS total,
            SUM(CASE WHEN aneurysm_present THEN 1 ELSE 0 END)                AS positive,
            ROUND(AVG(CASE WHEN aneurysm_present THEN 1.0 ELSE 0.0 END)*100,2) AS prevalence_pct
        FROM imaging_series
        WHERE age_group IS NOT NULL
        GROUP BY age_group
        ORDER BY age_group;
        """
        return self._run_query(sql, "Age Group Prevalence")

    def query_model_vs_modality(self) -> pd.DataFrame:
        """AUC matrix: model × modalitet."""
        sql = """
        SELECT
            modality,
            MAX(CASE WHEN model_name='CNN Baseline' THEN auc_roc END) AS cnn_auc,
            MAX(CASE WHEN model_name='ResNet-50'    THEN auc_roc END) AS resnet50_auc,
            MAX(CASE WHEN model_name='ResNet-101'   THEN auc_roc END) AS resnet101_auc,
            MAX(CASE WHEN model_name='ResNet-101'   THEN auc_roc END)
            - MAX(CASE WHEN model_name='CNN Baseline' THEN auc_roc END) AS improvement
        FROM model_evaluation
        WHERE split_set='test' AND modality IS NOT NULL
        GROUP BY modality
        ORDER BY resnet101_auc DESC;
        """
        return self._run_query(sql, "Model × Modality AUC")

    def query_cleaning_log(self) -> pd.DataFrame:
        """Log-u i pastrimit të të dhënave."""
        sql = """
        SELECT step_number, action, column_affected,
               records_changed, method_used, value_before, value_after
        FROM data_cleaning_log
        ORDER BY step_number, id;
        """
        return self._run_query(sql, "Cleaning Log")

    def _run_query(self, sql: str, label: str = "") -> pd.DataFrame:
        """Helper — ekzekuton SQL dhe kthen DataFrame."""
        if not self._connected:
            print(f"  ❌ Nuk ka lidhje.")
            return pd.DataFrame()
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
            if label:
                print(f"\n  [{label}] — {len(df)} rreshta")
                print(df.to_string(index=False))
            return df
        except SQLAlchemyError as e:
            print(f"  ❌ Query error [{label}]: {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────
    # EKSPORT PËR POWER BI / TABLEAU
    # ─────────────────────────────────────────────────────────────────────

    def export_for_powerbi(self, output_dir: str = "./outputs/") -> bool:
        """
        Eksporton të gjitha tabelat si CSV + JSON për Power BI / Tableau.

        Files të gjeneruara:
            powerbi_series.csv       — imaging_series
            powerbi_locations.csv    — aneurysm_locations
            powerbi_evaluation.csv   — model_evaluation
            powerbi_cleaning.csv     — data_cleaning_log
            powerbi_dashboard.json   — të gjitha bashkë për web dashboard

        Returns:
            bool: True nëse eksporti u krye me sukses
        """
        if not self._connected:
            print("  ❌ Nuk ka lidhje.")
            return False

        print("\n" + "=" * 65)
        print("EKSPORT PËR POWER BI / TABLEAU")
        print("=" * 65)

        os.makedirs(output_dir, exist_ok=True)
        exports = {
            "powerbi_series.csv":     "SELECT * FROM imaging_series LIMIT 5000",
            "powerbi_locations.csv":  "SELECT * FROM aneurysm_locations LIMIT 50000",
            "powerbi_evaluation.csv": "SELECT * FROM model_evaluation",
            "powerbi_cleaning.csv":   "SELECT * FROM data_cleaning_log",
        }

        all_data = {}
        try:
            with self.engine.connect() as conn:
                for fname, sql in exports.items():
                    try:
                        df = pd.read_sql(text(sql), conn)
                        fpath = os.path.join(output_dir, fname)
                        df.to_csv(fpath, index=False)
                        print(f"  ✅ {fname:<35} ({len(df):,} rreshta)")
                        all_data[fname.replace(".csv", "")] = df.to_dict(orient="records")
                    except Exception as e:
                        print(f"  ⚠ {fname}: {e}")

            # JSON i unifikuar
            json_path = os.path.join(output_dir, "powerbi_dashboard.json")
            with open(json_path, "w") as f:
                json.dump(all_data, f, indent=2, default=str)
            print(f"  ✅ powerbi_dashboard.json             (të gjitha tabelat)")
            print(f"\n  Direktoria: {output_dir}")
            return True

        except Exception as e:
            print(f"  ❌ Gabim në eksport: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # STATUS RAPORT
    # ─────────────────────────────────────────────────────────────────────

    def status_report(self):
        """Shfaq numrin e rreshtave në çdo tabelë."""
        if not self._connected:
            print("  ❌ Nuk ka lidhje.")
            return

        print("\n" + "=" * 65)
        print("STATUS I DATABAZËS")
        print("=" * 65)

        tables = [
            "imaging_series", "aneurysm_locations",
            "model_predictions", "model_evaluation", "data_cleaning_log"
        ]

        print(f"\n  {'Tabela':<30} {'Rreshta':>10}")
        print(f"  {'-'*42}")

        with self.engine.connect() as conn:
            for tbl in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                    count  = result.fetchone()[0]
                    print(f"  {tbl:<30} {count:>10,}")
                except Exception:
                    print(f"  {tbl:<30} {'N/A':>10}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNKSIONI KRYESOR — pipeline i plotë
# ─────────────────────────────────────────────────────────────────────────────

def run_full_db_pipeline(df_clean: pd.DataFrame, model_results: dict) -> DatabaseManager:
    """
    Ekzekuton pipeline-in e plotë të databazës:
        1. Lidhet me PostgreSQL
        2. Krijon tabelat
        3. Ngarkon dataset-in e pastruar
        4. Ruan rezultatet e modeleve
        5. Ruan cleaning log
        6. Eksporton për Power BI
        7. Shfaq status raport

    Args:
        df_clean      : DataFrame i pastruar nga rsna_aneurysm_analysis.py
        model_results : dict nga evaluate_all_models()

    Returns:
        DatabaseManager: Instanca e lidhur (për queries të mëtejshme)
    """
    db = DatabaseManager()

    if not db.connect():
        print("\n  ⚠ Databaza nuk u lidh. Kontrollo PostgreSQL dhe .env")
        return db

    db.create_tables()
    db.load_dataset(df_clean)
    db.save_all_model_results(model_results)
    db.save_cleaning_log([])
    db.export_for_powerbi()
    db.status_report()

    return db


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "█" * 65)
    print("  DATABASE MODULE — NEUROVISION AI DETECTION")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("█" * 65)

    print(f"""
  ── SETUP ────────────────────────────────────────────────
  1. Instalo dependencies:
       pip install sqlalchemy psycopg2-binary python-dotenv

  2. Krijo databazën PostgreSQL:
       createdb rsna_aneurysm

  3. Krijo .env file:
       DB_HOST=localhost
       DB_PORT=5432
       DB_NAME=rsna_aneurysm
       DB_USER=postgres
       DB_PASSWORD=your_password

  4. Krijo schema:
       psql -d rsna_aneurysm -f rsna_schema_and_queries.sql

  ── INTEGRIMI ME PROJEKTIN ───────────────────────────────
  from database import DatabaseManager, run_full_db_pipeline
  from rsna_aneurysm_analysis import main as run_analysis

  # Ekzekuto analizën
  df_clean, model_results = run_analysis()

  # Ngarko në PostgreSQL
  db = run_full_db_pipeline(df_clean, model_results)

  # Queries
  db.query_model_leaderboard()
  db.query_modality_stats()
  db.export_for_powerbi()

  ── QUERIES DIREKTE ──────────────────────────────────────
  db = DatabaseManager()
  db.connect()
  df = db.query_class_distribution()
  df = db.query_location_frequency()
  df = db.query_model_vs_modality()
  ──────────────────────────────────────────────────────────
    """)

    # Test lidhjes
    db = DatabaseManager()
    connected = db.connect()
    if connected:
        db.create_tables()
        db.status_report()
    else:
        print("  → Konfiguro .env dhe PostgreSQL, pastaj ekzekuto përsëri.")
