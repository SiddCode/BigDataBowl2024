from datetime import datetime
import pandas as pd
from psycopg2.extensions import AsIs, register_adapter

from calc_vectors import calculate_vector
from chart_play import *


def adapt_int64(int64):
    return AsIs(int64)


def adapt_bool(boolean):
    return AsIs(boolean)


def upload_games(path):
    """Upload games to psql db"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    df = pd.read_csv(path)
    for i in range(len(df)):
        cur.execute("""INSERT INTO games VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (df["gameId"][i], df["week"][i], datetime.strptime(df["gameDate"][i], "%m/%d/%Y"),
                     df["gameTimeEastern"][i], df["homeTeamAbbr"][i], df["visitorTeamAbbr"][i], df["homeFinalScore"][i],
                     df["visitorFinalScore"][i]))
    conn.commit()


def upload_players(path):
    """Upload players to psql db"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    df = pd.read_csv(path)
    for i in range(len(df)):
        cur.execute("""INSERT INTO players VALUES (%s, %s, %s, %s, %s, %s)""",
                    (df["nflId"][i], df["height"][i], df["weight"][i], df["collegeName"][i], df["position"][i],
                     df["displayName"][i]))
    conn.commit()


def upload_plays(path):
    """Upload plays to psql db"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    df = pd.read_csv(path)
    df = df.drop(["ballCarrierDisplayName", "yardlineSide", "yardlineNumber", "foulNFLId1", "foulNFLId2"], axis=1)
    col = df.pop("absoluteYardlineNumber")
    df.insert(9, col.name, col)
    for col in ["passLength", "penaltyYards", "foulName1", "foulName2"]:
        for i in range(len(df[col])):
            if df.loc[i, col] == "N/A":
                df.loc[i, col] = None
    for i in range(len(df["playNullifiedByPenalty"])):
        df.loc[i, "playNullifiedByPenalty"] = df.loc[i, "playNullifiedByPenalty"] == "Y"
    df = df.astype(object).replace(np.nan, None)
    for i in range(len(df)):
        cur.execute("INSERT INTO plays VALUES (" + "%s, " * 29 + "%s)", [df[col][i] for col in df.columns])
    conn.commit()


def upload_tackles(path):
    """Upload tackle data to psql db"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    df = pd.read_csv(path)
    for i in range(len(df)):
        cur.execute("INSERT INTO tackles VALUES (%s, %s, %s, CAST(%s as BOOL), CAST(%s as BOOL), CAST(%s as BOOL),"
                    " CAST(%s as BOOL))", [df[col][i] for col in df.columns])
    conn.commit()


def upload_tracking(path):
    """Upload tracking data to psql db"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    df = pd.read_csv(path)
    df = df.drop(["displayName", "jerseyNumber"], axis=1)
    for col in ["o", "dir"]:
        for i in range(len(df[col])):
            if df.loc[i, col] == "NA":
                df.loc[i, col] = None
    col = df["nflId"].astype(object).replace(np.nan, 1)
    df["nflId"] = col
    df = df.astype(object).replace(np.nan, None)
    for i in range(len(df)):
        data = [df[col][i] for col in df.columns]
        if data[5] == "football":
            data[5] = "FB"
        cur.execute("INSERT INTO tracking VALUES (" + "%s, " * 14 + "%s)", data)
    conn.commit()


def compute_sa_vectors():
    """Compute speed/acceleration vectors for tracking data"""
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    cur.execute("SELECT game_id, play_id, player_id, frame_id, speed, acceleration, direction FROM tracking "
                "WHERE player_id!=1 AND event='handoff'")
    data = cur.fetchall()
    for frame in data:
        speed_x, speed_y = calculate_vector(frame[4], frame[6])
        acc_x, acc_y = calculate_vector(frame[5], frame[6])
        cur.execute("UPDATE tracking SET speed_x=%s, speed_y=%s, acc_x=%s, acc_y=%s "
                    "WHERE game_id=%s AND play_id=%s AND player_id=%s AND frame_id=%s",
                    [speed_x, speed_y, acc_x, acc_y] + list(frame[:4]))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    compute_sa_vectors()
