from joblib import dump, load
import math
import numpy as np
from sklearn.linear_model import LogisticRegression

import psycopg2


def star_tackles_missed():
    '''       
    Calculates the total 1-star tackles missed
    
    For each play, fetch the probability of tackling the ball carrier from all defensive players
    and retrive the tacklers that had a probability of 0.95 or more but did not make the tackle

    Args: None
       
    Returns: None
    '''

    # Get plays
    model = load("distance.joblib")
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    for i in range(1, 6):
        cur.execute("UPDATE players SET star_{0}_missed=0".format(i))
    cur.execute("SELECT game_id, play_id, ball_carrier FROM plays")
    plays = cur.fetchall()

    for play in plays:
        cur.execute("SELECT x, y, lr, team FROM tracking WHERE game_id=%s AND play_id=%s AND player_id=%s "
                    "AND event='pass_arrived'", play)
        ball_carrier = cur.fetchone()
        if ball_carrier is None:
            continue
        cur.execute("SELECT player_id, x, y, speed, acceleration FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id "
            "NOT IN (SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
            "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[3]] + list(play[:2]))
        non_tacklers = cur.fetchall()
                
        # Calculate the distance of each tackler with the ball carrier, and predict the probability of tackle
        non_tacklers_x = [[math.sqrt(abs(ball_carrier[1] - tackler[2])),
                       math.sqrt(math.sqrt((ball_carrier[0] - tackler[1]) ** 2 + (ball_carrier[1] - tackler[2]) ** 2)),
                       tackler[3], tackler[4]]
                      for tackler in non_tacklers]
        tackler_probability = model.predict_proba(non_tacklers_x) if non_tacklers_x else []
        for i, tackler in enumerate(tackler_probability):
            if tackler[1] > 0.90:
                stars = "1"
            elif tackler[1] > 0.75:
                stars = "2"
            elif tackler[1] > 0.5:
                stars = "3"
            elif tackler[1] > 0.25:
                stars = "4"
            else:
                stars = "5"
            cur.execute("UPDATE players SET star_{0}_missed=star_{0}_missed+1 WHERE id=%s".format(stars),
                        (non_tacklers[i][0],))
    conn.commit()
    conn.close()


def star_tackles_made():
    '''       
    Calculate total star tackles made by each player"    
    
    For each play, fetch the probability of tackling the ball carrier from all defensive players.
    Each tackle is given a number of stars based on the probability of making that tackle, 
    then retrieve the total number star tackles a player has made.

    Args: None
       
    Returns: None
    '''
        
    # Get plays
    model = load("distance.joblib")
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    for i in range(1, 6):
        cur.execute("UPDATE players SET star_{0}_made=0".format(i))
    cur.execute("SELECT game_id, play_id, ball_carrier FROM plays")
    plays = cur.fetchall()

    for play in plays:
        cur.execute("SELECT x, y, lr, team FROM tracking WHERE game_id=%s AND play_id=%s AND player_id=%s "
                    "AND event='pass_arrived'", play)
        ball_carrier = cur.fetchone()
        if ball_carrier is None:
            continue
        cur.execute("SELECT player_id, x, y, speed, acceleration FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id "
            "IN (SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
            "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[3]] + list(play[:2]))
        tacklers = cur.fetchall()

        # Calculate the distance of each tackler with the ball carrier, and predict the probability of tackle
        tacklers_x = [[math.sqrt(abs(ball_carrier[1] - tackler[2])),
                       math.sqrt(math.sqrt((ball_carrier[0] - tackler[1]) ** 2 + (ball_carrier[1] - tackler[2]) ** 2)),
                       tackler[3], tackler[4]]
                      for tackler in tacklers]
        tackler_probability = model.predict_proba(tacklers_x) if tacklers_x else []

        # Classify each tackle by stars 1-5 based on the probability to make a tackle
        for i, tackler in enumerate(tackler_probability):
            if tackler[1] > 0.90:
                stars = "1"
            elif tackler[1] > 0.75:
                stars = "2"
            elif tackler[1] > 0.5:
                stars = "3"
            elif tackler[1] > 0.25:
                stars = "4"
            else:
                stars = "5"
            cur.execute("UPDATE players SET star_{0}_made=star_{0}_made+1 WHERE id=%s".format(stars),
                        (tacklers[i][0], ))
    conn.commit()
    conn.close()


def calculate_vector(magnitude: float, degrees: float):
    '''       
    Calculate the vector of a defender or ball_carrier's speed or acceleration

    Args:
        magnitude (float) - magnitude of player
        degrees (float) - degree the player facing       
        
    Returns: the player's speed or acceleration (float)
    '''

    actual_degrees = (450 - degrees) % 360
    rads = actual_degrees * math.pi / 180
    return [magnitude * math.cos(rads), magnitude * math.sin(rads)]


def expected_tackles():
    '''       
    Calculates the expected tackles for each player
    
    For each play, fetch the expectation of a selected player based on metric whether 
    the tackler is expected to tackle and whether the player actually tackled or not. 

    Args: None
       
    Returns: None
    '''

    # Get plays
    model = load("distance.joblib")
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    cur.execute("UPDATE players SET tackles_above_expected=0")
    cur.execute("UPDATE teams SET tackles_above_expected=0")
    cur.execute("SELECT game_id, play_id, ball_carrier FROM plays")
    plays = cur.fetchall()

    for play in plays:
        cur.execute("SELECT x, y, lr, team FROM tracking WHERE game_id=%s AND play_id=%s AND player_id=%s "
                    "AND event='pass_arrived'", play)
        ball_carrier = cur.fetchone()
        if ball_carrier is None:
            continue
        cur.execute("SELECT player_id, x, y, speed, acceleration, team FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id "
                    "IN (SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
                    "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[3]] + list(play[:2]))
        tacklers = cur.fetchall()
        cur.execute("SELECT player_id, x, y, speed, acceleration, team FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id "
                    "NOT IN (SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
                    "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[3]] + list(play[:2]))
        non_tacklers = cur.fetchall()

        # Calculate the distance of each player with the ball carrier, and predict the probability of tackle
        tacklers_x = [[math.sqrt(abs(ball_carrier[1] - tackler[2])),
                       math.sqrt(math.sqrt((ball_carrier[0] - tackler[1]) ** 2 + (ball_carrier[1] - tackler[2]) ** 2)),
                       tackler[3], tackler[4]]
                      for tackler in tacklers]
        non_tacklers_x = [[math.sqrt(abs(ball_carrier[1] - tackler[2])),
                           math.sqrt(math.sqrt((ball_carrier[0] - tackler[1]) ** 2 + (ball_carrier[1] - tackler[2]) ** 2)),
                           tackler[3], tackler[4]]
                          for tackler in non_tacklers]
        tackler_probability = model.predict_proba(tacklers_x) if tacklers_x else []
        non_tackler_probability = model.predict_proba(non_tacklers_x)
        
        # create two dictionaries between plays that the player tackled or did not, and update accordingly
        tp, ntp = {}, {}
        for i, tackler in enumerate(tackler_probability):
            tp[tacklers[i][0]] = tackler[1]
        for i, tackler in enumerate(non_tackler_probability):
            ntp[non_tacklers[i][0]] = tackler[1]
        ntp = dict(sorted(ntp.items(), key=lambda item: -item[1]))

        # update tackling expectation based on if the tackler is expected to tackle or not, and if the tackler actually tackled or not       
        if not tacklers:
            for nt in ntp:
                cur.execute("UPDATE players SET tackles_above_expected=tackles_above_expected-%s WHERE id=%s",
                            (ntp.get(nt), nt))
                cur.execute("UPDATE teams SET tackles_above_expected=tackles_above_expected-%s WHERE name=%s",
                            (ntp.get(nt), non_tacklers[0][5]))
                break
        else:
            for t in tp:
                cur.execute("UPDATE players SET tackles_above_expected=tackles_above_expected+%s WHERE id=%s",
                            (1 - tp.get(t), t))
                cur.execute("UPDATE teams SET tackles_above_expected=tackles_above_expected+%s WHERE name=%s",
                            (1 - tp.get(t), non_tacklers[0][5]))
            for nt in ntp:
                if ntp.get(nt) < max(tp.values()) + 0.25:
                    break
                cur.execute("UPDATE players SET tackles_above_expected=tackles_above_expected-%s WHERE id=%s",
                            (ntp.get(nt), nt))
                cur.execute("UPDATE teams SET tackles_above_expected=tackles_above_expected-%s WHERE name=%s",
                            (ntp.get(nt), non_tacklers[0][5]))
    conn.commit()
    conn.close()


def plot_distance():
    '''       
    Plot the distance between a defender and the ball-carrier    
    
    For each play, fetch the expectation of a selected player based on metric whether 
    the tackler is expected to tackle and whether the player actually tackled or not. 

    Args: None
       
    Returns: None
    '''

    # Get plays
    conn = psycopg2.connect("dbname=BigDataBowl user=cschneider")
    cur = conn.cursor()
    cur.execute("SELECT game_id, play_id, ball_carrier FROM plays")
    plays = cur.fetchall()
    vertical = [[], []]
    lateral = [[], []]
    distance = [[], []]
    players = [[], []]
    vec = [[], []]
    speed = [[], []]
    acceleration = [[], []]
    for play in plays:
        cur.execute("SELECT x, y, team, speed_x, speed_y FROM tracking WHERE game_id=%s AND play_id=%s AND player_id=%s "
                    "AND event='pass_arrived'", play)
        ball_carrier = cur.fetchone()
        if ball_carrier is None:
            continue
        cur.execute("SELECT x, y, speed, acceleration FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id IN "
                    "(SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
                    "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[2]] + list(play[:2]))
        players[0] = cur.fetchall()
        cur.execute("SELECT x, y, speed, acceleration FROM tracking WHERE game_id=%s AND play_id=%s AND team!=%s AND team!='FB' AND player_id NOT IN "
                    "(SELECT player_id FROM tackles WHERE game_id=%s AND play_id=%s AND (tackle='t' OR assist='t')) "
                    "AND event='pass_arrived'", list(play[:2]) + [ball_carrier[2]] + list(play[:2]))
        players[1] = cur.fetchall()
        np.seterr(all="raise")

        # calculate the distance from the veritical and lateral distance using pythagoras 
        for i in range(2):
            for player in players[i]:
                vertical[i].append(math.sqrt(abs(ball_carrier[0] - player[0])))
                lateral[i].append(math.sqrt(abs(ball_carrier[1] - player[1])))
                distance[i].append(math.sqrt(math.sqrt((ball_carrier[0] - player[0]) ** 2
                                                       + (ball_carrier[1] - player[1]) ** 2)))
                speed[i].append(player[2])
                acceleration[i].append(player[3])
   
    # plot the distance
    x = [i for i in zip(lateral[0] + lateral[1], distance[0] + distance[1], speed[0] + speed[1], acceleration[0] + acceleration[1])]
    y = [1] * len(distance[0]) + [0] * len(distance[1])
    print(np.corrcoef(vertical[0] + vertical[1], y))
    print(np.corrcoef(distance[0] + distance[1], y))
    print(np.corrcoef(acceleration[0] + acceleration[1], y))
    print(np.corrcoef(distance[0] + distance[1], vertical[0] + vertical[1]))
    model = LogisticRegression(class_weight="balanced").fit(x, y)
    print(sum(model.predict(x)))
    print(model.score(x, y))  # This is an F1-score
    print(model.coef_, model.intercept_)
    dump(model, "distance.joblib")


if __name__ == "__main__":
    star_tackles_missed()
    star_tackles_made()
