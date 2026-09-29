"""
2022 FIFA World Cup Final (Argentina vs France) - a beginner's tour of
football data analysis with statsbombpy (to get the data) and mplsoccer
(to draw pitches and networks).

This script produces three PNG charts:
  1. shot_map.png              - every shot in the match, sized by xG
  2. xg_timeline.png            - cumulative xG for each team, minute by minute
  3. pass_network_argentina.png - who passed to whom for Argentina's starting XI

It also prints two things to the console: a shot-by-shot table for the
whole match, and each team's total xG with and without penalties.

Every step below has a comment explaining WHY it is there, not just what it
does, since this is meant to be read and learned from.
"""

import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from mplsoccer import Pitch, football_shirt_marker, scatter_football
from mplsoccer.soccer.markers import HandlerFootball

from statsbombpy import sb

# statsbombpy warns on every call that "no credentials were supplied".
# That's expected: StatsBomb's open data is free and needs no login, so we
# silence the warning to keep the console output easy to read.
warnings.filterwarnings("ignore")

# Colors and pitch look are reused across all three charts so the set feels
# like one cohesive report. The pitch itself uses mplsoccer's textured
# "grass" background with mowing stripes, for a real-football-field look,
# and every shot is drawn as a little football shirt instead of a plain dot.
PITCH_KWARGS = dict(pitch_type="statsbomb", pitch_color="grass", line_color="white", stripe=True)
ARGENTINA_STYLE = dict(color="#6CABDD", edgecolor="#132257")  # sky blue shirt
FRANCE_STYLE = dict(color="#FFFFFF", edgecolor="#0055A4")      # white shirt


def find_final_match():
    """
    StatsBomb's open data is organised as:
        competition -> season -> match -> events
    So to get one specific match, we first look up the competition/season,
    then the match, then (later) its events. Doing this lookup in code
    instead of hardcoding a match_id shows how the data is structured and
    means the script keeps working even if StatsBomb reshuffles IDs.
    """
    competitions = sb.competitions()

    # Find the row for the men's World Cup, 2022 edition.
    world_cup_2022 = competitions[
        (competitions["competition_name"] == "FIFA World Cup")
        & (competitions["season_name"] == "2022")
    ].iloc[0]

    matches = sb.matches(
        competition_id=world_cup_2022["competition_id"],
        season_id=world_cup_2022["season_id"],
    )

    # Every StatsBomb match has a "competition_stage" - the final is the one we want.
    final_row = matches[matches["competition_stage"] == "Final"].iloc[0]

    return (
        int(final_row["match_id"]),
        final_row["home_team"],
        final_row["away_team"],
    )


def unpack_xy(df, location_col, x_col, y_col):
    """
    StatsBomb stores a pitch position as a single column holding a
    two-item list, e.g. location = [92.4, 30.0]. matplotlib/mplsoccer want
    plain numeric columns, so this helper splits that list into an x and a
    y column we can plot directly.
    """
    df[x_col] = df[location_col].apply(lambda point: point[0])
    df[y_col] = df[location_col].apply(lambda point: point[1])
    return df


# StatsBomb's "minute" column restarts at the top of every period: the
# second half always starts at minute 45 (not 46, 47, ... continuing on from
# first-half stoppage time), extra time's first half always starts at 90,
# and its second half always starts at 105. That matches how commentators
# talk ("45+2'", then "second half, 45:00"), but it means a deep
# stoppage-time minute from one period can be numerically BIGGER than an
# early minute of the next period - so sorting by raw minute alone can put
# events in the wrong order once you cross a half-time/full-time boundary.
PERIOD_START_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
PERIOD_END_MINUTE = {1: 45, 2: 90, 3: 105, 4: 120}


def format_broadcast_minute(period, minute):
    """
    Convert StatsBomb's period-relative minute into the "23'" / "45+2'"
    style you'd see on a TV scoreboard. StatsBomb's minute is zero-indexed
    (minute 0 is the first minute of play), so minute 22 is "the 23rd
    minute" - we add 1. Once minute reaches the scheduled end of that
    period (45/90/105/120), we switch to "+" stoppage-time notation instead
    of just letting the number climb past 90, 105, etc.
    """
    regulation_end = PERIOD_END_MINUTE[period]
    if minute < regulation_end:
        return f"{minute + 1}'"
    stoppage_minute = minute - regulation_end + 1
    return f"{regulation_end}+{stoppage_minute}'"


def build_continuous_clock(events):
    """
    Turn each period's "Half End" event into that period's real elapsed
    duration (including its own stoppage time), then use those durations to
    convert any (period, minute, second) into one continuous match clock,
    in minutes, that only ever counts upward. This is what lets us sort and
    add up shots from different periods in the order they actually
    happened, instead of by their period-relative minute.
    """
    half_ends = events[events["type"] == "Half End"].drop_duplicates("period")

    period_duration_seconds = {}
    for _, row in half_ends.iterrows():
        period = row["period"]
        elapsed = (row["minute"] - PERIOD_START_MINUTE[period]) * 60 + row["second"]
        period_duration_seconds[period] = elapsed

    # The offset for a period is simply the sum of every earlier period's
    # real duration - i.e. how much match time had already been played
    # before this period's kick-off.
    period_offset_seconds = {}
    running_total = 0
    for period in sorted(period_duration_seconds):
        period_offset_seconds[period] = running_total
        running_total += period_duration_seconds[period]

    def to_clock_minute(period, minute, second):
        elapsed_in_period = (minute - PERIOD_START_MINUTE[period]) * 60 + second
        return (period_offset_seconds[period] + elapsed_in_period) / 60

    return to_clock_minute, period_offset_seconds


def spread_overlapping_shots(df, x_col, y_col, gap=7):
    """
    A few shots - like two penalties re-spotted on the exact same penalty
    mark - can share the exact same pitch location, so their markers would
    stack precisely on top of one another and hide all but the last one
    drawn. This nudges shots within such a group a little further apart
    (sideways), purely so every marker stays visible - it's a drawing-only
    adjustment and doesn't change any underlying data.
    """
    df = df.copy()
    for _, idx in df.groupby([x_col, y_col]).groups.items():
        idx = list(idx)
        n = len(idx)
        if n < 2:
            continue
        offsets = (np.arange(n) - (n - 1) / 2) * gap
        df.loc[idx, y_col] = df.loc[idx, y_col] + offsets
    return df


def build_display_names(lineups):
    """
    One player_name -> short display name lookup covering both teams,
    preferring each player's StatsBomb nickname (e.g. "Lionel Messi") over
    their full legal name - useful since Spanish/French full names often
    include extra surnames a viewer wouldn't recognise at a glance.
    """
    display_name = {}
    for team_lineup in lineups.values():
        for _, row in team_lineup.iterrows():
            display_name[row["player_name"]] = (
                row["player_nickname"] if pd.notna(row["player_nickname"]) else row["player_name"]
            )
    return display_name


def add_footer(fig, how_to_read, wrap_width=100):
    """
    A short "how to read this" caption plus a data-source credit, printed
    on every chart so each PNG can stand on its own outside this script.
    A negative y pushes the text below the axes entirely (pitch charts fill
    almost the whole figure), and bbox_inches="tight" on save then expands
    the saved image to include it, instead of the pitch's own border line
    cutting through the caption. The caption is wrapped onto multiple lines
    so a longer note can't run into the "Data: StatsBomb" credit sharing
    that same row.
    """
    wrapped = textwrap.fill(f"How to read: {how_to_read}", width=wrap_width)
    fig.text(0.01, -0.03, wrapped, fontsize=8.5, color="#444444", ha="left", va="top")
    fig.text(0.99, -0.03, "Data: StatsBomb",
              fontsize=8.5, color="#444444", ha="right", va="top", style="italic")


# ---------------------------------------------------------------------------
# Chart 1: Shot map for both teams, marker size = xG
# ---------------------------------------------------------------------------
def make_shot_map(events, home_team, away_team, display_name, save_path):
    """
    xG (expected goals) is StatsBomb's estimate of how likely a shot was to
    result in a goal, based on things like distance, angle and shot type.
    A shot map plots every shot at the pitch location it was taken from,
    and we use a bigger marker for a higher xG so the "best chances" jump
    out visually.
    """
    # Keep only shots taken during normal play/extra-time (period 1-4).
    # StatsBomb also logs the penalty shoot-out as "Shot" events (period 5),
    # but those aren't in-game shots and don't belong on a shot map.
    shots = events[(events["type"] == "Shot") & (events["period"] < 5)].copy()
    shots = unpack_xy(shots, "location", "x", "y")

    # StatsBomb already records each team's OWN events as if that team is
    # always attacking from left to right, regardless of which real end of
    # the pitch they were defending at the time - so both teams' raw shots
    # already point the same way (towards x=120). We mirror the away team's
    # coordinates (flip both x and y) purely so the two teams land on
    # opposite halves of one shared pitch image, which makes them easy to
    # compare side by side. It's a drawing choice, not a reflection of who
    # was attacking which end of the real pitch.
    is_away = shots["team"] == away_team
    shots.loc[is_away, "x"] = 120 - shots.loc[is_away, "x"]
    shots.loc[is_away, "y"] = 80 - shots.loc[is_away, "y"]

    # A couple of shots - here, two penalties re-spotted on the same
    # penalty mark - land on the exact same pitch location. Spread any such
    # shots a little apart so every marker stays visible (see the
    # function's docstring for why).
    shots = spread_overlapping_shots(shots, "x", "y")

    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor("white")

    team_styles = {home_team: ARGENTINA_STYLE, away_team: FRANCE_STYLE}

    for team, style in team_styles.items():
        team_shots = shots[shots["team"] == team]

        # Every shot is drawn as a little football shirt using mplsoccer's
        # built-in shirt marker. Marker size scales with xG: size = a base
        # size + xG * a scale factor, so even a very low-xG shot is still
        # visible, and a high-xG shot (like a close-range chance) is clearly
        # bigger.
        pitch.scatter(
            team_shots["x"], team_shots["y"],
            s=150 + team_shots["shot_statsbomb_xg"] * 2200,
            marker=football_shirt_marker,
            c=style["color"], edgecolors=style["edgecolor"],
            linewidth=1.2, alpha=0.9, ax=ax, zorder=2,
        )

    # Flag every goal with a small football icon placed just OFF to the side
    # of its shot marker - not on top of it - so the flag never covers up
    # the shirt underneath and hides that shot's xG-sized marker. We nudge
    # the flag towards the centre circle (so it stays on the pitch, clear of
    # the goal frame) and slightly "up", away from the shot itself.
    goals = shots[shots["shot_outcome"] == "Goal"].copy()
    goals["flag_x"] = goals["x"] + np.where(goals["x"] > 60, -7, 7)
    goals["flag_y"] = goals["y"] - 7

    # A thin line from the shot to its flag makes the link between them
    # unambiguous - without it, a flag can look like it belongs to whichever
    # shot happens to be nearest, especially once shots have been nudged
    # apart by spread_overlapping_shots.
    for _, goal in goals.iterrows():
        ax.plot(
            [goal["x"], goal["flag_x"]], [goal["y"], goal["flag_y"]],
            color="black", linewidth=1, alpha=0.8, zorder=3,
        )

    goal_hexagons, _ = scatter_football(goals["flag_x"], goals["flag_y"], ax=ax, s=260, zorder=4)

    # Label every goal with the scorer's name and a broadcast-style match
    # minute (e.g. "23'", or "45+2'" during stoppage time), so all six goals
    # in this match are individually identifiable, not just visible dots.
    for _, goal in goals.iterrows():
        label = (
            f"{display_name.get(goal['player'], goal['player'])} "
            f"{format_broadcast_minute(goal['period'], goal['minute'])}"
        )
        goal_text = ax.annotate(
            label, xy=(goal["flag_x"], goal["flag_y"]), xytext=(0, 11),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black", zorder=5,
        )
        goal_text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    # Build a small custom legend by hand (rather than from the scatter calls
    # above) so it only shows 3 clean entries instead of several duplicates.
    # The "Goal" entry reuses the real football scatter plot as its handle
    # (via mplsoccer's HandlerFootball) so the legend icon matches the pitch.
    legend_handles = [
        Line2D([0], [0], marker=football_shirt_marker, color="none",
               markerfacecolor=ARGENTINA_STYLE["color"], markeredgecolor=ARGENTINA_STYLE["edgecolor"],
               markersize=16, label=f"{home_team} shot"),
        Line2D([0], [0], marker=football_shirt_marker, color="none",
               markerfacecolor=FRANCE_STYLE["color"], markeredgecolor=FRANCE_STYLE["edgecolor"],
               markersize=16, label=f"{away_team} shot"),
        goal_hexagons,
    ]
    legend_labels = [f"{home_team} shot", f"{away_team} shot", "Goal"]
    legend = ax.legend(handles=legend_handles, labels=legend_labels, loc="upper left",
                        labelcolor="black", framealpha=0.85, fontsize=10, facecolor="white",
                        handler_map={goal_hexagons: HandlerFootball()})
    legend.get_frame().set_edgecolor("black")

    ax.set_title(
        f"{home_team} vs {away_team} - Shot Map (marker size = xG)\n"
        "2022 FIFA World Cup Final - shoot-out penalties excluded",
        color="black", fontsize=13, pad=14,
    )
    add_footer(
        fig,
        "shirt size = shot's xG (chance quality); a line links each football flag to the goal it marks, "
        "labelled with scorer and minute; shots sharing the exact same spot are nudged apart so every one is visible.",
    )

    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Chart 2: Cumulative xG timeline by minute
# ---------------------------------------------------------------------------
def make_xg_timeline(events, home_team, away_team, save_path):
    """
    Instead of looking at one shot at a time, this chart adds up each team's
    xG as the match goes on, so we can see who was creating the better
    chances at each stage of the game (a "running total" or "step" chart).
    """
    # Build the continuous match clock described above, and use it instead
    # of the raw "minute" column so shots sort in the order they actually
    # happened, even across a half-time/full-time boundary.
    to_clock_minute, period_offset_seconds = build_continuous_clock(events)

    shots = events[(events["type"] == "Shot") & (events["period"] < 5)].copy()
    shots["clock"] = shots.apply(
        lambda r: to_clock_minute(r["period"], r["minute"], r["second"]), axis=1
    )

    # The match's last shot on the continuous clock - used so both teams'
    # lines are drawn out to the same point on the x-axis.
    final_clock = shots["clock"].max()

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    team_colors = {home_team: ARGENTINA_STYLE["color"], away_team: "#0055A4"}

    for team in (home_team, away_team):
        team_shots = shots[shots["team"] == team].sort_values("clock")

        # A running total of xG: start at (minute 0, xG 0), add each shot's
        # xG at the minute it happened, and hold that value flat until the
        # next shot (that's what a "step" plot does, vs. a straight line).
        cum_xg = team_shots["shot_statsbomb_xg"].cumsum()
        clocks = [0] + team_shots["clock"].tolist() + [final_clock]
        values = [0] + cum_xg.tolist() + [cum_xg.iloc[-1] if len(cum_xg) else 0]

        ax.step(clocks, values, where="post", color=team_colors[team],
                 linewidth=2.5, label=f"{team} (final: {values[-1]:.2f} xG)")

        # Mark actual goals on top of the running total with the same
        # football icon used on the shot map, so we can see how a team's
        # score compares to the chances it created. scatter_football works
        # on any matplotlib axes, not just an mplsoccer pitch.
        goal_mask = team_shots["shot_outcome"] == "Goal"
        scatter_football(
            team_shots.loc[goal_mask, "clock"], cum_xg.loc[goal_mask],
            ax=ax, s=260, zorder=5,
        )

    # Dashed reference lines at the real boundary between periods (the exact
    # moment one ends and the next kicks off), using the same period offsets
    # that built the continuous clock - not just the last shot's minute.
    period_labels = {2: "Half-time", 3: "Full-time", 4: "End of ET1"}
    top_of_chart = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    for period, label in period_labels.items():
        boundary_clock = period_offset_seconds[period] / 60
        ax.axvline(boundary_clock, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(boundary_clock, top_of_chart, label, color="grey",
                rotation=90, va="top", ha="right", fontsize=8)

    # The x-axis is the continuous clock, not the raw broadcast minute, so a
    # plain "20, 40, 60, ..." axis would be misleading (e.g. Mbappé's 80'
    # goal actually sits at clock-minute ~87, after first-half stoppage time
    # is folded in). Instead, we only label the clock-minute where each
    # period *kicks off* at its nominal broadcast minute (0, 45, 90, 105),
    # plus where the 120th minute would fall - the same to_clock_minute
    # conversion used for the goals and the dashed period lines above.
    nominal_tick_minutes = [(1, 0), (2, 45), (3, 90), (4, 105), (4, 120)]
    tick_positions = [to_clock_minute(period, minute, 0) for period, minute in nominal_tick_minutes]
    tick_labels = [f"{minute}'" for _, minute in nominal_tick_minutes]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Match time (including stoppage time)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title(
        f"{home_team} vs {away_team} - Cumulative xG Timeline\n"
        "2022 FIFA World Cup Final - footballs mark actual goals, shoot-out excluded",
        fontsize=13,
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    # A little breathing room after the last shot, so its marker isn't cut
    # off right at the edge of the chart.
    ax.set_xlim(0, final_clock + 3)
    ax.set_ylim(bottom=0)

    add_footer(
        fig,
        "each step is a team's running total of shot quality (xG); football icons mark actual goals.",
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Chart 3: Pass network for Argentina
# ---------------------------------------------------------------------------
def make_pass_network(events, display_name, team, save_path):
    """
    A pass network summarises an entire match into one picture:
      - a dot for each player, placed at their average on-pitch position
      - a line between two players, thicker the more they passed to each other
    We only use passes made before the team's FIRST substitution, so every
    pass in the chart was made by the original starting eleven - mixing in
    a substitute would distort the "average position" of whoever they
    replaced.
    """
    team_subs = events[(events["type"] == "Substitution") & (events["team"] == team)]
    first_sub_minute = team_subs["minute"].min() if not team_subs.empty else events["minute"].max()

    passes = events[
        (events["type"] == "Pass")
        & (events["team"] == team)
        & (events["minute"] < first_sub_minute)
    ].copy()

    # In StatsBomb data, a successful pass has NO value in "pass_outcome"
    # (it's only filled in for incomplete, out-of-play, offside, etc. passes).
    # A pass network should only show passes that actually reached a teammate.
    passes = passes[passes["pass_outcome"].isna()]

    passes = unpack_xy(passes, "location", "x", "y")
    passes = unpack_xy(passes, "pass_end_location", "end_x", "end_y")

    # --- Average position of each player ---
    # We approximate where a player operated by averaging every spot they
    # passed FROM, together with every spot they received a pass AT.
    avg_rows = []
    for player in passes["player"].unique():
        made = passes[passes["player"] == player]
        received = passes[passes["pass_recipient"] == player]
        all_x = pd.concat([made["x"], received["end_x"]])
        all_y = pd.concat([made["y"], received["end_y"]])
        avg_rows.append({"player": player, "x": all_x.mean(), "y": all_y.mean(), "passes_made": len(made)})
    avg_locations = pd.DataFrame(avg_rows)

    # --- How many times each pair of players passed to one another ---
    # We treat "A to B" and "B to A" as the same connection (sorted pair),
    # since the line on the chart represents the link between them, not
    # the direction of any single pass.
    passes["pair"] = passes.apply(
        lambda r: tuple(sorted([r["player"], r["pass_recipient"]])), axis=1
    )
    pair_counts = passes.groupby("pair").size().reset_index(name="pass_count")
    # Drop one-off passes so the busiest connections aren't lost in clutter.
    pair_counts = pair_counts[pair_counts["pass_count"] >= 2]

    pitch = Pitch(**PITCH_KWARGS)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor("white")

    # Draw the connecting lines first, so the player dots sit on top of them.
    for _, row in pair_counts.iterrows():
        player_a, player_b = row["pair"]
        loc_a = avg_locations.loc[avg_locations["player"] == player_a]
        loc_b = avg_locations.loc[avg_locations["player"] == player_b]
        if loc_a.empty or loc_b.empty:
            continue
        pitch.lines(
            loc_a["x"].values[0], loc_a["y"].values[0],
            loc_b["x"].values[0], loc_b["y"].values[0],
            lw=row["pass_count"] * 0.9, color="white", alpha=0.65, zorder=1, ax=ax,
        )

    # Draw a dot for each player - bigger dot = that player made more passes.
    pitch.scatter(
        avg_locations["x"], avg_locations["y"],
        s=avg_locations["passes_made"] * 20 + 250,
        color=ARGENTINA_STYLE["color"], edgecolors=ARGENTINA_STYLE["edgecolor"],
        linewidth=1.5, zorder=2, ax=ax,
    )

    # Label each dot with the player's short name. A black outline around the
    # white text (a "halo") keeps names readable even where a thick white
    # pass line crosses right behind them.
    for _, row in avg_locations.iterrows():
        text = ax.annotate(
            display_name.get(row["player"], row["player"]),
            xy=(row["x"], row["y"]), xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold", zorder=3,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])

    ax.set_title(
        f"{team} - Pass Network (2022 World Cup Final)\n"
        f"Passes before the first substitution (before minute {first_sub_minute:.0f})",
        color="black", fontsize=13, pad=14,
    )
    add_footer(fig, "dot size = passes made; line thickness = passes between two players (min. 2).")

    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Console report: a shot-by-shot table, plus each team's xG totals
# ---------------------------------------------------------------------------
def build_shot_table(events, display_name):
    """
    One row per in-game shot (the penalty shoot-out is excluded, same as on
    every chart above), with a broadcast-style "23'" minute instead of
    StatsBomb's raw period-relative one - handy for sanity-checking the
    charts, or for exploring the match without opening any image.
    """
    shots = events[(events["type"] == "Shot") & (events["period"] < 5)].copy()

    # Sort by when each shot actually happened - not by minute/second alone,
    # which (as build_continuous_clock's docstring explains) can put a
    # deep-stoppage-time shot after an early-next-period shot even though it
    # happened first.
    to_clock_minute, _ = build_continuous_clock(events)
    shots["_clock"] = shots.apply(
        lambda r: to_clock_minute(r["period"], r["minute"], r["second"]), axis=1
    )
    shots = shots.sort_values("_clock")

    table = pd.DataFrame({
        "Minute": shots.apply(lambda r: format_broadcast_minute(r["period"], r["minute"]), axis=1),
        "Team": shots["team"],
        "Player": shots["player"].map(lambda p: display_name.get(p, p)),
        "xG": shots["shot_statsbomb_xg"],
        "Shot Type": shots["shot_type"],
        "Outcome": shots["shot_outcome"],
    })
    return table.reset_index(drop=True)


def print_shot_table(events, display_name):
    table = build_shot_table(events, display_name)
    print("\nAll shots in the match (penalty shoot-out excluded):")
    print(table.to_string(index=False, formatters={"xG": "{:.2f}".format}))


def print_team_xg_totals(events, home_team, away_team):
    """
    A penalty's xG (StatsBomb gives every penalty the same ~0.78) can make a
    team's total look better than the run-of-play chances they actually
    created, so we print the total both with and without penalties included.
    """
    shots = events[(events["type"] == "Shot") & (events["period"] < 5)]
    print("\nTeam xG totals (penalty shoot-out excluded):")
    for team in (home_team, away_team):
        team_shots = shots[shots["team"] == team]
        total_xg = team_shots["shot_statsbomb_xg"].sum()
        non_penalty_xg = team_shots.loc[team_shots["shot_type"] != "Penalty", "shot_statsbomb_xg"].sum()
        print(f"  {team}: {total_xg:.2f} xG total, {non_penalty_xg:.2f} xG excluding penalties")


def main():
    print("Looking up the 2022 World Cup final...")
    match_id, home_team, away_team = find_final_match()
    print(f"Found match {match_id}: {home_team} vs {away_team}")

    print("Downloading match events from StatsBomb open data...")
    events = sb.events(match_id=match_id)

    print("Downloading team lineups (needed for player names on the charts)...")
    lineups = sb.lineups(match_id=match_id)
    display_name = build_display_names(lineups)

    print_shot_table(events, display_name)
    print_team_xg_totals(events, home_team, away_team)

    make_shot_map(events, home_team, away_team, display_name, "shot_map.png")
    make_xg_timeline(events, home_team, away_team, "xg_timeline.png")
    make_pass_network(events, display_name, home_team, "pass_network_argentina.png")

    print("\nDone! Check shot_map.png, xg_timeline.png and pass_network_argentina.png")


if __name__ == "__main__":
    main()
