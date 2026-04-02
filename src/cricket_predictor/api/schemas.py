from typing import Literal

from pydantic import BaseModel, Field


class AutoMatchPredictionRequest(BaseModel):
    """Minimal request — recent form, batting/bowling strength are pulled
    automatically from the live ESPN Cricinfo points table."""

    team_a: str = Field(..., description="Full name, short code, or alias  e.g. 'SRH' or 'Sunrisers Hyderabad'")
    team_b: str = Field(..., description="Full name, short code, or alias  e.g. 'KKR' or 'Kolkata Knight Riders'")
    venue: str
    match_format: Literal["T20", "ODI", "Test"] = "T20"
    pitch_type: Literal["batting", "bowling", "balanced"] = "balanced"
    toss_winner: str = Field(..., description="Same aliases accepted as team_a/team_b")
    toss_decision: Literal["bat", "bowl"] = "bat"
    head_to_head_win_pct_team_a: float = Field(default=0.5, ge=0.0, le=1.0)
    # Optional match condition overrides
    dew_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="0=no dew, 1=heavy dew (night match)")
    pitch_batting_bias: float = Field(default=0.0, ge=-1.0, le=1.0, description="-1=bowler friendly, +1=batting friendly")
    night_match: bool = Field(default=True, description="True for day-night / floodlit match")


class MatchPredictionRequest(BaseModel):
    team_a: str = Field(..., description="Home or first-listed team")
    team_b: str = Field(..., description="Away or second-listed team")
    venue: str
    match_format: Literal["T20", "ODI", "Test"]
    pitch_type: Literal["batting", "bowling", "balanced"]
    toss_winner: str
    toss_decision: Literal["bat", "bowl"]
    team_a_recent_form: float = Field(..., ge=0.0, le=1.0)
    team_b_recent_form: float = Field(..., ge=0.0, le=1.0)
    team_a_batting_strength: float = Field(..., ge=0.0, le=100.0)
    team_b_batting_strength: float = Field(..., ge=0.0, le=100.0)
    team_a_bowling_strength: float = Field(..., ge=0.0, le=100.0)
    team_b_bowling_strength: float = Field(..., ge=0.0, le=100.0)
    head_to_head_win_pct_team_a: float = Field(..., ge=0.0, le=1.0)
    venue_advantage_team_a: float = Field(..., ge=-1.0, le=1.0)
    # Optional match condition overrides
    dew_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="0=no dew, 1=heavy dew (night match)")
    pitch_batting_bias: float = Field(default=0.0, ge=-1.0, le=1.0, description="-1=bowler friendly, +1=batting friendly")
    night_match: bool = Field(default=True, description="True for day-night / floodlit match")


class PlayerPredictionRequest(BaseModel):
    player_name: str
    team: str
    opponent_team: str
    venue: str
    match_format: Literal["T20", "ODI", "Test"]
    pitch_type: Literal["batting", "bowling", "balanced"]
    batting_position: int = Field(..., ge=1, le=11)
    career_average: float = Field(..., ge=0.0)
    strike_rate: float = Field(..., ge=0.0)
    recent_form_runs: float = Field(..., ge=0.0)
    opponent_bowling_strength: float = Field(..., ge=0.0, le=100.0)
    venue_batting_average: float = Field(..., ge=0.0)
    # Optional match condition overrides
    dew_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    pitch_batting_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    night_match: bool = Field(default=True)
