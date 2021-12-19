import logging
import copy
from enum import Enum, auto
from random import uniform

from millify import millify

from TwitchChannelPointsMiner.utils import char_decision_as_index, float_round

logger = logging.getLogger(__name__)


class Strategy(Enum):
    MOST_VOTED = auto()
    HIGH_ODDS = auto()
    PERCENTAGE = auto()
    SMART = auto()
    GENSHIN = auto()

    def __str__(self):
        return self.name


class Condition(Enum):
    GT = auto()
    LT = auto()
    GTE = auto()
    LTE = auto()

    def __str__(self):
        return self.name


class OutcomeKeys(object):
    # Real key on Bet dict ['']
    PERCENTAGE_USERS = "percentage_users"
    ODDS_PERCENTAGE = "odds_percentage"
    ODDS = "odds"
    TOP_POINTS = "top_points"
    # Real key on Bet dict [''] - Sum()
    TOTAL_USERS = "total_users"
    TOTAL_POINTS = "total_points"
    # This key does not exist
    DECISION_USERS = "decision_users"
    DECISION_POINTS = "decision_points"


class DelayMode(Enum):
    FROM_START = auto()
    FROM_END = auto()
    PERCENTAGE = auto()

    def __str__(self):
        return self.name


class FilterCondition(object):
    __slots__ = [
        "by",
        "where",
        "value",
    ]

    def __init__(self, by=None, where=None, value=None, decision=None):
        self.by = by
        self.where = where
        self.value = value

    def __repr__(self):
        return f"FilterCondition(by={self.by.upper()}, where={self.where}, value={self.value})"


class BetSettings(object):
    __slots__ = [
        "strategy",
        "percentage",
        "percentage_genshin",
        "genshin_chances",
        "percentage_gap",
        "max_points",
        "minimum_points",
        "stealth_mode",
        "filter_condition",
        "delay",
        "delay_mode",
    ]

    def __init__(
        self,
        strategy: Strategy = None,
        percentage: int = None,
        percentage_genshin: int = None,
        genshin_chances: dict[str:int] = None,
        percentage_gap: int = None,
        max_points: int = None,
        minimum_points: int = None,
        stealth_mode: bool = None,
        filter_condition: FilterCondition = None,
        delay: float = None,
        delay_mode: DelayMode = None,
    ):
        self.strategy = strategy
        self.percentage = percentage
        self.percentage_genshin = percentage_genshin
        self.genshin_chances = genshin_chances
        self.percentage_gap = percentage_gap
        self.max_points = max_points
        self.minimum_points = minimum_points
        self.stealth_mode = stealth_mode
        self.filter_condition = filter_condition
        self.delay = delay
        self.delay_mode = delay_mode
        self.default()

    def default(self):
        self.strategy = self.strategy or Strategy.SMART
        self.percentage = self.percentage or 5
        self.percentage_genshin = self.percentage_genshin or self.percentage
        self.genshin_chances = self.genshin_chances or {}
        for t, c in {
            "artifact": 7,
            "fortune": 18.75,
            "weekly_boss_items": 40.6,
            "boss_2_or_3_mats": 45,
            "weekly_boss_3_mats": 40,
        }.items():
            self.genshin_chances.setdefault(t, c)
        self.percentage_gap = self.percentage_gap or 20
        self.max_points = self.max_points or 50000
        self.minimum_points = self.minimum_points or 0
        self.stealth_mode = self.stealth_mode or False
        self.delay = self.delay or 6
        self.delay_mode = self.delay_mode or DelayMode.FROM_END

    def __repr__(self):
        return f"BetSettings(strategy={self.strategy}, percentage={self.percentage}, percentage_genshin={self.percentage_genshin}, genshin_chances={self.genshin_chances}, percentage_genshin={self.percentage_genshin}, percentage_gap={self.percentage_gap}, max_points={self.max_points}, minimum_points={self.minimum_points}, stealth_mode={self.stealth_mode})"


class Bet(object):
    __slots__ = ["outcomes", "decision", "total_users", "total_points", "settings"]

    def __init__(self, outcomes: list, settings: BetSettings, title: str = ""):
        self.outcomes = outcomes
        self.__clear_outcomes()
        self.decision: dict = {}
        self.total_users = 0
        self.total_points = 0
        self.settings = settings

    def update_outcomes(self, outcomes):
        for index in range(0, len(self.outcomes)):
            self.outcomes[index][OutcomeKeys.TOTAL_USERS] = int(
                outcomes[index][OutcomeKeys.TOTAL_USERS]
            )
            self.outcomes[index][OutcomeKeys.TOTAL_POINTS] = int(
                outcomes[index][OutcomeKeys.TOTAL_POINTS]
            )
            if outcomes[index]["top_predictors"] != []:
                # Sort by points placed by other users
                outcomes[index]["top_predictors"] = sorted(
                    outcomes[index]["top_predictors"],
                    key=lambda x: x["points"],
                    reverse=True,
                )
                # Get the first elements (most placed)
                top_points = outcomes[index]["top_predictors"][0]["points"]
                self.outcomes[index][OutcomeKeys.TOP_POINTS] = top_points

        self.total_users = (
            self.outcomes[0][OutcomeKeys.TOTAL_USERS]
            + self.outcomes[1][OutcomeKeys.TOTAL_USERS]
        )
        self.total_points = (
            self.outcomes[0][OutcomeKeys.TOTAL_POINTS]
            + self.outcomes[1][OutcomeKeys.TOTAL_POINTS]
        )

        if (
            self.total_users > 0
            and self.outcomes[0][OutcomeKeys.TOTAL_POINTS] > 0
            and self.outcomes[1][OutcomeKeys.TOTAL_POINTS] > 0
        ):
            for index in range(0, len(self.outcomes)):
                self.outcomes[index][OutcomeKeys.PERCENTAGE_USERS] = float_round(
                    (100 * self.outcomes[index][OutcomeKeys.TOTAL_USERS])
                    / self.total_users
                )
                self.outcomes[index][OutcomeKeys.ODDS] = float_round(
                    self.total_points / self.outcomes[index][OutcomeKeys.TOTAL_POINTS]
                )
                self.outcomes[index][OutcomeKeys.ODDS_PERCENTAGE] = float_round(
                    100 / self.outcomes[index][OutcomeKeys.ODDS]
                )

        self.__clear_outcomes()

    def __repr__(self):
        return f"Bet(total_users={millify(self.total_users)}, total_points={millify(self.total_points)}), decision={self.decision})\n\t\tOutcome A({self.get_outcome(0)})\n\t\tOutcome B({self.get_outcome(1)})"

    def get_decision(self, parsed=False):
        decision = self.outcomes[0 if self.decision["choice"] == "A" else 1]
        return decision if parsed is False else Bet.__parse_outcome(decision)

    @staticmethod
    def __parse_outcome(outcome):
        return f"{outcome['title']} ({outcome['color']}), Points: {millify(outcome[OutcomeKeys.TOTAL_POINTS])}, Users: {millify(outcome[OutcomeKeys.TOTAL_USERS])} ({outcome[OutcomeKeys.PERCENTAGE_USERS]}%), Odds: {outcome[OutcomeKeys.ODDS]} ({outcome[OutcomeKeys.ODDS_PERCENTAGE]}%)"

    def get_outcome(self, index):
        return Bet.__parse_outcome(self.outcomes[index])

    def __clear_outcomes(self):
        for index in range(0, len(self.outcomes)):
            keys = copy.deepcopy(list(self.outcomes[index].keys()))
            for key in keys:
                if key not in [
                    OutcomeKeys.TOTAL_USERS,
                    OutcomeKeys.TOTAL_POINTS,
                    OutcomeKeys.TOP_POINTS,
                    OutcomeKeys.PERCENTAGE_USERS,
                    OutcomeKeys.ODDS,
                    OutcomeKeys.ODDS_PERCENTAGE,
                    "title",
                    "color",
                    "id",
                ]:
                    del self.outcomes[index][key]
            for key in [
                OutcomeKeys.PERCENTAGE_USERS,
                OutcomeKeys.ODDS,
                OutcomeKeys.ODDS_PERCENTAGE,
                OutcomeKeys.TOP_POINTS,
            ]:
                if key not in self.outcomes[index]:
                    self.outcomes[index][key] = 0

    def __return_choice(self, key) -> str:
        return "A" if self.outcomes[0][key] > self.outcomes[1][key] else "B"

    def skip(self) -> bool:
        if self.settings.filter_condition is not None:
            # key == by , condition == where
            key = self.settings.filter_condition.by
            condition = self.settings.filter_condition.where
            value = self.settings.filter_condition.value

            fixed_key = (
                key
                if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS]
                else key.replace("decision", "total")
            )
            if key in [OutcomeKeys.TOTAL_USERS, OutcomeKeys.TOTAL_POINTS]:
                compared_value = (
                    self.outcomes[0][fixed_key] + self.outcomes[1][fixed_key]
                )
            else:
                outcome_index = char_decision_as_index(self.decision["choice"])
                compared_value = self.outcomes[outcome_index][fixed_key]

            # Check if condition is satisfied
            if condition == Condition.GT:
                if compared_value > value:
                    return False, compared_value
            elif condition == Condition.LT:
                if compared_value < value:
                    return False, compared_value
            elif condition == Condition.GTE:
                if compared_value >= value:
                    return False, compared_value
            elif condition == Condition.LTE:
                if compared_value <= value:
                    return False, compared_value
            return True, compared_value  # Else skip the bet
        else:
            return False, 0  # Default don't skip the bet

    def calculate(self, balance: int, title: str = "") -> dict:
        self.decision = {"choice": None, "amount": 0, "id": None}
        if self.settings.strategy == Strategy.MOST_VOTED:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.TOTAL_USERS)
        elif self.settings.strategy == Strategy.HIGH_ODDS:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS)
        elif self.settings.strategy == Strategy.PERCENTAGE:
            self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS_PERCENTAGE)
        elif self.settings.strategy == Strategy.SMART:
            difference = abs(
                self.outcomes[0][OutcomeKeys.PERCENTAGE_USERS]
                - self.outcomes[1][OutcomeKeys.PERCENTAGE_USERS]
            )
            self.decision["choice"] = (
                self.__return_choice(OutcomeKeys.ODDS)
                if difference < self.settings.percentage_gap
                else self.__return_choice(OutcomeKeys.TOTAL_USERS)
            )
        elif self.settings.strategy == Strategy.GENSHIN:

            def event_chance(
                event_odds: float,
                multiplier_scaling: float = None,
                odds_label: str = [],
                percentage_label: str = [],
                strict: bool = False,
            ) -> None:
                if not isinstance(odds_label, list):
                    odds_label = [odds_label]
                # multiplier_scaling: Odd difference from event / multiplier scaling = multiplier (Max 2x)
                if multiplier_scaling is None:
                    multiplier_scaling = event_odds
                if not isinstance(percentage_label, list):
                    percentage_label = [percentage_label]
                odds, label = max(
                    (
                        self.outcomes[0][OutcomeKeys.ODDS],
                        self.outcomes[0]["title"].lower(),
                    ),
                    (
                        self.outcomes[1][OutcomeKeys.ODDS],
                        self.outcomes[1]["title"].lower(),
                    ),
                )
                if strict:
                    in_labels = lambda title, labels: any((title in label) for label in labels)
                    if in_labels(self.outcomes[0]["title"].lower(), odds_label) and in_labels(self.outcomes[1]["title"].lower(), percentage_label):
                        odds, label = (self.outcomes[0][OutcomeKeys.ODDS], self.outcomes[0]["title"].lower())
                    elif in_labels(self.outcomes[1]["title"].lower(), odds_label) and in_labels(self.outcomes[0]["title"].lower(), percentage_label):
                        odds, label = (self.outcomes[1][OutcomeKeys.ODDS], self.outcomes[1]["title"].lower())
                multiplier = 1 + min(abs(odds - event_odds) / multiplier_scaling - 1, 1)
                if odds > event_odds:
                    self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS)
                    if len(odds_label) > 0 and not any(label in label.lower() for label in odds_label):
                        logger.warning(
                            f"Event odds label not correct - Label: {label}, Expected: {odds_label}\n{title}: {self.outcomes[0]['title']} - {self.outcomes[1]['title']}"
                        )
                        if strict:
                            self.decision["amount"] = 10
                            return
                    # Normalize multiplier for odds event chance
                    multiplier /= event_odds
                else:
                    self.decision["choice"] = self.__return_choice(
                        OutcomeKeys.ODDS_PERCENTAGE
                    )
                    if len(percentage_label) > 0 and not any(label in label.lower() for label in percentage_label):
                        logger.warning(
                            f"Event percentage label not correct - Label: {label}, Expected: {odds_label}\n{title}: {self.outcomes[0]['title']} - {self.outcomes[1]['title']}"
                        )
                        if strict:
                            self.decision["amount"] = 10
                            return
                    # Normalize multiplier for percentage event chance
                    multiplier *= 1 - 1 / event_odds
                self.decision["amount"] = int(
                    balance * (self.settings.percentage_genshin * multiplier / 100)
                )

            percent_to_odds = lambda percent: (1 - percent / 100) / (percent / 100) + 1
            title = title.lower()
            if "good" in title and "artifact" in title:
                event_chance(
                    percent_to_odds(self.settings.genshin_chances["artifact"]),
                    odds_label="y",
                    percentage_label="n",
                )
            elif "fortune" in title:
                event_chance(
                    percent_to_odds(self.settings.genshin_chances["fortune"]),
                    odds_label="mis",
                    percentage_label="fortune",
                )
            elif ("proto" in title or "billet" in title) and "solvent" in title:
                event_chance(
                    percent_to_odds(self.settings.genshin_chances["weekly_boss_items"]),
                    odds_label=["yes", "prayge"],
                    percentage_label=["no", "pepeloser"],
                    strict=True,
                )
            elif ("2" in title or "two" in title) and "or" in title and ("3" in title or "three" in title):
                event_chance(
                    percent_to_odds(
                        self.settings.genshin_chances["boss_2_or_3_mats"]
                    ),
                    odds_label=["2", "two"],
                    percentage_label=["3", "three"],
                    strict=True,
                )
            elif ("3" in title or "three" in title) and "mat" in title:
                event_chance(
                    percent_to_odds(
                        self.settings.genshin_chances["weekly_boss_3_mats"]
                    ),
                    odds_label="yes",
                    percentage_label="no",
                    strict=True,
                )
            else:
                self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS)

        if self.decision["choice"] is not None:
            index = char_decision_as_index(self.decision["choice"])
            self.decision["id"] = self.outcomes[index]["id"]
            if self.decision["amount"] == 0:
                self.decision["amount"] = balance * (self.settings.percentage / 100)
            self.decision["amount"] = min(
                self.decision["amount"], self.settings.max_points, balance
            )
            if (
                self.settings.stealth_mode is True
                and self.decision["amount"]
                >= self.outcomes[index][OutcomeKeys.TOP_POINTS]
            ):
                reduce_amount = uniform(1, 5)
                self.decision["amount"] = (
                    self.outcomes[index][OutcomeKeys.TOP_POINTS] - reduce_amount
                )
            self.decision["amount"] = int(self.decision["amount"])
        return self.decision
