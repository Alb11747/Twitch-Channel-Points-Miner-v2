from enum import Enum, auto
from typing import Union
import logging
import copy
import math
import re

from millify import millify

from TwitchChannelPointsMiner.classes.Settings import Settings
from TwitchChannelPointsMiner.utils import char_decision_as_index, float_round

logger = logging.getLogger(__name__)


class Strategy(Enum):
    MOST_VOTED = auto()
    HIGH_ODDS = auto()
    PERCENTAGE = auto()
    SMART = auto()

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


class BetEvent(object):
    __slots__ = ["title", "chance", "odds_labels", "percentage_labels", "max_points", "strict"]

    def __init__(
        self,
        title: str,
        chance: float,
        odds_labels: Union[str, list] = [],
        percentage_labels: Union[str, list] = [],
        max_points: int = None,
        strict: bool = True,
    ):
        self.title = title
        self.chance = chance
        self.odds_labels = odds_labels
        self.percentage_labels = percentage_labels
        self.max_points = max_points
        self.strict = strict

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{attr}={getattr(self, attr)}' for attr in self.__slots__)})"


class BetSettings(object):
    __slots__ = [
        "strategy",
        "percentage",
        "event_percentage",
        "events",
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
        percentage_gap: int = None,
        events: list[BetEvent] = None,
        event_percentage: int = None,
        max_points: int = None,
        minimum_points: int = None,
        stealth_mode: bool = None,
        filter_condition: FilterCondition = None,
        delay: float = None,
        delay_mode: DelayMode = None,
    ):
        self.strategy = strategy
        self.percentage = percentage
        self.percentage_gap = percentage_gap
        self.events = events
        self.event_percentage = event_percentage
        self.max_points = max_points
        self.minimum_points = minimum_points
        self.stealth_mode = stealth_mode
        self.filter_condition = filter_condition
        self.delay = delay
        self.delay_mode = delay_mode

    def default(self):
        self.strategy = self.strategy if self.strategy is not None else Strategy.SMART
        self.percentage = self.percentage if self.percentage is not None else 5
        self.events = self.events if self.events is not None else []
        self.event_percentage = self.event_percentage if self.event_percentage is not None else 20
        self.percentage_gap = self.percentage_gap if self.percentage_gap is not None else 20
        self.max_points = self.max_points if self.max_points is not None else 50000
        self.minimum_points = self.minimum_points if self.minimum_points is not None else 0
        self.stealth_mode = self.stealth_mode if self.stealth_mode is not None else False
        self.delay = self.delay if self.delay is not None else 6
        self.delay_mode = self.delay_mode if self.delay_mode is not None else DelayMode.FROM_END

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{attr}={getattr(self, attr)}' for attr in self.__slots__)})"


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
            self.outcomes[index][OutcomeKeys.TOTAL_USERS] = int(outcomes[index][OutcomeKeys.TOTAL_USERS])
            self.outcomes[index][OutcomeKeys.TOTAL_POINTS] = int(outcomes[index][OutcomeKeys.TOTAL_POINTS])
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

        self.total_users = self.outcomes[0][OutcomeKeys.TOTAL_USERS] + self.outcomes[1][OutcomeKeys.TOTAL_USERS]
        self.total_points = self.outcomes[0][OutcomeKeys.TOTAL_POINTS] + self.outcomes[1][OutcomeKeys.TOTAL_POINTS]

        if self.total_users > 0 and self.outcomes[0][OutcomeKeys.TOTAL_POINTS] > 0 and self.outcomes[1][OutcomeKeys.TOTAL_POINTS] > 0:
            for index in range(0, len(self.outcomes)):
                self.outcomes[index][OutcomeKeys.PERCENTAGE_USERS] = float_round(
                    (100 * self.outcomes[index][OutcomeKeys.TOTAL_USERS]) / self.total_users
                )
                self.outcomes[index][OutcomeKeys.ODDS] = float_round(self.total_points / self.outcomes[index][OutcomeKeys.TOTAL_POINTS])
                self.outcomes[index][OutcomeKeys.ODDS_PERCENTAGE] = float_round(100 / self.outcomes[index][OutcomeKeys.ODDS])

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

            fixed_key = key if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS] else key.replace("decision", "total")
            if key in [OutcomeKeys.TOTAL_USERS, OutcomeKeys.TOTAL_POINTS]:
                compared_value = self.outcomes[0][fixed_key] + self.outcomes[1][fixed_key]
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

    def odds_strategy(self, balance: int) -> None:
        if self.outcomes[0][OutcomeKeys.ODDS] > self.outcomes[1][OutcomeKeys.ODDS]:
            odds, percentage, odds_choice = self.outcomes[0], self.outcomes[1], "A"
        else:
            odds, percentage, odds_choice = self.outcomes[1], self.outcomes[0], "B"
        self.decision["choice"] = odds_choice
        total_points = self.outcomes[0][OutcomeKeys.TOTAL_POINTS] + self.outcomes[1][OutcomeKeys.TOTAL_POINTS]
        self.decision["amount"] = int(
            min(  # Stop from swaping Odds and Percentage based on points bet
                min(balance, total_points) * (self.settings.percentage / 100),
                (percentage[OutcomeKeys.TOTAL_POINTS] - odds[OutcomeKeys.TOTAL_POINTS])
                * 0.5,  # (0-1) Maximum differnce from 50% to change
            )
        )

    def smart_strategy(self, balance: int) -> None:
        percentage_users_A = self.outcomes[0][OutcomeKeys.PERCENTAGE_USERS]
        percentage_users_B = self.outcomes[1][OutcomeKeys.PERCENTAGE_USERS]
        if abs(percentage_users_A - percentage_users_B) < self.settings.percentage_gap:
            self.odds_strategy(balance)
        else:
            total_points = self.outcomes[0][OutcomeKeys.TOTAL_POINTS] + self.outcomes[1][OutcomeKeys.TOTAL_POINTS]
            if percentage_users_A > self.outcomes[0][OutcomeKeys.ODDS_PERCENTAGE]:
                self.decision["choice"] = "A"
                amount = abs(percentage_users_A - self.outcomes[0][OutcomeKeys.ODDS_PERCENTAGE]) / 5
            else:
                self.decision["choice"] = "B"
                amount = abs(percentage_users_B - self.outcomes[1][OutcomeKeys.ODDS_PERCENTAGE]) / 5
            self.decision["amount"] = int(
                min(
                    amount ** 2 / 20 * balance * (self.settings.percentage / 100),
                    min(balance, total_points) * (self.settings.percentage / 100),
                    amount / 100 * total_points * 0.5,
                )
            )

    def chance_amount_calculations(
        self,
        balance: int,
        event_chance: float,
        odds_labels: str = [],
        percentage_labels: str = [],
        max_points: int = None,
        strict: bool = True,
        title: str = "Title",
    ) -> None:
        event_odds = 1 / (event_chance / 100)
        if not isinstance(odds_labels, list):
            odds_labels = [odds_labels]
        if not isinstance(percentage_labels, list):
            percentage_labels = [percentage_labels]

        if self.outcomes[0][OutcomeKeys.ODDS] > self.outcomes[1][OutcomeKeys.ODDS]:
            odds, odds_choice = self.outcomes[0], "A"
            percentage, percentage_choice = self.outcomes[1], "B"
        else:
            odds, odds_choice = self.outcomes[1], "B"
            percentage, percentage_choice = self.outcomes[0], "A"

        log_extra = {"color": Settings.logger.color_palette.BET_FAILED}
        in_labels = lambda title, labels: any((label in title.lower()) for label in labels)

        if (
            len(odds_labels) > 0
            and len(percentage_labels) > 0
            and not (in_labels(odds["title"], odds_labels) and in_labels(percentage["title"], percentage_labels))
        ):
            event_str = f"Label: {odds['title']} - Expected: {percentage['title']} - \
{title}: {self.outcomes[0]['title']}:{self.outcomes[0][OutcomeKeys.ODDS_PERCENTAGE]} - {self.outcomes[1]['title']}:{self.outcomes[1][OutcomeKeys.ODDS_PERCENTAGE]}"
            if strict:
                if in_labels(self.outcomes[0]["title"], odds_labels) and in_labels(self.outcomes[1]["title"], percentage_labels):
                    logger.warning(f"Odds label does not match (Overriding due to strict mode) - {event_str}", log_extra)
                    odds, odds_choice = self.outcomes[0], "A"
                    percentage, percentage_choice = self.outcomes[1], "B"
                elif in_labels(self.outcomes[1]["title"], odds_labels) and in_labels(self.outcomes[0]["title"], percentage_labels):
                    logger.warning(f"Odds label does not match (Overriding due to strict mode) - {event_str}", log_extra)
                    odds, odds_choice = self.outcomes[1], "B"
                    percentage, percentage_choice = self.outcomes[0], "A"
                else:
                    logger.warning(f"Odds label does not match (Can't find match) - {event_str}", log_extra)
            else:
                logger.warning(f"Odds label does not match (Enable strict mode to override) - {event_str}", log_extra)

        b, p, t = balance, self.settings.event_percentage / 100, self.total_points
        if odds[OutcomeKeys.ODDS] > event_odds:
            self.decision["choice"] = odds_choice
            if len(odds_labels) > 0 and not in_labels(odds["title"], odds_labels):
                logger.warning(f"Event odds label not correct - {odds['title']} - Expected: {odds_labels}", log_extra)
                if strict:
                    self.smart_strategy(balance)
                    return
            o, c, m = odds[OutcomeKeys.TOTAL_POINTS], event_odds, event_odds - 1
        else:
            self.decision["choice"] = percentage_choice
            if len(percentage_labels) > 0 and not in_labels(percentage["title"], percentage_labels):
                logger.warning(f"Event percentage label not correct - {percentage['title']} - Expected: {percentage_labels}", log_extra)
                if strict:
                    self.smart_strategy(balance)
                    return
            o, c, m = percentage[OutcomeKeys.TOTAL_POINTS], 1 / (1 - 1 / event_odds), 1 / (1 - 1 / event_odds) - 1
        # Account for Bet Amount, Scale based on Difference, and Normalize for Bet Odds
        self.decision["amount"] = (
            math.sqrt((b * c * p - b * p + c * m * o) ** 2 - 4 * c * m * (b * c * o * p - b * p * t)) - b * c * p + b * p - c * m * o
        ) / (2 * c * m)
        if max_points is not None:
            self.decision["amount"] = min(self.decision["amount"], max_points / c)  # c = choice_odds
        self.decision["amount"] = int(min(self.decision["amount"], balance / c))

    def calculate(self, balance: int, title: str = None) -> dict:
        self.decision = {"choice": None, "amount": None, "id": None}
        is_event = False

        if title is not None:
            for event in self.settings.events:
                if re.search(event.title, title, re.IGNORECASE):
                    self.chance_amount_calculations(
                        balance,
                        event.chance,
                        odds_labels=event.odds_labels,
                        percentage_labels=event.percentage_labels,
                        max_points=event.max_points,
                        strict=event.strict,
                        title=title,
                    )
                    is_event = True
                    break
        else:
            logger.warning("Missing Title", {"color": Settings.logger.color_palette.BET_FAILED})

        if not is_event:
            if self.settings.strategy == Strategy.MOST_VOTED:
                self.decision["choice"] = self.__return_choice(OutcomeKeys.TOTAL_USERS)
            elif self.settings.strategy == Strategy.HIGH_ODDS:
                self.odds_strategy(balance)
            elif self.settings.strategy == Strategy.PERCENTAGE:
                self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS_PERCENTAGE)
            elif self.settings.strategy == Strategy.SMART:
                self.smart_strategy(balance)

        if self.decision["choice"] is not None:
            index = char_decision_as_index(self.decision["choice"])
            self.decision["id"] = self.outcomes[index]["id"]
            if not self.decision["amount"]:
                total_points = self.outcomes[0][OutcomeKeys.TOTAL_POINTS] + self.outcomes[1][OutcomeKeys.TOTAL_POINTS]
                self.decision["amount"] = int(min(balance, total_points) * (self.settings.percentage / 100))
            self.decision["amount"] = max(self.decision["amount"], self.settings.minimum_points)
            if not is_event:
                self.decision["amount"] = min(self.decision["amount"], self.settings.max_points)
            if self.settings.stealth_mode is True:
                self.decision["amount"] = min(self.decision["amount"], self.outcomes[index][OutcomeKeys.TOP_POINTS] - 1)
            self.decision["amount"] = int(min(self.decision["amount"], balance))
        return self.decision
