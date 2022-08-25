from enum import Enum, auto
import random
from typing import Callable
import logging
import copy
import math
import re

from millify import millify

from TwitchChannelPointsMiner.classes.Settings import Settings
from TwitchChannelPointsMiner.utils import float_round

logger = logging.getLogger(__name__)


class Strategy(Enum):
    MOST_VOTED = auto()
    HIGH_ODDS = auto()
    PERCENTAGE = auto()
    SMART_MONEY = auto()
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

    def __init__(self, by=None, where=None, value=None):
        self.by = by
        self.where = where
        self.value = value

    def __repr__(self):
        return f"FilterCondition(by={self.by.upper()}, where={self.where}, value={self.value})"


class BetEvent(object):
    __slots__ = ["title", "event_chances", "max_points", "strict", "filter", "regex_flags"]

    def __init__(
        self,
        title: str,
        event_chances: dict[str, float],
        max_points: int = None,
        strict: bool = True,
        filter: Callable[["Bet"], bool] = None,
        regex_flags: re.RegexFlag = re.IGNORECASE,
    ):
        """
        Object that represents a bet event with a title regex and chances of each outcome.

        :param title: Regex for the title of the bet
        :param event_chances: Dictionary of outcome labels and chances of each outcome
        :param max_points: Chance normalized maximum points to place on the bet
        :param strict: If true, outcome labels must match exactly. Otherwise the same outcome label can be used multiple times and not all outcomes must be present
        :param filter: Function to filter the bets before placing them. Return True if the bet should be placed, False otherwise
        :param regex_flags: Flags to use when compiling the regex
        """

        self.title = title
        self.event_chances = event_chances
        self.max_points = max_points
        self.strict = strict
        self.filter = filter
        self.regex_flags = regex_flags

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{attr}={getattr(self, attr)!r}' for attr in self.__slots__)})"


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

    def __init__(self, outcomes: list, settings: BetSettings):
        self.outcomes = outcomes
        self.__clear_outcomes()
        self.decision: dict = {}
        self.total_users = 0
        self.total_points = 0
        self.settings = settings

    def update_outcomes(self, outcomes):
        for i, outcome in enumerate(self.outcomes):
            outcome[OutcomeKeys.TOTAL_USERS] = int(outcomes[i][OutcomeKeys.TOTAL_USERS])
            outcome[OutcomeKeys.TOTAL_POINTS] = int(outcomes[i][OutcomeKeys.TOTAL_POINTS])
            if outcomes[i]["top_predictors"] != []:
                # Sort by points placed by other users
                outcomes[i]["top_predictors"] = sorted(
                    outcomes[i]["top_predictors"],
                    key=lambda x: x["points"],
                    reverse=True,
                )
                # Get the first elements (most placed)
                top_points = outcomes[i]["top_predictors"][0]["points"]
                outcome[OutcomeKeys.TOP_POINTS] = top_points

        self.total_users = sum(outcome[OutcomeKeys.TOTAL_USERS] for outcome in self.outcomes)
        self.total_points = sum(outcome[OutcomeKeys.TOTAL_POINTS] for outcome in self.outcomes)

        if self.total_users > 0 and self.total_points > 0:
            for outcome in self.outcomes:
                outcome[OutcomeKeys.PERCENTAGE_USERS] = float_round((100 * outcome[OutcomeKeys.TOTAL_USERS]) / self.total_users)
                outcome[OutcomeKeys.ODDS_PERCENTAGE] = float_round(100 * outcome[OutcomeKeys.TOTAL_POINTS] / self.total_points)

                outcome_points = outcome[OutcomeKeys.TOTAL_POINTS]
                if outcome_points > 0:
                    outcome[OutcomeKeys.ODDS] = float_round(self.total_points / outcome_points)

        self.__clear_outcomes()

    def __repr__(self):
        outcome_str = "".join(f"\n\t\tOutcome {chr(ord('A') + i)}({self.get_outcome(i)})" for i in range(len(self.outcomes)))
        return f"Bet(total_users={millify(self.total_users)}, total_points={millify(self.total_points)}), decision={self.decision}){outcome_str}"

    def get_decision(self, parsed=False):
        decision = self.outcomes[self.decision["choice"]]
        return decision if parsed is False else Bet.__parse_outcome(decision)

    @staticmethod
    def __parse_outcome(outcome):
        return f"{outcome['title']} ({outcome['color']}), Points: {millify(outcome[OutcomeKeys.TOTAL_POINTS])}, Users: {millify(outcome[OutcomeKeys.TOTAL_USERS])} ({outcome[OutcomeKeys.PERCENTAGE_USERS]}%), Odds: {outcome[OutcomeKeys.ODDS]} ({outcome[OutcomeKeys.ODDS_PERCENTAGE]}%)"

    def get_outcome(self, index):
        return Bet.__parse_outcome(self.outcomes[index])

    def __clear_outcomes(self):
        for index in range(len(self.outcomes)):
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
        return max(range(len(self.outcomes)), key=lambda i: self.outcomes[i][key])

    def skip(self) -> bool:
        if self.settings.filter_condition is not None:
            # key == by , condition == where
            key = self.settings.filter_condition.by
            condition = self.settings.filter_condition.where
            value = self.settings.filter_condition.value

            fixed_key = key if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS] else key.replace("decision", "total")
            if key in [OutcomeKeys.TOTAL_USERS, OutcomeKeys.TOTAL_POINTS]:
                compared_value = sum(outcome[fixed_key] for outcome in self.outcomes)
            else:
                compared_value = self.outcomes[self.decision["choice"]][fixed_key]

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

    def calculate(self, balance: int, title: str = None) -> dict:
        self.decision = {"choice": None, "amount": None, "id": None}

        if all(outcome[OutcomeKeys.TOTAL_POINTS] == 0 for outcome in self.outcomes):
            return self.decision

        is_event = False
        if title is None:
            logger.warning("Missing Title", {"color": Settings.logger.color_palette.BET_FAILED})
        else:
            for event in self.settings.events:
                if (
                    isinstance(event.title, str)
                    and re.search(event.title, title, flags=event.regex_flags)
                    or callable(event.title)
                    and event.title(title)
                ):
                    if event.filter is not None and not event.filter(self):
                        continue
                    if self.event_strategy(balance, event):
                        is_event = True
                        break

        if not is_event:
            # Choose random empty outcome
            if any(outcome[OutcomeKeys.TOTAL_POINTS] == 0 for outcome in self.outcomes):
                self.decision["choice"] = random.choice(
                    list(filter(lambda i: self.outcomes[i][OutcomeKeys.TOTAL_POINTS] == 0, range(len(self.outcomes))))
                )

            elif self.settings.strategy == Strategy.MOST_VOTED:
                self.decision["choice"] = self.__return_choice(OutcomeKeys.TOTAL_USERS)
            elif self.settings.strategy == Strategy.HIGH_ODDS:
                self.odds_strategy(balance)
            elif self.settings.strategy == Strategy.PERCENTAGE:
                self.decision["choice"] = self.__return_choice(OutcomeKeys.ODDS_PERCENTAGE)
            elif self.settings.strategy == Strategy.SMART:
                self.smart_strategy(balance)

        if self.decision["choice"] is not None:
            self.decision["id"] = self.outcomes[self.decision["choice"]]["id"]
            if not self.decision["amount"]:
                self.decision["amount"] = min(balance, self.total_points) * (self.settings.percentage / 100)
            self.decision["amount"] = max(self.decision["amount"], self.settings.minimum_points)
            if not is_event:
                self.decision["amount"] = min(self.decision["amount"], self.settings.max_points)
            if self.settings.stealth_mode is True:
                self.decision["amount"] = min(self.decision["amount"], self.outcomes[self.decision["choice"]][OutcomeKeys.TOP_POINTS] - 1)
            self.decision["amount"] = int(min(self.decision["amount"], balance, 250000))
        return self.decision

    def odds_strategy(self, balance: int) -> None:
        first, second = sorted(range(len(self.outcomes)), key=lambda i: self.outcomes[i][OutcomeKeys.TOTAL_POINTS])[:2]
        self.decision["choice"] = first
        self.decision["amount"] = int(
            min(  # Keep bet from changing which outcome has the highest odds
                min(balance, self.total_points) * (self.settings.percentage / 100),
                (self.outcomes[second][OutcomeKeys.TOTAL_POINTS] - self.outcomes[first][OutcomeKeys.TOTAL_POINTS])
                * 0.5,  # (0-1) Buffer to prevent changing on the odds outcome
            )
        )

    def smart_strategy(self, balance: int) -> None:
        for i in range(len(self.outcomes)):
            for j in range(i + 1, len(self.outcomes)):
                difference = abs(self.outcomes[i][OutcomeKeys.PERCENTAGE_USERS] - self.outcomes[j][OutcomeKeys.PERCENTAGE_USERS])
                if difference > self.settings.percentage_gap:
                    self.decision["choice"] = self.__return_choice(OutcomeKeys.PERCENTAGE_USERS)
                    return
        self.odds_strategy(balance)

    def event_strategy(self, balance: int, event: BetEvent) -> bool:
        failed_logger_extra = {"color": Settings.logger.color_palette.BET_FAILED}

        if event.strict and len(self.outcomes) != len(event.event_chances):
            logger.info("Event outcomes counts don't match", failed_logger_extra)
            return False

        total_chance = 0
        outcome_chances = [0.0] * len(self.outcomes)
        available_event_chances = set(event.event_chances.keys())

        for i, decision_outcome in enumerate(self.outcomes):
            for event_outcome, chance in event.event_chances.items():
                if event_outcome in available_event_chances:
                    if re.search(event_outcome, decision_outcome["title"], flags=event.regex_flags):
                        if event.strict:
                            available_event_chances.remove(event_outcome)
                        outcome_chances[i] = chance
                        total_chance += chance
                        break
            else:
                logger.info("Event outcome not found", failed_logger_extra)
                return False

        if event.strict and len(available_event_chances) > 0:
            logger.info("Event outcomes left after matching", failed_logger_extra)
            return False

        for i in range(len(outcome_chances)):
            outcome_chances[i] /= total_chance
        max_bet, max_amount = None, 0

        for i, outcome in enumerate(self.outcomes):
            if outcome[OutcomeKeys.TOTAL_POINTS] / self.total_points >= outcome_chances[i]:
                continue

            # Account for Bet amount and scale based on difference of actual chance and bet reward
            p = balance * (self.settings.event_percentage / 100) * (outcome_chances[i] ** 2)
            t, o, c = self.total_points, outcome[OutcomeKeys.TOTAL_POINTS], 1 / outcome_chances[i]
            bet_amount = (math.sqrt((c * p + o - p) ** 2 - 4 * (c * o * p - p * t)) - c * p - o + p) / 2

            if bet_amount > max_amount:
                max_bet = i
                max_amount = bet_amount

        self.decision["choice"] = max_bet
        self.decision["amount"] = max_amount

        if event.max_points is not None:
            self.decision["amount"] = min(self.decision["amount"], event.max_points * outcome_chances[i])
        self.decision["amount"] = min(self.decision["amount"], balance * outcome_chances[i])

        return True
