import copy
import logging
import math
import re
from enum import Enum, auto
from typing import Any, Callable, Optional, Union

from millify import millify

from TwitchChannelPointsMiner.classes.Settings import Settings

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

    def __init__(
        self,
        by: OutcomeKeys,
        where: Condition,
        value: float,
    ):
        self.by = by
        self.where = where
        self.value = value

    def __repr__(self):
        return f"FilterCondition(by={str(self.by).upper()}, where={self.where}, value={self.value})"


class BetEvent(object):
    __slots__ = [
        "title",
        "event_chances",
        "max_points",
        "strict",
        "filter",
        "regex_flags",
    ]

    def __init__(
        self,
        title: Union[str, re.Pattern[str], None],
        event_chances: Union[dict[str, float], list[tuple[Union[str, re.Pattern[str]], float]]],
        max_points: Optional[int] = None,
        strict: bool = True,
        filter: Optional[Callable[["BetEventData"], bool]] = None,
        regex_flags: re.RegexFlag = re.IGNORECASE,
    ):
        """
        Object that represents a bet event with a title regex and chances of each outcome.

        :param title: Regex for the title of the bet event or None to match all events
        :param event_chances: Outcome labels and probabilities pairs. Can be a dict or a list of tuples.
        :param max_points: Chance normalized maximum points to place on the bet
        :param strict: If true, outcome labels must match exactly. Otherwise the same outcome label can be used multiple times and not all outcomes must be present
        :param filter: Function to filter the bets before placing them. Return True if the bet should be placed, False otherwise
        :param regex_flags: Flags to use when compiling the regex

        Probability of the outcomes will be normalized to 1.

        For bets based on order of outcomes, use ".*" as a wildcard for the outcome label in a list of tuples and set strict to True.
        E.g The probability of the first outcome is 80% and the probability of the second outcome is 20%: [(".*", 0.8), (".*", 0.2)]
        """

        if isinstance(title, str):
            try:
                title = re.compile(title, regex_flags)
            except re.error as e:
                raise ValueError(f"Invalid regex for title: {title}") from e

        if isinstance(event_chances, dict):
            event_chances = list(event_chances.items())

        new_event_chances: list[tuple[re.Pattern[str], float]] = []
        for outcome, chance in event_chances:
            if isinstance(outcome, str):
                try:
                    outcome = re.compile(outcome, regex_flags)
                except re.error as e:
                    raise ValueError(f"Invalid regex for outcome: {outcome}") from e
            new_event_chances.append((outcome, chance))

        self.title = title
        self.event_chances = new_event_chances
        self.max_points = None if max_points is None else int(max_points)
        self.strict = strict
        self.filter = filter
        self.regex_flags = regex_flags

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{attr}={getattr(self, attr)!r}' for attr in self.__slots__)})"


class BetEventData(object):
    __slots__ = [
        "bet",
        "event",
        "title",
        "balance",
    ]

    def __init__(self, bet: "Bet", event: "BetEvent", title: str, balance: int):
        self.bet = bet
        self.event = event
        self.title = title
        self.balance = balance


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
        strategy: Optional[Strategy] = None,
        percentage: Optional[int] = None,
        percentage_gap: Optional[int] = None,
        events: Optional[list[BetEvent]] = None,
        event_percentage: Optional[int] = None,
        max_points: Optional[int] = None,
        minimum_points: Optional[int] = None,
        stealth_mode: Optional[bool] = None,
        filter_condition: Optional[FilterCondition] = None,
        delay: Optional[float] = None,
        delay_mode: Optional[DelayMode] = None,
    ):
        self.strategy: Strategy = strategy  # type: ignore
        self.percentage: int = percentage  # type: ignore
        self.percentage_gap: int = percentage_gap  # type: ignore
        self.events: list[BetEvent] = events  # type: ignore
        self.event_percentage: int = event_percentage  # type: ignore
        self.max_points: int = max_points  # type: ignore
        self.minimum_points: int = minimum_points  # type: ignore
        self.stealth_mode: bool = stealth_mode  # type: ignore
        self.filter_condition: FilterCondition = filter_condition  # type: ignore
        self.delay: float = delay  # type: ignore
        self.delay_mode: DelayMode = delay_mode  # type: ignore

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
    __slots__ = [
        "outcomes",
        "decision",
        "total_users",
        "total_points",
        "settings",
    ]

    def __init__(self, outcomes: list[dict[str, Any]], settings: BetSettings):
        self.outcomes = outcomes
        self.__clear_outcomes()
        self.decision: dict[str, Any] = {}
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
                outcome[OutcomeKeys.PERCENTAGE_USERS] = 100 * outcome[OutcomeKeys.TOTAL_USERS] / self.total_users
                outcome[OutcomeKeys.ODDS_PERCENTAGE] = 100 * outcome[OutcomeKeys.TOTAL_POINTS] / self.total_points

                outcome_points = outcome[OutcomeKeys.TOTAL_POINTS]
                if outcome_points > 0:
                    outcome[OutcomeKeys.ODDS] = self.total_points / outcome_points

        self.__clear_outcomes()

    def __repr__(self):
        outcome_str = "".join(f"\n\t\tOutcome {chr(ord('A') + i)}({self.get_outcome(i)})" for i in range(len(self.outcomes)))
        return f"Bet(total_users={millify(self.total_users)}, total_points={millify(self.total_points)}), decision={self.decision}){outcome_str}"

    def get_decision(self, parsed=False):
        decision = self.outcomes[self.decision["choice"]]
        return decision if parsed is False else Bet.__parse_outcome(decision)

    @staticmethod
    def __parse_outcome(outcome):
        return f"{outcome['title']} ({outcome['color']}), Points: {millify(outcome[OutcomeKeys.TOTAL_POINTS])}, Users: {millify(outcome[OutcomeKeys.TOTAL_USERS])} ({outcome[OutcomeKeys.PERCENTAGE_USERS]:.2f}%), Odds: {outcome[OutcomeKeys.ODDS]:.2f} ({outcome[OutcomeKeys.ODDS_PERCENTAGE]:.2f}%)"

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

    def __return_choice(self, key) -> int:
        return max(range(len(self.outcomes)), key=lambda i: self.outcomes[i][key])

    def skip(self) -> tuple[bool, float]:
        if self.settings.filter_condition is not None:
            # key == by , condition == where
            key = str(self.settings.filter_condition.by)
            condition = self.settings.filter_condition.where
            value = self.settings.filter_condition.value

            fixed_key = key if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS] else key.replace("decision", "total")
            if key in [OutcomeKeys.TOTAL_USERS, OutcomeKeys.TOTAL_POINTS]:
                compared_value = sum(outcome[fixed_key] for outcome in self.outcomes)
            else:
                outcome_index = self.decision["choice"]
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

    def calculate(self, balance: int, title: Optional[str] = None) -> dict:
        self.decision = {
            "choice": None,
            "amount": None,
            "id": None,
        }

        if all(outcome[OutcomeKeys.TOTAL_POINTS] == 0 for outcome in self.outcomes):
            return self.decision

        is_event = False
        if title is None:
            logger.warning(
                "Missing Title",
                {"color": Settings.logger.color_palette.BET_FAILED},
            )
        elif self.settings.events is not None:
            for event in self.settings.events:
                if event.title is None or event.title.search(title):
                    if event.filter is not None and not event.filter(BetEventData(self, event, title, balance)):
                        continue
                    if self.event_strategy(balance, event):
                        is_event = True
                        break

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
            self.decision["id"] = self.outcomes[self.decision["choice"]]["id"]
            if not self.decision["amount"]:
                self.decision["amount"] = min(balance, self.total_points) * (self.settings.percentage / 100)
            self.decision["amount"] = max(self.decision["amount"], self.settings.minimum_points)
            if not is_event:
                self.decision["amount"] = min(self.decision["amount"], self.settings.max_points)
            if self.settings.stealth_mode is True:
                self.decision["amount"] = min(
                    self.decision["amount"],
                    self.outcomes[self.decision["choice"]][OutcomeKeys.TOP_POINTS] - 1,
                )
            self.decision["amount"] = int(min(self.decision["amount"], balance, 250000))
        return self.decision

    def odds_strategy(self, balance: int) -> None:
        first, second = sorted(
            range(len(self.outcomes)),
            key=lambda i: self.outcomes[i][OutcomeKeys.TOTAL_POINTS],
        )[:2]
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
        if self.settings.event_percentage is None:
            return False

        failed_logger_extra = {"color": Settings.logger.color_palette.BET_FAILED}

        if event.strict and len(self.outcomes) != len(event.event_chances):
            logger.info("Event outcomes counts don't match", failed_logger_extra)
            return False

        total_chance = 0
        outcome_chances = [0.0] * len(self.outcomes)
        available_event_titles = {outcome for outcome, chance in event.event_chances}

        for i, decision_outcome in enumerate(self.outcomes):
            for event_outcome, chance in event.event_chances:
                if event_outcome in available_event_titles:
                    if event_outcome.search(decision_outcome["title"]):
                        if event.strict:
                            available_event_titles.remove(event_outcome)
                        outcome_chances[i] = chance
                        total_chance += chance
                        break
            else:
                logger.info("Event outcome not found", failed_logger_extra)
                return False

        if event.strict and len(available_event_titles) > 0:
            logger.info("Event outcomes left after matching", failed_logger_extra)
            return False

        # Normalize chances
        for i in range(len(outcome_chances)):
            outcome_chances[i] /= total_chance

        decision = None
        max_expected_value = 0

        for i, outcome in enumerate(self.outcomes):
            if outcome[OutcomeKeys.TOTAL_POINTS] / self.total_points >= outcome_chances[i]:
                continue

            t, o, c = (
                self.total_points,
                outcome[OutcomeKeys.TOTAL_POINTS],
                1 / outcome_chances[i],
            )

            # Account for Bet amount and scale based on difference of actual chance and bet reward
            p = balance * (self.settings.event_percentage / 100) * (outcome_chances[i] ** 2)
            bet_amount = (math.sqrt((c * p + o - p) ** 2 - 4 * (c * o * p - p * t)) - c * p - o + p) / 2

            # Limit bet amount to bet with max expected value and twitch limit
            bet_amount = min(bet_amount, math.sqrt(-(c - 1) * o * (o - t)) / (c - 1) - o, 250000)

            odds_after_bet = (self.total_points + bet_amount) / (outcome[OutcomeKeys.TOTAL_POINTS] + bet_amount)
            expected_value = odds_after_bet * bet_amount * outcome_chances[i] - bet_amount

            if expected_value > max_expected_value:
                decision = (i, bet_amount)
                max_expected_value = expected_value

        if decision is None:
            logger.error("No profitable outcome found", failed_logger_extra)
            return True

        self.decision["choice"], self.decision["amount"] = decision
        if event.max_points is not None:
            self.decision["amount"] = min(self.decision["amount"], event.max_points * outcome_chances[decision[0]])
        self.decision["amount"] = min(self.decision["amount"], balance * outcome_chances[decision[0]])

        return True
