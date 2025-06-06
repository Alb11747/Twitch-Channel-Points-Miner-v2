import copy
import logging
import re
from enum import Enum, auto
from typing import Any, Callable, Optional, Union

from millify import millify

from TwitchChannelPointsMiner.classes.Settings import Events, Settings

logger = logging.getLogger(__name__)


class Strategy(Enum):
    MOST_VOTED = auto()
    HIGH_ODDS = auto()
    PERCENTAGE = auto()
    SMART_MONEY = auto()
    SMART = auto()
    NUMBER_1 = auto()
    NUMBER_2 = auto()
    NUMBER_3 = auto()
    NUMBER_4 = auto()
    NUMBER_5 = auto()
    NUMBER_6 = auto()
    NUMBER_7 = auto()
    NUMBER_8 = auto()

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
        "confidence",
        "max_points",
        "strict",
        "filter",
        "regex_flags",
        "name",
    ]

    def __init__(
        self,
        title: Union[str, re.Pattern[str], None],
        event_chances: Union[dict[str, float], list[tuple[Union[str, re.Pattern[str]], float]]],
        confidence: float = 1.0,
        max_points: Optional[int] = None,
        strict: bool = True,
        filter: Optional[Callable[["BetEventData"], bool]] = None,
        regex_flags: re.RegexFlag = re.IGNORECASE,
        name: Optional[str] = None,
    ):
        """
        Object that represents a bet event with a title regex and chances of each outcome.

        :param title: Regex for the title of the bet event or None to match all events
        :param event_chances: Outcome labels and probabilities pairs. Can be a dict or a list of tuples.
        :param confidence: Confidence level of the bet event, used to determine how much to bet on the event. [0.0, 1.0]
        :param max_points: Chance normalized maximum points to place on the bet
        :param strict: If true, outcome labels must match exactly. Otherwise the same outcome label can be used multiple times and not all outcomes must be present
        :param filter: Function to filter the bets before placing them. Return True if the bet should be placed, False otherwise
        :param regex_flags: Flags to use when compiling the regex
        :param name: Name of the event for logging purposes

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

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        self.title = title
        self.event_chances = new_event_chances
        self.confidence = confidence
        self.max_points = None if max_points is None else int(max_points)
        self.strict = strict
        self.filter = filter
        self.regex_flags = regex_flags
        self.name = name

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
        outcome_str = "".join(
            f"\n\t\tOutcome {chr(ord('A') + i)}({self.get_outcome(i)})" for i in range(len(self.outcomes))
        )
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

            fixed_key = (
                key
                if key not in [OutcomeKeys.DECISION_USERS, OutcomeKeys.DECISION_POINTS]
                else key.replace("decision", "total")
            )
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
            logger.warning("Missing Title", {"color": Settings.logger.color_palette.get(Events.BET_FAILED)})
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
                difference = abs(
                    self.outcomes[i][OutcomeKeys.PERCENTAGE_USERS] - self.outcomes[j][OutcomeKeys.PERCENTAGE_USERS]
                )
                if difference > self.settings.percentage_gap:
                    self.decision["choice"] = self.__return_choice(OutcomeKeys.PERCENTAGE_USERS)
                    return
        self.odds_strategy(balance)

    def event_strategy(self, balance: int, event: BetEvent) -> bool:
        """
        Kelly Criterion betting strategy that calculates optimal bet amounts for long-term growth.

        This method implements the Kelly Criterion, a mathematically proven formula for optimal
        betting that maximizes logarithmic utility (long-term growth rate). It solves the
        circular dependency where placing a bet changes the odds, but Kelly requires final odds.

        Mathematical Foundation:
        ----------------------
        Standard Kelly Criterion: f* = (bp - q) / b
        Where:
        - f* = fraction of capital to wager
        - b = odds received on the wager (decimal odds - 1)
        - p = probability of winning
        - q = probability of losing (1 - p)

        For our betting context: f* = (odds * probability - 1) / (odds - 1)

        Circular Dependency Problem:
        ---------------------------
        The challenge: Our bet amount changes the final odds, but Kelly needs those final odds
        to calculate the optimal bet amount. This creates a circular dependency:
        - Need bet_amount to calculate final_odds = (total + bet_amount) / (outcome + bet_amount)
        - Need final_odds to calculate Kelly fraction
        - Need Kelly fraction to calculate bet_amount

        Mathematical Solution:
        ---------------------
        We solve this by deriving a direct formula that accounts for the odds impact:

        Starting with Kelly: f* = (odds * p - 1) / (odds - 1)
        Where odds = (total + x) / (outcome + x) and x = balance * f*

        Algebraic manipulation yields:
        x = balance * (total * p - outcome) / ((total - outcome) - balance * (p - 1))

        This formula directly calculates the Kelly-optimal bet amount while accounting
        for how that bet will change the final odds, eliminating the circular dependency.

        Variable Definitions:
        -------------------
        - total: Total points in the betting pool
        - outcome: Points currently on the outcome we're considering
        - p: Probability of this outcome winning
        - balance: Available capital to bet with
        - x: Optimal bet amount (Kelly-optimal)

        Example Calculation:
        ------------------
        Given: balance=100k, probability=60%, total=500k, outcome=200k
        - p = 0.6
        - numerator = 100k * (500k * 0.6 - 200k) = 100k * 100k = 10B
        - denominator = (500k - 200k) - 100k * (0.6 - 1) = 300k + 40k = 340k
        - x = 10B / 340k ≈ 29,412 points

        Risk Management Features:
        ------------------------
        1. Kelly Criterion: Mathematically optimal for long-term growth
           - Prevents overbetting by optimizing log utility
           - Automatically scales bet size based on edge and odds

        2. Conservative Scaling: Uses event_percentage as a fraction of Kelly
           - Acts as a "fractional Kelly" for more conservative betting
           - Reduces variance while maintaining positive expected growth

        3. Platform Constraints: Respects Twitch's 250,000 point maximum bet limit

        4. Balance Protection: Never bets more than the available balance

        5. Edge Detection: Only bets when there's positive expected value

        Advantages of the Strategy:
        --------------------------
        - Mathematically optimal: Kelly Criterion is proven for long-term growth
        - Self-correcting: Automatically adjusts for market impact of the bet
        - Risk-proportional: Bet size scales with edge strength and bankroll
        - Conservative: Fractional Kelly implementation reduces variance
        - Robust: Handles edge cases and prevents negative expected value bets

        Args:
            balance: User's current channel point balance
            event: BetEvent containing outcome patterns and expected probabilities

        Returns:
            bool: True if a profitable bet was calculated and decision was set, False otherwise

        Note:
            This method modifies self.decision with the calculated Kelly-optimal bet choice and amount.
            The algorithm will skip outcomes where there is no positive expected value.
        """

        if self.settings.event_percentage is None:
            return False

        failed_logger_extra = {"color": Settings.logger.color_palette.get(Events.BET_FAILED)}

        if self.total_points <= 0:
            logger.error("Total points in betting pool is zero, cannot calculate bet", failed_logger_extra)
            return False

        # Print identifiable event name
        bet_info = "(Unknown Bet)"
        if event.name is not None:
            bet_info = f"(Name: {event.name})"
        elif event.title is not None:
            bet_info = f"(Title: {event.title.pattern})"
        elif event.filter is not None and getattr(event.filter, "__name__", None) not in ["<lambda>", None]:
            bet_info = f"(Filter: {event.filter.__name__})"

        # Validate that outcome counts match in strict mode
        if event.strict and len(self.outcomes) != len(event.event_chances):
            logger.info(
                f"Event outcomes counts don't match {bet_info}: {len(self.outcomes)=} != {len(event.event_chances)=}",
                failed_logger_extra,
            )
            return False

        # Match bet outcomes with event probability patterns
        total_chance = 0
        outcome_chances = [0.0] * len(self.outcomes)
        available_event_titles = [outcome for outcome, chance in event.event_chances]

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
                logger.info(f"Event outcome not found {bet_info}: {decision_outcome['title']}", failed_logger_extra)
                return False

        # In strict mode, ensure all event outcomes were matched
        if event.strict and len(available_event_titles) > 0:
            outcomes_left = ", ".join(f"'{outcome.pattern}'" for outcome in available_event_titles)
            logger.info(f"Event outcomes left after matching {bet_info}: {outcomes_left}", failed_logger_extra)
            return False

        # Normalize probabilities to sum to 1.0
        for i in range(len(outcome_chances)):
            outcome_chances[i] /= total_chance  # Find the most profitable betting opportunity using Kelly Criterion
        decision: Optional[int] = None
        decision_bet_amount = 0
        decision_odds_after_bet = 0
        decision_expected_value = 0

        for i, outcome in enumerate(self.outcomes):
            # Skip outcomes where current odds are already worse than expected (no positive edge)
            current_implied_probability = outcome[OutcomeKeys.TOTAL_POINTS] / self.total_points
            if current_implied_probability >= outcome_chances[i]:
                continue

            # Kelly Criterion variables
            total = self.total_points  # Total points in betting pool
            outcome_points = outcome[OutcomeKeys.TOTAL_POINTS]  # Points on this outcome
            probability = outcome_chances[i]  # True probability of this outcome winning

            effective_balance = balance * (self.settings.event_percentage / 100)  # Effective balance for betting
            if effective_balance <= 0:
                continue  # Skip if no effective balance to bet

            current_odds = total / outcome_points if outcome_points > 0 else float("inf")
            if current_odds <= 1:
                continue

            # Solve the circular dependency directly using the Kelly Criterion
            # Standard Kelly: f* = (odds * p - 1) / (odds - 1)
            # But since our bet changes the odds, we solve for x where:
            # x = balance * f* and final_odds = (total + x) / (outcome + x)
            #
            # Mathematical derivation yields:
            # x = balance * (total * p - outcome) / ((total - outcome) - balance * (p - 1))

            numerator = effective_balance * (total * probability - outcome_points)
            denominator = (total - outcome_points) - effective_balance * (probability - 1)

            # Check for mathematical validity
            if denominator <= 0:
                # This occurs when the bet would be too large relative to the pool
                # or when the probability is too close to 1.0
                continue

            # Calculate the Kelly-optimal bet amount that accounts for odds impact
            bet_amount = numerator / denominator

            # Verify the bet amount is positive (negative indicates no edge)
            if bet_amount <= 0:
                continue

            # Apply conservative scaling (fractional Kelly) to reduce variance
            bet_amount *= event.confidence

            # Apply safety constraints
            if bet_amount <= 0:
                continue

            # Limit to Twitch platform maximum
            bet_amount = min(bet_amount, 250000)

            # Ensure we don't bet more than the event's maximum points if specified
            if event.max_points is not None:
                bet_amount = min(bet_amount, event.max_points)

            # Ensure we don't bet more than the effective balance
            bet_amount = min(bet_amount, effective_balance)

            # Calculate expected value of this Kelly-optimal bet
            odds_after_bet = (total + bet_amount) / (outcome_points + bet_amount)
            expected_value = odds_after_bet * bet_amount * probability - bet_amount

            # Track the best opportunity
            if expected_value > decision_expected_value:
                decision = i
                decision_bet_amount = bet_amount
                decision_odds_after_bet = odds_after_bet
                decision_expected_value = expected_value

        if decision is None:
            logger.error(f"No profitable outcome found {bet_info}", failed_logger_extra)
            return True

        # Apply final constraints to the chosen bet
        self.decision["choice"] = decision
        self.decision["amount"] = decision_bet_amount

        # Logging error if a single '%' is used rather than double '%%'
        logger.info(
            f"Calculated bet for event {bet_info}: "
            f"New Bet Percent: {1 / decision_odds_after_bet:.2%}% (Odds: {decision_odds_after_bet:.2f}), "
            f"Expected Chance: {outcome_chances[decision]:.2%}%, "
            f"Expected Value: {decision_expected_value:.2f}",
            {"color": Settings.logger.color_palette.get(Events.BET_GENERAL)},
        )

        return True
