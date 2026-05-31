"""Entry point for cached LLM inference."""

import logging
from datetime import datetime

import os
import pandas as pd

from . import configs as cfg
from . import logprob_classifier as lc
from .agreement_config import StrategyConfig
from .agreement_runner import run_phase, _run_dir, _next_output_label

logger = logging.getLogger(__name__)


BANKING77_INTENT_NAMES = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


# -------------------------------------------------------------------
# data loading
# -------------------------------------------------------------------

def _load_clinc_oos_data(split: str = "plus", samples_per_class: int | None = None):
    """Load CLINC-OOS dataset.

    Returns (sample_list, intent_list, all_labels, label_descriptions, split).
    Intent 42 = 'oos' is the OOD class.
    """
    parquet_path = os.path.join(cfg.clinc_dir, split, "test-00000-of-00001.parquet")
    df = pd.read_parquet(parquet_path)

    if samples_per_class is not None:
        df = df.groupby("intent", group_keys=False).head(samples_per_class)
        logger.info(f"Subsampled to {samples_per_class} per class -> {len(df)} total")

    intent_names = {
        0: "restaurant_reviews", 1: "nutrition_info", 2: "account_blocked",
        3: "oil_change_how", 4: "time", 5: "weather", 6: "redeem_rewards",
        7: "interest_rate", 8: "gas_type", 9: "accept_reservations",
        10: "smart_home", 11: "user_name", 12: "report_lost_card",
        13: "repeat", 14: "whisper_mode", 15: "what_are_your_hobbies",
        16: "order", 17: "jump_start", 18: "schedule_meeting",
        19: "meeting_schedule", 20: "freeze_account", 21: "what_song",
        22: "meaning_of_life", 23: "restaurant_reservation", 24: "traffic",
        25: "make_call", 26: "text", 27: "bill_balance",
        28: "improve_credit_score", 29: "change_language", 30: "no",
        31: "measurement_conversion", 32: "timer", 33: "flip_coin",
        34: "do_you_have_pets", 35: "balance", 36: "tell_joke",
        37: "last_maintenance", 38: "exchange_rate", 39: "uber",
        40: "car_rental", 41: "credit_limit", 42: "oos",
        43: "shopping_list", 44: "expiration_date", 45: "routing",
        46: "meal_suggestion", 47: "tire_change", 48: "todo_list",
        49: "card_declined", 50: "rewards_balance", 51: "change_accent",
        52: "vaccines", 53: "reminder_update", 54: "food_last",
        55: "change_ai_name", 56: "bill_due", 57: "who_do_you_work_for",
        58: "share_location", 59: "international_visa", 60: "calendar",
        61: "translate", 62: "carry_on", 63: "book_flight",
        64: "insurance_change", 65: "todo_list_update", 66: "timezone",
        67: "cancel_reservation", 68: "transactions", 69: "credit_score",
        70: "report_fraud", 71: "spending_history", 72: "directions",
        73: "spelling", 74: "insurance", 75: "what_is_your_name",
        76: "reminder", 77: "where_are_you_from", 78: "distance",
        79: "payday", 80: "flight_status", 81: "find_phone", 82: "greeting",
        83: "alarm", 84: "order_status", 85: "confirm_reservation",
        86: "cook_time", 87: "damaged_card", 88: "reset_settings",
        89: "pin_change", 90: "replacement_card_duration", 91: "new_card",
        92: "roll_dice", 93: "income", 94: "taxes", 95: "date",
        96: "who_made_you", 97: "pto_request", 98: "tire_pressure",
        99: "how_old_are_you", 100: "rollover_401k", 101: "pto_request_status",
        102: "how_busy", 103: "application_status", 104: "recipe",
        105: "calendar_update", 106: "play_music", 107: "yes",
        108: "direct_deposit", 109: "credit_limit_change", 110: "gas",
        111: "pay_bill", 112: "ingredients_list", 113: "lost_luggage",
        114: "goodbye", 115: "what_can_i_ask_you", 116: "book_hotel",
        117: "are_you_a_bot", 118: "next_song", 119: "change_speed",
        120: "plug_type", 121: "maybe", 122: "w2", 123: "oil_change_when",
        124: "thank_you", 125: "shopping_list_update", 126: "pto_balance",
        127: "order_checks", 128: "travel_alert", 129: "fun_fact",
        130: "sync_device", 131: "schedule_maintenance", 132: "apr",
        133: "transfer", 134: "ingredient_substitution", 135: "calories",
        136: "current_location", 137: "international_fees", 138: "calculator",
        139: "definition", 140: "next_holiday", 141: "update_playlist",
        142: "mpg", 143: "min_payment", 144: "change_user_name",
        145: "restaurant_suggestion", 146: "travel_notification",
        147: "cancel", 148: "pto_used", 149: "travel_suggestion",
        150: "change_volume",
    }
    intent_list = [intent_names[idx] for idx in df["intent"].tolist()]
    sample_list = df["text"].tolist()
    all_labels = [intent_names[i] for i in range(151)]

    label_descriptions = {
        "w2": "W2 form for Wage and Tax Statement",
        "no": "saying no, refusing",
        "oos": "out of set, out of distribution",
        "apr": "Annual Percentage Rate",
        "mpg": "Miles Per Gallon",
        "pto_used": "used Paid Time Off",
    }
    return sample_list, intent_list, all_labels, label_descriptions, split


def _load_banking77_data(samples_per_class: int | None = None):
    """Load BANKING77 test data.

    Returns (sample_list, intent_list, all_labels, label_descriptions, split).
    BANKING77 has 77 in-domain banking intents and no explicit OOD class.
    """
    parquet_path = os.path.join(cfg.banking77_dir, "data", "test-00000-of-00001-ca30debcf4e3446b.parquet")
    df = pd.read_parquet(parquet_path)

    if samples_per_class is not None:
        df = df.groupby("label", group_keys=False).head(samples_per_class)
        logger.info(f"Subsampled to {samples_per_class} per class -> {len(df)} total")

    intent_names = {idx: name for idx, name in enumerate(BANKING77_INTENT_NAMES)}
    intent_list = [intent_names[idx] for idx in df["label"].tolist()]
    sample_list = df["text"].tolist()
    all_labels = list(BANKING77_INTENT_NAMES)
    label_descriptions = {
        name: name.replace("_", " ").replace("?", "")
        for name in BANKING77_INTENT_NAMES
    }
    return sample_list, intent_list, all_labels, label_descriptions, "banking77_test"


def _load_dataset(dataset: str, samples_per_class: int | None = None):
    if dataset == "banking77":
        return _load_banking77_data(samples_per_class=samples_per_class)
    if dataset == "clinc_oos":
        return _load_clinc_oos_data(samples_per_class=samples_per_class)
    raise ValueError(f"Unsupported dataset: {dataset}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Weak-strong agreement experiments\n"
        "Usage:\n"
        "  Run inference (output saved to results/dual_cache/<output>/):\n"
        "     python -m src.run_task --model MODEL --output DIR [--quick N]\n"
        "     e.g. --output Qwen3.6-27B-run0"
    )
    parser.add_argument("--quick", type=int, default=None, metavar="N",
        help="Quick test: N samples per class")
    parser.add_argument("--model", type=str, default=None,
        help="Model name (e.g. cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4)")
    parser.add_argument("--dataset", choices=["banking77", "clinc_oos"], default="banking77",
        help="Dataset to use. Default: banking77")
    parser.add_argument("--output", type=str, default=None,
        help="Output directory name (e.g. Qwen3.6-27B-run0). Omitted -> auto")
    args = parser.parse_args()

    if args.model:
        model = args.model
        sample_list, intent_list, all_labels, label_descriptions, split = (
            _load_dataset(args.dataset, samples_per_class=args.quick)
        )
        output_label = args.output or _next_output_label(model, split)
        run_d = _run_dir(output_label)
        os.makedirs(run_d, exist_ok=True)
        config = StrategyConfig(model_A=model, model_B=model, num_runs=3)
        logfile_path = run_d + f"/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cfg.configure_logger(logfile_path, mode="w", reset=True, logging_level=logging.INFO)

        prompt_template = lc.PROMPT_TEMPLATES["Rank"]["ranking"]
        logger.info(f"=== Running model: {model} (output: {output_label}) ===")
        run_phase(
            model, sample_list, all_labels, label_descriptions,
            config, split, prompt_template, output_label=output_label,
        )
        logger.info(f"Done. Cache: {output_label}")
    else:
        parser.print_help()
