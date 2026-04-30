import logging
import os
import sys


def setup_logger(log_path="workflow.log", verbose=False, mode="a"):
    """Sets up a logger that outputs to both a file and the console."""
    logger = logging.getLogger("AutoFlow-SRXN")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers to allow re-configuration and avoid duplicates
    # (e.g., if a default logger was initialized before the specific prefix-log)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure log directory exists
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        import os

        print(f"Warning: Could not setup file logging at {os.path.relpath(log_path)}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_workflow_logger():
    return logging.getLogger("AutoFlow-SRXN")


def log_stage_title(logger, stage_name, description):
    """Logs a standardized stage header."""
    logger.info(f"{stage_name}: {description}")


def log_energy_comparison(logger, label, e_init, e_final):
    """Logs a standardized energy comparison between two states."""
    delta = e_final - e_init
    logger.info(
        f"  [{label}] E_initial: {e_init:12.4f} eV, E_final: {e_final:12.4f} eV, Delta: {delta:10.4f} eV"
    )


def log_results_table(logger, summary_data, title="Optimization Summary"):
    """Logs a formatted table of results including ID, mechanism, and energies."""
    if not summary_data:
        return

    # Find best (lowest e_final) for each mechanism group
    best_by_mech = {}
    for row in summary_data:
        m = row["mech"]
        if m not in best_by_mech or row["e_final"] < best_by_mech[m]["e_final"]:
            best_by_mech[m] = row

    best_ids = {res["id"] for res in best_by_mech.values()}

    logger.info("\n" + "=" * 95)
    logger.info(f" {title}")
    logger.info("-" * 95)
    logger.info(
        f"{'ID':<4} | {'Mechanism':<15} | {'E_final (eV)':<15} | {'Delta (eV)':<10} | {'E_ads (eV)':<10} | {'Note'}"
    )
    logger.info("-" * 105)
    for row in summary_data:
        marker = "* (Best Pose)" if row["id"] in best_ids else ""
        e_ads = row.get("e_ads", 0.0)
        logger.info(
            f"{row['id']:<4} | {row['mech'][:15]:<15} | {row['e_final']:15.4f} | {row['delta']:10.4f} | {e_ads:10.4f} | {marker}"
        )
    logger.info("=" * 105 + "\n")
