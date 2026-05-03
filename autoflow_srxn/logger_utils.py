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
        m = row.get("mech", "unknown")
        e_final = row.get("e_final")
        if e_final is not None:
            if m not in best_by_mech or e_final < best_by_mech[m].get("e_final", 1e10):
                best_by_mech[m] = row

    best_ids = {res["id"] for res in best_by_mech.values()} if best_by_mech else set()

    logger.info("\n" + "=" * 135)
    logger.info(f" {title}")
    logger.info("-" * 135)
    
    # Dynamic header based on available keys
    has_e_final = any("e_final" in r for r in summary_data)
    has_e_init = any("e_initial" in r for r in summary_data)
    has_stage = any("stage" in r for r in summary_data)
    
    header = f"{'ID':<4} | "
    if has_stage: header += f"{'Stage':<10} | "
    header += f"{'Mechanism':<15} | "
    if has_e_init: header += f"{'E_initial (eV)':<15} | "
    if has_e_final: header += f"{'E_final (eV)':<15} | {'Delta (eV)':<10} | "
    header += f"{'E_ads (eV)':<10} | {'Note'}"
    
    logger.info(header)
    logger.info("-" * 135)
    
    for row in summary_data:
        marker = "* (Best Pose)" if row.get("id") in best_ids else ""
        e_ads = row.get("e_ads", 0.0)
        mech = row.get("mech", "unknown")
        
        line = f"{row.get('id', 0):<4} | "
        if has_stage: line += f"{row.get('stage', ''):<10} | "
        line += f"{mech[:15]:<15} | "
        if has_e_init:
            line += f"{row.get('e_initial', 0.0):15.4f} | "
        if has_e_final:
            line += f"{row.get('e_final', 0.0):15.4f} | {row.get('delta', 0.0):10.4f} | "
        line += f"{e_ads:10.4f} | {row.get('note', marker)}"
        
        logger.info(line)
    logger.info("=" * 135 + "\n")
