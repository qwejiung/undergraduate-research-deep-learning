import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, DefaultDict, Hashable, Iterable, List

from predcancer.utils import (deep_freeze, extract_config,
                              file_path_to_exp_name, my_tqdm)

SORT_KEYS = tuple(["cancer_type", "drop_cols"])


def _is_list_like(x: Any) -> bool:
    """Return True if x is list-like for our purposes."""
    return isinstance(x, (list, tuple))


def _all_numbers(L):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in L)


def _all_strings(L):
    return all(isinstance(v, str) for v in L)


def _sorted_safely(seq: Iterable[Any]) -> List[Any]:
    lst = list(seq)

    assert _all_numbers(lst) or _all_strings(lst)
    return sorted(lst)


def modify_dict(d: dict, sort_keys: List[str], exp_name: str = None):
    """
    Apply in-place modifications:
      1) ...
      2) For each key in sort_keys, if the value is list-like, sort it.
    """
    if d["model"] == "Transfer":
        if "ft_lr" in d:  # with fine-tuning
            if d.get("lora") is True:
                d["ft_head_only"] = False
            if "lora" not in d:
                d["lora"] = False
        if d.get("sp_reg") is False:
            if "sp_reg_coeff" in d:
                del d["sp_reg_coeff"]
        if d["use_epoch_loader"] is False and "pre_epoch" in d:
            del d["pre_epoch"]
        if "use_es" in d:
            d["ft_use_es"] = d["use_es"]
            d["pre_use_es"] = d["use_es"]
            del d["use_es"]
    if d["use_cv"] is False:
        if "cv_id" in d:
            del d["cv_id"]
        if "N_SPLITS" in d:
            del d["N_SPLITS"]

    if exp_name is not None:
        d["exp_name"] = exp_name

    if "optimizer" in d:
        del d["optimizer"]

    # Sort the fields in sort_keys if they are list-like
    for k in sort_keys:
        if k not in d:
            continue
        assert _is_list_like(
            d[k]
        ), f"Expected list-like for sort_keys, got {type(d[k])}"
        before = list(d[k]) if not isinstance(d[k], list) else d[k][:]
        sorted_value = _sorted_safely(d[k])
        # Ensure JSON-serializable list type
        d[k] = list(sorted_value)
        if before != d[k]:
            # Mark change via print; write happens at caller
            print(f"[INFO] Sorted field '{k}'")


def modify_json_files(directory: Path):
    """
    Load all JSON files from the given directory, modify them, and optionally deduplicate.
    When dedup is True, any file whose normalized content (after ignore_keys removal) matches
    a previously seen file will be removed.
    """
    # Keep a map from content-hash -> first-kept file path

    for file in my_tqdm(sorted(directory.glob("*.json"))):
        exp_name = file_path_to_exp_name(file)
        with open(file, "r", encoding="utf-8") as f:
            d: dict = json.load(f)

        old_d = deepcopy(d)
        modify_dict(d, SORT_KEYS, exp_name)

        # If modified, inform and write back before dedup comparison
        if old_d != d:
            print(f"[INFO] Modified {directory}/{file.name}")

            with open(file, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=4, ensure_ascii=False)

        file_name = file.stem
        if file_name != exp_name:
            print(
                f"[WARN] Filename does not match exp_name field: {file.name} vs {exp_name}"
            )
            file.rename(file.with_name(f"{exp_name}.json"))
            print(f"[INFO] Renamed file to match exp_name: {exp_name}.json")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Modify JSON files in a directory (and optionally remove duplicates)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing JSON files.",
    )
    args = parser.parse_args()

    directory = Path(args.dir)

    if not directory.exists() or not directory.is_dir():
        print(f"[ERROR] Provided path is not a valid directory: {directory}")
        return

    modify_json_files(directory)


class ConfigManager:
    def __init__(self, dict_list: List[dict]):
        self.dict_list = dict_list
        self.config_list = [extract_config(d) for d in dict_list]

        # Build a hash index for O(1) average-time lookups.
        # Key: frozen representation of config; Value: list of original dicts matching it
        self._index: DefaultDict[Hashable, List[dict]] = defaultdict(list)
        for original, cfg in my_tqdm(
            zip(self.dict_list, self.config_list),
            desc="Building index",
            total=len(self.dict_list),
        ):
            key = deep_freeze(cfg)
            self._index[key].append(original)

    def find(self, query_dict: dict):
        """
        Return all original dicts whose extracted config equals `query_dict`.
        Complexity: O(1) average (hash lookup) instead of O(N) scan.
        """
        key = deep_freeze(query_dict)

        # Return a copy to avoid accidental external mutation of internal list
        return list(self._index.get(key, []))

    def append(self, new_dict: dict):
        """
        Append a new item and update both the list and the index in sync.
        Complexity: O(log K) for sorting inside deep_freeze of dicts (K = keys),
        typically small; amortized O(1) for the index insertion.
        """
        self.dict_list.append(new_dict)
        cfg = extract_config(new_dict)
        self.config_list.append(cfg)
        self._index[deep_freeze(cfg)].append(new_dict)
