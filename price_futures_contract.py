"""
Compare a real PJM Western Hub power futures contract (e.g. quoted on ICE)
against this project's own forward-curve prediction, to check for potential
mispricing.

Usage:
    python price_futures_contract.py --delivery-month 2026-09 --peak-type on_peak --quoted-price 42.50
    python price_futures_contract.py --delivery-month 2027-01 --peak-type monthly_avg --quoted-price 55 --scenario hot

Scoped to PJM Western Hub only. Prices are supplied manually — there's no
live ICE data feed here (enterprise-only, out of reach).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from temperature_modeling.futures_pricer import price_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", default="pjm", help="ISO code (only 'pjm' is supported right now)")
    parser.add_argument("--delivery-month", required=True, help="Delivery month, YYYY-MM")
    parser.add_argument("--peak-type", required=True, choices=["monthly_avg", "on_peak", "off_peak"])
    parser.add_argument("--quoted-price", required=True, type=float, help="The market-quoted $/MWh price")
    parser.add_argument("--scenario", default="base", choices=["cold", "base", "hot"],
                         help="Weather scenario to compare against (default: base)")
    args = parser.parse_args()

    try:
        result = price_contract(
            iso=args.iso,
            delivery_month=args.delivery_month,
            peak_type=args.peak_type,
            quoted_price=args.quoted_price,
            scenario=args.scenario,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    band = (f"${result['band_low']:.2f} - ${result['band_high']:.2f}"
            if result["band_source"] == "cqr" else "unavailable")

    print(f"\n{result['iso'].upper()} {result['delivery_month']} {result['peak_type']} ({result['scenario']} scenario)")
    print("-" * 60)
    print(f"  Model price:      ${result['model_price']:.2f}/MWh")
    print(f"  90% band:         {band}")
    print(f"  Quoted price:     ${result['quoted_price']:.2f}/MWh")
    pct = f" ({result['spread_pct']:+.1f}%)" if result["spread_pct"] is not None else ""
    print(f"  Spread:           ${result['spread_usd_mwh']:+.2f}/MWh{pct}")
    print(f"  Signal:           {result['signal']}")
    print(f"  Lead time:        {result['lead_months']} months")
    print(f"  Model source:     {result['model_source']}")
    print(f"  Confidence:       {result['confidence']}")
    if result["confidence_notes"]:
        print("  Notes:")
        for note in result["confidence_notes"]:
            print(f"    - {note}")
    print()


if __name__ == "__main__":
    main()
