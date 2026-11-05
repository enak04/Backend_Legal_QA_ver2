"""
src/jurisdiction.py - Hierarchical Indian jurisdiction resolution and statute mapping.

Maps Indian metropolitan cities, state capitals, and industrial hubs to:
- State / Union Territory
- Territorial High Court jurisdiction
- State-specific labor and tenancy enactments
"""

import re
from typing import Any

# Major Indian city to State & High Court mappings
INDIAN_JURISDICTION_REGISTRY: dict[str, dict[str, str]] = {
    # Karnataka
    "bangalore": {
        "city": "Bengaluru",
        "state": "Karnataka",
        "high_court": "High Court of Karnataka",
        "state_shop_act": "Karnataka Shops and Commercial Establishments Act, 1961",
        "state_rent_act": "Karnataka Rent Act, 1999",
    },
    "bengaluru": {
        "city": "Bengaluru",
        "state": "Karnataka",
        "high_court": "High Court of Karnataka",
        "state_shop_act": "Karnataka Shops and Commercial Establishments Act, 1961",
        "state_rent_act": "Karnataka Rent Act, 1999",
    },
    "mysore": {
        "city": "Mysuru",
        "state": "Karnataka",
        "high_court": "High Court of Karnataka",
        "state_shop_act": "Karnataka Shops and Commercial Establishments Act, 1961",
        "state_rent_act": "Karnataka Rent Act, 1999",
    },
    # Maharashtra
    "mumbai": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "high_court": "Bombay High Court",
        "state_shop_act": "Maharashtra Shops and Establishments (Regulation of Employment and Conditions of Service) Act, 2017",
        "state_rent_act": "Maharashtra Rent Control Act, 1999",
    },
    "pune": {
        "city": "Pune",
        "state": "Maharashtra",
        "high_court": "Bombay High Court",
        "state_shop_act": "Maharashtra Shops and Establishments (Regulation of Employment and Conditions of Service) Act, 2017",
        "state_rent_act": "Maharashtra Rent Control Act, 1999",
    },
    "nagpur": {
        "city": "Nagpur",
        "state": "Maharashtra",
        "high_court": "Bombay High Court (Nagpur Bench)",
        "state_shop_act": "Maharashtra Shops and Establishments (Regulation of Employment and Conditions of Service) Act, 2017",
        "state_rent_act": "Maharashtra Rent Control Act, 1999",
    },
    # Delhi NCT
    "delhi": {
        "city": "Delhi",
        "state": "NCT of Delhi",
        "high_court": "Delhi High Court",
        "state_shop_act": "Delhi Shops and Establishments Act, 1954",
        "state_rent_act": "Delhi Rent Control Act, 1958",
    },
    "new delhi": {
        "city": "New Delhi",
        "state": "NCT of Delhi",
        "high_court": "Delhi High Court",
        "state_shop_act": "Delhi Shops and Establishments Act, 1954",
        "state_rent_act": "Delhi Rent Control Act, 1958",
    },
    # Telangana
    "hyderabad": {
        "city": "Hyderabad",
        "state": "Telangana",
        "high_court": "High Court for the State of Telangana",
        "state_shop_act": "Telangana Shops and Establishments Act, 1988",
        "state_rent_act": "Telangana Buildings (Lease, Rent and Eviction) Control Act, 1960",
    },
    "secunderabad": {
        "city": "Secunderabad",
        "state": "Telangana",
        "high_court": "High Court for the State of Telangana",
        "state_shop_act": "Telangana Shops and Establishments Act, 1988",
        "state_rent_act": "Telangana Buildings (Lease, Rent and Eviction) Control Act, 1960",
    },
    # Tamil Nadu
    "chennai": {
        "city": "Chennai",
        "state": "Tamil Nadu",
        "high_court": "Madras High Court",
        "state_shop_act": "Tamil Nadu Shops and Establishments Act, 1947",
        "state_rent_act": "Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017",
    },
    "coimbatore": {
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "high_court": "Madras High Court",
        "state_shop_act": "Tamil Nadu Shops and Establishments Act, 1947",
        "state_rent_act": "Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017",
    },
    # West Bengal
    "kolkata": {
        "city": "Kolkata",
        "state": "West Bengal",
        "high_court": "Calcutta High Court",
        "state_shop_act": "West Bengal Shops and Establishments Act, 1963",
        "state_rent_act": "West Bengal Premises Tenancy Act, 1997",
    },
    # Haryana (NCR)
    "gurgaon": {
        "city": "Gurugram",
        "state": "Haryana",
        "high_court": "Punjab and Haryana High Court",
        "state_shop_act": "Punjab Shops and Commercial Establishments Act, 1958 (as applied to Haryana)",
        "state_rent_act": "Haryana Urban (Control of Rent and Eviction) Act, 1973",
    },
    "gurugram": {
        "city": "Gurugram",
        "state": "Haryana",
        "high_court": "Punjab and Haryana High Court",
        "state_shop_act": "Punjab Shops and Commercial Establishments Act, 1958 (as applied to Haryana)",
        "state_rent_act": "Haryana Urban (Control of Rent and Eviction) Act, 1973",
    },
    "faridabad": {
        "city": "Faridabad",
        "state": "Haryana",
        "high_court": "Punjab and Haryana High Court",
        "state_shop_act": "Punjab Shops and Commercial Establishments Act, 1958 (as applied to Haryana)",
        "state_rent_act": "Haryana Urban (Control of Rent and Eviction) Act, 1973",
    },
    # Uttar Pradesh (NCR)
    "noida": {
        "city": "Noida",
        "state": "Uttar Pradesh",
        "high_court": "Allahabad High Court",
        "state_shop_act": "Uttar Pradesh Dookan Aur Vanijya Adhishthan Adhiniyam, 1962",
        "state_rent_act": "Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972",
    },
    "lucknow": {
        "city": "Lucknow",
        "state": "Uttar Pradesh",
        "high_court": "Allahabad High Court (Lucknow Bench)",
        "state_shop_act": "Uttar Pradesh Dookan Aur Vanijya Adhishthan Adhiniyam, 1962",
        "state_rent_act": "Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972",
    },
    # Gujarat
    "ahmedabad": {
        "city": "Ahmedabad",
        "state": "Gujarat",
        "high_court": "Gujarat High Court",
        "state_shop_act": "Gujarat Shops and Establishments (Regulation of Employment and Conditions of Service) Act, 2019",
        "state_rent_act": "Gujarat Rents, Hotel and Lodging House Rates Control Act, 1947",
    },
    # Kerala
    "kochi": {
        "city": "Kochi",
        "state": "Kerala",
        "high_court": "Kerala High Court",
        "state_shop_act": "Kerala Shops and Commercial Establishments Act, 1960",
        "state_rent_act": "Kerala Buildings (Lease and Rent Control) Act, 1965",
    },
    # Rajasthan
    "jaipur": {
        "city": "Jaipur",
        "state": "Rajasthan",
        "high_court": "Rajasthan High Court",
        "state_shop_act": "Rajasthan Shops and Commercial Establishments Act, 1958",
        "state_rent_act": "Rajasthan Rent Control Act, 2001",
    },
    # Punjab & Chandigarh
    "chandigarh": {
        "city": "Chandigarh",
        "state": "Chandigarh (UT)",
        "high_court": "Punjab and Haryana High Court",
        "state_shop_act": "Punjab Shops and Commercial Establishments Act, 1958",
        "state_rent_act": "East Punjab Urban Rent Restriction Act, 1949",
    },
}


def resolve_indian_jurisdiction(location_text: str | None) -> dict[str, str] | None:
    """
    Resolves a raw city, district, or location name to its governing State and High Court.
    Returns metadata dictionary or None if not recognized.
    """
    if not location_text:
        return None

    cleaned = re.sub(r"[^\w\s]", " ", location_text.lower()).strip()
    words = cleaned.split()

    # 1. Exact match on multi-word keys
    for key, data in INDIAN_JURISDICTION_REGISTRY.items():
        if key in cleaned:
            return data

    # 2. Match on individual words
    for word in words:
        if word in INDIAN_JURISDICTION_REGISTRY:
            return INDIAN_JURISDICTION_REGISTRY[word]

    return None


def canonicalize_jurisdiction(raw_jurisdiction: str | None) -> str | None:
    """
    Returns an enriched canonical jurisdiction string e.g.:
    'Bangalore' -> 'Bangalore, Karnataka'
    """
    if not raw_jurisdiction:
        return raw_jurisdiction

    resolved = resolve_indian_jurisdiction(raw_jurisdiction)
    if resolved:
        city = resolved["city"]
        state = resolved["state"]
        if state.lower() not in raw_jurisdiction.lower():
            return f"{city}, {state}"
        return f"{city}, {state}"

    return raw_jurisdiction
