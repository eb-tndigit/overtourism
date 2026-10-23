# SPDX-License-Identifier: Apache-2.0
"""Country-name normalization for Vodafone ``userCountry`` values."""

import pandas as pd

ITALY = ("ITA", "Italia")

# (ISO alpha-3, English name) of European countries
EUROPE = (
    ("ALB", "Albania"),
    ("AND", "Andorra"),
    ("AUT", "Austria"),
    ("BLR", "Bielorussia"),
    ("BEL", "Belgio"),
    ("BIH", "Bosnia ed Erzegovina"),
    ("BGR", "Bulgaria"),
    ("HRV", "Croazia"),
    ("CYP", "Cipro"),
    ("CZE", "Repubblica Ceca"),
    ("DNK", "Danimarca"),
    ("EST", "Estonia"),
    ("FIN", "Finlandia"),
    ("FRA", "Francia"),
    ("DEU", "Germania"),
    ("GRC", "Grecia"),
    ("HUN", "Ungheria"),
    ("ISL", "Islanda"),
    ("IRL", "Irlanda"),
    ("XKX", "Kosovo"),
    ("LVA", "Lettonia"),
    ("LIE", "Liechtenstein"),
    ("LTU", "Lituania"),
    ("LUX", "Lussemburgo"),
    ("MLT", "Malta"),
    ("MDA", "Moldavia"),
    ("MCO", "Principato di Monaco"),
    ("MNE", "Montenegro"),
    ("NLD", "Paesi Bassi"),
    ("MKD", "Macedonia del Nord"),
    ("NOR", "Norvegia"),
    ("POL", "Polonia"),
    ("PRT", "Portogallo"),
    ("ROU", "Romania"),
    ("RUS", "Russia"),
    ("SMR", "San Marino"),
    ("SRB", "Serbia"),
    ("SVK", "Slovacchia"),
    ("SVN", "Slovenia"),
    ("ESP", "Spagna"),
    ("SWE", "Svezia"),
    ("CHE", "Svizzera"),
    ("TUR", "Turchia"),
    ("UKR", "Ucraina"),
    ("GBR", "Regno Unito"),
    ("VAT", "Città del Vaticano"),
)


def _build_lookup() -> dict:
    """casefolded spelling -> English name of the country (Italy included)."""
    lookup = {k.casefold(): ITALY[1] for k in ITALY}
    for iso3, name in EUROPE:
        for key in (iso3, name):
            lookup[key.casefold()] = name
    return lookup


_LOOKUP = _build_lookup()


def country_name(value):
    """Return the English name for known values; otherwise retain the supplied country label."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    return _LOOKUP.get(value.casefold(), value)
