"""
Vietnamese legal abbreviation expansion utilities.

Simplified version - keeps only essential expansion logic.
"""

from typing import Dict, List


# Common Vietnamese legal abbreviations
LEGAL_ABBREVIATIONS: Dict[str, str] = {
    # Organization roles
    "HĐQT": "Hội đồng quản trị",
    "HĐTV": "Hội đồng thành viên",
    "ĐHĐCĐ": "Đại hội đồng cổ đông",
    "TGĐ": "Tổng giám đốc",
    "GĐ": "Giám đốc",
    "BKS": "Ban kiểm soát",
    "KSV": "Kiểm soát viên",
    "CT": "Chủ tịch",
    "PCT": "Phó Chủ tịch",

    # Company types
    "CTCP": "Công ty cổ phần",
    "TNHH": "Trách nhiệm hữu hạn",
    "DNTN": "Doanh nghiệp tư nhân",
    "HTX": "Hợp tác xã",
    "DN": "Doanh nghiệp",

    # Registration/Legal terms
    "GCNĐKKD": "Giấy chứng nhận đăng ký kinh doanh",
    "GCNĐKDN": "Giấy chứng nhận đăng ký doanh nghiệp",
    "ĐKKD": "Đăng ký kinh doanh",
    "ĐKDN": "Đăng ký doanh nghiệp",
    "VĐL": "Vốn điều lệ",

    # Government agencies
    "UBND": "Ủy ban nhân dân",
    "HĐND": "Hội đồng nhân dân",
    "CP": "Chính phủ",
    "QH": "Quốc hội",
    "BTC": "Bộ Tài chính",
    "BKHĐT": "Bộ Kế hoạch và Đầu tư",

    # Legal documents
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "QĐ": "Quyết định",
    "NQ": "Nghị quyết",
}


def expand_abbreviations(text: str, abbrev_dict: Dict[str, str] = None) -> str:
    """
    Expand Vietnamese abbreviations in text.

    Args:
        text: Input text with abbreviations
        abbrev_dict: Custom abbreviation dictionary (optional)

    Returns:
        Text with expanded abbreviations
    """
    if abbrev_dict is None:
        abbrev_dict = LEGAL_ABBREVIATIONS

    result = text
    for abbrev, full_form in abbrev_dict.items():
        # Replace whole word only (avoid partial matches)
        result = result.replace(f" {abbrev} ", f" {full_form} ")
        result = result.replace(f" {abbrev},", f" {full_form},")
        result = result.replace(f" {abbrev}.", f" {full_form}.")
        result = result.replace(f"({abbrev})", f"({full_form})")

    return result


def get_abbreviation_variants(term: str) -> List[str]:
    """
    Get all known variants of a term (abbreviation + full form).

    Args:
        term: Input term (abbreviation or full form)

    Returns:
        List of variants (including original)
    """
    variants = [term]

    # Check if term is abbreviation
    if term in LEGAL_ABBREVIATIONS:
        variants.append(LEGAL_ABBREVIATIONS[term])

    # Check if term is full form
    for abbrev, full_form in LEGAL_ABBREVIATIONS.items():
        if term.lower() == full_form.lower():
            variants.append(abbrev)

    return list(set(variants))


def expand_search_query(query: str) -> List[str]:
    """
    Expand search query with abbreviation variations.

    Args:
        query: Search query

    Returns:
        List of query variants
    """
    queries = [query]

    # Expand known abbreviations
    expanded = expand_abbreviations(query)
    if expanded != query:
        queries.append(expanded)

    # For each word in query, add abbreviation variants
    words = query.split()
    for word in words:
        variants = get_abbreviation_variants(word)
        if len(variants) > 1:
            # Create new query with variant
            for variant in variants:
                if variant != word:
                    new_query = query.replace(word, variant)
                    queries.append(new_query)

    return list(set(queries))
