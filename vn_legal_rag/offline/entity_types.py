"""
Legal domain entity types for Vietnamese legal document NER extraction.

Entity types designed for:
- Vietnamese legal documents (Luật, Nghị định, Thông tư)
- Corporate law focus (Luật Doanh nghiệp)
- Support for both Vietnamese abbreviations and full forms

Ported from semantica/legal/entity_types.py
"""

from enum import Enum
from typing import Dict, List


class LegalEntityType(Enum):
    """Entity types for Vietnamese legal documents."""

    # Organizations & Legal Entities
    ORGANIZATION = "ORGANIZATION"  # doanh nghiệp, công ty, tổ chức, cơ quan

    # Persons & Roles
    PERSON_ROLE = "PERSON_ROLE"  # Giám đốc, TGĐ, thành viên HĐQT, Chủ tịch

    # Legal Terms & Concepts
    LEGAL_TERM = "LEGAL_TERM"  # vốn điều lệ, cổ phần, ĐHĐCĐ, hợp đồng

    # Legal References (articles, clauses, laws)
    LEGAL_REFERENCE = "LEGAL_REFERENCE"  # Điều 5, Khoản 1, Luật Doanh nghiệp

    # Quantities & Values
    MONETARY = "MONETARY"  # 10 triệu đồng, 50% vốn điều lệ
    PERCENTAGE = "PERCENTAGE"  # 51%, trên 50%, dưới 35%
    DURATION = "DURATION"  # 30 ngày, 06 tháng, 2 năm

    # Locations & Geographic
    LOCATION = "LOCATION"  # Việt Nam, Hà Nội, quận/huyện

    # Logic & Conditions
    CONDITION = "CONDITION"  # nếu, trường hợp, khi, trừ trường hợp

    # Actions & Activities
    ACTION = "ACTION"  # thành lập, giải thể, đăng ký, chuyển nhượng

    # Consequences & Penalties
    PENALTY = "PENALTY"  # phạt tiền, đình chỉ hoạt động, tước quyền


# Entity types list for config
LEGAL_ENTITY_TYPES: List[str] = [e.value for e in LegalEntityType]


# Vietnamese regex patterns for entity extraction (pattern-based fallback)
ENTITY_PATTERNS: Dict[LegalEntityType, List[str]] = {
    LegalEntityType.ORGANIZATION: [
        r"(công ty(?:\s+(?:cổ phần|TNHH|hợp danh|tư nhân))?)",
        r"(doanh nghiệp(?:\s+(?:nhà nước|tư nhân|có vốn đầu tư nước ngoài))?)",
        r"(tổ chức(?:\s+(?:kinh tế|tín dụng|phi lợi nhuận))?)",
        r"(cơ quan(?:\s+(?:nhà nước|quản lý))?)",
        r"(hợp tác xã)",
        r"(chi nhánh)",
        r"(văn phòng đại diện)",
        r"\b(CTCP|TNHH|DNTN|DNNN|HTX)\b",
    ],
    LegalEntityType.PERSON_ROLE: [
        r"((?:Chủ tịch|Phó Chủ tịch)(?:\s+(?:Hội đồng quản trị|Hội đồng thành viên|công ty))?)",
        r"((?:Giám đốc|Phó giám đốc|Tổng giám đốc|Phó Tổng giám đốc))",
        r"((?:thành viên|Thành viên)(?:\s+(?:Hội đồng quản trị|Hội đồng thành viên|Ban kiểm soát))?)",
        r"((?:cổ đông|Cổ đông)(?:\s+(?:sáng lập|lớn|nhỏ))?)",
        r"((?:người|Người)(?:\s+(?:đại diện theo pháp luật|quản lý|điều hành)))",
        r"((?:Kiểm soát viên|kiểm soát viên))",
        r"((?:Kế toán trưởng|kế toán trưởng))",
        r"\b(GĐ|TGĐ|HĐQT|HĐTV|BKS|KSV|ĐHĐCĐ)\b",
    ],
    LegalEntityType.LEGAL_TERM: [
        r"(vốn(?:\s+(?:điều lệ|góp|pháp định|chủ sở hữu)))",
        r"(cổ phần(?:\s+(?:phổ thông|ưu đãi|biểu quyết))?)",
        r"(phần vốn góp)",
        r"((?:Điều lệ|điều lệ)(?:\s+công ty)?)",
        r"(giấy(?:\s+(?:chứng nhận đăng ký doanh nghiệp|phép kinh doanh))?)",
        r"(hợp đồng(?:\s+(?:lao động|thương mại|dân sự))?)",
        r"(biên bản(?:\s+(?:họp|nghị quyết))?)",
        r"(quyền(?:\s+(?:sở hữu|sử dụng|biểu quyết))?)",
        r"(nghĩa vụ(?:\s+(?:tài chính|thuế))?)",
        r"\b(GCNĐKDN|ĐKKD|HĐLĐ)\b",
    ],
    LegalEntityType.LEGAL_REFERENCE: [
        r"(Điều\s+\d+(?:\s+(?:và|,)\s+Điều\s+\d+)*)",
        r"(Khoản\s+\d+(?:\s+(?:và|,)\s+Khoản\s+\d+)*)",
        r"(Điểm\s+[a-zđ](?:\s+(?:và|,)\s+Điểm\s+[a-zđ])*)",
        r"(Chương\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X|\d+))",
        r"(Mục\s+\d+)",
        r"((?:Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+(?:số\s+)?\d+/\d+/[A-ZĐ\-]+)",
    ],
    LegalEntityType.LOCATION: [
        r"(Việt Nam)",
        r"((?:nước|quốc gia)\s+(?:ngoài|khác))",
        r"(Hà Nội|Hồ Chí Minh|Đà Nẵng|Hải Phòng|Cần Thơ)",
    ],
    LegalEntityType.MONETARY: [
        r"(\d+(?:[.,]\d+)?(?:\s+(?:triệu|tỷ|nghìn|tỉ))?\s*(?:đồng|VND|USD|EUR))",
        r"(\d+(?:[.,]\d+)?\s*%\s*(?:vốn|vốn điều lệ|tổng vốn|giá trị))",
    ],
    LegalEntityType.PERCENTAGE: [
        r"(\d+(?:[.,]\d+)?\s*%)",
        r"((?:trên|dưới|ít nhất|hơn|không quá)\s+\d+(?:[.,]\d+)?\s*%)",
    ],
    LegalEntityType.DURATION: [
        r"((?:trong\s+)?(?:thời hạn\s+)?(\d+)\s*(ngày|tháng|năm|giờ|tuần))",
        r"((?:không quá|tối đa|ít nhất|tối thiểu)\s+(\d+)\s*(ngày|tháng|năm))",
    ],
    LegalEntityType.CONDITION: [
        r"((?:nếu|Nếu)(?:\s+như)?)",
        r"((?:trường hợp|Trường hợp)(?:\s+(?:nếu|này|khác|đặc biệt))?)",
        r"((?:khi|Khi)(?:\s+(?:nào|đó))?)",
        r"((?:trừ|Trừ)(?:\s+(?:trường hợp|khi|phi))?)",
    ],
    LegalEntityType.ACTION: [
        r"(thành lập(?:\s+(?:công ty|doanh nghiệp))?)",
        r"(giải thể(?:\s+(?:công ty|doanh nghiệp))?)",
        r"((?:đăng ký|Đăng ký)(?:\s+(?:kinh doanh|doanh nghiệp|thay đổi))?)",
        r"((?:chuyển nhượng|Chuyển nhượng)(?:\s+(?:cổ phần|vốn|quyền))?)",
        r"(sáp nhập(?:\s+(?:công ty|doanh nghiệp))?)",
    ],
    LegalEntityType.PENALTY: [
        r"(phạt\s+tiền(?:\s+(?:từ|đến|tối đa|tối thiểu))?\s*\d*(?:[.,]\d+)?\s*(?:triệu|tỷ)?(?:\s*đồng)?)",
        r"(đình chỉ(?:\s+(?:hoạt động|kinh doanh|thi công))?)",
        r"((?:tước|thu hồi)(?:\s+(?:quyền|giấy phép|chứng chỉ))?)",
    ],
}


# Vietnamese NER prompt template
LEGAL_NER_PROMPT_VI = """Bạn là chuyên gia trích xuất thực thể từ văn bản pháp luật Việt Nam.

Trích xuất các loại thực thể sau từ văn bản:

1. ORGANIZATION: tên công ty, tổ chức, cơ quan nhà nước
2. PERSON_ROLE: chức vụ, vai trò trong tổ chức
3. LEGAL_TERM: thuật ngữ pháp lý, khái niệm pháp luật
4. LEGAL_REFERENCE: tham chiếu đến điều khoản, văn bản pháp luật
5. LOCATION: địa danh, vị trí địa lý
6. MONETARY: số tiền, giá trị tài chính
7. DURATION: thời hạn, kỳ hạn
8. PERCENTAGE: tỷ lệ phần trăm
9. CONDITION: điều kiện áp dụng
10. ACTION: hành vi pháp lý
11. PENALTY: hình phạt, chế tài

Lưu ý:
- Giữ nguyên các từ viết tắt (HĐQT, TGĐ, TNHH, CTCP, ĐHĐCĐ)
- "doanh nghiệp", "hộ kinh doanh" LUÔN LÀ ORGANIZATION

Văn bản cần trích xuất:
{text}

Trả về JSON array với format:
[{{"text": "...", "label": "ENTITY_TYPE", "start_char": N, "end_char": M}}]
"""


# Common abbreviation mappings for entity type disambiguation
ABBREVIATION_TO_ENTITY_TYPE: Dict[str, LegalEntityType] = {
    # Organization abbreviations
    "CTCP": LegalEntityType.ORGANIZATION,
    "TNHH": LegalEntityType.ORGANIZATION,
    "DNTN": LegalEntityType.ORGANIZATION,
    "DNNN": LegalEntityType.ORGANIZATION,
    "HTX": LegalEntityType.ORGANIZATION,
    # Person/Role abbreviations
    "GĐ": LegalEntityType.PERSON_ROLE,
    "TGĐ": LegalEntityType.PERSON_ROLE,
    "HĐQT": LegalEntityType.PERSON_ROLE,
    "HĐTV": LegalEntityType.PERSON_ROLE,
    "BKS": LegalEntityType.PERSON_ROLE,
    "KSV": LegalEntityType.PERSON_ROLE,
    "ĐHĐCĐ": LegalEntityType.PERSON_ROLE,
    # Legal term abbreviations
    "GCNĐKDN": LegalEntityType.LEGAL_TERM,
    "ĐKKD": LegalEntityType.LEGAL_TERM,
    "HĐLĐ": LegalEntityType.LEGAL_TERM,
}


# Known abbreviations for context
LEGAL_ABBREVIATIONS = {
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


# Synonym mappings for entity deduplication
LEGAL_SYNONYMS: Dict[str, List[str]] = {
    # Organizations
    "doanh nghiệp": ["Doanh nghiệp", "DOANH NGHIỆP", "doanh-nghiệp", "DN"],
    "công ty": ["Công ty", "CÔNG TY", "cong ty", "CTY"],
    "công ty cổ phần": ["Công ty cổ phần", "CTCP", "cty cổ phần"],
    "công ty tnhh": ["Công ty TNHH", "công ty trách nhiệm hữu hạn", "TNHH"],
    # Roles
    "giám đốc": ["Giám đốc", "GĐ", "giam doc", "GIÁM ĐỐC"],
    "tổng giám đốc": ["Tổng giám đốc", "TGĐ", "tong giam doc"],
    "chủ tịch": ["Chủ tịch", "CT", "chu tich"],
    "hội đồng quản trị": ["Hội đồng quản trị", "HĐQT", "hoi dong quan tri"],
    "hội đồng thành viên": ["Hội đồng thành viên", "HĐTV"],
    "đại hội đồng cổ đông": ["Đại hội đồng cổ đông", "ĐHĐCĐ"],
    "cổ đông": ["Cổ đông", "cổ-đông", "co dong"],
    # Legal terms
    "vốn điều lệ": ["Vốn điều lệ", "VĐL", "von dieu le"],
    "cổ phần": ["Cổ phần", "CP", "co phan"],
    # Locations
    "việt nam": ["Việt Nam", "VN", "viet nam", "VIỆT NAM"],
    "hà nội": ["Hà Nội", "HA NOI", "ha noi", "HN"],
    "hồ chí minh": ["Hồ Chí Minh", "HCM", "TP.HCM", "TPHCM", "ho chi minh"],
}
