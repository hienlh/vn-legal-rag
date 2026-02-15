"""
Legal domain entity types for Vietnamese legal document NER extraction.

Entity types designed for:
- Vietnamese legal documents (Luật, Nghị định, Thông tư)
- Corporate law focus (Luật Doanh nghiệp)
- Support for both Vietnamese abbreviations and full forms
"""

from enum import Enum
from typing import List


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


# Entity types list for NER extraction
LEGAL_ENTITY_TYPES: List[str] = [e.value for e in LegalEntityType]
