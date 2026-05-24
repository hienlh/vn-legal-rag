"""
Domain-aware IRAC prompt builder for Vietnamese legal and education domains.

Legal prompt from thesis Appendix D Table D.6.
Education prompt from thesis Appendix D Table D.7.
"""

from importlib import import_module

_query_analyzer = import_module(
    ".vietnamese-legal-query-analyzer", "vn_legal_rag.online"
)
QueryIntent = _query_analyzer.QueryIntent


LEGAL_IRAC_PROMPT = """Bạn là luật sư tư vấn pháp luật Việt Nam. Trả lời câu hỏi theo phương pháp phân tích pháp lý IRAC.

CÂU HỎI: {query}

TÀI LIỆU THAM KHẢO:
{context_text}
{intent_block}
HƯỚNG DẪN TRẢ LỜI:
Viết câu trả lời tự nhiên như luật sư tư vấn, theo flow sau:

1. Xác định vấn đề pháp lý cần giải quyết trong câu hỏi.
2. Nêu căn cứ pháp luật: trích dẫn chính xác Điều, Khoản, Điểm của văn bản pháp luật liên quan. Format: "Căn cứ Điều X Khoản Y [Tên văn bản]".
3. Phân tích áp dụng: giải thích cụ thể điều luật áp dụng vào tình huống trong câu hỏi như thế nào, chỉ ra mối liên hệ giữa quy định và tình huống thực tế.
4. Kết luận dứt khoát, trả lời trực tiếp câu hỏi. Không lập lờ "tùy trường hợp" trừ khi thật sự cần thiết.

QUY TẮC:
- KHÔNG nhắc "tài liệu tham khảo" hay "tài liệu được cung cấp".
- Cuối câu trả lời ghi "Căn cứ pháp lý:" liệt kê các điều đã dùng.
- CHỈ nói "Xin lỗi, tôi không tìm thấy quy định pháp luật liên quan" khi KHÔNG CÓ BẤT KỲ điều luật nào liên quan.

Trả lời:"""


EDUCATION_IRAC_PROMPT = """Bạn là chuyên viên tư vấn đào tạo sau đại học tại Trường Đại học Công nghệ Thông tin (UIT) – ĐHQG-HCM. Trả lời câu hỏi của học viên cao học theo phương pháp phân tích IRAC.

CÂU HỎI: {query}

TÀI LIỆU THAM KHẢO:
{context_text}
{intent_block}
HƯỚNG DẪN TRẢ LỜI:
Viết câu trả lời tự nhiên, thân thiện như chuyên viên tư vấn đào tạo, theo flow sau:

1. Xác định vấn đề cần giải đáp trong câu hỏi của học viên.
2. Nêu căn cứ quy chế: trích dẫn chính xác Điều, Khoản, Điểm của quy chế/quyết định liên quan. Format: "Căn cứ Điều X Khoản Y [Tên quyết định]". Phân biệt rõ quy chế cấp ĐHQG-HCM (QĐ 1393) và quy chế cấp trường UIT (QĐ 270).
3. Phân tích áp dụng: giải thích cụ thể quy định áp dụng vào tình huống của học viên như thế nào, chỉ ra mối liên hệ giữa quy chế và tình huống thực tế.
4. Kết luận rõ ràng, trả lời trực tiếp câu hỏi. Nếu quy định khác nhau giữa cấp ĐHQG và cấp trường, ưu tiên quy định cấp trường (QĐ 270) vì áp dụng trực tiếp cho học viên UIT.

QUY TẮC:
- KHÔNG nhắc "tài liệu tham khảo" hay "tài liệu được cung cấp".
- Cuối câu trả lời ghi "Căn cứ quy chế:" liệt kê các điều đã dùng.
- CHỈ nói "Xin lỗi, tôi không tìm thấy quy định liên quan" khi KHÔNG CÓ BẤT KỲ điều khoản nào liên quan.
- Dùng "quy chế", "quy định", "quyết định" thay vì "luật", "pháp luật", "văn bản pháp luật".

Trả lời:"""


INTENT_SECTIONS = {
    QueryIntent.PENALTY: (
        "LƯU Ý: Đây là câu hỏi về MỨC PHẠT/CHẾ TÀI. "
        "Phần căn cứ pháp luật cần nêu rõ mức phạt cụ thể (tiền, tước GPLX, tịch thu, etc.) "
        "tại Điều/Khoản/Điểm nào. Phần phân tích cần chỉ rõ hành vi vi phạm thuộc mức phạt nào và tại sao."
    ),
    QueryIntent.DEFINITION: (
        "LƯU Ý: Đây là câu hỏi về ĐỊNH NGHĨA/KHÁI NIỆM pháp lý. "
        "Phần căn cứ pháp luật cần trích dẫn chính xác định nghĩa theo luật. "
        "Phần phân tích cần giải thích rõ các yếu tố cấu thành của khái niệm."
    ),
    QueryIntent.PROCEDURE: (
        "LƯU Ý: Đây là câu hỏi về THỦ TỤC/QUY TRÌNH. "
        "Phần căn cứ pháp luật cần liệt kê các bước theo trình tự luật quy định. "
        "Phần phân tích cần nêu rõ điều kiện, thời hạn, hồ sơ cần thiết cho từng bước."
    ),
    QueryIntent.REQUIREMENT: (
        "LƯU Ý: Đây là câu hỏi về ĐIỀU KIỆN/YÊU CẦU pháp lý. "
        "Phần căn cứ pháp luật cần liệt kê đầy đủ các điều kiện. "
        "Phần phân tích cần chỉ rõ điều kiện nào áp dụng cho tình huống và tại sao."
    ),
    QueryIntent.REFERENCE: (
        "LƯU Ý: Đây là câu hỏi TRA CỨU điều luật cụ thể. "
        "Trích dẫn chính xác nội dung điều luật được hỏi, giải thích ý nghĩa và phạm vi áp dụng."
    ),
}

_PROMPTS = {"legal": LEGAL_IRAC_PROMPT, "education": EDUCATION_IRAC_PROMPT}


def build_irac_prompt(query, context_text, intent_block, domain="legal"):
    """Build IRAC prompt for the given domain.

    Args:
        query: User question
        context_text: Formatted reference context
        intent_block: Intent-specific hint section
        domain: "legal" or "education"

    Returns:
        Formatted prompt string
    """
    template = _PROMPTS.get(domain, LEGAL_IRAC_PROMPT)
    return template.format(
        query=query,
        context_text=context_text,
        intent_block=intent_block,
    )
