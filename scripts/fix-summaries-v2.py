"""Fix summaries: prepend routing hints to ORIGINAL content, don't replace."""
import json

# === Fix document_summaries.json ===
ds = json.load(open("data/kg_enhanced/document_summaries.json", "r", encoding="utf-8"))

# Restore original scope and prepend routing tags
ORIGINAL_SCOPES = {
    "59-2020-QH14": "Luật này quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp, bao gồm công ty trách nhiệm hữu hạn, công ty cổ phần, công ty hợp danh và doanh nghiệp tư nhân; quy định về nhóm công ty.",
    "01-2021-ND": "Nghị định này quy định chi tiết về hồ sơ, trình tự, thủ tục đăng ký doanh nghiệp; đăng ký hộ kinh doanh; quy định về Cơ quan đăng ký kinh doanh và quản lý nhà nước về đăng ký doanh nghiệp, đăng ký hộ kinh doanh.\n\n2.",
    "168-2025-ND": "Nghị định này quy định về hồ sơ, trình tự, thủ tục đăng ký doanh nghiệp; quy định về đăng ký và hoạt động của hộ kinh doanh; quy định việc liên thông thủ tục đăng ký doanh nghiệp, đăng ký hộ kinh doanh; đăng ký doanh nghiệp, đăng ký hộ kinh doanh qua mạng thông tin điện tử; cung cấp thông tin đăng ký doanh nghiệp, đăng ký hộ kinh doanh, khai thác và chia sẻ thông tin doanh nghiệp; quy định về cơ quan đăng ký kinh doanh đối với doanh nghiệp, hộ kinh doanh và quản lý nhà nước về đăng ký doanh nghi.",
    "36-2024-QH15": "Luật này quy định về quy tắc, phương tiện, người tham gia giao thông đường bộ, chỉ huy, điều khiển, tuần tra, kiểm soát, giải quyết tai nạn giao thông đường bộ, trách nhiệm quản lý nhà nước và trách nhiệm của cơ quan, tổ chức, cá nhân có liên quan đến trật tự, an toàn giao thông đường bộ.",
    "168-2024-ND-CP": "Nghị định này quy định về:\n\na) Xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ bao gồm: hành vi vi phạm hành chính; hình thức, mức xử phạt, biện pháp khắc phục hậu quả đối với từng hành vi vi phạm hành chính; thẩm quyền lập biên bản, thẩm quyền xử phạt, mức phạt tiền cụ thể theo từng chức danh đối với hành vi vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ;\n\nb) Mức trừ điểm giấy phép lái xe đối với từng hành vi vi .",
    "100-2019-ND-CP": "Nghị định này quy định về hành vi vi phạm hành chính; hình thức, mức xử phạt, biện pháp khắc phục hậu quả đối với từng hành vi vi phạm hành chính; thẩm quyền lập biên bản, thẩm quyền xử phạt, mức phạt tiền cụ thể theo từng chức danh đối với hành vi vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt.\n\n2.",
}

ORIGINAL_TITLES = {
    "01-2021-ND": "Nghị định 01/2021/NĐ-CP đăng ký doanh nghiệp",
    "168-2025-ND": "Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp",
    "100-2019-ND-CP": "Nghị định 100/2019/NĐ-CP xử phạt vi phạm hành chính lĩnh vực giao thông đường bộ và đường sắt",
}

# Routing prefixes (will be prepended to scope)
SCOPE_PREFIXES = {
    "59-2020-QH14": "[LUẬT GỐC - ưu tiên khi hỏi về quyền, nghĩa vụ, cơ cấu tổ chức DN] ",
    "01-2021-ND": "[NĐ hướng dẫn thủ tục ĐKDN - đã thay bằng NĐ 168/2025] ",
    "168-2025-ND": "[NĐ hướng dẫn thủ tục ĐKDN - hiện hành, thay NĐ 01/2021] ",
    "36-2024-QH15": "[LUẬT GỐC - ưu tiên khi hỏi về quy tắc GT, GPLX, điều kiện xe] ",
    "168-2024-ND-CP": "[NĐ xử phạt GT - tra khi hỏi mức phạt cụ thể, trừ điểm GPLX] ",
    "100-2019-ND-CP": "[NĐ xử phạt GT cũ - chỉ còn phần đường sắt] ",
}

LOAI_MAP = {
    "59-2020-QH14": "Luật",
    "36-2024-QH15": "Luật",
    "01-2021-ND": "Nghị định",
    "47-2021-ND": "Nghị định",
    "23-2022-ND": "Nghị định",
    "65-2022-ND": "Nghị định",
    "16-2023-ND": "Nghị định",
    "89-2024-ND": "Nghị định",
    "168-2024-ND-CP": "Nghị định",
    "100-2019-ND-CP": "Nghị định",
    "44-2025-ND": "Nghị định",
    "168-2025-ND": "Nghị định",
    "248-2025-ND": "Nghị định",
}

for doc_id, doc in ds.items():
    # Fix loai_van_ban
    if doc_id in LOAI_MAP:
        doc["loai_van_ban"] = LOAI_MAP[doc_id]

    # Restore original scope and prepend prefix
    if doc_id in ORIGINAL_SCOPES:
        doc["scope"] = SCOPE_PREFIXES.get(doc_id, "") + ORIGINAL_SCOPES[doc_id]

    # Restore original titles
    if doc_id in ORIGINAL_TITLES:
        doc["ten_van_ban"] = ORIGINAL_TITLES[doc_id]

with open("data/kg_enhanced/document_summaries.json", "w", encoding="utf-8") as f:
    json.dump(ds, f, ensure_ascii=False, indent=2)
print("Fixed document_summaries.json")

# === Fix chapter_summaries.json ===
cs = json.load(open("data/kg_enhanced/chapter_summaries.json", "r", encoding="utf-8"))

# For chapter descriptions: restore original format (article_range + keywords)
# then prepend a short routing tag
CHAPTER_TAGS = {
    "59-2020-QH14": "[LUẬT GỐC 59/2020] ",
    "36-2024-QH15": "[LUẬT GỐC 36/2024] ",
    "168-2024-ND-CP": "[NĐ 168/2024 xử phạt] ",
    "100-2019-ND-CP": "[NĐ 100/2019 xử phạt cũ] ",
    "01-2021-ND": "[NĐ 01/2021 thủ tục ĐKDN - hết hiệu lực] ",
    "168-2025-ND": "[NĐ 168/2025 thủ tục ĐKDN - hiện hành] ",
}

for ch_id, ch in cs.items():
    doc_prefix = ch_id.split(":")[0]
    tag = ""
    for doc_id, t in CHAPTER_TAGS.items():
        if doc_prefix == doc_id:
            tag = t
            break

    # Restore description to: tag + article_range + ". Nội dung: " + keywords
    ar = ch.get("article_range", "")
    kw = ch.get("keywords", "")
    if kw and ar:
        ch["description"] = f"{tag}{ar}. Nội dung: {kw}"
    elif kw:
        ch["description"] = f"{tag}Nội dung: {kw}"

with open("data/kg_enhanced/chapter_summaries.json", "w", encoding="utf-8") as f:
    json.dump(cs, f, ensure_ascii=False, indent=2)
print("Fixed chapter_summaries.json")
print("Done!")
