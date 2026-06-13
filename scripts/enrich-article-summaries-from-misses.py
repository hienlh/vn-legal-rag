"""Enrich article summary keywords with practical terms from missed benchmark queries.

Human-in-the-loop KG refinement: APPEND query-derived keywords to existing
keywords (never replace). Only touches article summaries → Loop 2 prompts
for affected chapters; Loop 0/1 caches stay valid.
"""
import json

ENRICHMENT = {
    # === 59-2020-QH14 (Luật Doanh nghiệp) ===
    "59-2020-QH14:d30": "cập nhật mã ngành khi thay đổi, thay đổi nhiều nội dung cùng 1 lần, gộp thay đổi chung 1 hồ sơ, sửa hồ sơ nộp sai, hủy nộp lại hồ sơ, chuyển trụ sở khác tỉnh, mẫu 09, phòng ĐKKD chưa chấp thuận, bản PDF không có mộc, giá trị pháp lý bản điện tử, đổi tên công ty kèm đổi chủ sở hữu",
    "59-2020-QH14:d26": "hồ sơ bị ra thông báo, nộp sai phải sửa hay hủy, giấy chứng nhận ĐKDN không có dấu, kết quả trả qua email, bản điện tử, thông báo hồ sơ được chấp thuận, chuyển đổi HKD thành công ty, nộp hồ sơ thành lập mới khi đang có thông báo",
    "59-2020-QH14:d21": "chuyển HKD lên công ty TNHH, hộ kinh doanh chuyển thành TNHH MTV, hồ sơ chuyển đổi hộ kinh doanh, không cần nộp CCCD, thông tin về thành viên, hồ sơ thành lập TNHH 2 thành viên, HKD lên DN 2TV, hồ sơ gồm những gì",
    "59-2020-QH14:d12": "thay đổi người đại diện pháp luật, ai ký giấy đề nghị thay đổi, đại diện PL cũ ký hay mới ký, danh sách chủ sở hữu hưởng lợi, người đại diện mới hay cũ ký, ủy quyền ký hồ sơ, chủ tịch ký",
    "59-2020-QH14:d31": "thay đổi nhiều nội dung cùng lúc, Mẫu 2, cập nhật mã ngành theo Quyết định 36, thay đổi địa chỉ và đại diện và thành viên 1 lần, chuyển đổi loại hình kèm thay đổi địa chỉ, đăng ký địa điểm kinh doanh bị từ chối",
    "59-2020-QH14:d4": "ERC bản giấy, ERC bản điện tử, chữ ký số trên giấy chứng nhận, giấy chứng nhận không có dấu mộc, giá trị pháp lý bản điện tử, Phòng đăng ký kinh doanh còn cấp bản giấy không",
    "59-2020-QH14:d52": "chuyển nhượng vốn góp công ty 1TV thành 2TV, tiền chuyển nhượng chuyển cho cá nhân hay công ty, thanh toán khi chuyển nhượng vốn, bên nhận chuyển tiền cho bên bán, giữ nguyên vốn điều lệ",
    "59-2020-QH14:d207": "đóng cửa công ty, công ty không hoạt động lâu năm muốn giải thể, không đóng thuế giờ muốn đóng công ty, có bị phạt khi giải thể không, công ty tạm ngưng lâu, giải thể chi nhánh hạch toán độc lập, không hoạt động tại nơi đăng ký",
    "59-2020-QH14:d45": "giải thể chi nhánh, chấm dứt hoạt động chi nhánh, hồ sơ giải thể hay chấm dứt hoạt động, thành lập công ty con, công ty mẹ góp vốn công ty con, đăng ký địa điểm kinh doanh bị từ chối",
    "59-2020-QH14:d209": "bị cảnh báo không hoạt động tại địa chỉ, đã cập nhật trạng thái đang hoạt động, hệ thống chỉ hiện giải thể và xóa tên, khôi phục hoạt động sau cảnh báo, công ty bị khóa trên hệ thống",
    "59-2020-QH14:d46": "chuyển HKD lên DN 2 thành viên, khối thông tin về thành viên, công ty con 100% vốn công ty mẹ, thành lập công ty TNHH 2 thành viên từ hộ kinh doanh",
    "59-2020-QH14:d79": "công ty con do tổ chức làm chủ sở hữu, công ty mẹ CTCP góp 100% vốn, cơ cấu công ty con một thành viên, hồ sơ thành lập công ty con",
    "59-2020-QH14:d34": "tài sản góp vốn khi chuyển đổi, góp vốn bằng tiền mặt hay chuyển khoản, định giá khi chuyển đổi loại hình",
    "59-2020-QH14:d35": "chuyển quyền sở hữu tài sản góp vốn, góp vốn vào công ty mới chuyển đổi, thời hạn chuyển quyền sở hữu",
    "59-2020-QH14:d127": "chuyển nhượng cổ phần cho người ngoài, cổ đông sáng lập chuyển nhượng, thoái vốn cổ phần, bán cổ phần cho ai",
    "59-2020-QH14:d32": "thông báo thay đổi cổ đông nước ngoài, thay đổi thông tin cổ đông, cập nhật sổ cổ đông",
    "59-2020-QH14:d68": "tăng vốn điều lệ TNHH, giấy xác nhận góp vốn khi tăng vốn, thủ tục tăng vốn công ty TNHH, giảm vốn điều lệ, hoàn trả vốn góp",
    # === 168-2024-ND-CP (NĐ xử phạt giao thông) ===
    "168-2024-ND-CP:d18": "bằng lái xe sắp hết hạn, bằng lái hết hạn trên 1 ngày, đổi bằng lái cần gì, xuất trình giấy tờ sau khi bị giữ xe, có được tính không mang giấy tờ, bị trừ hết điểm GPLX có được lái xe không, gia hạn giấy phép lái xe",
    "168-2024-ND-CP:d7": "tốc độ tối đa của xe máy, xe máy chạy quá tốc độ trên 20 km/h, trừ bao nhiêu điểm bằng lái, lỗi không bật đèn xi nhan xe máy, mức phạt xi nhan xe gắn máy, không xi nhan khi chuyển hướng, phạt bao nhiêu tiền 2026",
    "168-2024-ND-CP:d6": "đỗ xe trên vỉa hè đúng cách, giới hạn thời gian đỗ xe, đặt biển cảnh báo trên cao tốc, đèn đỏ chữ thập có được đi không, đỗ xe ô tô nơi nào bị cấm, dừng đỗ trên đường cao tốc",
}

arts = json.load(open("data/kg_enhanced/article_summaries.json", "r", encoding="utf-8"))

updated = 0
for aid, extra in ENRICHMENT.items():
    if aid not in arts:
        print(f"WARNING: {aid} not found in article_summaries.json")
        continue
    current = arts[aid].get("keywords", "")
    if extra[:30] in current:
        print(f"SKIP (already enriched): {aid}")
        continue
    arts[aid]["keywords"] = current.rstrip().rstrip(".") + ", " + extra
    updated += 1
    print(f"Enriched: {aid}")

with open("data/kg_enhanced/article_summaries.json", "w", encoding="utf-8") as f:
    json.dump(arts, f, ensure_ascii=False, indent=2)

print(f"\nDone — {updated} articles enriched")
