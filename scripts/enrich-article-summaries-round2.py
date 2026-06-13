"""Round 2 enrichment: append keywords for newly-missed articles (Run 3 misses).

APPEND only, never replace. Skip STT 279 (vague spam query expecting ~50 governance
articles — unanswerable). Targets articles not covered in round 1.
"""
import json

ENRICHMENT = {
    # === 59-2020-QH14 — registration / formation ===
    "59-2020-QH14:d22": "hồ sơ thành lập công ty cổ phần, không cần nộp CCCD người đại diện, không cần CCCD người ủy quyền, hồ sơ thành lập mới CP, đăng ký mã ngành kho bãi, ngành nghề kinh doanh có điều kiện",
    "59-2020-QH14:d29": "mã số doanh nghiệp đồng thời mã số thuế, bao lâu nhận bản chính giấy chứng nhận, nhận ERC có chữ ký, mã số thuế khi thành lập, đăng ký mã ngành kho bãi",
    "59-2020-QH14:d44": "địa điểm kinh doanh bị từ chối, đăng ký địa điểm kinh doanh, kho bãi ở mảnh đất khác, địa điểm ở nơi khác trụ sở, lập địa điểm kinh doanh, chi nhánh văn phòng đại diện",
    "59-2020-QH14:d47": "tăng vốn điều lệ cần giấy xác nhận góp vốn, giấy chứng nhận phần vốn góp, file mẫu giấy xác nhận góp vốn, hồ sơ tăng vốn TNHH 2TV, thông tin góp vốn thực tế, scan hồ sơ tăng vốn",
    "59-2020-QH14:d25": "danh sách thành viên khi chuyển HKD lên TNHH 2TV, khối thông tin về thành viên, danh sách cổ đông sáng lập, hồ sơ chuyển hộ kinh doanh sang công ty, nhập thông tin thành viên trên hệ thống",
    "59-2020-QH14:d24": "điều lệ công ty khi chuyển HKD lên TNHH, hồ sơ chuyển hộ kinh doanh gồm điều lệ, mẫu điều lệ công ty 2 thành viên",
    "59-2020-QH14:d8": "nghĩa vụ doanh nghiệp sau khi thành lập, sau khi có giấy phép cần làm gì bên thuế, nghĩa vụ thuế sau chuyển đổi HKD, khai thuế ban đầu, nghĩa vụ kê khai khi chuyển trụ sở",
    "59-2020-QH14:d74": "chuyển HKD thành công ty TNHH một thành viên, TNHH MTV ở 2 quận khác nhau, hồ sơ chuyển đổi sang TNHH MTV",
    "59-2020-QH14:d27": "bị ra thông báo khi nộp hồ sơ thành lập CP, cấp giấy chứng nhận sau thông báo, chuyển loại hình khi đang có thông báo",
    "59-2020-QH14:d28": "bao lâu nhận bản chính có chữ ký phòng ĐKKD, nội dung giấy chứng nhận đăng ký, không có bản giấy chỉ có bản điện tử",
    # === 36-2024-QH15 — traffic law prohibited acts / enforcement ===
    "36-2024-QH15:d9": "nháy pha báo hiệu CSGT, ra hiệu cảnh báo có công an, hành vi bị nghiêm cấm giao thông, CSGT yêu cầu dừng xe đo nồng độ cồn khi không vi phạm, cản trở người thi hành công vụ",
    "36-2024-QH15:d67": "app tra cứu phạt nguội, biện pháp phát hiện vi phạm giao thông, thiết bị kỹ thuật nghiệp vụ, CSGT dừng xe kiểm tra nồng độ cồn, tra cứu vi phạm trên ứng dụng, camera giám sát giao thông",
    "36-2024-QH15:d62": "lấy lại giấy phép lái xe bị giữ, cấp lại GPLX sau vi phạm, thu hồi giấy phép lái xe, đổi cấp lại bằng lái",
    # === 168-2024-ND-CP — penalty procedures ===
    "168-2024-ND-CP:d4": "biên bản vi phạm từ năm 2022 còn hiệu lực không, thời hiệu xử phạt vi phạm giao thông, lấy lại GPLX sau khi bị lập biên bản lâu, vi phạm cũ còn bị phạt không, lỗi lấn làn từ lâu",
    "168-2024-ND-CP:d14": "xuất trình giấy tờ tại nơi tạm giữ xe, có được tính không mang đăng ký xe, không mang giấy phép lái xe xe máy, điều kiện phương tiện xe máy",
    "168-2024-ND-CP:d41": "trụ sở CSGT chuyển về đâu, phân định thẩm quyền xử phạt, CSGT huyện xử phạt, nộp phạt ở đâu",
    "168-2024-ND-CP:d43": "thẩm quyền xử phạt công an, CSGT huyện trụ sở, nơi nộp phạt vi phạm, công an xử phạt giao thông",
    "168-2024-ND-CP:d12": "nháy pha cảnh báo CSGT, sử dụng đèn pha báo hiệu, hành vi khác vi phạm quy tắc giao thông, dùng lòng đường vỉa hè sai mục đích",
    "168-2024-ND-CP:d47": "app tra cứu phạt nguội, thủ tục xử phạt chủ phương tiện, nguyên tắc xử phạt phạt nguội, tra cứu vi phạm phương tiện",
    "168-2024-ND-CP:d5": "CSGT dừng xe đo nồng độ cồn, tước quyền sử dụng giấy phép có thời hạn, kiểm tra nồng độ cồn khi không vi phạm",
}

arts = json.load(open("data/kg_enhanced/article_summaries.json", "r", encoding="utf-8"))

updated = 0
for aid, extra in ENRICHMENT.items():
    if aid not in arts:
        print(f"WARNING: {aid} not found")
        continue
    current = arts[aid].get("keywords", "")
    if extra[:25] in current:
        print(f"SKIP (already): {aid}")
        continue
    arts[aid]["keywords"] = current.rstrip().rstrip(".") + ", " + extra
    updated += 1
    print(f"Enriched: {aid}")

with open("data/kg_enhanced/article_summaries.json", "w", encoding="utf-8") as f:
    json.dump(arts, f, ensure_ascii=False, indent=2)
print(f"\nDone — {updated} articles enriched (round 2)")
