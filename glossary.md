# Glossary — Từ điển thuật ngữ Orbit Wars (Anh → Việt)

Giải thích các từ tiếng Anh hay gặp trong game & code, kèm ví dụ.

---

## 1. Các vật thể trong game

| Từ tiếng Anh | Tiếng Việt | Giải thích |
|---|---|---|
| **Planet** | Hành tinh | Vật thể đứng yên hoặc quay quanh sun. Bạn đứng trên planet để đẻ ships và phóng fleets. |
| **Sun** | Mặt trời | Ngôi sao ở tâm map (vị trí 50,50), bán kính 10. Fleet nào bay qua là **chết sạch**. |
| **Fleet** | Hạm đội | Nhóm ships đang bay từ planet này sang planet khác. Bay theo đường thẳng, không đổi hướng được. |
| **Ship** | Phi thuyền (tàu) | Đơn vị cơ bản. Đếm số ship để tính điểm và tính sức mạnh combat. |
| **Comet** | Sao chổi | Planet **tạm thời**, bay qua map theo quỹ đạo elip. Cũng chiếm được, đẻ 1 ship/turn. Hết đường bay thì biến mất (ships trên đó mất luôn). |
| **Home planet** | Hành tinh nhà | Planet xuất phát của bạn, bắt đầu với 10 ships. |
| **Neutral planet** | Hành tinh trung lập | Planet chưa có ai sở hữu (owner = -1). |
| **Board** | Bản đồ / sân chơi | Khu vực 100×100 để chơi. Fleet bay ra ngoài là mất. |

---

## 2. Thuộc tính (attributes) của planet

Planet = `[id, owner, x, y, radius, ships, production]`

| Từ | Nghĩa | Ví dụ |
|---|---|---|
| **id** | Mã định danh | Số nguyên 0, 1, 2... để nhận diện planet |
| **owner** | Chủ sở hữu | `-1` = neutral, `0/1/2/3` = player 0/1/2/3 |
| **x, y** | Tọa độ | Vị trí trên bản đồ, kiểu (50.3, 42.1) |
| **radius** | Bán kính (kích thước) | Planet càng to càng dễ bị trúng, công thức: `1 + ln(production)` |
| **ships** | Số phi thuyền đang đóng trên planet | Đây chính là **garrison** (đội phòng thủ) |
| **production** | Sản lượng / tốc độ đẻ ships | Số nguyên 1-5. Mỗi turn planet bạn sở hữu sinh ra từng này ships |
| **garrison** | **Đội đồn trú** = số ships đang đứng trên planet | Khi có fleet địch đến thì garrison đánh lại |

**Ví dụ**: Planet `[3, 1, 70.0, 30.0, 1.69, 25, 2]`
→ Planet id=3, thuộc player 1, ở (70, 30), bán kính 1.69, đang có **25 ships (garrison)**, mỗi turn đẻ thêm **2 ships**.

---

## 3. Thuộc tính của fleet

Fleet = `[id, owner, x, y, angle, from_planet_id, ships]`

| Từ | Nghĩa |
|---|---|
| **angle** | Góc bay (radian) | 0 = sang phải, π/2 = xuống, π = sang trái |
| **from_planet_id** | Id của planet xuất phát | |
| **ships** | Số phi thuyền trong hạm đội | Không đổi suốt đường bay |

---

## 4. Cơ chế di chuyển (movement)

| Từ tiếng Anh | Tiếng Việt | Giải thích |
|---|---|---|
| **Speed** | Tốc độ | Số units/turn fleet di chuyển. Fleet càng nhiều ships bay càng nhanh (công thức log, xem README) |
| **Orbit / Orbiting** | Quay quanh sun | Planet ở gần tâm tự quay quanh sun mỗi turn |
| **Angular velocity** | Vận tốc góc | Tốc độ quay của planet (radian/turn), từ 0.025-0.05. Mỗi game random một giá trị |
| **Static** | Tĩnh (đứng yên) | Planet xa tâm, không quay |
| **Trajectory / path** | Quỹ đạo / đường bay | Đường mà comet hoặc fleet đi theo |
| **ETA** (Estimated Time of Arrival) | Thời gian đến dự kiến | Sau bao nhiêu turn nữa fleet sẽ tới mục tiêu |
| **Out of bounds** | Bay ra ngoài map | Fleet mất khi vượt ranh giới 100×100 |

---

## 5. Combat (chiến đấu)

| Từ | Tiếng Việt | Giải thích |
|---|---|---|
| **Combat** | Giao tranh / trận đánh | Khi fleet chạm vào planet |
| **Garrison** | **Quân đồn trú / lực lượng đóng trên planet** | Chính là số ships đang đứng trên planet. Đánh lại fleet tấn công |
| **Attacker** | Bên tấn công | Fleet bay đến |
| **Capture** | Chiếm được planet | Khi attacker > garrison, planet đổi chủ |
| **Takeover** | Đoạt lấy | Đồng nghĩa capture |
| **Sweep** | Quét trúng | Planet đang quay "chạm vào" fleet đang bay qua → combat |
| **Collision** | Va chạm | Fleet chạm planet hoặc sun |
| **Continuous collision detection** | Kiểm tra va chạm liên tục | Check cả đoạn đường bay, không chỉ điểm cuối |
| **Surviving / Survivor** | Sống sót | Ships còn lại sau combat |

**Ví dụ combat**:
- Planet có garrison=20 (của player 0)
- Player 1 gửi fleet 30 ships đến
- → 30 - 20 = 10 ships của player 1 sống sót → chiếm planet → garrison mới = 10 (của player 1)

---

## 6. Actions (hành động của agent)

| Từ | Tiếng Việt | Giải thích |
|---|---|---|
| **Move** | Lệnh di chuyển | 1 lệnh phóng fleet: `[from_planet_id, angle, num_ships]` |
| **Launch** | Phóng (fleet) | Gửi ships từ planet đi |
| **Turn / Step** | Lượt | 1 chu kỳ game. Game tổng cộng 500 turns |
| **Action** | Hành động | List các moves agent trả về mỗi turn |
| **Observation (obs)** | Quan sát | Dữ liệu đầu vào agent nhận mỗi turn: planets, fleets, player id... |
| **Agent** | Bot / hàm chơi game | Hàm `agent(obs)` bạn viết |

---

## 7. Thuật ngữ chiến thuật (strategy)

Các từ hay gặp trong notebook top 1:

| Từ tiếng Anh | Tiếng Việt | Giải thích |
|---|---|---|
| **Snipe** | Bắn tỉa | Gửi đúng số ships tối thiểu để chiếm planet (garrison + 1) |
| **Swarm** | Tấn công bầy đàn | Nhiều fleet từ nhiều planet cùng đến 1 target một lúc |
| **Reinforce** | Tăng viện | Gửi ships đến planet nhà để tăng phòng thủ |
| **Rescue** | Giải cứu | Gửi ships đến planet mình **đang sắp mất** để cứu |
| **Recapture** | Chiếm lại | Lấy lại planet đã bị địch chiếm |
| **Capture** | Chiếm lần đầu | Lấy planet neutral hoặc của địch |
| **Hold** | Giữ | Duy trì sở hữu planet trước đợt tấn công |
| **Crash exploit** | Khai thác tình huống hủy diệt | Khi 2 phe đánh nhau tie (hòa) → cả hai chết sạch → nhảy vào chiếm planet trống |
| **Salvage** | Tận dụng / trục vớt | Dùng ships thừa còn lại cho mục đích phụ |
| **Rear funneling** | Chuyển ships từ hậu phương ra tiền tuyến | Gom ships từ planet an toàn ra planet đang chiến đấu |
| **Pressure** | Gây áp lực | Liên tục tấn công để địch không kịp phòng thủ |
| **Commitment** | Cam kết (phóng) | Sau khi đã quyết định phóng fleet, cập nhật "future state" |
| **Legal shot** | Đường bắn hợp lệ | Góc bắn không cắt qua sun |
| **Doomed planet** | Planet sắp mất | Planet không thể cứu được nữa (đừng phí ships) |
| **Future state** | Trạng thái tương lai | Dự đoán map sẽ thế nào sau N turn |
| **Forecast** | Dự báo | Tính trước ai sẽ sở hữu planet tại ETA |

---

## 8. Code / thuật ngữ lập trình

| Từ | Giải thích |
|---|---|
| **Tuple / Named tuple** | Bộ giá trị. `Planet(*p)` → truy cập `p.id`, `p.x` thay vì `p[0]`, `p[2]` |
| **atan2(dy, dx)** | Hàm toán tính góc từ điểm này đến điểm kia |
| **hypot(dx, dy)** | Tính khoảng cách Euclid `sqrt(dx² + dy²)` |
| **Notebook** | File `.ipynb` của Jupyter — kết hợp code + text |
| **Kernel** | Notebook đã được publish trên Kaggle |
| **Submission** | Bản nộp (file `main.py` hoặc `.tar.gz`) |
| **Replay** | File JSON ghi lại toàn bộ 1 trận để xem lại |
| **Episode** | 1 trận đấu hoàn chỉnh |
| **Leaderboard** | Bảng xếp hạng |
| **Baseline** | Giải pháp đơn giản làm nền, để so sánh với phiên bản cải tiến |

---

## 9. Các từ hay lặp lại trong notebook v11

- **Arrival-time ownership** = Tính xem "đến thời điểm fleet đến nơi thì planet đang thuộc về ai"
- **Reinforce-to-hold** = Tăng viện để giữ planet
- **Multi-source** = Nhiều planet cùng bắn về 1 target
- **Synchronized arrival** = Tính toán để các fleet đến cùng lúc (đỡ bị đánh lẻ)
- **Follow-up** = Đợt tấn công tiếp theo (sau khi đòn đầu đã thành công)
- **Tie** = Hòa (2 phe ngang ships → cả hai chết sạch)

---

## Mẹo đọc code

Khi thấy từ lạ trong code/notebook, tra bảng trên trước. Nếu vẫn khó hiểu, hỏi lại tôi — tôi sẽ giải thích kèm ví dụ.
