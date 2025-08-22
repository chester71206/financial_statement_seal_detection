# financial_statement_seal_detection

![detected_seal_result](https://github.com/user-attachments/assets/785700af-6a4c-4fd7-8f0b-28b16aa14c5c)
傳統上，在印章與文字顏色分割清楚的情況，我們可以將圖片用HSV空間讀取，藉由設定H值分割印章顏色與文字顏色(因為顏色的RGB會受亮度影響，而HSV的顏色判斷並不會受亮度影響)，我們可以將印章與文字區別開來，準確率可達100%


然而財報經過影印機黑白掃描後，便難以區分印章與文字的顏色，只能透過印章的輪廓去取得印章，需先做一次高斯模糊去除雜訊，再對每個點做擴張，確保印章足夠緊密，接下來透過HoughLinesP偵測過長的直線，避免偵測到圖表，再透過一次HoughLinesP偵測叫短的直線，將偵測到相鄰的短直線合併，計算圍起來的面積，與長寬比，如果符合域值才會被保留，如下圖
![ezgif com-animated-gif-maker (1)](https://github.com/user-attachments/assets/9088fa0e-7295-4baf-b25f-d0fbcb31e2d4)

即使經過重重關卡，還是會有誤判的情況，驗證集準確率約為95%，因此透過cv2的結果經過篩選後，當成訓練資料，訓練paddledetection與yolo，驗證集準確率提高至99%
