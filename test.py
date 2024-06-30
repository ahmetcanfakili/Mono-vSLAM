import cv2
import numpy as np

# Kamera nesnesini başlat
cap = cv2.VideoCapture(0)  # 0, dahili kamerayı temsil eder. Farklı bir indeks kullanarak harici kamerayı seçebilirsiniz.

# ORB dedektörünü oluştur
orb = cv2.ORB_create(nfeatures=1000)

# İlk kareyi okuma
ret, old_frame = cap.read()
if not ret:
    print("Kamera açılmadı")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(old_gray, None)

# Kameranın başlangıç pozisyonu
pose = np.eye(4)

# Boş bir siyah görüntü oluştur
trajectory = np.zeros((600, 800, 3), dtype=np.uint8)

# Pencereleri oluşturma
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('trajectory', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü griye dönüştürme
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Özellik çıkarma ve eşleştirme
    kp2, des2 = orb.detectAndCompute(frame_gray, None)
    
    if des2 is None:
        continue
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Eşleşen noktaları çıkarma
    if len(matches) >= 8:  # Minimum eşleşme sayısını artırma
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Hareket tahmini
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts)
        
        if mask_pose.sum() > 8:  # Yeterli sayıda pozitif eşleşme olduğundan emin olma
            # Pozisyon güncelleme
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t.flatten()
            pose = pose @ transformation
            
            # Görüntü ve özellikleri güncelleme
            old_gray = frame_gray.copy()
            kp1 = kp2
            des1 = des2
            
            # Kameranın pozisyonunu alın
            x, z = pose[0, 3], pose[2, 3]
            
            # Pozisyonu çizgi olarak çizme
            cv2.circle(trajectory, (int(x) + 400, int(z) + 100), 1, (255, 0, 0), 2)
        
    # Görüntüyü gösterme
    cv2.imshow('frame', frame)
    
    # Çizgiyi gösterme
    cv2.imshow('trajectory', trajectory)
    
    # 'q' tuşuna basıldığında döngüden çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera nesnesini ve pencereleri serbest bırakma
cap.release()
cv2.destroyAllWindows()
