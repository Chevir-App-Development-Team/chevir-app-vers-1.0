// MediaPipe Hands standart 21-nöqtə sxemi. İndekslər:
//  0      bilək (wrist)
//  1-4    baş barmaq: CMC, MCP, IP, TIP
//  5-8    işarə barmağı: MCP, PIP, DIP, TIP
//  9-12   orta barmaq: MCP, PIP, DIP, TIP
//  13-16  adsız barmaq: MCP, PIP, DIP, TIP
//  17-20  çeçələ barmaq: MCP, PIP, DIP, TIP
//
// Koordinatlar təxminidir (real AzSLD məlumatı hələ yoxdur) - məqsəd yalnız
// anatomik cəhətdən inandırıcı barmaq uzunluğu/açı nisbətlərini saxlamaqdır,
// dəqiq ölçü deyil. Real gest məlumatı gələndə bu massivlər eyni formatda
// (21 x [x, y, z]) əvəz olunmalıdır - qalan kod (hand3d.js) dəyişmədən qalır.
// Vahid: normallaşdırılmış lokal ölçü, bilək mərkəzdə (0,0,0), ovuc +z-ə baxır.

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // baş barmaq
  [0, 5], [5, 6], [6, 7], [7, 8], // işarə
  [5, 9], [9, 10], [10, 11], [11, 12], // orta
  [9, 13], [13, 14], [14, 15], [15, 16], // adsız
  [13, 17], [17, 18], [18, 19], [19, 20], // çeçələ
  [0, 17], // ovucun bağlanması (bilək - çeçələ kökü)
]

// Açıq əl - bütün barmaqlar açıq və yayılıb.
export const HAND_POSE_OPEN = [
  [0, 0, 0],
  [0.25, 0.15, 0.1], [0.42, 0.32, 0.18], [0.55, 0.48, 0.22], [0.65, 0.62, 0.25],
  [0.35, 0.55, 0.05], [0.38, 0.78, 0.05], [0.4, 0.93, 0.04], [0.41, 1.05, 0.03],
  [0.12, 0.6, 0.05], [0.13, 0.85, 0.05], [0.14, 1.02, 0.04], [0.15, 1.16, 0.03],
  [-0.12, 0.58, 0.03], [-0.13, 0.82, 0.03], [-0.14, 0.98, 0.02], [-0.15, 1.1, 0.01],
  [-0.33, 0.5, 0], [-0.34, 0.72, -0.02], [-0.36, 0.85, -0.04], [-0.37, 0.95, -0.05],
]

// "Salam" pozası - hazırkı AzSL uyğunluğu təsdiqlənməyib, sonra dəyişəcək.
// Açıq əl (HAND_POSE_OPEN) bazasından bilək ətrafında yüngül fırlanmış iki
// variant yaradılır; hand3d.js-nin mövcud poza-keçid mexanizmi (dəyişdirilməyib)
// bunlar arasında morflayaraq yellənmə effekti verir.
function rotateAroundWrist(pose, degrees) {
  const angle = (degrees * Math.PI) / 180
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  return pose.map(([x, y, z]) => [x * cos + z * sin, y, -x * sin + z * cos])
}

export const HAND_POSE_WAVE_LEFT = rotateAroundWrist(HAND_POSE_OPEN, -10)
export const HAND_POSE_WAVE_RIGHT = rotateAroundWrist(HAND_POSE_OPEN, 10)

export const HAND_POSES = [HAND_POSE_WAVE_LEFT, HAND_POSE_WAVE_RIGHT]
